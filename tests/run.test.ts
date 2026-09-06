import { expect, test } from 'vitest';
import { IDEAL_MMA } from './fixtures';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { evaluateDecodeAtBatch } from '../src/core/engine/sim/run/decode';
import { memoryFootprint } from '../src/core/engine/sim/run/memory';
import { partitionIntoStages } from '../src/core/engine/sim/lowering/stages';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { naiveOpCost } from '../src/core/engine/sim/cost/helpers/naiveOpCost';
import { OpId } from '../src/core/engine/sim/ir/ops';
import { localElems, tt } from '../src/core/engine/sim/ir/tensors';
import { runnableOn, validateInput } from '../src/core/engine/sim/run/validate';
import { Deployment, makeMesh, MoeDispatch } from '../src/core/engine/surface/deploy';
import { ChipSpec, CHIPS, CHIPS_BY_ID, peakFlops, runsAs } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { gqa } from '../src/core/model/block/attn';
import { MlpConfig, moeMlp } from '../src/core/model/block/mlp';
import {
  ALL_BF16,
  BF16_ALL,
  DTYPES,
  DTYPE_BYTES,
  FP8_ALL,
  PrecisionSpec,
  type Dtype,
} from '../src/core/model/dtype';
import { ModelSpec, MODEL_PRESETS } from '../src/core/model/models';
import { flopsPerPrefillToken, kvBytesPerSeq, weightBytesTotal } from '../src/core/model/utils';
import { matmulSeconds } from '../src/core/engine/roofline';

const backend = makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 });
const h100 = CHIPS_BY_ID['h100-sxm'];
const hbm = h100.hbmBandwidth * h100.realizableHbmBwFrac;
const idealTiles: ChipSpec = { ...h100, mmaShapes: IDEAL_MMA };

test('validateInput gates weights past HBM capacity', () => {
  const kimi = MODEL_PRESETS.find((m) => m.name.startsWith('Kimi K2.6'))!;
  const diags = validateInput({
    model: kimi,
    deployment: singleChip(),
    workload: { prefillLen: 128, generateLen: 0 },
  });
  expect(diags.map((x) => x.code)).toContain('weights-dont-fit');
});

test('prefill capacity counts DPA groups and pipeline microbatches', () => {
  const model = MODEL_PRESETS.find((m) => m.name === 'LLaMA 3 8B')!;
  const workload = { prefillLen: 512, generateLen: 256 };
  const axes = deployedAxes(h100.interconnect, { domain: 8, nodes: 1 });
  const mesh = makeMesh(
    axes,
    { DPA: ['D1'], TP: [], EP: [], ETP: ['D1'], PP: ['D0'] },
    { D: [2, 4] },
  );
  const deployment = (chip: ChipSpec): Deployment => ({
    chip,
    mesh,
    moeDispatch: 'ring-of-experts',
  });
  const stages = partitionIntoStages(model, 2);
  const perStage = stages.map((stage) =>
    memoryFootprint(model, deployment(idealTiles), [stage], workload.prefillLen),
  );
  const chipWithCapacity = (residentSeqs: number): ChipSpec => ({
    ...idealTiles,
    hbmCapacity: Math.max(
      ...perStage.map(
        (memory) => memory.weightBytesPerChip + residentSeqs * memory.kvBytesPerSeqPerChip,
      ),
    ),
  });
  const input = { model, deployment: deployment(chipWithCapacity(2)), workload };

  expect(memoryFootprint(model, input.deployment, stages, workload.prefillLen)).toMatchObject({
    maxResidentSeqsPerChip: 2,
  });
  expect(
    memoryFootprint(model, input.deployment, stages, workload.prefillLen + workload.generateLen),
  ).toMatchObject({ maxResidentSeqsPerChip: 1 });
  expect(evaluatePrefill(input, 4, 'throughput', { costBackend: backend }).ok).toBe(true);

  const overflow = evaluatePrefill(input, 8, 'throughput', { costBackend: backend });
  expect(overflow.ok).toBe(false);
  expect(overflow.diags.map((diag) => diag.code)).toContain('kv-no-room');
  expect(
    evaluatePrefill(input, 8, 'throughput', { costBackend: backend, ignoreKvCapacity: true }).ok,
  ).toBe(true);

  const ttftInput = { model, deployment: deployment(chipWithCapacity(1)), workload };
  expect(memoryFootprint(model, ttftInput.deployment, stages, workload.prefillLen)).toMatchObject({
    maxResidentSeqsPerChip: 1,
  });
  expect(evaluatePrefill(ttftInput, 1, 'ttft', { costBackend: backend }).ok).toBe(true);
});

test('a width the chip has no unit for widens to one it has, with a warning', () => {
  // Kimi K3 asks for MXFP8 expert activations; a TPU v6e has a bf16 matmul
  // unit and nothing narrower, so those GEMMs upconvert exactly as a stack
  // without an fp8 path does. The config stays priceable and says so rather
  // than being refused.
  const k3 = MODEL_PRESETS.find((m) => m.name.startsWith('Kimi K3'))!;
  const v6e = CHIPS_BY_ID['tpu-v6e'];
  expect(v6e.formats.mxfp8).toBeUndefined();
  expect(runsAs(v6e, 'mxfp8')).toBe('bf16');

  const diags = validateInput({
    model: k3,
    deployment: { ...singleChip(), chip: v6e },
    workload: { prefillLen: 128, generateLen: 0 },
  });
  expect(diags.filter((d) => d.severity === 'error').map((d) => d.code)).toEqual([
    'weights-dont-fit', // 1.5 TB of weights on one 32 GB chip, but not a dtype refusal
  ]);
  expect(diags.map((d) => d.code)).toContain('dtype-widened');
});

test('weights price packed on every chip, and a missing kernel only warns', () => {
  // gpt-oss ships MXFP4 experts. An H200 serves them through a shipped
  // widening kernel; a v6e has none, but the sim assumes one could be
  // written and prices the checkpoint at its packed size there too. The
  // difference is surfaced, not priced: a known kernel is an info note, a
  // missing one is a warning that serving means writing it.
  const oss = MODEL_PRESETS.find((m) => m.name.startsWith('gpt-oss-120b'))!;
  const [h200, v6e] = [CHIPS_BY_ID['h200-sxm'], CHIPS_BY_ID['tpu-v6e']];
  expect(h200.formats.mxfp4).toBe('kernel-widened');
  expect(v6e.formats.mxfp4).toBeUndefined();

  const bytesOn = (c: ChipSpec) => weightBytesTotal(runnableOn(oss, c));
  expect(bytesOn(h200)).toBe(weightBytesTotal(oss));
  expect(bytesOn(v6e)).toBe(weightBytesTotal(oss));

  const severities = (c: ChipSpec) =>
    validateInput({
      model: oss,
      deployment: { ...singleChip(), chip: c },
      workload: { prefillLen: 128, generateLen: 0 },
    }).map((d) => `${d.severity}:${d.code}`);
  expect(severities(h200)).toEqual(['info:weights-unpacked']);
  expect(severities(v6e)).toContain('warning:weights-kernel-missing');
});

test('a widened width prices exactly as if it had been declared wide', () => {
  // the fallback is a substitution, not an approximation: on a chip with no
  // fp8 unit, asking for fp8 experts has to cost what asking for bf16 costs,
  // in the matmul rate and the activation traffic and the wire payload
  // alike. A fallback that reached only the rate would show up here.
  const bf16Only: ChipSpec = { ...h100, formats: { bf16: peakFlops(h100, 'bf16')! } };
  const mlp = moeMlp({ experts: 8, topK: 2, expertDim: 512 });
  const asksForFp8: PrecisionSpec = {
    ...ALL_BF16,
    activations: { ...BF16_ALL, routedExperts: 'fp8' },
  };

  expect(a2aCost(mlp, 'expanded-a2a', asksForFp8, bf16Only)).toEqual(
    a2aCost(mlp, 'expanded-a2a', ALL_BF16, bf16Only),
  );
  // and on a chip that does have the unit, the two must differ
  expect(a2aCost(mlp, 'expanded-a2a', asksForFp8).comms).toBeLessThan(
    a2aCost(mlp, 'expanded-a2a', ALL_BF16).comms,
  );
});

function singleChip(chip = idealTiles): Deployment {
  return {
    chip,
    mesh: makeMesh(deployedAxes(h100.interconnect, { domain: 1, nodes: 1 }), {
      DPA: [],
      TP: [],
      EP: [],
      ETP: [],
      PP: [],
    }),
    moeDispatch: 'ring-of-experts',
  };
}

// The lowered per-op costs must reproduce the closed-form roofline
// exactly (idealTiles prices utilization out; T is large enough that
// virtually every expert activates).
function prefillCompute(model: (typeof MODEL_PRESETS)[number], T: number): number {
  const r = evaluatePrefill(
    { model, deployment: singleChip(), workload: { prefillLen: T, generateLen: 0 } },
    1,
    'throughput',
    { costBackend: backend },
  );
  if (!r.ok) throw new Error(`eval failed`);
  return r.cost.busy.compute;
}

test('single-chip dense prefill matches the closed-form roofline', () => {
  const model = MODEL_PRESETS.find((m) => m.name === 'LLaMA 3 8B')!;
  const T = 512;
  const ideal = T * matmulSeconds(flopsPerPrefillToken(model, T), h100, h100.realizableFlopsFrac)!;
  expect(prefillCompute(model, T) / ideal).toBeCloseTo(1, 6);
});

test('single-chip MoE prefill matches the closed-form roofline', () => {
  const model = MODEL_PRESETS.find((m) => m.name === 'gpt-oss-120b MXFP4/BF16')!;
  const T = 8192;
  const ideal = T * matmulSeconds(flopsPerPrefillToken(model, T), h100, h100.realizableFlopsFrac)!;
  expect(prefillCompute(model, T) / ideal).toBeCloseTo(1, 6);
});

test('mixed MoE layers use their own routing fraction', () => {
  const base = MODEL_PRESETS.find((m) => m.name === 'LLaMA 3 8B')!;
  const model: ModelSpec = {
    ...base,
    blocks: [
      {
        repeat: 1,
        pattern: [4, 8].map((topK, i) => ({
          count: 1,
          block: {
            attn: base.blocks[0].pattern[0].block.attn,
            mlp: moeMlp({ experts: 64, topK, expertDim: 128 * (i + 1) }),
          },
        })),
      },
    ],
  };
  const r = evaluateDecodeAtBatch(
    { model, deployment: singleChip(), workload: { prefillLen: 1, generateLen: 0 } },
    1,
    1,
    { costBackend: backend },
  );
  if (!r.ok) throw new Error('eval failed');
  const ops = r.perStageTrace[0].flatMap((s) => s.ops);
  expect(
    ops.flatMap((op) => (op.kind === 'gemm' && op.label === 'experts-in' ? [op.groups] : [])),
  ).toEqual([4, 8]);
  expect(
    ops.flatMap((op) =>
      op.kind === 'weight-load' && op.label === 'experts-in-weight' ? [op.loadFraction] : [],
    ),
  ).toEqual([1 / 16, 1 / 8]);
});

test('single-chip dense prefill memory matches weights plus KV writes', () => {
  const model = MODEL_PRESETS.find((m) => m.name === 'LLaMA 3 8B')!;
  const T = 512;
  const r = evaluatePrefill(
    { model, deployment: singleChip(), workload: { prefillLen: T, generateLen: 0 } },
    1,
    'throughput',
    { costBackend: backend },
  );
  if (!r.ok) throw new Error(`eval failed`);
  // activation charges are read off the trace and netted out: the
  // conservation claim here is about weights and KV
  const dims = singleChip().mesh.dims;
  const acts = r.perStageTrace[0].reduce(
    (s, seg) =>
      s +
      seg.repeat *
        seg.ops.reduce(
          (t, op) =>
            op.kind === 'gemm'
              ? t + (localElems(op.x, dims) + localElems(op.out, dims)) * DTYPE_BYTES[op.dtype]
              : t,
          0,
        ),
    0,
  );

  // the embedding gather is not lowered, so one embedding copy is not read
  const inputEmb = model.vocab * model.modelDim * DTYPE_BYTES[model.precision.weights.embeddings];
  const bytes =
    weightBytesTotal(model) - inputEmb + kvBytesPerSeq(model, DTYPE_BYTES[model.precision.kv], T);
  expect((r.cost.busy.memory - acts / hbm) / (bytes / hbm)).toBeCloseTo(1, 6);
});

function gemmCost(
  chip: ChipSpec,
  m: number,
  k = 128,
  n = 128,
  groups?: number,
  dtype: Dtype = 'bf16',
) {
  return naiveOpCost(
    {
      kind: 'gemm',
      id: 'g' as OpId,
      label: 'g',
      deps: [],
      x: tt([m, k]),
      w: tt(groups === undefined ? [k, n] : [groups, k, n]),
      out: tt([m, n]),
      dtype,
      groups,
    },
    singleChip(chip),
  );
}

test('dense GEMMs use the chip shapes for M, N and K', () => {
  const time = (m: number, k: number, n: number) => gemmCost(h100, m, k, n).compute;
  const full = time(128, 128, 128);
  expect(time(64, 128, 128) / full).toBeCloseTo(0.5, 9);
  expect(time(128, 64, 128) / full).toBeCloseTo(0.5, 9);
  expect(time(128, 128, 64) / full).toBeCloseTo(0.5, 9);
  expect(time(128, 8, 128)).toBe(time(128, 16, 128));
  expect(time(8, 128, 128)).toBe(time(128, 128, 8));
  expect(time(256, 128, 128) / full).toBeCloseTo(2, 9);
});

test('grouped GEMMs require at least one tile per active expert', () => {
  const time = (chip: ChipSpec, m: number, groups?: number) =>
    gemmCost(chip, m, 128, 128, groups).compute;
  // 256 rows over 111 groups: swapped warp paths use eight-row tiles;
  // Rubin currently models only wide instructions with 128-row output tiles.
  for (const [id, rows, rate] of [
    ['h100-sxm', 8, 0.67],
    ['a100-sxm', 8, 1],
    ['b200', 8, 0.25],
    ['vr100-nvl72', 128, 1],
  ] as const) {
    const chip = CHIPS_BY_ID[id];
    expect(time(chip, 256, 111) / time(chip, 256)).toBeCloseTo((111 * rows) / rate / 256, 9);
  }
  const dense = time(h100, 256);
  expect(time(h100, 2, 1) / dense).toBeCloseTo(8 / 0.67 / 256, 9);
  // Register-A/shared-B WGMMA reaches near-full throughput at 32 rows per group.
  expect(time(h100, 4096, 128) / dense).toBeCloseTo(4096 / 256, 9);
  expect(time(h100, 8192, 128) / dense).toBeCloseTo(8192 / 256, 9);
});

test('Blackwell small GEMMs trade padding against measured instruction rates', () => {
  for (const id of ['b200', 'gb200-nvl72', 'b300']) {
    const chip = CHIPS_BY_ID[id];
    for (const dtype of ['bf16', 'fp8'] as const) {
      const time = (m: number, n = 4096) => gemmCost(chip, m, 128, n, undefined, dtype).compute;
      const full = time(128);
      // Equivalent full-rate rows, including the slower small-instruction path.
      for (const [m, bf16Rows, fp8Rows] of [
        [1, 32, 64],
        [8, 32, 64],
        [16, 64, 80],
        [32, 80, 80],
        [64, 96, 96],
      ]) {
        expect(time(m) / full).toBeCloseTo((dtype === 'bf16' ? bf16Rows : fp8Rows) / 128, 12);
        expect(time(m)).toBe(time(4096, m));
      }
      // Both output dimensions can be small; a wide tile or warp alone loses.
      expect(time(64, 64) / time(128, 128)).toBeCloseTo(0.5, 12);
      expect(time(64, 32) / time(128, 128)).toBeCloseTo(0.375, 12);
    }
  }
});

test('Blackwell block-scaled GEMMs use narrower N without assuming native M64', () => {
  for (const id of ['b200', 'gb200-nvl72']) {
    for (const dtype of ['mxfp8', 'fp4', 'mxfp4', 'nvfp4'] as const) {
      const k = dtype === 'mxfp8' ? 32 : 64;
      const time = (m: number, n = 4096, reduction = k) =>
        gemmCost(CHIPS_BY_ID[id], m, reduction, n, undefined, dtype).compute;
      expect(time(32) / time(128)).toBeCloseTo(80 / 128, 12);
      expect(time(64) / time(128)).toBeCloseTo(96 / 128, 12);
      expect(time(32)).toBe(time(4096, 32));
      expect(time(64, 64) / time(128, 128)).toBeCloseTo(0.75, 12);
      expect(time(128, 128, k / 2)).toBe(time(128, 128));
      expect(time(128, 128, k + 1) / time(128, 128)).toBeCloseTo(2, 12);
    }
  }
});

test('B300 FP4 reaches its higher peak only with larger tiles', () => {
  for (const dtype of ['fp4', 'mxfp4', 'nvfp4'] as const) {
    const time = (id: string, m: number, k: number, n: number) =>
      gemmCost(CHIPS_BY_ID[id], m, k, n, undefined, dtype).compute;
    // Equal work below the larger tile still runs at B200's throughput.
    expect(time('b300', 128, 192, 128) / time('b200', 128, 192, 128)).toBeCloseTo(1, 12);
    expect(time('b300', 128, 192, 256) / time('b200', 128, 192, 256)).toBeCloseTo(0.75, 12);
    expect(time('b300', 256, 192, 256) / time('b200', 256, 192, 256)).toBeCloseTo(2 / 3, 12);
    // The estimator can choose either reduction width instead of always padding to 96.
    expect(time('b300', 128, 64, 128) / time('b300', 128, 96, 128)).toBeCloseTo(2 / 3, 12);
  }
});

test('Rubin GEMMs use format-specific reduction widths', () => {
  const chip = CHIPS_BY_ID['vr100-nvl72'];
  for (const [dtype, k] of [
    ['bf16', 16],
    ['fp8', 64],
    ['mxfp8', 64],
    ['fp4', 128],
    ['mxfp4', 128],
    ['nvfp4', 128],
  ] as const) {
    const time = (reduction: number, groups = 1) =>
      gemmCost(chip, 128, reduction, 128, groups, dtype).compute;
    expect(time(k / 2)).toBe(time(k));
    expect(time(k + 1) / time(k)).toBeCloseTo(2, 12);
    expect(time(k, 2) / time(k)).toBeCloseTo(2, 12);
    expect(gemmCost(chip, 128, k, 128, undefined, dtype).compute).toBe(time(k));
  }
});

test('TPU expert work uses streamed rows and generation-specific array widths', () => {
  for (const id of ['tpu-v5p', 'tpu-v6e', 'tpu-v7x']) {
    const chip = CHIPS_BY_ID[id];
    const width = id === 'tpu-v5p' ? 128 : 256;
    const time = (m: number, k = width, n = width, groups = 1) =>
      gemmCost(chip, m, k, n, groups).compute;
    expect(time(1)).toBe(time(8));
    expect(time(8)).toBe(time(width, width, 8));
    expect(time(16) / time(8)).toBeCloseTo(2, 12);
    expect(time(16, width, width, 8) / time(16)).toBeCloseTo(4, 12);
    expect(time(8, width / 2, width / 2)).toBe(time(8));
    expect(time(8, width + 1) / time(8)).toBeCloseTo(2, 12);
  }
});

test('Neuron uses either orientation and Trainium2 FP8 doubles contraction width', () => {
  for (const id of ['inferentia2', 'trainium1', 'trainium2']) {
    const chip = CHIPS_BY_ID[id];
    expect(gemmCost(chip, 1).compute).toBe(gemmCost(chip, 64).compute);
    expect(gemmCost(chip, 64).compute).toBe(gemmCost(chip, 128, 128, 64).compute);
  }
  const chip = CHIPS_BY_ID['trainium2'];
  const time = (k: number, dtype: Dtype) => gemmCost(chip, 64, k, 128, 1, dtype).compute;
  expect(time(128, 'fp8')).toBe(time(256, 'fp8'));
  expect(time(256, 'bf16') / time(128, 'bf16')).toBeCloseTo(2, 12);
  expect(time(256, 'fp8') / time(256, 'bf16')).toBeCloseTo(
    peakFlops(chip, 'bf16')! / peakFlops(chip, 'fp8')!,
    12,
  );
});

test('Gaudi pads the output array while streaming the reduction dimension', () => {
  for (const id of ['gaudi2', 'gaudi3']) {
    const chip = CHIPS_BY_ID[id];
    const time = (m: number, k: number, n: number) => gemmCost(chip, m, k, n).compute;
    expect(time(128, 17, 128)).toBe(time(256, 17, 256));
    expect(time(256, 34, 256) / time(256, 17, 256)).toBeCloseTo(2, 12);
  }
});

test('every chip declares valid shapes for each arithmetic dtype', () => {
  for (const chip of CHIPS)
    for (const dtype of DTYPES) {
      expect(chip.mmaShapes[dtype].length).toBeGreaterThan(0);
      for (const s of chip.mmaShapes[dtype]) {
        for (const dim of [s.m, s.n, s.k]) expect(Number.isInteger(dim) && dim > 0).toBe(true);
        expect(s.rate).toBeGreaterThan(0);
        expect(s.rate).toBeLessThanOrEqual(1);
      }
    }
});

test('dense and one-group GEMMs have identical costs on every chip and native dtype', () => {
  for (const chip of CHIPS)
    for (const dtype of DTYPES.filter((d) => peakFlops(chip, d)))
      for (const m of [2, 16.25, 64, 129])
        expect(gemmCost(chip, m, 33, 19, 1, dtype)).toEqual(
          gemmCost(chip, m, 33, 19, undefined, dtype),
        );
});

test('Hopper small FP8 shapes use the widening rate, large shapes use native FP8', () => {
  const time = (m: number, dtype: Dtype) => gemmCost(h100, m, 128, 128, undefined, dtype).compute;
  expect(time(2, 'fp8') / time(2, 'bf16')).toBeCloseTo(1, 2);
  expect(time(128, 'fp8') / time(128, 'bf16')).toBeCloseTo(0.5, 3);
  // Native FP8 at 16 rows is ~60% of FP8 peak; 32 rows is near full rate.
  expect(time(16, 'fp8') / time(128, 'fp8')).toBeCloseTo(16 / 0.6 / 128, 12);
  for (const dtype of ['bf16', 'fp8'] as const) {
    expect(time(32, dtype) / time(128, dtype)).toBeCloseTo(0.25, 12);
    expect(time(32, dtype)).toBe(gemmCost(h100, 128, 128, 32, undefined, dtype).compute);
  }
});

test('shape costs include N/K padding and rate without changing memory traffic', () => {
  const chip: ChipSpec = {
    ...h100,
    mmaShapes: {
      ...h100.mmaShapes,
      bf16: [
        { m: 8, n: 128, k: 128, rate: 1 },
        { m: 16, n: 8, k: 16, rate: 0.5 },
      ],
    },
  };
  // The larger M wins for thin N/K despite running at half rate.
  const narrow = gemmCost(chip, 8, 16, 8);
  const ideal = gemmCost(idealTiles, 8, 16, 8);
  expect(narrow.compute / ideal.compute).toBeCloseTo(4, 9);
  expect(narrow.memory).toBe(ideal.memory);
  expect(gemmCost(chip, 8).compute).toBe(gemmCost(idealTiles, 8).compute);
  expect(gemmCost(chip, 0, 128, 128, 111).compute).toBe(0);
  // Unit tiles also leave fractional analytical dimensions unrounded.
  expect(gemmCost(idealTiles, 0.5, 16.25, 8.5, 111).compute).toBe(
    (2 * 0.5 * 16.25 * 8.5) / (peakFlops(h100, 'bf16')! * h100.realizableFlopsFrac),
  );
});

test('gpt-oss-120b decode on one H200 stays under the measured step time', () => {
  // InferenceX gptoss-fp4-h200-trt, TP=1, 1k/1k, concurrency 64: median
  // TPOT 22.5 ms (github.com/SemiAnalysisAI/InferenceX, run 26016892349)
  const measured = 22.5e-3;
  const model = MODEL_PRESETS.find((m) => m.name === 'gpt-oss-120b MXFP4/BF16')!;
  const h200 = CHIPS_BY_ID['h200-sxm'];
  const result = evaluateDecodeAtBatch(
    { model, deployment: singleChip(h200), workload: { prefillLen: 1024, generateLen: 1024 } },
    64,
    1,
    { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0.9, commsOverlap: 0.65 }) },
  );
  if (!result.ok) throw new Error(result.diags.map((x) => x.message).join(', '));
  expect(result.stepTime).toBeLessThan(measured);
  // Weight traffic dominates decode at this batch.
  expect(result.cost.busy.memory).toBeGreaterThan(result.cost.busy.compute);
});

test('a gemm charges its activation streams to memory', () => {
  const [m, k, n] = [256, 512, 1024];
  const cost = gemmCost(h100, m, k, n);
  // rows in and rows out; the weights are priced by their own load node
  expect(cost.memory).toBeCloseTo((m * (k + n) * 2) / hbm, 15);
});

// One MoE block on a 4-chip mesh with DPA = EP = the whole mesh and
// TP = 1, which leaves the dispatch and combine a2as as the only priced
// collectives: every byte on the wire is a routed token.
function a2aCost(
  mlp: MlpConfig,
  moeDispatch: MoeDispatch,
  precision: PrecisionSpec = ALL_BF16,
  chip = h100,
): Record<'compute' | 'memory' | 'comms', number> {
  const model: ModelSpec = {
    name: 'pin',
    modelDim: 1024,
    vocab: 32000,
    tiedEmbeddings: false,
    precision,
    blocks: [
      {
        pattern: [{ block: { attn: gqa({ N: 8, H: 128, kvHeads: 2 }), mlp }, count: 1 }],
        repeat: 2,
      },
    ],
  };
  // latency zeroed: these tests pin the payload accounting, and the
  // per-step constant would bend the exact linear ratios
  const axes = deployedAxes(h100.interconnect, { domain: 4, nodes: 1 }).map((ax) => ({
    ...ax,
    latency: 0,
  }));
  const mesh = makeMesh(axes, { DPA: ['D'], TP: [], EP: ['D'], ETP: [], PP: [] });
  const r = evaluatePrefill(
    {
      model,
      deployment: { chip, mesh, moeDispatch },
      workload: { prefillLen: 512, generateLen: 0 },
    },
    4,
    'throughput',
    { costBackend: backend },
  );
  if (!r.ok) throw new Error(`eval failed`);
  return r.cost.busy;
}

test('coalesced a2a ships the expected distinct-shard payload', () => {
  // 8 experts on EP = 4 shards of 2, topK = 2: a shard is missed with
  // probability C(6,2)/C(8,2) = 15/28, so a token reaches 4 * 13/28 =
  // 13/7 distinct shards vs its 2 expanded copies. Both a2as are linear
  // in the shipped copies, so the whole bill scales by exactly (13/7) / 2.
  const mlp = moeMlp({ experts: 8, topK: 2, expertDim: 512 });
  expect(a2aCost(mlp, 'coalesced-a2a').comms / a2aCost(mlp, 'expanded-a2a').comms).toBeCloseTo(
    13 / 7 / 2,
    9,
  );
});

test('a latent MoE ships the dispatch at the latent width', () => {
  // Routed experts running at L instead of D scale that same bill by
  // exactly L/D: a latent width has to shrink what crosses the wire, not
  // just the expert GEMMs, or there would be no reason to model it apart
  // from expertDim.
  const opts = { experts: 8, topK: 2, expertDim: 512 };
  const latent = a2aCost(moeMlp({ ...opts, latentDim: 256 }), 'expanded-a2a').comms;
  expect(latent / a2aCost(moeMlp(opts), 'expanded-a2a').comms).toBeCloseTo(256 / 1024, 9);
});

test('the running example decodes end to end', () => {
  const model = MODEL_PRESETS.find((m) => m.name.startsWith('Kimi K2.6'))!;
  const d: Deployment = {
    chip: CHIPS_BY_ID['tpu-v5p'],
    mesh: makeMesh(
      [
        { name: 'X', size: 4, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true },
        { name: 'Y', size: 2, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true },
        { name: 'Z', size: 2, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true },
      ],
      { DPA: ['Z'], TP: ['X', 'Y'], EP: ['X', 'Y'], ETP: ['Z'], PP: [] },
    ),
    moeDispatch: 'ring-of-experts',
  };
  const r = evaluateDecodeAtBatch(
    { model, deployment: d, workload: { prefillLen: 4096, generateLen: 0 } },
    256,
    1,
    { costBackend: backend },
  );
  if (!r.ok) throw new Error(`eval failed: ${r.diags.map((x) => x.message).join(', ')}`);
  expect(r.cost.busy.compute).toBeGreaterThan(0);
  expect(r.cost.busy.memory).toBeGreaterThan(0);
  // the boundary reshards now materialize as priced collectives
  expect(r.cost.busy.comms).toBeGreaterThan(0);
  expect(r.perStageTrace[0].flatMap((s) => s.ops).some((op) => op.kind === 'collective')).toBe(
    true,
  );
  expect(Number.isFinite(r.stepTime)).toBe(true);
});

test('pipelined prefill pays a send between stages', () => {
  const model = MODEL_PRESETS.find((m) => m.name === 'LLaMA 3 8B')!;
  const d: Deployment = {
    chip: h100,
    mesh: makeMesh(deployedAxes(h100.interconnect, { domain: 2, nodes: 1 }), {
      DPA: [],
      TP: [],
      EP: [],
      ETP: [],
      PP: ['D'],
    }),
    moeDispatch: 'ring-of-experts',
  };
  const r = evaluatePrefill(
    { model, deployment: d, workload: { prefillLen: 512, generateLen: 0 } },
    1,
    'throughput',
    { costBackend: backend },
  );
  if (!r.ok) throw new Error(`eval failed`);
  // the last stage has no outgoing send (and owns the unembed, so it is
  // the critical one) - price the first stage directly
  const stage0 = backend(d).priceTrace(r.perStageTrace[0]);
  expect(stage0.busy.comms).toBeGreaterThan(0);
});

test('quantizing only the experts narrows only their share', () => {
  // The same block priced three ways: everything bf16, everything fp8, and
  // the mixed release that runs its routed experts and the dispatch feeding
  // them at fp8 while attention stays wide. The mixed reading has to land
  // strictly between the two -- landing on either end would mean a width
  // leaked across the category boundary in one direction or the other.
  const mlp = moeMlp({ experts: 8, topK: 2, expertDim: 512 });
  const wide = a2aCost(mlp, 'expanded-a2a');
  const mixedPrecision: PrecisionSpec = {
    ...ALL_BF16,
    activations: { ...BF16_ALL, routedExperts: 'fp8' },
  };
  const mixed = a2aCost(mlp, 'expanded-a2a', mixedPrecision);
  // the narrow end forces the residual down too, else its combine leg would
  // ride the bf16 stream and the midpoint identity below would not close
  const narrow = a2aCost(mlp, 'expanded-a2a', {
    ...ALL_BF16,
    activations: FP8_ALL,
    residual: 'fp8',
  });

  // attention and the router keep their width, so compute lands strictly
  // between: on the wide end nothing narrowed, on the narrow end the
  // attention GEMMs leaked into a rate they do not run at
  expect(mixed.compute).toBeLessThan(wide.compute);
  expect(mixed.compute).toBeGreaterThan(narrow.compute);

  // With TP = 1 and no shared experts the only priced collectives are the
  // dispatch and the combine, which carry equal payloads. Exactly one of
  // them narrows -- tokens go out at the experts' width, their outputs come
  // back as partial sums in the stream's -- so the mixed reading is exactly
  // the midpoint. Landing on `narrow` would mean partial sums were being
  // reduced in fp8; landing on `wide` would mean the dispatch never
  // narrowed at all.
  expect(mixed.comms).toBeCloseTo((wide.comms + narrow.comms) / 2, 15);
});
