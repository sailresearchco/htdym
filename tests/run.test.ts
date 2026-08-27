import { expect, test } from 'vitest';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { evaluateDecodeAtBatch } from '../src/core/engine/sim/run/decode';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { naiveOpCost } from '../src/core/engine/sim/cost/helpers/naiveOpCost';
import { OpId } from '../src/core/engine/sim/ir/ops';
import { localElems, tt } from '../src/core/engine/sim/ir/tensors';
import { runnableOn, validateInput } from '../src/core/engine/sim/run/validate';
import { Deployment, makeMesh, MoeDispatch } from '../src/core/engine/surface/deploy';
import { ChipSpec, CHIPS_BY_ID, peakFlops, runsAs } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { gqa } from '../src/core/model/block/attn';
import { MlpConfig, moeMlp } from '../src/core/model/block/mlp';
import { ALL_BF16, BF16_ALL, DTYPE_BYTES, FP8_ALL, PrecisionSpec } from '../src/core/model/dtype';
import { ModelSpec, MODEL_PRESETS } from '../src/core/model/models';
import { flopsPerPrefillToken, kvBytesPerSeq, weightBytesTotal } from '../src/core/model/utils';
import { matmulSeconds } from '../src/core/engine/roofline';

const backend = makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 });
const h100 = CHIPS_BY_ID['h100-sxm'];
const hbm = h100.hbmBandwidth * h100.realizableHbmBwFrac;
// 1x1 matmul tiles: the closed-form tests check FLOP conservation, so
// tile-padding utilization is priced out (and pinned separately below)
const idealTiles: typeof h100 = { ...h100, matmulSatRows: 1 };

test('validateInput gates weights past HBM capacity', () => {
  const kimi = MODEL_PRESETS.find((m) => m.name.startsWith('Kimi K2.6'))!;
  const diags = validateInput({
    model: kimi,
    deployment: singleChip(),
    workload: { prefillLen: 128, generateLen: 0 },
  });
  expect(diags.map((x) => x.code)).toContain('weights-dont-fit');
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

test('thin GEMM dims pay tile padding', () => {
  const ctx = singleChip(h100);
  const time = (m: number, k: number, n: number) =>
    naiveOpCost(
      {
        kind: 'gemm',
        id: 'g' as OpId,
        label: 'g',
        deps: [],
        x: tt([m, k]),
        w: tt([k, n]),
        out: tt([m, n]),
        dtype: 'bf16',
      },
      ctx,
    ).compute;

  // any dim below the 128 tile costs the same as the padded full tile
  const full = time(128, 128, 128);
  expect(time(64, 128, 128)).toBeCloseTo(full, 15);
  expect(time(128, 64, 128)).toBeCloseTo(full, 15);
  expect(time(128, 128, 64)).toBeCloseTo(full, 15);
  // aligned shapes pay none: pure flops ratio
  expect(time(256, 128, 128) / full).toBeCloseTo(2, 9);
});

test('a gemm charges its activation streams to memory', () => {
  const ctx = singleChip(h100);
  const [m, k, n] = [256, 512, 1024];
  const cost = naiveOpCost(
    {
      kind: 'gemm',
      id: 'g' as OpId,
      label: 'g',
      deps: [],
      x: tt([m, k]),
      w: tt([k, n]),
      out: tt([m, n]),
      dtype: 'bf16',
    },
    ctx,
  );
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
