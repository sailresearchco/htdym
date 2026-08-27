import { expect } from 'vitest';
import { fuzzTest } from './fuzz';
import { randAxes, randPlacement } from './gen';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { memoryFootprint } from '../src/core/engine/sim/run/memory';
import { partitionIntoStages } from '../src/core/engine/sim/lowering/stages';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { Deployment, makeMesh, roleSize } from '../src/core/engine/surface/deploy';
import { localElems } from '../src/core/engine/sim/ir/tensors';
import { ChipSpec } from '../src/core/hardware/chips';
import { matmulSeconds } from '../src/core/engine/roofline';
import { PhysAxis } from '../src/core/hardware/topology';
import { ModelSpec } from '../src/core/model/models';
import { gqa, linearAttn, mla } from '../src/core/model/block/attn';
import { denseMlp, moeMlp } from '../src/core/model/block/mlp';
import { DTYPE_BYTES, type Dtype } from '../src/core/model/dtype';
import {
  expertReadFraction,
  flopsPerPrefillToken,
  kvBytesPerSeq,
  routedExpertParams,
  weightBytesTotal,
} from '../src/core/model/utils';

// The closed forms in model/utils compute FLOPs and bytes algebraically
// from the spec, while lowering builds sharded tensors op by op. They
// share nothing but the model, so agreement over random architectures
// checks the whole lowering path. memoryFootprint is a third
// independently written accounting of who holds which weights.

const backend = makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 });

const chip: ChipSpec = {
  id: 'fuzz',
  name: 'fuzz',
  vendor: 'fuzz',
  // several peaks so a model may run its categories at different widths
  formats: { bf16: 5e14, fp8: 1.1e15, mxfp8: 1.05e15, fp4: 2.3e15, mxfp4: 2.2e15 },
  hbmCapacity: 1e12,
  hbmBandwidth: 1e12,
  interconnect: { bandwidthPerChip: 4e11, latency: 0, domainSize: 64 },
  realizableFlopsFrac: 0.8,
  realizableHbmBwFrac: 0.85,
  // 1x1 matmul tiles: these oracles check conservation, so random
  // unaligned dims must not pay tile padding
  matmulSatRows: 1,
};
const hbm = chip.hbmBandwidth * chip.realizableHbmBwFrac;

function generator(rand: () => number) {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];

  const model = (): ModelSpec => {
    const bytes = (): Dtype => pick<Dtype>(['fp4', 'mxfp4', 'fp8', 'mxfp8', 'bf16']);
    const attn = () => {
      const roll = rand();
      return roll < 1 / 3
        ? gqa({
            N: pick([4, 8, 16]),
            H: pick([64, 128]),
            kvHeads: pick([1, 2, 4]),
            window: rand() < 0.3 ? 256 : null,
            kEqV: rand() < 0.3,
          })
        : roll < 2 / 3
          ? mla({
              N: pick([8, 16]),
              H: 128,
              dc: pick([128, 256]),
              dqc: rand() < 0.5 ? 384 : undefined,
              dRope: 64,
              valueHeadDim: 128,
              outputGate: rand() < 0.5,
              dsa:
                rand() < 0.3
                  ? { topk: 512, indexHeads: 4, indexHeadDim: 32, shareEvery: 2 }
                  : undefined,
            })
          : linearAttn({
              N: pick([4, 8]),
              H: pick([32, 64]),
              valueHeads: pick([8, 16]),
              valueHeadDim: pick([32, 64]),
              stateBytes: pick([2, 4]),
            });
    };
    const mlp = () =>
      rand() < 0.5
        ? denseMlp(pick([1024, 2048]), pick(['gated', 'standard']))
        : moeMlp({
            experts: pick([8, 16, 32]),
            topK: pick([1, 2, 4]),
            expertDim: pick([256, 512]),
            sharedExperts: pick([0, 1, 2]),
            variant: pick(['gated', 'standard']),
            latentDim: rand() < 0.5 ? pick([256, 384]) : undefined,
          });
    return {
      name: 'fuzz',
      modelDim: pick([512, 768, 1024]),
      vocab: pick([32000, 50000]),
      tiedEmbeddings: rand() < 0.5,
      precision: {
        weights: {
          attention: bytes(),
          denseMlp: bytes(),
          sharedExperts: bytes(),
          routedExperts: bytes(),
          router: bytes(),
          embeddings: bytes(),
        },
        // per category, so a wrongly-wired GEMM prices at a width the
        // closed form buckets somewhere else
        activations: {
          attention: bytes(),
          denseMlp: bytes(),
          sharedExperts: bytes(),
          routedExperts: bytes(),
          router: bytes(),
          embeddings: bytes(),
        },
        residual: bytes(),
        kv: pick<Dtype>(['fp8', 'bf16']),
      },
      // groups of run patterns with counts and repeats, so the
      // closed-form-vs-trace oracles cover the repeat accounting too
      blocks: Array.from({ length: pick([1, 2, 3]) }, () => ({
        pattern: Array.from({ length: pick([1, 1, 2]) }, () => ({
          block: { attn: attn(), mlp: mlp() },
          count: pick([1, 1, 3]),
        })),
        repeat: pick([1, 2, 5]),
      })),
    };
  };

  return { pick, model };
}

function singleChip(): Deployment {
  const axes: PhysAxis[] = [
    { name: 'D', size: 1, kind: 'switch', bandwidth: 4e11, latency: 0, wrap: false },
  ];
  return {
    chip,
    mesh: makeMesh(axes, { DPA: [], TP: [], EP: [], ETP: [], PP: [] }),
    moeDispatch: 'ring-of-experts',
  };
}

fuzzTest('stage partitioning preserves the flattened layer stack', (rand) => {
  const { model } = generator(rand);
  const m = model();
  const flat = (groups: (typeof m)['blocks']) =>
    groups.flatMap((g) =>
      Array.from({ length: g.repeat }, () =>
        g.pattern.flatMap((r) => Array<(typeof r)['block']>(r.count).fill(r.block)),
      ).flat(),
    );

  const layers = flat(m.blocks);
  const pp = 1 + Math.floor(rand() * layers.length);
  const stages = partitionIntoStages(m, pp);

  // stages concatenate back to exactly the model's layers, evenly split
  expect(stages.flatMap((s) => flat(s.groups))).toEqual(layers);
  for (const s of stages) {
    const n = flat(s.groups).length;
    expect(n).toBe(Math.floor(layers.length / pp) + (s.index < layers.length % pp ? 1 : 0));
  }
});

fuzzTest('single-chip prefill conserves the closed-form FLOPs and bytes', (rand) => {
  const { model } = generator(rand);
  // large and tile-aligned so utilization is exactly 1 and virtually
  // every expert is activated
  const T = 16384;

  const m = model();
  const r = evaluatePrefill(
    { model: m, deployment: singleChip(), workload: { prefillLen: T, generateLen: 0 } },
    1,
    'throughput',
    { costBackend: backend },
  );
  if (!r.ok) throw new Error(`eval failed`);

  const ideal = T * matmulSeconds(flopsPerPrefillToken(m, T), chip, chip.realizableFlopsFrac)!;
  expect(r.cost.busy.compute / ideal).toBeCloseTo(1, 9);

  // The conservation claim is about weights and KV, which have anchors
  // (checkpoint footprint, KV algebra). GEMM activation traffic has no
  // independent closed form worth maintaining, so its charge is read
  // back off the trace and netted out rather than re-derived.
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

  // the embedding gather is not lowered, and routed experts stream only
  // the activated fraction
  const emb = m.tiedEmbeddings
    ? 0
    : m.vocab * m.modelDim * DTYPE_BYTES[m.precision.weights.embeddings];
  const inactive =
    routedExpertParams(m) *
    DTYPE_BYTES[m.precision.weights.routedExperts] *
    (1 - expertReadFraction(m, T));
  const bytes =
    weightBytesTotal(m) -
    emb -
    inactive +
    kvBytesPerSeq(m, DTYPE_BYTES[m.precision.kv], T, 'store');
  expect((r.cost.busy.memory - acts / hbm) / (bytes / hbm)).toBeCloseTo(1, 9);
});

fuzzTest('sharded traces load exactly the weights memoryFootprint says they hold', (rand) => {
  const { pick, model } = generator(rand);
  const T = 512;

  const m = model();
  const axes: PhysAxis[] = ['X', 'Y'].map((name) => ({
    name,
    size: pick([2, 4]),
    kind: 'ring',
    bandwidth: 9e10,
    latency: 0,
    wrap: true,
  }));
  // random role partition of the mesh per phase
  const attnRoles: ('DPA' | 'TP')[] = axes.map(() => (rand() < 0.5 ? 'DPA' : 'TP'));
  const moeRoles: ('EP' | 'ETP')[] = axes.map(() => (rand() < 0.5 ? 'EP' : 'ETP'));
  const roles = {
    DPA: axes.filter((_, i) => attnRoles[i] === 'DPA').map((a) => a.name),
    TP: axes.filter((_, i) => attnRoles[i] === 'TP').map((a) => a.name),
    EP: axes.filter((_, i) => moeRoles[i] === 'EP').map((a) => a.name),
    ETP: axes.filter((_, i) => moeRoles[i] === 'ETP').map((a) => a.name),
    PP: [],
  };
  const d: Deployment = {
    chip,
    mesh: makeMesh(axes, roles),
    moeDispatch: pick(['ring-of-experts', 'expanded-a2a', 'coalesced-a2a']),
  };

  const seqs = roleSize(d.mesh, 'DPA');
  const r = evaluatePrefill(
    { model: m, deployment: d, workload: { prefillLen: T, generateLen: 0 } },
    seqs,
    'throughput',
    { costBackend: backend },
  );
  // the random partition can be infeasible for the random model
  // (TP > heads, EP > experts), which the validator rightly rejects
  if (!r.ok && r.diags.some((x) => x.severity === 'error')) return;
  if (!r.ok) throw new Error(`eval failed`);

  // full residency: what the trace would load at loadFraction 1
  const traceBytes = r.perStageTrace[0].reduce(
    (s, seg) =>
      s +
      seg.repeat *
        seg.ops.reduce(
          (t, op) =>
            op.kind === 'weight-load'
              ? t + localElems(op.out, d.mesh.dims) * DTYPE_BYTES[op.dtype]
              : t,
          0,
        ),
    0,
  );
  const tp = roleSize(d.mesh, 'TP');
  const inputEmb = m.tiedEmbeddings
    ? 0
    : (m.vocab * m.modelDim * DTYPE_BYTES[m.precision.weights.embeddings]) / tp;
  const held = memoryFootprint(m, d, partitionIntoStages(m, 1), T).weightBytesPerChip - inputEmb;
  expect(traceBytes / held).toBeCloseTo(1, 9);
});

// Coalescing only merges same-destination copies of the a2a payload: the
// expert compute and HBM traffic must not move, and the comms bill can
// only shrink. With one expert per shard every copy already has its own
// destination, so the two dispatches must price identically - a
// too-good-to-be-true fence that a with-replacement or interpolated
// distinct-shard count would trip.
fuzzTest('coalesced a2a matches expanded compute and never ships more bytes', (rand) => {
  const { pick, model } = generator(rand);
  const T = 512;

  const axes: PhysAxis[] = ['X', 'Y'].map((name) => ({
    name,
    size: pick([2, 4]),
    kind: 'ring',
    bandwidth: 9e10,
    latency: 0,
    wrap: true,
  }));
  const attnRoles: ('DPA' | 'TP')[] = axes.map(() => (rand() < 0.5 ? 'DPA' : 'TP'));
  const moeRoles: ('EP' | 'ETP')[] = axes.map(() => (rand() < 0.5 ? 'EP' : 'ETP'));
  const mesh = makeMesh(axes, {
    DPA: axes.filter((_, i) => attnRoles[i] === 'DPA').map((a) => a.name),
    TP: axes.filter((_, i) => attnRoles[i] === 'TP').map((a) => a.name),
    EP: axes.filter((_, i) => moeRoles[i] === 'EP').map((a) => a.name),
    ETP: axes.filter((_, i) => moeRoles[i] === 'ETP').map((a) => a.name),
    PP: [],
  });

  const ep = roleSize(mesh, 'EP');
  const experts = rand() < 0.3 ? ep : ep * pick([2, 4, 8]);
  const topK = pick([1, 2, 4].filter((k) => k <= experts));

  const m = model();
  for (const g of m.blocks)
    for (const r of g.pattern)
      r.block.mlp = moeMlp({
        experts,
        topK,
        expertDim: 512,
        sharedExperts: pick([0, 1]),
        latentDim: rand() < 0.5 ? 384 : undefined,
      });

  const evalWith = (moeDispatch: 'expanded-a2a' | 'coalesced-a2a') =>
    evaluatePrefill(
      {
        model: m,
        deployment: { chip, mesh, moeDispatch },
        workload: { prefillLen: T, generateLen: 0 },
      },
      roleSize(mesh, 'DPA'),
      'throughput',
      { costBackend: backend },
    );

  const exp = evalWith('expanded-a2a');
  if (!exp.ok && exp.diags.some((x) => x.severity === 'error')) return;
  const coal = evalWith('coalesced-a2a');
  if (!exp.ok || !coal.ok) throw new Error(`eval failed`);

  expect(coal.cost.busy.compute / exp.cost.busy.compute).toBeCloseTo(1, 9);
  expect(coal.cost.busy.memory / exp.cost.busy.memory).toBeCloseTo(1, 9);
  expect(coal.cost.busy.comms).toBeLessThanOrEqual(exp.cost.busy.comms * (1 + 1e-9));
  if (experts === ep)
    expect(Math.abs(coal.cost.busy.comms - exp.cost.busy.comms)).toBeLessThanOrEqual(
      1e-9 * exp.cost.busy.comms,
    );
});

// busyPerOp is what a UI draws per node, so it has to be the same money the
// backend already charged, split up: one entry per op, summing to the
// sequential total, and scaling with repeats rather than counting a repeated
// segment once. Comparing against the trace's own op list (not against a
// second walk of the same helper) is what makes a missing or double-counted
// op visible.
fuzzTest('busyPerOp partitions the trace cost it is attributed from', (rand) => {
  const { pick, model } = generator(rand);
  const m = model();
  const axes = randAxes(rand, pick, [1, 2, 4]);
  const d: Deployment = {
    chip,
    mesh: randPlacement(axes, rand),
    moeDispatch: pick(['ring-of-experts', 'expanded-a2a', 'coalesced-a2a']),
  };

  const r = evaluatePrefill(
    { model: m, deployment: d, workload: { prefillLen: 512, generateLen: 0 } },
    roleSize(d.mesh, 'DPA'),
    'throughput',
    { costBackend: backend },
  );
  if (!r.ok && r.diags.some((x) => x.severity === 'error')) return;
  if (!r.ok) throw new Error(`eval failed`);

  const bound = backend(d);
  for (const trace of r.perStageTrace) {
    const cost = bound.priceTrace(trace);
    const perOp = cost.busyPerOp!;
    const ids = trace.flatMap((s) => s.ops.map((op) => op.id));

    expect([...perOp.keys()].sort()).toEqual([...ids].sort());
    for (const v of perOp.values())
      for (const r of ['compute', 'memory', 'comms'] as const)
        expect(v[r]).toBeGreaterThanOrEqual(0);

    const attributed = [...perOp.values()].reduce((a, c) => a + c.compute + c.memory + c.comms, 0);
    const total = cost.busy.compute + cost.busy.memory + cost.busy.comms;
    expect(attributed / total).toBeCloseTo(1, 9);

    // repeats are real work, not a label: doubling every one doubles the bill
    const doubled = bound.priceTrace(trace.map((s) => ({ ...s, repeat: s.repeat * 2 })));
    for (const [id, v] of doubled.busyPerOp!)
      for (const r of ['compute', 'memory', 'comms'] as const)
        expect(v[r]).toBeCloseTo(perOp.get(id)![r] * 2, 12);
  }
});
