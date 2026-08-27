import { expect, test } from 'vitest';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { Deployment, makeMesh } from '../src/core/engine/surface/deploy';
import { localElems } from '../src/core/engine/sim/ir/tensors';
import { ChipSpec } from '../src/core/hardware/chips';
import { matmulSeconds } from '../src/core/engine/roofline';
import { PhysAxis } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';
import { DTYPE_BYTES } from '../src/core/model/dtype';
import {
  expertReadFraction,
  flopsPerPrefillToken,
  kvBytesPerSeq,
  routedExpertParams,
  weightBytesTotal,
} from '../src/core/model/utils';

const backend = makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 });
const chip: ChipSpec = {
  id: 'fuzz',
  name: 'fuzz',
  vendor: 'fuzz',
  formats: { bf16: 5e14, fp8: 1.1e15, mxfp8: 1.05e15, fp4: 2.3e15, mxfp4: 2.2e15 },
  hbmCapacity: 1e15,
  hbmBandwidth: 1e12,
  interconnect: { bandwidthPerChip: 4e11, latency: 0, domainSize: 64 },
  realizableFlopsFrac: 0.8,
  realizableHbmBwFrac: 0.85,
  matmulSatRows: 1,
};
const hbm = chip.hbmBandwidth * chip.realizableHbmBwFrac;
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

// The lowering fuzz checks this invariant over synthetic architectures, but
// its generator only emits the knobs that existed when it was written: no
// compressed KV, no low-rank grouped output, no hyper-connections. Running
// the same oracle over the real presets is what covers those, and it is the
// only automated check that the trace and the closed forms in model/utils
// still agree on an architecture either one of them could get wrong alone.
test('every preset conserves the closed-form FLOPs and bytes', () => {
  const T = 16384;
  for (const m of MODEL_PRESETS) {
    const r = evaluatePrefill(
      { model: m, deployment: singleChip(), workload: { prefillLen: T, generateLen: 0 } },
      1,
      'throughput',
      { costBackend: backend },
    );
    if (!r.ok)
      throw new Error(`${m.name} failed to evaluate: ${r.diags.map((d) => d.code).join(',')}`);
    const ideal = T * matmulSeconds(flopsPerPrefillToken(m, T), chip, chip.realizableFlopsFrac)!;
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
    const fl = r.cost.busy.compute / ideal,
      by = (r.cost.busy.memory - acts / hbm) / (bytes / hbm);
    expect(fl).toBeCloseTo(1, 9);
    expect(by).toBeCloseTo(1, 9);
  }
});
