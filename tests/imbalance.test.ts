import { expect, test } from 'vitest';
import { evaluateDecodeAtBatch } from '../src/core/engine/sim/run/decode';
import { validateInput } from '../src/core/engine/sim/run/validate';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import type { MoeDispatch } from '../src/core/engine/surface/deploy';
import type { SimInput } from '../src/core/engine/surface/api';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';

// DeepSeek V4 Flash on one B300 node, DEP8 (TP1 x DPA8, EP8) or EP1.
const model = MODEL_PRESETS.find((m) => m.name.startsWith('DeepSeek V4 Flash'))!;
const chip = CHIPS_BY_ID['b300'];
const axes = deployedAxes(chip.interconnect, { domain: 8, nodes: 1 });
const opts = { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0.9, commsOverlap: 0.65 }) };

function input(ep: number, routingSkew: number, moeDispatch: MoeDispatch = 'coalesced-a2a') {
  const mesh = enumeratePlacements(axes, { PP: 1, TP: 1, DPA: 8, EP: ep, ETP: 8 / ep })[0];
  return {
    model,
    deployment: { chip, mesh, moeDispatch },
    workload: { prefillLen: 1024, generateLen: 1024, routingSkew },
  } satisfies SimInput;
}

// KV residency is ignored: these pin the routed path, not batch capacity
function decode(inp: SimInput, batch: number) {
  const r = evaluateDecodeAtBatch(inp, batch, 1, { ...opts, ignoreKvCapacity: true });
  if (!r.ok) throw new Error(r.diags.map((d) => d.code).join(','));
  return r;
}

// Rows the hottest rank's routed-expert GEMM runs, over the balanced
// share B*topK/EP: the imbalance factor the lowering priced.
function expertRowsFactor(inp: SimInput, batch: number): number {
  const r = decode(inp, batch);
  const op = r.perStageTrace[0]
    .flatMap((s) => s.ops)
    .find((o) => o.kind === 'gemm' && o.label === 'experts-in');
  if (!op || op.kind !== 'gemm') throw new Error(`no routed-expert gemm`);
  return op.x.shape[0] / (batch * moe().topK);
}

function moe() {
  const mlp = model.blocks.flatMap((g) => g.pattern).find((p) => p.block.mlp.kind === 'moe')!.block
    .mlp;
  if (mlp.kind !== 'moe') throw new Error(`no moe block`);
  return mlp;
}

test('no imbalance without expert parallelism', () => {
  expect(expertRowsFactor(input(1, 1), 8)).toBe(1);
  // with every expert on every rank there is no hotter rank to wait for
  expect(decode(input(1, 1.6), 64).stepTime).toBe(decode(input(1, 1), 64).stepTime);
});

test('balanced routing at very large batch converges to the mean rank', () => {
  // 2^20 tokens: the finite-batch spread is ~ 1/sqrt(B) of the mean
  const f = expertRowsFactor(input(8, 1), 1 << 20);
  expect(f).toBeGreaterThan(1);
  expect(f).toBeLessThan(1.01);
});

test('step time rises monotonically with routing skew', () => {
  for (const dispatch of ['coalesced-a2a', 'expanded-a2a', 'ring-of-experts'] as const)
    for (const batch of [8, 1024]) {
      const times = [1, 1.15, 1.3, 1.6, 2].map(
        (s) => decode(input(8, s, dispatch), batch).stepTime,
      );
      for (let i = 1; i < times.length; i++) expect(times[i]).toBeGreaterThan(times[i - 1]);
    }
});

test('small batches suffer more imbalance', () => {
  const factors = [8, 64, 512, 4096].map((b) => expertRowsFactor(input(8, 1), b));
  for (let i = 1; i < factors.length; i++) expect(factors[i]).toBeLessThan(factors[i - 1]);
  // and never past every token putting all it can on one rank: min(topK,
  // E/EP) pairs against the mean rank's topK/EP
  const { topK, experts } = moe();
  expect(expertRowsFactor(input(8, 1), 1)).toBeLessThanOrEqual(
    (Math.min(topK, experts / 8) * 8) / topK,
  );
});

test('skew stacks on the finite-batch spread', () => {
  // the spread is additive in units of the mean rank's load
  const base = expertRowsFactor(input(8, 1), 256);
  expect(expertRowsFactor(input(8, 1.3), 256)).toBeCloseTo(base + 0.3, 12);
});

test('routing skew below 1 is rejected', () => {
  expect(validateInput(input(8, 0.9)).map((d) => d.code)).toContain('routing-skew-invalid');
});
