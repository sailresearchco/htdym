import { expect, test } from 'vitest';
import { cheapestNodeRatio, evaluateDisagg } from '../src/core/engine/sim/run/disagg';
import type { DisaggInput } from '../src/core/engine/sim/run/disagg';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import type { Deployment } from '../src/core/engine/surface/deploy';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { DTYPE_BYTES } from '../src/core/model/dtype';
import { MODEL_PRESETS } from '../src/core/model/models';
import { kvBytesPerSeq } from '../src/core/model/utils';

const opts = { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 }) };
const h100 = CHIPS_BY_ID['h100-sxm'];
const llama8 = MODEL_PRESETS.find((m) => m.name === 'LLaMA 3 8B')!;
const llama70 = MODEL_PRESETS.find((m) => m.name === 'LLaMA 3 70B')!;

function pool(chips: number, sizes: { TP: number; DPA: number }): Deployment {
  const axes = deployedAxes(h100.interconnect, { domain: chips, nodes: 1 });
  const mesh = enumeratePlacements(axes, { PP: 1, EP: 1, ETP: sizes.TP * sizes.DPA, ...sizes })[0];
  return { chip: h100, mesh, moeDispatch: 'ring-of-experts' };
}

function run(over: Partial<DisaggInput> = {}) {
  const e = evaluateDisagg(
    {
      model: llama8,
      prefill: pool(1, { TP: 1, DPA: 1 }),
      decode: pool(1, { TP: 1, DPA: 1 }),
      workload: { prefillLen: 1024, generateLen: 256 },
      decodeBatch: 32,
      ...over,
    },
    opts,
  );
  if (!e.ok) throw new Error(e.diags.map((d) => d.message).join('; '));
  return e;
}

test('transfer bytes are the model KV at P and scale with P and with model size', () => {
  const a = run();
  const b = run({ workload: { prefillLen: 2048, generateLen: 256 } });
  const big = run({
    model: llama70,
    prefill: pool(4, { TP: 4, DPA: 1 }),
    decode: pool(4, { TP: 4, DPA: 1 }),
  });
  const kv = (m: typeof llama8, len: number) => kvBytesPerSeq(m, DTYPE_BYTES[m.precision.kv], len);

  expect(a.kvBytesPerRequest).toBe(kv(llama8, 1024));
  expect(b.kvBytesPerRequest).toBeCloseTo(2 * a.kvBytesPerRequest);
  expect(big.kvBytesPerRequest).toBe(kv(llama70, 1024));
  expect(big.kvBytesPerRequest).toBeGreaterThan(a.kvBytesPerRequest);
  // TP4 splits 70B's 8 KV heads four ways on each side, in parallel
  expect(big.decode.transferBytesPerChip).toBeCloseTo(big.kvBytesPerRequest / 4);
  expect(big.transferTime).toBeCloseTo(
    big.link.latency + big.kvBytesPerRequest / 4 / big.link.bandwidthPerChip,
  );
});

test('decode layout sets bytes per destination chip and the link rate', () => {
  const tp1 = run();
  const tp2 = run({ decode: pool(2, { TP: 2, DPA: 1 }) });
  const dpa2 = run({ decode: pool(2, { TP: 1, DPA: 2 }) });
  expect(tp2.decode.transferBytesPerChip).toBeCloseTo(tp1.decode.transferBytesPerChip / 2);
  // DPA keeps each request whole on one chip, but two groups share the load
  expect(dpa2.decode.transferBytesPerChip).toBe(tp1.decode.transferBytesPerChip);
  expect(dpa2.decode.linkReqPerSecPerChip).toBeCloseTo(tp1.decode.linkReqPerSecPerChip);
});

test('prefix hits cut prefill load and transfer', () => {
  const cold = run();
  const warm = run({ prefixHitRate: 0.5 });
  // half the prompt is computed, but it attends over (and reads) the whole
  // prefix, so the rate gains less than the 2x half the tokens suggests
  const gain = warm.prefill.computeReqPerSecPerChip / cold.prefill.computeReqPerSecPerChip;
  expect(gain).toBeGreaterThan(1.2);
  expect(gain).toBeLessThan(2);
  expect(warm.kvBytesPerRequest).toBeCloseTo(cold.kvBytesPerRequest / 2);
  expect(warm.decode.computeReqPerSecPerChip).toBe(cold.decode.computeReqPerSecPerChip);
  expect(warm.decodeChipsPerPrefillChip).toBeCloseTo(gain * cold.decodeChipsPerPrefillChip);
});

test('a slow link, not compute, bounds the rate', () => {
  const fast = run();
  expect(fast.linkBound).toBe(false);

  const slow = run({ link: { bandwidthPerChip: 1e6, latency: 0 } });
  expect(slow.linkBound).toBe(true);
  expect(slow.prefill.linkUtilisation).toBe(1);
  expect(slow.prefill.reqPerSecPerChip).toBeCloseTo(1e6 / slow.kvBytesPerRequest);
  expect(slow.usdPerMOutput).toBeGreaterThan(fast.usdPerMOutput);
});

test('the pool ratio balances the two rates', () => {
  const e = run();
  const p = e.prefill.computeReqPerSecPerChip;
  const d = e.decode.computeReqPerSecPerChip;
  expect(e.decodeChipsPerPrefillChip).toBeCloseTo(p / d);
  // one prefill chip and its ratio of decode chips serve p requests/s
  expect(e.reqPerSecPerChip * (1 + e.decodeChipsPerPrefillChip)).toBeCloseTo(p);
  expect(e.usdPerRequest).toBeCloseTo((h100.costPerHour! / 3600) * (1 / p + 1 / d));
  expect(e.usdPerMInput / e.usdPerMOutput).toBeCloseTo(256 / 1024);

  // the whole-node pick is the brute-force cheapest pair
  const n = cheapestNodeRatio(e, 8);
  let best = Infinity;
  for (let x = 1; x <= 8; x++)
    for (let y = 1; y <= 8; y++) best = Math.min(best, (x + y) / Math.min(x * p, y * d));
  expect(n.usdPerRequest).toBeCloseTo((best * h100.costPerHour!) / 3600);
  expect(n.reqPerSec).toBeCloseTo(Math.min(n.prefillNodes * p, n.decodeNodes * d));
});

test('a decode layout that replicates KV heads makes the source send each copy', () => {
  // two KV heads over TP4: every head is held twice on the decode side
  const narrow = {
    ...llama8,
    blocks: llama8.blocks.map((g) => ({
      ...g,
      pattern: g.pattern.map((r) => ({
        ...r,
        block: { ...r.block, attn: { ...r.block.attn, kvHeads: 2 } },
      })),
    })),
  } as typeof llama8;
  const e = run({ model: narrow, decode: pool(4, { TP: 4, DPA: 1 }) });
  expect(e.decode.transferBytesPerChip).toBeCloseTo(e.kvBytesPerRequest / 2);
  expect(e.prefill.transferBytesPerChip).toBeCloseTo(2 * e.kvBytesPerRequest);
  // and a source that replicates splits one copy among its chips
  const back = run({ model: narrow, prefill: pool(4, { TP: 4, DPA: 1 }) });
  expect(back.prefill.transferBytesPerChip).toBeCloseTo(back.kvBytesPerRequest / 4);
});

test('a hit rate that rounds to the whole prompt never yields NaN', () => {
  const e = evaluateDisagg(
    {
      model: llama8,
      prefill: pool(1, { TP: 1, DPA: 1 }),
      decode: pool(1, { TP: 1, DPA: 1 }),
      workload: { prefillLen: 1024, generateLen: 256 },
      decodeBatch: 32,
      prefixHitRate: 0.99999,
    },
    opts,
  );
  // a one-token tail packs more prompts than fit: rejected, never NaN
  if (e.ok) expect(Number.isFinite(e.usdPerMOutput)).toBe(true);
  else expect(e.diags.map((d) => d.code)).toContain('kv-no-room');
  const fits = run({ prefixHitRate: 0.99999, prefillSeqs: 8 });
  expect(Number.isFinite(fits.usdPerMOutput)).toBe(true);
  expect(fits.kvBytesPerRequest).toBeGreaterThan(0);
});
