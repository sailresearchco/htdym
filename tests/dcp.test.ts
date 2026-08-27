import { expect, test } from 'vitest';
import { MODEL_PRESETS } from '../src/core/model/models';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { kvFraction, makeMesh, validateSizes } from '../src/core/engine/surface/deploy';
import { decodeContextParallels } from '../src/core/engine/optimizer/search';
import { memoryFootprint } from '../src/core/engine/sim/run/memory';
import { partitionIntoStages } from '../src/core/engine/sim/lowering/stages';
import { deployedAxes } from '../src/core/hardware/topology';
import { minKvHeads } from '../src/core/model/utils';

// The TP ranks split two ways: heads first, up to what a block has, then
// DCP spends whatever is left on the sequence.
test('kvFraction spends TP ranks on heads first, then on the sequence', () => {
  // one KV head (MLA): heads can use exactly one rank, so every rank the
  // model is given beyond the first is dead weight until DCP claims it
  expect(kvFraction(1, 8, 1)).toBe(1);
  expect(kvFraction(1, 8, 8)).toBe(1 / 8);

  // two heads on eight ranks: heads cover two, DCP can claim the other four
  expect(kvFraction(2, 8, 1)).toBe(1 / 2);
  expect(kvFraction(2, 8, 4)).toBe(1 / 8);

  // heads that already cover every rank leave DCP nothing to claim
  expect(kvFraction(8, 8, 1)).toBe(1 / 8);
  expect(kvFraction(8, 8, 4)).toBe(1 / 8);
  expect(kvFraction(96, 8, 2)).toBe(kvFraction(96, 8, 1));

  // and DCP is never the reason a chip holds MORE
  for (const k of [1, 2, 3, 5, 8, 96])
    for (const tp of [1, 2, 4, 8, 16])
      for (const dcp of [1, 2, 4, 8].filter((d) => tp % d === 0))
        expect(kvFraction(k, tp, dcp)).toBeLessThanOrEqual(kvFraction(k, tp, 1));
});

test('DCP shrinks a replicated latent cache and leaves a sharded one alone', () => {
  const chip = CHIPS_BY_ID['b300'];
  const axes = deployedAxes(chip.interconnect, { domain: 8, nodes: 1 });
  const names = axes.map((a) => a.name);
  const m = MODEL_PRESETS.find((x) => x.name.startsWith('Kimi K3'))!;
  const kv = (dcp: number) =>
    memoryFootprint(
      m,
      {
        chip,
        mesh: makeMesh(axes, { DPA: [], TP: names, EP: names, ETP: [], PP: [] }),
        moeDispatch: 'ring-of-experts',
        decodeContextParallel: dcp,
      },
      partitionIntoStages(m, 1),
      32768,
    ).kvBytesPerSeqPerChip;

  // K3's 24 MLA layers replicate on all 8 TP ranks; its 69 KDA layers hold
  // a 96-head recurrent state that already shards 8 ways. So DCP=8 divides
  // the MLA part by 8 and cannot touch the KDA part at all, which is why
  // the total falls by well under 8x.
  const kdaState = 69 * 96 * 128 * 128 * 4;
  expect(kv(1) - kdaState / 8).toBeCloseTo(8 * (kv(8) - kdaState / 8), -3);
  expect(kv(8)).toBeGreaterThan(kdaState / 8);
});

test('only the DCP degrees that shrink the cache are priced', () => {
  const k3 = MODEL_PRESETS.find((x) => x.name.startsWith('Kimi K3'))!;
  const gemma = MODEL_PRESETS.find((x) => x.name.startsWith('Gemma 4 31B'))!;

  // K3's narrowest cache is one MLA latent, so every divisor of TP helps
  expect(minKvHeads(k3)).toBe(1);
  expect(decodeContextParallels(k3, 8)).toEqual([1, 2, 4, 8]);

  // Gemma's globals have 4 KV heads, so past ceil(8/4) = 2 the cache stops
  // shrinking and only the combine would be left to pay
  expect(minKvHeads(gemma)).toBe(4);
  expect(decodeContextParallels(gemma, 8)).toEqual([1, 2]);

  // and a model whose heads already cover TP is never given the option
  expect(decodeContextParallels(gemma, 2)).toEqual([1]);
});

test('DCP is rejected when it does not divide TP, and flagged when it is futile', () => {
  const k3 = MODEL_PRESETS.find((x) => x.name.startsWith('Kimi K3'))!;
  const codes = (s: Parameters<typeof validateSizes>[1]) => validateSizes(k3, s).map((d) => d.code);

  expect(codes({ TP: 8, PP: 1, EP: 1, DCP: 3 })).toContain('dcp-not-divisor');
  expect(codes({ TP: 8, PP: 1, EP: 1, DCP: 4 })).not.toContain('dcp-not-divisor');

  // Qwen3.6's linear layers shard 32 ways, so at TP=2 heads already cover
  // every rank and DCP would buy nothing while still owing the combine
  const qwen = MODEL_PRESETS.find((x) => x.name.startsWith('Qwen3.6'))!;
  expect(validateSizes(qwen, { TP: 2, PP: 1, EP: 1, DCP: 2 }).map((d) => d.code)).toContain(
    'dcp-without-benefit',
  );
});
