import { expect, test } from 'vitest';
import { evaluateDecodeAtBatch } from '../src/core/engine/sim/run/decode';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { evaluateServingStep, tokensPerVerify } from '../src/core/engine/sim/run/serving';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import type { Deployment } from '../src/core/engine/surface/deploy';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { blockKvBytes } from '../src/core/model/block';
import { MODEL_PRESETS, type ModelSpec } from '../src/core/model/models';
import { kvBytesPerSeq, weightBytesTotal } from '../src/core/model/utils';

const opts = { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 }) };
const b300 = CHIPS_BY_ID['b300'];
const v41 = MODEL_PRESETS.find((m) => m.name.startsWith('DeepSeek V4.1 Flash'))!;
const v4 = MODEL_PRESETS.find((m) => m.name.startsWith('DeepSeek V4 Flash'))!;

function dep8(): Deployment {
  const axes = deployedAxes(b300.interconnect, { domain: 8, nodes: 1 });
  const mesh = enumeratePlacements(axes, { PP: 1, TP: 1, DPA: 8, EP: 8, ETP: 1 })[0];
  return { chip: b300, mesh, moeDispatch: 'coalesced-a2a' };
}

const input = (model: ModelSpec, prefillLen = 1024, generateLen = 1024) => ({
  model,
  deployment: dep8(),
  workload: { prefillLen, generateLen },
});

test('tokensPerVerify spans one token to the full draft plus bonus', () => {
  expect(tokensPerVerify(5, 0)).toBe(1);
  expect(tokensPerVerify(5, 1)).toBe(6);
  expect(tokensPerVerify(2, 0.5)).toBeCloseTo(1 + 0.5 + 0.25);
});

test('V4.1 weights and cache reconcile with the checkpoint and the kernels', () => {
  // ~510 GB on disk, less ~203 GB of host-resident Engram tables, the
  // draft's ~7 GB of experts and the ~1 GB vision tower
  expect(weightBytesTotal(v41) / 1e9).toBeGreaterThan(290);
  expect(weightBytesTotal(v41) / 1e9).toBeLessThan(305);
  // long-context bytes per token: 3 pair-compressed sources at 264 B, one
  // full-length source at 528 B, and 4 mxfp4 indexer key caches
  const perToken = (kvBytesPerSeq(v41, 1, 2 ** 20) - kvBytesPerSeq(v41, 1, 2 ** 19)) / 2 ** 19;
  expect(perToken).toBeCloseTo(3 * 264 + 528 + (3 * 68) / 2 + 68, 0);
});

test('a layer sharing a cache stores only its window but reads the shared entries', () => {
  const shared = v41.blocks
    .flatMap((g) => g.pattern)
    .find(
      (r) =>
        r.block.attn.kind === 'csa' &&
        'sharedCache' in r.block.attn &&
        r.block.attn.sharedCache &&
        r.block.attn.compressionRate === 2,
    )!.block;
  expect(blockKvBytes(shared, 1, 65536, 'store')).toBe(128 * 528);
  expect(blockKvBytes(shared, 1, 65536, 'read')).toBe((128 + 512) * 528);
});

test('a speculative verify reads each cache once however many queries it scores', () => {
  const plain = evaluateServingStep(input(v41), 256, {}, opts);
  const verify = evaluateServingStep(
    input(v41),
    256,
    { speculative: { draftTokens: 5, acceptance: 0 } },
    opts,
  );
  if (!plain.ok || !verify.ok) throw new Error('infeasible');
  const kv = (trace: typeof plain.perStageTrace) =>
    trace
      .flat()
      .filter((s) => !s.label.startsWith('draft:'))
      .flatMap((s) => s.ops.map((o) => (o.kind === 'attention' ? s.repeat * o.kvReadBytes : 0)))
      .reduce((a, b) => a + b, 0);
  expect(kv(verify.perStageTrace)).toBeCloseTo(kv(plain.perStageTrace));
  // six queries cost more than one, but far less than six steps
  expect(verify.targetTime).toBeGreaterThan(plain.stepTime);
  expect(verify.stepTime).toBeLessThan(6 * plain.stepTime);
  // the draft's ops keep ids of their own beside the target's
  const ids = verify.perStageTrace.flat().flatMap((s) => s.ops.map((o) => o.id));
  expect(new Set(ids).size).toBe(ids.length);
});

test('speculation needs a draft proposing the tokens it verifies', () => {
  const none = evaluateServingStep(
    input(v4),
    256,
    { speculative: { draftTokens: 5, acceptance: 0.8 } },
    opts,
  );
  expect(none.ok).toBe(false);
  const wrong = evaluateServingStep(
    input(v41),
    256,
    { speculative: { draftTokens: 3, acceptance: 0.8 } },
    opts,
  );
  expect(wrong.ok).toBe(false);
});

test('a cached prefix is read by the prompt that extends it', () => {
  const at = (cached: number) => {
    const r = evaluatePrefill(input(v4, 8192, 1), 8, 'throughput', opts, cached);
    if (!r.ok) throw new Error('infeasible');
    return r.perStageTrace
      .flat()
      .flatMap((s) => s.ops.map((o) => (o.kind === 'attention' ? s.repeat * o.kvReadBytes : 0)))
      .reduce((a, b) => a + b, 0);
  };
  expect(at(0)).toBe(0);
  expect(at(4096)).toBeGreaterThan(0);
});

test('higher acceptance buys tokens at the same step time', () => {
  const at = (a: number) => {
    const r = evaluateServingStep(
      input(v41),
      2048,
      { speculative: { draftTokens: 5, acceptance: a } },
      opts,
    );
    if (!r.ok) throw new Error('infeasible');
    return r;
  };
  const [lo, hi] = [at(0.3), at(0.8)];
  expect(hi.stepTime).toBeCloseTo(lo.stepTime);
  expect(hi.outputTokPerSecPerChip / lo.outputTokPerSecPerChip).toBeCloseTo(
    tokensPerVerify(5, 0.8) / tokensPerVerify(5, 0.3),
  );
  // the draft is priced and is a minority of the step
  expect(lo.draftTime).toBeGreaterThan(0);
  expect(lo.draftTime).toBeLessThan(lo.targetTime);
});

test('mixed prefill completes prompts at the output rate and prefix hits cut its cost', () => {
  const decodeOnly = evaluateServingStep(input(v41, 4096, 512), 1024, {}, opts);
  const mixed = evaluateServingStep(input(v41, 4096, 512), 1024, { mixedPrefill: true }, opts);
  const cached = evaluateServingStep(
    input(v41, 4096, 512),
    1024,
    { mixedPrefill: true, prefixHitRate: 0.75 },
    opts,
  );
  if (!decodeOnly.ok || !mixed.ok || !cached.ok) throw new Error('infeasible');
  expect(mixed.inputTokPerSecPerChip).toBeCloseTo((mixed.outputTokPerSecPerChip * 4096) / 512);
  expect(mixed.stepTime).toBeGreaterThan(decodeOnly.stepTime);
  expect(cached.stepTime).toBeLessThan(mixed.stepTime);
  expect(cached.prefillTokensPerGroup).toBeCloseTo(mixed.prefillTokensPerGroup / 4);
  // a token budget below the step's work rejects it
  const capped = evaluateServingStep(
    input(v41, 4096, 512),
    1024,
    { mixedPrefill: true, maxBatchedTokens: 512 },
    opts,
  );
  expect(capped.ok).toBe(false);
});

test("prefill skips the tail-only layers for all but each prompt's last tokens", () => {
  const untailed: ModelSpec = {
    ...v41,
    blocks: v41.blocks.map((g) => ({
      ...g,
      pattern: g.pattern.map((r) => ({ ...r, block: { ...r.block, prefillTail: undefined } })),
    })),
  };
  const withTail = evaluatePrefill(input(v41, 8192, 1), 8, 'throughput', opts);
  const without = evaluatePrefill(input(untailed, 8192, 1), 8, 'throughput', opts);
  if (!withTail.ok || !without.ok) throw new Error('infeasible');
  // half the layers run on 128 of 8192 tokens
  expect(withTail.tokPerSecPerChip / without.tokPerSecPerChip).toBeGreaterThan(1.6);
});

test('plain decode through the serving step matches evaluateDecodeAtBatch', () => {
  const a = evaluateDecodeAtBatch(input(v4), 512, 1, opts);
  const b = evaluateServingStep(input(v4), 512, {}, opts);
  if (!a.ok || !b.ok) throw new Error('infeasible');
  expect(b.stepTime).toBeCloseTo(a.stepTime);
  expect(b.outputTokPerSecPerChip).toBeCloseTo(a.tokPerSecPerChip);
});

test('a prompt longer than the step budget re-reads its earlier chunks', () => {
  const reads = (maxBatchedTokens?: number) => {
    const r = evaluateServingStep(
      input(v41, 32768, 1024),
      256,
      { mixedPrefill: true, maxBatchedTokens },
      opts,
    );
    if (!r.ok) throw new Error(r.diags.map((d) => d.message).join('; '));
    return r.perStageTrace
      .flat()
      .flatMap((s) => s.ops.map((o) => (o.kind === 'attention' ? s.repeat * o.kvReadBytes : 0)))
      .reduce((a, b) => a + b, 0);
  };
  expect(reads(8192)).toBeGreaterThan(reads());
});

test('pipeline splits that separate a shared cache from its readers are rejected', () => {
  const r = evaluateServingStep(
    {
      model: v41,
      deployment: {
        chip: b300,
        mesh: enumeratePlacements(deployedAxes(b300.interconnect, { domain: 8, nodes: 1 }), {
          PP: 4,
          TP: 1,
          DPA: 2,
          EP: 2,
          ETP: 1,
        })[0],
        moeDispatch: 'coalesced-a2a',
      },
      workload: { prefillLen: 1024, generateLen: 1024 },
    },
    64,
    {},
    opts,
  );
  expect(r.ok).toBe(false);
  if (!r.ok) expect(r.diags.map((d) => d.code)).toContain('pp-splits-shared-cache');
});
