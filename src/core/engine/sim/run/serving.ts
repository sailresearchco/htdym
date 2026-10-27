import { roleSize, totalChips } from '../../surface/deploy';
import type { Diagnostic } from '../../surface/deploy';
import type { CostBackend, TracePriceOf } from '../cost/types';
import { lowerStage } from '../lowering/lower';
import { partitionIntoStages, type Stage } from '../lowering/stages';
import type { EvalFailure, EvalOptions, MemoryFootprint, SimInput } from '../../surface/api';
import type { DraftSpec, ModelSpec } from '../../../model/models';
import type { OpId, Segment } from '../ir/ops';
import { memoryFootprint } from './memory';
import { runnableOn, validateInput } from './validate';
import { runPipeline } from './common';

// How the engine actually serves, beyond one isolated decode step.
export interface ServingOptions {
  // Speculative decoding: each step the draft proposes `draftTokens` per
  // sequence and the target scores them plus one bonus position in a
  // single verify pass. `acceptance` is the chance each draft token is
  // accepted given the ones before it were, so a step emits
  // (1 - a^(k+1)) / (1 - a) tokens per sequence. Absent = plain decode.
  speculative?: { draftTokens: number; acceptance: number };
  // Continuous batching: the decode step also carries the chunked prefill
  // of the requests that replace finished ones, so prompts and outputs
  // share every weight load. Off prices decode alone, as evaluateDecode
  // does (the prefill then needs machines of its own).
  mixedPrefill?: boolean;
  // Fraction of each prompt served from the prefix cache (no compute).
  prefixHitRate?: number;
  // vLLM's max_num_batched_tokens, per DPA group: a step whose decode
  // queries plus prefill chunk exceed it cannot keep up with arrivals.
  maxBatchedTokens?: number;
}

export type ServingEvaluation<TBackend extends CostBackend> =
  | EvalFailure
  | {
      ok: true;
      diags: Diagnostic[];
      memory: MemoryFootprint;
      stepTime: number;
      // output tokens one sequence gains per step
      tokensPerStep: number;
      // prompt tokens the step prefills per DPA group (0 without mixedPrefill)
      prefillTokensPerGroup: number;
      // decode queries + prefill chunk per DPA group, against maxBatchedTokens
      batchedTokensPerGroup: number;
      tpot: number;
      outputTokPerSecPerChip: number;
      // prompt tokens (cached ones included) completed per second per chip
      inputTokPerSecPerChip: number;
      // time split: target verify (or decode) step vs the draft's passes
      targetTime: number;
      draftTime: number;
      cost: TracePriceOf<TBackend>;
      perStageTrace: Segment[][];
    };

// Expected tokens per sequence per step under i.i.d. acceptance.
export function tokensPerVerify(draftTokens: number, acceptance: number): number {
  if (acceptance >= 1) return draftTokens + 1;
  return (1 - Math.pow(acceptance, draftTokens + 1)) / (1 - acceptance);
}

// One steady-state serving step at `batch` concurrent decoding sequences.
// Requests arrive as fast as they finish, so with mixed prefill every
// step prefills exactly the prompts of the requests it completes:
// batch * tokensPerStep / G of them.
export function evaluateServingStep<TBackend extends CostBackend>(
  input: SimInput,
  batch: number,
  serving: ServingOptions,
  opts: EvalOptions<TBackend>,
): ServingEvaluation<TBackend> {
  const { workload, deployment } = input;
  const diags = validateInput(input);
  const fail = (extra?: Diagnostic): EvalFailure => ({
    ok: false,
    diags: extra ? [...diags, extra] : diags,
  });
  if (diags.some((x) => x.severity === 'error')) return fail();

  const model = runnableOn(input.model, deployment.chip);
  // the draft runs only when serving speculatively
  const draft = serving.speculative ? model.draft : undefined;
  const dpa = roleSize(deployment.mesh, 'DPA');
  const pp = roleSize(deployment.mesh, 'PP');
  // Speculation is only credited with the draft that produces it: the
  // model must carry one, proposing exactly the tokens the verify scores.
  if (serving.speculative) {
    const proposed = draft ? draft.passes * draft.tokensPerPass : 0;
    if (proposed !== serving.speculative.draftTokens)
      return fail({
        severity: 'error',
        code: 'spec-draft-mismatch',
        message: draft
          ? `the draft proposes ${proposed} tokens per step, not ${serving.speculative.draftTokens}`
          : `${input.model.name} has no draft model to speculate with`,
      });
  }
  if (pp > 1 && serving.speculative)
    return fail({
      severity: 'error',
      code: 'spec-with-pp',
      message: 'speculative serving is only modeled without pipeline parallelism',
    });

  const P = workload.prefillLen;
  const G = workload.generateLen;
  const seqsPerGroup = batch / dpa;
  const spec = serving.speculative;
  const queries = spec ? spec.draftTokens + 1 : 1;
  const perStep = spec ? tokensPerVerify(spec.draftTokens, spec.acceptance) : 1;
  const hit = Math.min(Math.max(serving.prefixHitRate ?? 0, 0), 1);
  const cached = Math.round(P * hit);

  // prompts completed per group per step, and the tokens they actually prefill
  const promptSeqs = serving.mixedPrefill ? (seqsPerGroup * perStep) / G : 0;
  const prefillTokensPerGroup = promptSeqs * (P - cached);
  const decodeTokensPerGroup = seqsPerGroup * queries;
  const batchedTokensPerGroup = decodeTokensPerGroup + prefillTokensPerGroup;

  const stages = partitionIntoStages(model, pp);
  const memory = withDraftMemory(
    memoryFootprint(model, deployment, stages, P + G),
    model,
    draft,
    deployment,
    P + G,
  );
  if (!opts.ignoreKvCapacity && seqsPerGroup * pp > memory.maxResidentSeqsPerChip)
    return fail({
      severity: 'error',
      code: 'kv-no-room',
      message: `${seqsPerGroup * pp} resident sequences per chip exceed the ${memory.maxResidentSeqsPerChip} that fit in KV`,
    });
  if (serving.maxBatchedTokens !== undefined && batchedTokensPerGroup > serving.maxBatchedTokens)
    return fail({
      severity: 'error',
      code: 'token-budget',
      message: `${Math.ceil(batchedTokensPerGroup)} tokens per step per group exceed the ${serving.maxBatchedTokens} budget`,
    });

  const ctx = P + G / 2;
  const draftStage: Stage | undefined = draft && {
    index: 0,
    groups: draft.blocks,
    hasEmbedding: false,
    // the draft scores its proposals with the target's head
    hasUnembedding: true,
  };

  let targetTime = 0;
  let draftTime = 0;
  const run = runPipeline(input, stages, opts, (stage) => {
    const target = lowerStage(
      {
        model,
        deployment,
        phase: 'decode',
        tokensTotal: batchedTokensPerGroup * dpa,
        tokensPerGroup: batchedTokensPerGroup,
        seqsPerGroup,
        ctx,
        chunk:
          promptSeqs > 0
            ? {
                seqs: promptSeqs,
                ctx: P,
                cached,
                // the budget the decode queries leave is what a prompt gets
                chunkLen:
                  serving.maxBatchedTokens === undefined
                    ? undefined
                    : Math.max(1, serving.maxBatchedTokens - decodeTokensPerGroup),
              }
            : undefined,
        includeBoundary: pp > 1 && stage.index < stages.length - 1,
        routingSkew: workload.routingSkew,
      },
      stage,
    );
    if (!draftStage || stage.index !== stages.length - 1) return target;

    const passTokens = seqsPerGroup * draft!.tokensPerPass;
    const drafted = lowerStage(
      {
        model: { ...model, blocks: draft!.blocks },
        deployment,
        phase: 'decode',
        tokensTotal: passTokens * dpa,
        tokensPerGroup: passTokens,
        seqsPerGroup,
        ctx,
        includeBoundary: false,
        routingSkew: workload.routingSkew,
      },
      draftStage,
    ).map((s) => ({
      ...s,
      label: `draft:${s.label}`,
      repeat: s.repeat * draft!.passes,
      // a fresh builder numbers from zero again: keep its ids apart
      ops: s.ops.map((op) => ({
        ...op,
        id: `draft:${op.id}` as OpId,
        deps: op.deps.map((d) => `draft:${d}` as OpId),
      })),
    }));
    return [...target, ...drafted];
  });

  // split the critical stage's time into target and draft by their busy share
  const busyPerOp = run.cost.busyPerOp;
  if (busyPerOp) {
    let all = 0;
    let d = 0;
    for (const seg of run.perStageTrace[run.criticalStage])
      for (const op of seg.ops) {
        const b = busyPerOp.get(op.id);
        if (!b) continue;
        const t = b.compute + b.memory + b.comms;
        all += t;
        if (seg.label.startsWith('draft:')) d += t;
      }
    draftTime = all > 0 ? (run.stepTime * d) / all : 0;
  }
  targetTime = run.stepTime - draftTime;

  const chips = totalChips(deployment);
  const outPerSec = (batch * perStep) / run.stepTime;
  return {
    ok: true,
    diags,
    memory,
    stepTime: run.stepTime,
    tokensPerStep: perStep,
    prefillTokensPerGroup,
    batchedTokensPerGroup,
    tpot: (pp * run.stepTime) / perStep,
    outputTokPerSecPerChip: outPerSec / chips,
    inputTokPerSecPerChip: serving.mixedPrefill ? (outPerSec * P) / G / chips : 0,
    targetTime,
    draftTime,
    cost: run.cost,
    perStageTrace: run.perStageTrace,
  };
}

// The draft's weights and cache sit on every chip of the last stage next
// to the target's, so they shrink the room left for sequences.
function withDraftMemory(
  m: MemoryFootprint,
  model: ModelSpec,
  draft: DraftSpec | undefined,
  deployment: SimInput['deployment'],
  fullLen: number,
): MemoryFootprint {
  if (!draft) return m;
  const d = memoryFootprint(
    { ...model, blocks: draft.blocks },
    deployment,
    [{ index: 0, groups: draft.blocks, hasEmbedding: false, hasUnembedding: false }],
    fullLen,
  );
  const weights = m.weightBytesPerChip + d.weightBytesPerChip;
  const kv = m.kvBytesPerSeqPerChip + d.kvBytesPerSeqPerChip;
  const free = deployment.chip.hbmCapacity - weights;
  return {
    weightBytesPerChip: weights,
    kvBytesPerSeqPerChip: kv,
    maxResidentSeqsPerChip: kv > 0 ? Math.max(0, Math.floor(free / kv)) : Infinity,
  };
}
