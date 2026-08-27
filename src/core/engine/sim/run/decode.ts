import { roleSize, totalChips } from '../../surface/deploy';
import type { Diagnostic } from '../../surface/deploy';
import type { CostBackend } from '../cost/types';
import { lowerStage } from '../lowering/lower';
import { partitionIntoStages } from '../lowering/stages';
import type { DecodeEvaluation, EvalOptions, SimInput } from '../../surface/api';
import { memoryFootprint } from './memory';
import { runnableOn, validateInput } from './validate';
import { runPipeline } from './common';

export function evaluateDecodeAtBatch<TBackend extends CostBackend>(
  input: SimInput,
  batch: number,
  microbatches: number,
  opts: EvalOptions<TBackend>,
): DecodeEvaluation<TBackend> {
  const { workload, deployment } = input;

  const diags = validateInput(input);
  const fail = (extra?: Diagnostic): DecodeEvaluation<TBackend> => ({
    ok: false,
    diags: extra ? [...diags, extra] : diags,
  });
  if (diags.some((x) => x.severity === 'error')) return fail();

  // validate saw the model as written; run the model as this chip runs it
  const model = runnableOn(input.model, deployment.chip);

  const dpa = roleSize(deployment.mesh, 'DPA');
  const pp = roleSize(deployment.mesh, 'PP');

  const stages = partitionIntoStages(model, pp);
  const memory = memoryFootprint(
    model,
    deployment,
    stages,
    workload.prefillLen + workload.generateLen,
  );
  // each DPA group holds batch/dpa of every resident microbatch's KV
  if (!opts.ignoreKvCapacity && (batch / dpa) * microbatches > memory.maxResidentSeqsPerChip) {
    return fail({
      severity: 'error',
      code: 'kv-no-room',
      message: `${(batch / dpa) * microbatches} resident sequences per chip exceed the ${memory.maxResidentSeqsPerChip} that fit in KV`,
    });
  }

  const run = runPipeline(input, stages, opts, (stage) =>
    lowerStage(
      {
        model,
        deployment,
        phase: 'decode',
        tokensTotal: batch,
        tokensPerGroup: batch / dpa,
        seqsPerGroup: batch / dpa,
        ctx: workload.prefillLen + workload.generateLen / 2,
        includeBoundary: pp > 1 && stage.index < stages.length - 1,
      },
      stage,
    ),
  );

  return {
    ok: true,
    diags,
    memory,
    stepTime: run.stepTime,
    criticalStage: run.criticalStage,
    tpot: microbatches * run.stepTime,
    tokPerSecPerChip: batch / run.stepTime / totalChips(deployment),
    cost: run.cost,
    perStageTrace: run.perStageTrace,
  };
}
