import { roleSize, totalChips } from '../../surface/deploy';
import type { CostBackend } from '../cost/types';
import { lowerStage } from '../lowering/lower';
import { partitionIntoStages } from '../lowering/stages';
import type { EvalOptions, PrefillEvaluation, SimInput } from '../../surface/api';
import { runnableOn, validateInput } from './validate';
import { runPipeline } from './common';

export function evaluatePrefill<TBackend extends CostBackend>(
  input: SimInput,
  seqs: number,
  mode: 'throughput' | 'ttft',
  opts: EvalOptions<TBackend>,
): PrefillEvaluation<TBackend> {
  const { workload, deployment } = input;
  const T = workload.prefillLen;

  const diags = validateInput(input);
  if (diags.some((x) => x.severity === 'error')) return { ok: false, diags };

  // validate saw the model as written; lower the model as this chip runs it
  const model = runnableOn(input.model, deployment.chip);

  const dpa = roleSize(deployment.mesh, 'DPA');
  const pp = roleSize(deployment.mesh, 'PP');

  const stages = partitionIntoStages(model, pp);

  const run = runPipeline(input, stages, opts, (stage) =>
    lowerStage(
      {
        model,
        deployment,
        phase: 'prefill',
        tokensTotal: seqs * T,
        // TTFT: the single sequence lives on ONE batch group, no DPA speedup
        tokensPerGroup: mode === 'ttft' ? T : (seqs * T) / dpa,
        seqsPerGroup: mode === 'ttft' ? seqs : seqs / dpa,
        ctx: T,
        includeBoundary: pp > 1 && stage.index < stages.length - 1,
      },
      stage,
    ),
  );

  return {
    ok: true,
    diags,
    stepTime: run.stepTime,
    criticalStage: run.criticalStage,
    latency: run.latency,
    tokPerSecPerChip: (seqs * T) / run.stepTime / totalChips(deployment),
    cost: run.cost,
    perStageTrace: run.perStageTrace,
  };
}
