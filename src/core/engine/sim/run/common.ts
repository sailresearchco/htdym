import type { CostBackend, TracePriceOf } from '../cost/types';
import type { Stage } from '../lowering/stages';
import type { LoweredOp, Segment } from '../ir/ops';
import type { EvalOptions, SimInput } from '../../surface/api';
import { expandReshards } from '../lowering/expand';

// Slowest-stage metrics plus the sum of all stage times.
export interface PipelineRunOutput<TBackend extends CostBackend> {
  stepTime: number;
  criticalStage: number;
  // full backend-specific result for the slowest stage
  cost: TracePriceOf<TBackend>;
  perStageTrace: Segment[][];
  // sum of per-stage times (one-shot latency / TTFT)
  latency: number;
}

export function runPipeline<TBackend extends CostBackend>(
  input: SimInput,
  stages: Stage[],
  opts: EvalOptions<TBackend>,
  lower: (s: Stage) => Segment<LoweredOp>[],
): PipelineRunOutput<TBackend> {
  const backend = opts.costBackend(input.deployment);

  let stepTime = 0,
    criticalStage = 0,
    latency = 0;
  let criticalCost: TracePriceOf<TBackend> | undefined;

  const perStageTrace: Segment[][] = [];
  for (const s of stages) {
    const trace = expandReshards(lower(s), input.deployment.mesh, backend);
    perStageTrace.push(trace);
    const cost = backend.priceTrace(trace) as TracePriceOf<TBackend>;
    if (!criticalCost || cost.time > stepTime) {
      stepTime = cost.time;
      criticalStage = s.index;
      criticalCost = cost;
    }
    latency += cost.time;
  }

  if (!criticalCost) throw new Error('cannot run an empty pipeline');
  return { stepTime, criticalStage, cost: criticalCost, perStageTrace, latency };
}
