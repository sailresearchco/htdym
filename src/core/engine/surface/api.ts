import type { ModelSpec } from '../../model/models';
import type { Deployment, Diagnostic } from './deploy';

import type { Segment } from '../sim/ir/ops';
import type { CostBackend, TracePriceOf } from '../sim/cost/types';

export interface SimInput {
  model: ModelSpec;
  deployment: Deployment;
  workload: {
    // prefill (prompt) length T, tokens
    prefillLen: number;
    // decode (generation) length S, tokens
    generateLen: number;
  };
}

export interface EvalOptions<TBackend extends CostBackend> {
  // Skip the KV-residency feasibility gate: evaluate the step as if the
  // batch fit. Only for B_inf saturation diagnostics (batchSaturation's own
  // ceiling), never for reported operating points.
  ignoreKvCapacity?: boolean;
  // bound per evaluation, then prices traces and candidate collectives
  costBackend: TBackend;
}

export type HardwareResource = 'compute' | 'memory' | 'comms';

export interface EvalFailure {
  ok: false;
  diags: Diagnostic[];
}

export interface BaseEvaluation<TBackend extends CostBackend> {
  ok: true;
  diags: Diagnostic[];
  // steady-state time of one stage step (slowest stage)
  stepTime: number;
  // pipeline stage index that set stepTime / cost
  criticalStage: number;
  // per-chip token throughput at this batch
  tokPerSecPerChip: number;
  // full backend-specific result for the stage that set stepTime
  cost: TracePriceOf<TBackend>;
  // one per-chip expanded trace per pipeline stage (index = stage),
  // segment-structured: repeated layers stay one segment
  perStageTrace: Segment[][];
}

export interface MemoryFootprint {
  // resident weight bytes on the heaviest chip
  weightBytesPerChip: number;
  // KV bytes one full-length sequence costs its group's chips (worst stage)
  kvBytesPerSeqPerChip: number;
  // sequences one chip's free HBM holds KV for (worst stage). Each DPA
  // group holds its own sequences, so the machine total is dpa times this.
  maxResidentSeqsPerChip: number;
}

export type DecodeEvaluation<TBackend extends CostBackend> =
  | EvalFailure
  | (BaseEvaluation<TBackend> & {
      memory: MemoryFootprint;
      // time between successive tokens of one sequence (= PP * stepTime)
      tpot: number;
    });

export type PrefillEvaluation<TBackend extends CostBackend> =
  | EvalFailure
  | (BaseEvaluation<TBackend> & {
      // single-pass latency (TTFT when evaluating one sequence)
      latency: number;
    });
