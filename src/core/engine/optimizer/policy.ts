import { roleSize } from '../surface/deploy';
import { partitionIntoStages } from '../sim/lowering/stages';
import { memoryFootprint } from '../sim/run/memory';
import { evaluateDecodeAtBatch } from '../sim/run/decode';
import type { CostBackend } from '../sim/cost/types';
import type { EvalOptions, SimInput } from '../surface/api';

export interface ServingPolicy {
  // Optional interactivity SLO: minimum decode tok/s/user (1/TPOT). When set,
  // the decode operating batch becomes the largest batch that still meets it
  // (capped by KV capacity), overriding batching.
  sloTokPerSecPerUser?: number;
  // Decode batching policy when no SLO is set: 'max' (default) batches to KV
  // capacity: no latency constraint was stated, so aggregate throughput wins.
  // 'b1' runs one sequence per DPA group (a single sequence cannot split
  // across groups): step time is monotone in batch, so this is max tok/s/user.
  batching?: 'max' | 'b1';
}

// The decode batch the policy operates this deployment at, in steps of one
// sequence per DPA group, with PP microbatches resident so every stage
// stays busy. 0 when no feasible batch exists: KV cannot hold one sequence
// per group, or even the smallest batch misses the SLO.
export function operatingBatch<TBackend extends CostBackend>(
  input: SimInput,
  policy: ServingPolicy,
  opts: EvalOptions<TBackend>,
): number {
  const { model, deployment, workload } = input;
  const dpa = roleSize(deployment.mesh, 'DPA');
  const pp = roleSize(deployment.mesh, 'PP');

  const stages = partitionIntoStages(model, pp);
  const memory = memoryFootprint(
    model,
    deployment,
    stages,
    workload.prefillLen + workload.generateLen,
  );
  // batch sizes ONE of the pp microbatches
  const cap = Math.floor(memory.maxResidentSeqsPerChip / pp);
  if (cap < 1) return 0;

  const slo = policy.sloTokPerSecPerUser;
  if (slo === undefined) return (policy.batching === 'b1' ? 1 : cap) * dpa;

  const meets = (k: number) => {
    const res = evaluateDecodeAtBatch(input, k * dpa, pp, opts);
    return res.ok && 1 / res.tpot >= slo;
  };
  if (!meets(1)) return 0;

  // tpot grows with batch, so bisect for the largest step count meeting it
  let lo = 1;
  let hi = cap;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (meets(mid)) lo = mid;
    else hi = mid - 1;
  }
  return lo * dpa;
}
