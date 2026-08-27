import { ModelSpec } from '../../model/models';
import { hasMoeLayers, minKvHeads } from '../../model/utils';
import { ChipSpec } from '../../hardware/chips';
import { deployedAxes, PhysAxis, SliceTopology } from '../../hardware/topology';
import { divisors } from '../misc/combinatorics';
import { Deployment, MoeDispatch, roleSize, validateSizes } from '../surface/deploy';
import { ShardingRole } from '../sim/ir/sharding/roles';
import { enumeratePlacements } from '../surface/placements';
import { evaluateDecodeAtBatch } from '../sim/run/decode';
import { evaluatePrefill } from '../sim/run/prefill';
import type { CostBackend } from '../sim/cost/types';
import type { DecodeEvaluation, EvalOptions, PrefillEvaluation, SimInput } from '../surface/api';
import { operatingBatch, ServingPolicy } from './policy';

export type SearchPhase =
  | { kind: 'decode'; policy?: ServingPolicy }
  | { kind: 'prefill'; seqs: number; mode: 'throughput' | 'ttft' };

export interface SearchOptions<TBackend extends CostBackend> extends EvalOptions<TBackend> {
  phase: SearchPhase;
  // cross-machine rank key: throughput per chip (default) or per dollar of
  // chip.costPerHour. TTFT-mode prefill always ranks by latency instead.
  rank?: 'perChip' | 'perDollar';
}

export interface Candidate<TBackend extends CostBackend> {
  deployment: Deployment;
  // decode operating batch the policy chose (absent for prefill)
  batch?: number;
  result: Extract<DecodeEvaluation<TBackend> | PrefillEvaluation<TBackend>, { ok: true }>;
  // the sort key, higher is better (negated latency in TTFT mode)
  score: number;
}

// One priced role-size tuple of a running search.
export interface SearchStepOutput<TBackend extends CostBackend> {
  // tuples priced so far out of every size-valid tuple, so done/total
  // is the search's progress fraction
  done: number;
  total: number;
  // the tuple's best placement and dispatch, absent when every
  // placement was infeasible (weights or KV overflow, SLO unmeetable)
  candidate?: Candidate<TBackend>;
}

// The DCP degrees worth pricing at this TP. The cache spreads over the
// TP ranks heads-first, so DCP only buys anything while heads alone
// cannot cover them: past ceil(TP / K) the cache stops shrinking and the
// query gather and output combine are all that is left. 1 is always in.
export function decodeContextParallels(model: ModelSpec, tp: number): number[] {
  const k = minKvHeads(model);
  if (tp <= 1 || k >= tp) return [1];
  const useful = Math.ceil(tp / Math.max(1, k));
  return divisors(tp).filter((d) => d <= useful);
}

// Every size-valid (pp, tp, ep) tuple in search order (cheap arithmetic,
// so callers can size a search before running it). Likely winners sort
// first: observed winners prefer high EP, then minimal TP, and lower PP.
export function searchTuples(model: ModelSpec, chips: number): Record<ShardingRole, number>[] {
  const tuples: Record<ShardingRole, number>[] = [];
  for (const pp of divisors(chips))
    for (const tp of divisors(chips / pp))
      // dense models never read the expert plane, so EP > 1 tuples would
      // price identically to their EP = 1 twin
      for (const ep of hasMoeLayers(model) ? divisors(chips / pp) : [1]) {
        const sizes = { PP: pp, TP: tp, DPA: chips / pp / tp, EP: ep, ETP: chips / pp / ep };
        const diags = validateSizes(model, sizes);
        if (!diags.some((d) => d.severity === 'error')) tuples.push(sizes);
      }

  // Prefer higher EP, then lower TP, then lower PP
  return tuples.sort((a, b) => b.EP - a.EP || a.TP - b.TP || a.PP - b.PP);
}

// Search every sharding of the model onto one concrete machine, yielding
// each role-size tuple's best candidate as it finishes pricing (PP
// divides the machine, then each plane independently factors the rest;
// within a tuple every placement and MoE dispatch competes). Batch is
// chosen by the serving policy per candidate, never swept. Rank by
// `score` at the call site; a tuple's full placement table is one
// enumeratePlacements + evaluate away.
export function* searchShardings<TBackend extends CostBackend>(
  model: ModelSpec,
  chip: ChipSpec,
  // a catalog slice or domain x nodes resolves through the chip's own
  // fabric. raw axes stay possible for machines no catalog sells
  machine: SliceTopology | { domain: number; nodes: number } | PhysAxis[],
  workload: SimInput['workload'],
  opts: SearchOptions<TBackend>,
): Generator<SearchStepOutput<TBackend>> {
  const axes = Array.isArray(machine) ? machine : deployedAxes(chip.interconnect, machine);
  const chips = axes.reduce((p, ax) => p * ax.size, 1);

  const tuples = searchTuples(model, chips);
  for (const [i, sizes] of tuples.entries()) {
    // dispatch changes the plan, not the path to it, so it is swept here
    // rather than priced inside. With EP = 1 there is nothing to route
    // and every dispatch lowers to the same plan, so one stands for all.
    // expanded-a2a is skipped: it lowers to the coalesced trace with a
    // strictly larger a2a payload, so it can never win a search (the
    // optimizer fuzz holds this dominance to account); it stays
    // available to explicit evaluation
    const dispatches: MoeDispatch[] =
      hasMoeLayers(model) && sizes.EP > 1
        ? ['ring-of-experts', 'coalesced-a2a']
        : ['ring-of-experts'];

    // DCP re-spends TP ranks that a narrow-KV block cannot use on the
    // sequence instead. It costs no chips, so it is not part of the size
    // tuple; it is swept here like dispatch. Only sizes that shrink the
    // cache are worth pricing - past that it is pure combine overhead -
    // and which of them are placeable depends on the placement.
    const dcps = decodeContextParallels(model, sizes.TP);

    let best: Candidate<TBackend> | undefined;
    for (const mesh of enumeratePlacements(axes, sizes)) {
      for (const moeDispatch of dispatches) {
        for (const decodeContextParallel of dcps) {
          const cand = evaluateWithSearchOpts(
            { model, deployment: { chip, mesh, moeDispatch, decodeContextParallel }, workload },
            opts,
          );

          if (!cand) continue;
          if (!best || cand.score > best.score) best = cand;
        }
      }
    }

    yield { done: i + 1, total: tuples.length, candidate: best };
  }
}

// Evaluate one deployment, dropping infeasible results and computing its search score.
function evaluateWithSearchOpts<TBackend extends CostBackend>(
  input: SimInput,
  opts: SearchOptions<TBackend>,
): Candidate<TBackend> | null {
  const { phase } = opts;
  let batch: number | undefined;
  let result: Candidate<TBackend>['result'];
  let score: number;

  if (phase.kind === 'decode') {
    batch = operatingBatch(input, phase.policy ?? {}, opts);
    if (batch === 0) return null;

    const res = evaluateDecodeAtBatch(input, batch, roleSize(input.deployment.mesh, 'PP'), opts);
    if (!res.ok) return null;

    [result, score] = [res, res.tokPerSecPerChip];
  } else {
    const res = evaluatePrefill(input, phase.seqs, phase.mode, opts);
    if (!res.ok) return null;

    // a latency target has no per-dollar form, TTFT ranks by time alone
    [result, score] = [res, phase.mode === 'ttft' ? -res.latency : res.tokPerSecPerChip];
  }

  if (opts.rank === 'perDollar' && !(phase.kind === 'prefill' && phase.mode === 'ttft')) {
    const price = input.deployment.chip.costPerHour;
    if (!price) throw new Error(`missing price`);
    score = (score * 3600) / price; // tokens per dollar
  }

  return { deployment: input.deployment, batch, result, score };
}
