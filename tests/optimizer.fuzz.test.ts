import { expect } from 'vitest';
import { fuzzTest } from './fuzz';
import { randAxes, randPlacement, sizesOf } from './gen';
import { Candidate, searchShardings, SearchPhase } from '../src/core/engine/optimizer/search';
import { operatingBatch } from '../src/core/engine/optimizer/policy';
import { Deployment, roleSize } from '../src/core/engine/surface/deploy';
import { ROLES } from '../src/core/engine/sim/ir/sharding/roles';
import { evaluateDecodeAtBatch } from '../src/core/engine/sim/run/decode';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { roofline } from '../src/core/engine/roofline';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { MODEL_PRESETS, ModelSpec } from '../src/core/model/models';
import { hasMoeLayers } from '../src/core/model/utils';

// Oracles built from first principles against the search: a hand-built
// sharding must never outscore the table's best beyond the accepted
// whole-atom planner pessimism (a missing size tuple masquerading as an
// argmin), its size tuple must survive into the table (completeness,
// both held against the SLO-free table when a decode SLO is in play),
// every row must honor the SLO it was batched under, and no row may beat
// the model's roofline ceilings, which are derived independently of the
// sim: a config above the ceiling means the sim dropped work, not that
// the search found a clever sharding.

const backend = makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 });
const chip = CHIPS_BY_ID['h100-sxm'];
const workload = { prefillLen: 512, generateLen: 256 };

type Backend = typeof backend;

// the searched score, re-derived through the public evaluators
function scoreOf(model: ModelSpec, deployment: Deployment, phase: SearchPhase): number | null {
  const input = { model, deployment, workload };
  if (phase.kind === 'decode') {
    const batch = operatingBatch(input, phase.policy ?? {}, { costBackend: backend });
    if (batch === 0) return null;
    const r = evaluateDecodeAtBatch(input, batch, roleSize(deployment.mesh, 'PP'), {
      costBackend: backend,
    });
    return r.ok ? r.tokPerSecPerChip : null;
  }
  const r = evaluatePrefill(input, phase.seqs, phase.mode, { costBackend: backend });
  return r.ok ? (phase.mode === 'ttft' ? -r.latency : r.tokPerSecPerChip) : null;
}

function randPhase(rand: () => number, pick: <T>(xs: T[]) => T, chips: number): SearchPhase {
  if (rand() < 0.5)
    return {
      kind: 'decode',
      policy:
        rand() < 0.33
          ? { sloTokPerSecPerUser: pick([5, 20, 100]) }
          : { batching: pick(['max', 'b1'] as const) },
    };
  const mode = pick(['throughput', 'ttft'] as const);
  // chips seqs divide every candidate DPA, one seq is enough for TTFT
  return { kind: 'prefill', seqs: mode === 'throughput' ? chips : 1, mode };
}

fuzzTest('no hand-built sharding outscores the searched table', (rand) => {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];

  const axes = randAxes(rand, pick, [2, 3, 4], [1, 1, 2]);
  const chips = axes.reduce((p, ax) => p * ax.size, 1);
  const ref = randPlacement(axes, rand);
  const sizes = sizesOf(ref);
  const model = pick(MODEL_PRESETS);
  const phase = randPhase(rand, pick, chips);
  // expanded-a2a stays in the pick though the search skips it: a ref
  // using it must still lose to the table, which is exactly the
  // dominance claim the skip rests on
  const dispatch = pick(['ring-of-experts', 'expanded-a2a', 'coalesced-a2a'] as const);

  const refScore = scoreOf(model, { chip, mesh: ref, moeDispatch: dispatch }, phase);
  if (refScore === null) return; // infeasible for this model, gates rejected it

  const searchTable = (p: SearchPhase) =>
    [...searchShardings(model, chip, axes, workload, { costBackend: backend, phase: p })].flatMap(
      (s) => s.candidate ?? [],
    );
  const table = searchTable(phase);

  // An SLO breaks batch parity between the ref and the table: the
  // canonical twin may trail the fine-split ref by the accepted
  // whole-atom pessimism, and a target inside that gap gates the twin
  // down to a smaller operating batch or out of the table entirely. So
  // completeness and the fence hold against the SLO-free table (an SLO
  // only shrinks refScore, so the fence transfers); the SLO table itself
  // only owes its target, checked last. Every other policy batches the
  // ref and its twin identically and fences its own table
  const slo = phase.kind === 'decode' ? phase.policy?.sloTokPerSecPerUser : undefined;
  const fenced =
    slo === undefined ? table : searchTable({ kind: 'decode', policy: { batching: 'max' } });

  // the ref was feasible, so its canonical twin must have made the table.
  // Dense models only consume PP and the attention plane; the search pins
  // their expert plane to EP = 1, so those roles are fungible
  const consumed = hasMoeLayers(model) ? ROLES : ROLES.filter((r) => r !== 'EP' && r !== 'ETP');
  expect(
    fenced.some((c) => consumed.every((r) => roleSize(c.deployment.mesh, r) === sizes[r])),
  ).toBe(true);

  // a fine-split ref can outscore every canonical row by the accepted
  // whole-atom planner pessimism, so the fence is 2x in the time domain,
  // mirroring placements.fuzz
  const best = fenced.reduce((a, b) => (b.score > a.score ? b : a));
  if (phase.kind === 'prefill' && phase.mode === 'ttft')
    expect(-best.score).toBeLessThanOrEqual(-refScore * 2);
  else expect(best.score * 2).toBeGreaterThanOrEqual(refScore);

  // every row batched under an SLO must actually meet it
  if (slo !== undefined)
    for (const c of table as Candidate<Backend>[])
      if ('tpot' in c.result) expect(1 / c.result.tpot).toBeGreaterThanOrEqual(slo * (1 - 1e-9));
});

fuzzTest('no searched config beats the roofline ceilings', (rand) => {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];

  const axes = randAxes(rand, pick, [2, 3, 4], [1, 1, 2]);
  const chips = axes.reduce((p, ax) => p * ax.size, 1);
  const model = pick(MODEL_PRESETS);
  const ceil = roofline(model, chip, workload);
  if (!ceil) return; // chip lacks the model's precision

  const phase: SearchPhase =
    rand() < 0.5 ? { kind: 'decode' } : { kind: 'prefill', seqs: chips, mode: 'throughput' };
  const steps = searchShardings(model, chip, axes, workload, { costBackend: backend, phase });

  const ceiling = phase.kind === 'decode' ? ceil.decodeCeilingOverlapped : ceil.prefillCeiling;
  for (const s of steps)
    if (s.candidate) {
      expect(s.candidate.result.tokPerSecPerChip).toBeLessThanOrEqual(ceiling * (1 + 1e-9));

      // the B -> inf saturation rate (KV gate off) of a real sharding must
      // land between its own operating rate (throughput is monotone in
      // batch) and the ideal-sharding ceiling: above it means the KV-less
      // eval dropped work rather than freed capacity
      if (phase.kind === 'decode') {
        const d = s.candidate.deployment;
        // "B -> inf" has to actually exceed the operating batch, or the
        // comparison below reads monotonicity backwards. A flat constant
        // stopped being enough once DCP could shrink the KV far enough for
        // the operating batch to pass it.
        const sat = evaluateDecodeAtBatch(
          { model, deployment: d, workload },
          Math.max(65536 * roleSize(d.mesh, 'DPA'), 2 * (s.candidate.batch ?? 1)),
          roleSize(d.mesh, 'PP'),
          { costBackend: backend, ignoreKvCapacity: true },
        );
        if (sat.ok) {
          expect(sat.tokPerSecPerChip).toBeLessThanOrEqual(ceiling * (1 + 1e-9));
          expect(sat.tokPerSecPerChip * (1 + 1e-9)).toBeGreaterThanOrEqual(
            s.candidate.result.tokPerSecPerChip,
          );
        }
      }
    }
});
