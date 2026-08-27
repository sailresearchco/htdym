import { expect } from 'vitest';
import { fuzzTest } from './fuzz';
import { primeShuffle, randAxes, randPlacement, sizesOf } from './gen';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import {
  Deployment,
  makeMesh,
  Mesh,
  MoeDispatch,
  roleSize,
} from '../src/core/engine/surface/deploy';
import { cross, permutations } from '../src/core/engine/misc/combinatorics';
import { ROLES, ShardingRole } from '../src/core/engine/sim/ir/sharding/roles';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { MODEL_PRESETS } from '../src/core/model/models';
import { PhysAxis } from '../src/core/hardware/topology';

// Oracles built from first principles against the enumeration: a random
// directly-constructed valid placement must appear in the table
// (completeness, the failure mode a cost check cannot see), every table
// entry must carry the requested sizes (conservation), the table must
// hold no duplicates under the semantic key (canonicalMesh actually
// canonicalizes), no random placement may beat the table's best (a
// missing candidate masquerading as an argmin), and a fine-split
// spelling brackets its canonical twin (never cheaper for the twin,
// never more than 2x, the accepted whole-atom planner pessimism).

// the semantic identity of a placement: per axis, the inner-to-outer run
// of (size, attention owner, moe owner), adjacent equal-owner runs
// merged. Axes no cost model can tell apart (equal size, kind, bandwidth,
// latency, wrap) may be freely relabeled and the table keeps one
// representative per relabeling orbit, so the key is the
// minimum over those relabelings, rederived here from first principles.
function canonKey(m: Mesh, axes: PhysAxis[]): string {
  const runs = new Map<string, [number, string, string][]>();
  for (const d of m.dims) {
    if (d.size === 1) continue;
    const attn = m.roles.PP.includes(d.name) ? 'PP' : m.roles.DPA.includes(d.name) ? 'DPA' : 'TP';
    const moe = m.roles.PP.includes(d.name) ? 'PP' : m.roles.EP.includes(d.name) ? 'EP' : 'ETP';
    const rs = runs.get(d.axis.name) ?? [];
    const last = rs[rs.length - 1];
    if (last && last[1] === attn && last[2] === moe) last[0] *= d.size;
    else rs.push([d.size, attn, moe]);
    runs.set(d.axis.name, rs);
  }

  // a switch has no stride structure, so run order within it is cost-
  // invisible: sort so every spelling of one partition keys identically
  const kindOf = new Map(axes.map((a) => [a.name, a.kind]));
  for (const [name, rs] of runs)
    if (kindOf.get(name) === 'switch')
      rs.sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));

  const classes = new Map<string, string[]>();
  for (const ax of axes) {
    const k = JSON.stringify([ax.size, ax.kind, ax.bandwidth, ax.latency ?? 0, ax.wrap]);
    classes.set(k, [...(classes.get(k) ?? []), ax.name]);
  }
  const renamings = cross(
    [...classes.values()].map((names) =>
      permutations(names).map((perm) => names.map((n, i): [string, string] => [n, perm[i]])),
    ),
  ).map((maps) => Object.fromEntries(maps.flat()));

  return renamings
    .map((ren) =>
      JSON.stringify(
        [...runs].map(([ax, rs]) => [ren[ax], rs] as const).sort(([a], [b]) => (a < b ? -1 : 1)),
      ),
    )
    .reduce((a, b) => (a < b ? a : b));
}

fuzzTest('every random valid placement appears exactly once in the table', (rand) => {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];

  const axes = randAxes(rand, pick, [2, 3, 4, 6, 8]);
  const ref = randPlacement(axes, rand);
  const sizes = sizesOf(ref);

  const table = enumeratePlacements(axes, sizes);
  const keys = table.map((m) => canonKey(m, axes));
  expect(new Set(keys).size).toBe(keys.length);
  expect(keys).toContain(canonKey(ref, axes));
  for (const m of table) for (const r of ROLES) expect(roleSize(m, r)).toBe(sizes[r]);
});

const backend = makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 });

function stepTime(
  model: (typeof MODEL_PRESETS)[number],
  dispatch: MoeDispatch,
  mesh: Mesh,
  seqs: number,
): number | null {
  const d: Deployment = { chip: CHIPS_BY_ID['h100-sxm'], mesh, moeDispatch: dispatch };
  const r = evaluatePrefill(
    { model, deployment: d, workload: { prefillLen: 512, generateLen: 0 } },
    seqs,
    'throughput',
    { costBackend: backend },
  );
  return r.ok ? r.stepTime : null;
}

// The equivalence that lets enumeratePlacements keep one spelling per
// switch-axis group multiset: the same prime slots with the same owners,
// spelled forwards vs reversed on every switch axis, must price
// identically. If switch pricing ever grows position-dependence, this
// fires and the quotient is no longer exact.
fuzzTest('switch-axis spellings price order-blind', (rand) => {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];
  const axes = randAxes(rand, pick, [4, 8, 12], [1, 2]);
  if (!axes.some((a) => a.kind === 'switch')) return;

  const slotted = axes.map((ax) => ({
    ax,
    slots: primeShuffle(ax.size, rand).map((size) => ({
      size,
      attn: rand() < 0.15 ? ('PP' as const) : rand() < 0.5 ? ('DPA' as const) : ('TP' as const),
      moe: rand() < 0.5 ? ('EP' as const) : ('ETP' as const),
    })),
  }));

  const meshOf = (reversed: boolean) => {
    const splits: Record<string, number[]> = {};
    const roles: Record<ShardingRole, string[]> = { DPA: [], TP: [], EP: [], ETP: [], PP: [] };
    for (const { ax, slots } of slotted) {
      const order = reversed && ax.kind === 'switch' ? [...slots].reverse() : slots;
      if (order.length > 1) splits[ax.name] = order.map((s) => s.size);
      order.forEach((s, i) => {
        const name = order.length > 1 ? `${ax.name}${i}` : ax.name;
        if (s.attn === 'PP') roles.PP.push(name);
        else {
          roles[s.attn].push(name);
          roles[s.moe].push(name);
        }
      });
    }
    return makeMesh(axes, roles, splits);
  };

  const fwd = meshOf(false);
  const rev = meshOf(true);
  const model = pick(MODEL_PRESETS);
  const dispatch = pick(['ring-of-experts', 'expanded-a2a', 'coalesced-a2a'] as const);
  const a = stepTime(model, dispatch, fwd, sizesOf(fwd).DPA);
  const b = stepTime(model, dispatch, rev, sizesOf(rev).DPA);
  if (a === null || b === null) expect(a).toBe(b);
  else expect(Math.abs(a - b)).toBeLessThanOrEqual(a * 1e-9);
});

fuzzTest('no random placement prices below the best of the table', (rand) => {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];

  const axes = randAxes(rand, pick, [2, 3, 4]);
  const ref = randPlacement(axes, rand);
  const sizes = sizesOf(ref);
  const model = pick(MODEL_PRESETS);
  const dispatch = pick(['ring-of-experts', 'expanded-a2a', 'coalesced-a2a'] as const);

  const refTime = stepTime(model, dispatch, ref, sizes.DPA);
  if (refTime === null) return; // infeasible for this model, validator rejected
  const times = enumeratePlacements(axes, sizes)
    .map((m) => stepTime(model, dispatch, m, sizes.DPA))
    .filter((t): t is number => t !== null);
  // a fine-split spelling can price below every canonical candidate by
  // the accepted whole-atom pessimism, so the table's best is fenced at
  // 2x rather than strictly. Canonical spellings stay strict for free,
  // their twin is in the table
  expect(Math.min(...times)).toBeLessThanOrEqual(refTime * 2);
});

// The reshard planner deliberately treats mesh dims as whole atoms
// (sub-axis moves priced out as a perf and simplicity choice, see
// so a fine-split spelling can genuinely price below its canonical twin.
// Two bounds fence the choice. Twin below fine means the coarse planner
// cannot express a composed algorithm, always a bug (past catches: unfused
// a2as, stride-derated p2p hops, PP dims as scratch). Twin far above fine
// means the accepted sub-axis pessimism stopped being small, measured
// ~1-20% on comms-bound fabrics, fenced at 2x.
fuzzTest('a canonical twin never beats its fine spelling nor trails it past 2x', (rand) => {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];

  const axes = randAxes(rand, pick, [2, 3, 4]);
  const ref = randPlacement(axes, rand);
  const sizes = sizesOf(ref);
  const model = pick(MODEL_PRESETS);
  const dispatch = pick(['ring-of-experts', 'expanded-a2a', 'coalesced-a2a'] as const);

  const refTime = stepTime(model, dispatch, ref, sizes.DPA);
  if (refTime === null) return;
  const twin = enumeratePlacements(axes, sizes).find(
    (m) => canonKey(m, axes) === canonKey(ref, axes),
  );
  expect(twin).toBeDefined();
  const twinTime = stepTime(model, dispatch, twin!, sizes.DPA)!;
  expect(refTime).toBeLessThanOrEqual(twinTime * (1 + 1e-9));
  expect(twinTime).toBeLessThanOrEqual(refTime * 2);
});
