import { PhysAxis } from '../../hardware/topology';
import { cross, permutations, primes } from '../misc/combinatorics';
import { PLANES, ROLES, ShardingRole } from '../sim/ir/sharding/roles';
import { makeMesh, Mesh } from './deploy';

// One merged run of an axis spelling: `size` chips owned by the PP
// singleton or by one role per plane.
interface Group {
  size: number;
  owners: ShardingRole[];
}

// One way to spell a whole axis: its inner-to-outer groups, plus the
// factor each role's size gains from them.
interface Decoration {
  groups: Group[];
  contrib: Record<ShardingRole, number>;
}

// Find every legal assignment of role sizes onto the machine, as
// ready-to-price meshes, one candidate per COST-DISTINCT placement:
// axes no cost model can tell apart (equal size, kind, bandwidth,
// latency, wrap) may be freely relabeled, and only the canonical labeling
// of each such orbit is ever generated. Each axis independently picks
// a decoration, pruned by role-size divisibility; a candidate is a full
// pick whose role products land exactly on the requested sizes.
export function enumeratePlacements(axes: PhysAxis[], sizes: Record<ShardingRole, number>): Mesh[] {
  // every plane must account for the whole machine: PP times the plane's
  // role sizes equals the chip count, or no assignment of slots exists
  const chips = axes.reduce((p, ax) => p * ax.size, 1);
  for (const plane of PLANES)
    if (plane.reduce((p, r) => p * sizes[r], sizes.PP) !== chips)
      throw new Error(`role sizes do not multiply to the machine size`);

  // interchangeable axes iterate consecutively so the symmetry
  // constraint below sees its class-mates side by side
  const classKey = (ax: PhysAxis) =>
    `${ax.size} ${ax.kind} ${ax.bandwidth} ${ax.latency} ${ax.wrap}`;
  const order = [...axes].sort((a, b) => classKey(a).localeCompare(classKey(b)));
  const options = order.map((ax) => axisDecorations(ax));

  const out: Mesh[] = [];
  const chosen: Decoration[] = [];
  const pick = (i: number, minIdx: number, remaining: Record<ShardingRole, number>) => {
    if (i === order.length) {
      if (ROLES.every((r) => remaining[r] === 1)) out.push(buildMesh(axes, order, chosen));
      return;
    }
    // the symmetry quotient: interchangeable axes pick decorations in
    // non-decreasing index order (their option lists are identical), so
    // exactly one labeling per relabeling orbit is generated
    const start = i > 0 && classKey(order[i - 1]) === classKey(order[i]) ? minIdx : 0;
    for (let j = start; j < options[i].length; j++) {
      const d = options[i][j];
      if (!ROLES.every((r) => remaining[r] % d.contrib[r] === 0)) continue;
      chosen[i] = d;
      pick(
        i + 1,
        j,
        Object.fromEntries(ROLES.map((r) => [r, remaining[r] / d.contrib[r]])) as typeof remaining,
      );
    }
  };
  pick(0, 0, sizes);
  return out;
}

// One complete pick (a decoration per axis, parallel to the DFS `order`)
// spelled as the mesh a user would write by hand: each axis's groups
// become its split and its dims' role memberships, walked in the
// machine's original axes order so dim order is placement-independent.
function buildMesh(axes: PhysAxis[], order: PhysAxis[], chosen: Decoration[]): Mesh {
  const byAxis = new Map(order.map((ax, i) => [ax, chosen[i]]));
  const splits: Record<string, number[]> = {};
  const roles = Object.fromEntries(ROLES.map((r): [ShardingRole, string[]] => [r, []])) as Record<
    ShardingRole,
    string[]
  >;

  for (const ax of axes) {
    const { groups } = byAxis.get(ax)!;
    // a single group is the whole axis, no split entry needed
    if (groups.length > 1) splits[ax.name] = groups.map((g) => g.size);
    groups.forEach((g, i) => {
      // names must mirror what factorAxes mints inside makeMesh
      const name = groups.length > 1 ? `${ax.name}${i}` : ax.name;
      // the PP singleton registers once, a plane tuple once per plane
      for (const r of g.owners) roles[r].push(name);
    });
  }

  // one construction path for every mesh, structural checks included.
  // Groups are the planner's atoms: a fine-split spelling can genuinely
  // plan cheaper sub-axis moves, an accepted pessimism fenced at 2x by
  // the twin fuzz in placements.fuzz.test.ts.
  return makeMesh(axes, roles, splits);
}

// Every spelling of one axis: an ordering of its primes (distinct
// orderings are distinct stride structures) with each slot assigned an
// owner tuple, adjacent equal-owner slots merged into one coarser group.
// Merging collapses spellings that differ only below the group level
// ([2,3] and [3,2] both spell the whole-axis form), so the list is
// duplicate-free and identical axes get identical lists.
function axisDecorations(ax: PhysAxis): Decoration[] {
  const ownerOptions: ShardingRole[][] = [['PP'], ...cross(PLANES.map((p) => [...p]))];

  const out = new Map<string, Decoration>();
  for (const ordering of permutations(primes(ax.size)))
    for (const owners of cross(ordering.map(() => ownerOptions))) {
      const groups: Group[] = [];
      ordering.forEach((size, i) => {
        const last = groups[groups.length - 1];
        if (last && last.owners.join() === owners[i].join()) last.size *= size;
        else groups.push({ size, owners: owners[i] });
      });

      // orderings of one axis are distinct stride structures on a ring;
      // a switch has no strides, so spellings differing only in group
      // order are cost-identical and one representative suffices
      const spelled = groups.map((g) => `${g.size} ${g.owners.join()}`);
      const key = (ax.kind === 'switch' ? [...spelled].sort() : spelled).join('|');
      if (out.has(key)) continue;
      const contrib = Object.fromEntries(ROLES.map((r) => [r, 1])) as Decoration['contrib'];
      for (const g of groups) for (const r of g.owners) contrib[r] *= g.size;
      out.set(key, { groups, contrib });
    }
  return [...out.values()];
}
