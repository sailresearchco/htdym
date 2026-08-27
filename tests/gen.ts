import { makeMesh, Mesh, roleSize } from '../src/core/engine/surface/deploy';
import { ROLES, ShardingRole } from '../src/core/engine/sim/ir/sharding/roles';
import { PhysAxis } from '../src/core/hardware/topology';

export function primeShuffle(n: number, rand: () => number): number[] {
  const out: number[] = [];
  for (let p = 2; n > 1; p++)
    while (n % p === 0) {
      out.splice(Math.floor(rand() * (out.length + 1)), 0, p);
      n /= p;
    }
  return out;
}

// a random valid placement built directly, bypassing the enumeration:
// full prime split in a random order, every slot assigned by coin flips
export function randPlacement(axes: PhysAxis[], rand: () => number): Mesh {
  const splits: Record<string, number[]> = {};
  for (const ax of axes) {
    const factors = primeShuffle(ax.size, rand);
    if (factors.length > 1) splits[ax.name] = factors;
  }
  const roles: Record<ShardingRole, string[]> = { DPA: [], TP: [], EP: [], ETP: [], PP: [] };
  for (const ax of axes) {
    const count = splits[ax.name]?.length ?? 1;
    for (let i = 0; i < count; i++) {
      if (ax.size === 1) continue;
      const name = count > 1 ? `${ax.name}${i}` : ax.name;
      if (rand() < 0.15) roles.PP.push(name);
      else {
        roles[rand() < 0.5 ? 'DPA' : 'TP'].push(name);
        roles[rand() < 0.5 ? 'EP' : 'ETP'].push(name);
      }
    }
  }
  return makeMesh(axes, roles, splits);
}

export function randAxes(
  rand: () => number,
  pick: <T>(xs: T[]) => T,
  sizes: number[],
  counts = [1, 2, 2, 3],
): PhysAxis[] {
  const n = pick(counts);
  return Array.from({ length: n }, (_, i) => {
    const kind = pick(['ring', 'ring', 'switch'] as const);
    return {
      name: 'XYZ'[i],
      size: pick(sizes),
      kind,
      bandwidth: pick([5e9, 5e10, 4e11]),
      latency: pick([0, 1e-6, 5e-6]),
      wrap: kind === 'ring' && rand() < 0.5,
    };
  });
}

export function sizesOf(m: Mesh): Record<ShardingRole, number> {
  return Object.fromEntries(ROLES.map((r) => [r, roleSize(m, r)])) as Record<ShardingRole, number>;
}
