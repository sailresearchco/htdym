import type { MeshDim } from '../../../hardware/topology';

// A mesh dim name minted by the topology layer (factorAxes in
// hardware/topology.ts). The IR references dims by name only and encodes
// no geometry. That is what lets different backends price the same IR.
export type MeshAxis = string;

// A JAX/GSPMD-like tensor layout: global shape, the mesh axes each dim
// is split over, and the axes along which chips still hold unsummed
// partial sums (the result of contracting over a sharded dimension).
export interface TensorType {
  // global (logical) dim sizes, expectations may be fractional
  shape: number[];
  // mesh axes each dim is split over ([] = the whole dim on every chip)
  sharding: MeshAxis[][];
  // axes along which chips hold partial sums awaiting reduction
  unreduced: MeshAxis[];
}

export function tt(
  shape: number[],
  sharding: MeshAxis[][] = [],
  unreduced: MeshAxis[] = [],
): TensorType {
  if (sharding.length > shape.length) throw new Error(`rank mismatch`);
  const seen = new Set<MeshAxis>();
  let count = 0;
  for (const axes of sharding) for (const a of axes) count = (seen.add(a), count + 1);
  for (const a of unreduced) count = (seen.add(a), count + 1);
  if (seen.size !== count) throw new Error(`axis used twice`);
  // unlisted trailing dims are unsharded
  return { shape: [...shape], sharding: shape.map((_, i) => sharding[i] ?? []), unreduced };
}

export function sameAxes(a: MeshAxis[], b: MeshAxis[]): boolean {
  return a.length === b.length && a.every((x) => b.includes(x));
}

export function dimEq(a: number, b: number): boolean {
  return Math.abs(a - b) <= 1e-9 * Math.max(1, a);
}

export function typeEq(a: TensorType, b: TensorType): boolean {
  return (
    a.shape.length === b.shape.length &&
    a.shape.every((x, i) => dimEq(x, b.shape[i])) &&
    a.sharding.every((ax, i) => sameAxes(ax, b.sharding[i])) &&
    sameAxes(a.unreduced, b.unreduced)
  );
}

export function globalElems(type: TensorType): number {
  return type.shape.reduce((acc, shape) => acc * shape, 1);
}

// How many ways a dim sharded over axes is cut, given the mesh dims.
export function shardWays(axes: MeshAxis[], dims: MeshDim[]): number {
  return axes.reduce((p, ax) => {
    const dim = dims.find((d) => d.name === ax);
    if (!dim) throw new Error(`layout uses unknown mesh dim`);
    return p * dim.size;
  }, 1);
}

// Elements one chip holds: the global count divided through by the sharding.
export function localElems(t: TensorType, dims: MeshDim[]): number {
  return globalElems(t) / t.sharding.reduce((p, axes) => p * shardWays(axes, dims), 1);
}
