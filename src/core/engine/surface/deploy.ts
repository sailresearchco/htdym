import { ModelSpec } from '../../model/models';
import {
  hasMlaLayers,
  hasMoeLayers,
  layerCount,
  minKvHeads,
  minQueryHeads,
  moeExperts,
} from '../../model/utils';

import { ChipSpec } from '../../hardware/chips';
import { factorAxes, MeshDim, PhysAxis } from '../../hardware/topology';
import { PLANES, ShardingRole } from '../sim/ir/sharding/roles';
import { MeshAxis } from '../sim/ir/tensors';

export interface Deployment {
  chip: ChipSpec;
  mesh: Mesh;
  moeDispatch: MoeDispatch;
  // Decode context parallelism (SGLang's --decode-context-parallel-size):
  // how many of the TP ranks the KV cache is cut across along the sequence
  // instead of across heads. It buys no chips and shards no weights - it
  // re-spends TP ranks that a narrow-KV block cannot use, which is the only
  // way an MLA latent (one KV "head", replicated on every TP rank) divides
  // at all. Must divide TP. Absent = 1.
  decodeContextParallel?: number;
}

// The placement, fully resolved: the machine's mesh dims (each carrying
// its parent physical axis) and the dims each parallelism role owns.
// Role sizes are derived, the product of a role's dim sizes (roleSize).
// Built and checked by makeMesh(), not written as a literal.
export interface Mesh {
  dims: MeshDim[];
  roles: Record<ShardingRole, MeshAxis[]>;
}

// Place roles onto the whole acquired machine (deployedAxes), factoring
// its axes into mesh dims (factorAxes, unlisted axes stay whole). PP's
// dims carve the machine into stage meshes and never appear in tensor
// layouts. Throws on structurally malformed placements.
export function makeMesh(
  axes: PhysAxis[],
  roles: Record<ShardingRole, MeshAxis[]>,
  splits?: Record<string, number[]>,
): Mesh {
  const dims = factorAxes(axes, splits);
  const names = new Set(dims.map((x) => x.name));

  for (const role of Object.values(roles))
    for (const name of role) if (!names.has(name)) throw new Error(`role uses unknown mesh dim`);

  // PP carves the machine into stage meshes. Each plane's roles then
  // partition the stage mesh, so each plane plus PP must partition the
  // whole machine (dims of size 1 are exempt).
  for (const plane of PLANES) {
    const seen = new Set<string>();
    for (const name of [...roles.PP, ...plane.flatMap((role) => roles[role])]) {
      if (seen.has(name)) throw new Error(`mesh dim used twice in one plane`);
      seen.add(name);
    }
    for (const dim of dims)
      if (dim.size > 1 && !seen.has(dim.name)) throw new Error(`mesh dim owned by no role`);
  }

  return { dims, roles };
}

export type MoeDispatch = 'ring-of-experts' | 'coalesced-a2a' | 'expanded-a2a';

export type DiagnosticCode =
  | 'tp-exceeds-heads'
  | 'pp-exceeds-layers'
  | 'ep-exceeds-experts'
  | 'ep-not-divisor'
  | 'ep-without-moe'
  | 'kv-replicated'
  | 'mla-kv-replicated'
  | 'dcp-not-divisor'
  | 'dcp-unplaceable'
  | 'dcp-without-benefit'
  | 'dtype-widened'
  | 'weights-unpacked'
  | 'weights-kernel-missing'
  | 'weights-dont-fit'
  | 'kv-no-room';

export interface Diagnostic {
  severity: 'error' | 'warning' | 'info';
  code: DiagnosticCode;
  message: string;
}

export function roleSize(mesh: Mesh, role: ShardingRole): number {
  return mesh.roles[role].reduce((p, name) => {
    const dim = mesh.dims.find((x) => x.name === name);
    if (!dim) throw new Error(`role uses unknown mesh dim`);
    return p * dim.size;
  }, 1);
}

export function chipsPerStage(d: Deployment): number {
  return roleSize(d.mesh, 'TP') * roleSize(d.mesh, 'DPA');
}

export function dcpSize(d: Deployment): number {
  return d.decodeContextParallel ?? 1;
}

// The fraction of one block's KV cache a single chip holds.
//
// The TP ranks are the only ones a cache can spread over, and they split
// two ways: heads first, up to what the block actually has, then whatever
// ranks are left over go to DCP, which cuts the sequence instead. So a
// block with K heads on TP ranks with DCP = c shards its heads
// min(K, TP/c) ways and its context c ways.
//
// The ceil is the honest part: 3 heads over 2 rank-groups is 2 heads on
// one and 1 on the other, and the chip holding 2 is what sizes HBM.
export function kvFraction(kvHeads: number, tp: number, dcp: number): number {
  const headWays = Math.max(1, Math.min(kvHeads, Math.floor(tp / dcp)));
  return Math.ceil(kvHeads / headWays) / kvHeads / dcp;
}

// The TP mesh dims the DCP group occupies, or null when no subset of them
// multiplies to exactly dcp on this placement. Taken innermost-first: the
// query gather and the partial-output combine are small and run every
// layer of every step, so they want the fastest links available.
export function dcpAxes(mesh: Mesh, dcp: number): MeshAxis[] | null {
  if (dcp <= 1) return [];
  const out: MeshAxis[] = [];
  let n = 1;
  for (const name of mesh.roles.TP) {
    const dim = mesh.dims.find((d) => d.name === name);
    if (!dim || n * dim.size > dcp || dcp % (n * dim.size) !== 0) continue;
    out.push(name);
    n *= dim.size;
    if (n === dcp) return out;
  }
  return null;
}

export function totalChips(d: Deployment): number {
  return chipsPerStage(d) * roleSize(d.mesh, 'PP');
}

// Model-dependent feasibility of role sizes alone, so searches can gate a
// size tuple before enumerating its placements. Structural placement
// checks live in makeMesh(), which throws instead.
export function validateSizes(
  m: ModelSpec,
  s: Record<'TP' | 'PP' | 'EP', number> & { DCP?: number },
): Diagnostic[] {
  const diags: Diagnostic[] = [];
  const diag = (severity: 'error' | 'warning', code: DiagnosticCode, message: string) =>
    diags.push({ severity, code, message });
  const { TP: tp, PP: pp, EP: ep } = s;
  const dcp = s.DCP ?? 1;

  const nHeads = minQueryHeads(m);
  if (tp > nHeads)
    diag('error', 'tp-exceeds-heads', `TP = ${tp} exceeds N = ${nHeads} query heads`);
  if (pp > layerCount(m))
    diag('error', 'pp-exceeds-layers', `PP = ${pp} exceeds L = ${layerCount(m)} layers`);

  if (ep > 1) {
    const experts = moeExperts(m);
    if (!hasMoeLayers(m))
      diag('warning', 'ep-without-moe', `EP = ${ep} but ${m.name} has no experts`);
    else if (ep > experts)
      diag('error', 'ep-exceeds-experts', `EP = ${ep} exceeds E = ${experts} experts`);
    else if (experts % ep !== 0)
      diag('error', 'ep-not-divisor', `EP = ${ep} does not divide E = ${experts} experts`);
  }

  if (dcp > 1) {
    if (tp % dcp !== 0) diag('error', 'dcp-not-divisor', `DCP = ${dcp} does not divide TP = ${tp}`);
    // DCP only pays where heads cannot already spread the cache over every
    // TP rank. Once they can, it buys nothing and still owes the combine.
    else if (minKvHeads(m) >= tp)
      diag(
        'warning',
        'dcp-without-benefit',
        `DCP = ${dcp} but K = ${minKvHeads(m)} KV heads already shard over all ${tp} TP ranks: the cache does not shrink and the combine is still paid`,
      );
  }

  // replication is priced honestly (kvFraction ceils), so these are advisory
  const replicated = kvFraction(minKvHeads(m), tp, dcp) * tp;
  if (hasMlaLayers(m) && replicated > 1)
    diag(
      'warning',
      'mla-kv-replicated',
      `MLA latent KV is replicated on ${replicated} of the ${tp} TP ranks` +
        (dcp > 1 ? ` (DCP = ${dcp} already cut it ${dcp} ways)` : ''),
    );
  else if (replicated > 1)
    diag(
      'warning',
      'kv-replicated',
      `TP = ${tp} exceeds K = ${minKvHeads(m)} KV heads, KV replicated on ${replicated} ranks`,
    );

  return diags;
}
