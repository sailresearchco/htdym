import { CollectiveOp, ExpandedOp, JoinOp, OpId, LoweredOp, Segment, Unbuilt } from '../ir/ops';
import { dimEq, MeshAxis, TensorType, tt } from '../ir/tensors';
import { BoundBackend } from '../cost/types';
import { DTYPE_BYTES } from '../../../model/dtype';
import type { Mesh } from '../../surface/deploy';
import { subsets } from '../../misc/combinatorics';
import { BoundedCache } from '../../misc/boundedCache';

// A collective as the planner sees it: layout moves only. The wire format is
// the reshard's business, added when the winning plan is wired in.
type PlannedCollective = Omit<Unbuilt<CollectiveOp>, 'dtype'>;

// Replace every abstract reshard with its cheapest concrete collective
// chain that satisfies. The chain is wired in place of the reshard node.
// the last op inherits the reshard's id so downstream deps stay valid,
// and a free reshard leaves a dependency-only join.
// Mesh dims are the planner's atoms, whole axes only: sub-axis moves
// (borrowing or routing a FRACTION of a dim) can genuinely beat these
// plans, but exploring them blows up the search, so we just accept this
// and fuzz to make sure sub-axes aren't >2x better in placements.fuzz.test.ts.
export function expandReshards(
  trace: Segment<LoweredOp>[],
  mesh: Mesh,
  backend: Pick<BoundBackend, 'priceCollective' | 'priceCollectiveHash'>,
): Segment[] {
  // PP dims are not scratch space: they carve the machine into stages
  // that run concurrently on other microbatches, so this stage's
  // reshards cannot borrow their chips for slice-and-gather tricks
  const names = mesh.dims.filter((d) => !mesh.roles.PP.includes(d.name)).map((d) => d.name);
  const expand = (op: LoweredOp): ExpandedOp | ExpandedOp[] => {
    if (op.kind !== 'reshard') return op;

    const { collectives } = planReshard(
      op.x,
      op.out,
      names,
      DTYPE_BYTES[op.dtype],
      backend.priceCollective,
      backend.priceCollectiveHash,
    );

    if (!collectives.length)
      return {
        kind: 'join',
        id: op.id,
        label: 'reshard (free)',
        deps: op.deps,
        out: op.out,
      } satisfies JoinOp;

    return collectives.map((c, i): CollectiveOp => {
      const isLast = i === collectives.length - 1;
      return {
        ...c,
        dtype: op.dtype,
        id: isLast ? op.id : (`${op.id}/${i}` as OpId),
        deps: i === 0 ? op.deps : [`${op.id}/${i - 1}` as OpId],
        // trailing free slices mean the last collective's own output can
        // be a coarser layout than the reshard's, so pin the declared dst
        out: isLast ? op.out : c.out,
      };
    });
  };
  return trace.map((s) => ({ ...s, ops: s.ops.flatMap(expand) }));
}

// plans cached under the backend's declared pricing hash, so every bind
// that prices alike shares them. Both levels are bounded: a sweep moves
// through meshes and batch shapes and rarely returns to old ones, so the
// oldest namespaces and plans are dropped
const planCache = new BoundedCache<string, BoundedCache<string, ReturnType<typeof planReshard>>>(
  16,
);
const PLANS_PER_NAMESPACE = 8192;

// One layout a reshard toward some destination can pass through, and
// the single rewrites out of it. Rewrites read the layout, never the
// shape, so one graph per (destination layout, mesh dims) serves every
// shape, and a plan for a new shape only re-prices its edges.
interface LayoutNode {
  sharding: MeshAxis[][];
  unreduced: MeshAxis[];
  // order-insensitive identity, what Dijkstra settles by
  key: string;
  // the rewrites out of it, as parallel lists: what each one runs (null
  // for a free slice) and the layout it lands on
  edges?: { collective: (CollectiveDesc | null)[]; next: LayoutNode[] };
}
type CollectiveDesc = Pick<PlannedCollective, 'variant' | 'label' | 'axes'>;
// Graphs are kept under a budget of layouts, oldest graph out first: a
// sweep moves on to new destination layouts tuple by tuple, and a large
// mesh's graphs run to hundreds of thousands of layouts
interface LayoutGraph {
  nodes: Map<string, LayoutNode>;
  evicted: boolean;
}
const graphCache = new Map<string, LayoutGraph>();
const GRAPH_LAYOUT_BUDGET = 100_000;
let graphLayouts = 0;
function layoutGraph(key: string): LayoutGraph {
  let g = graphCache.get(key);
  if (!g) graphCache.set(key, (g = { nodes: new Map(), evicted: false }));
  return g;
}
function addLayout(g: LayoutGraph, k: string, n: LayoutNode) {
  g.nodes.set(k, n);
  if (g.evicted) return;
  graphLayouts++;
  for (const [key, old] of graphCache) {
    if (graphLayouts <= GRAPH_LAYOUT_BUDGET) break;
    graphCache.delete(key);
    old.evicted = true;
    graphLayouts -= old.nodes.size;
  }
}
// graphs repeat the same few hundred collectives across hundreds of
// thousands of edges, so each is stored once
const descs = new Map<string, CollectiveDesc>();
const descOf = ({ variant, label, axes }: CollectiveDesc): CollectiveDesc => {
  const k = `${variant} ${axes.join(' ')}`;
  let d = descs.get(k);
  if (!d) descs.set(k, (d = { variant, label, axes }));
  return d;
};

// a layout spelled exactly, axis order included
const layoutStr = (t: Pick<TensorType, 'sharding' | 'unreduced'>) =>
  t.sharding.map((a) => a.join('.')).join(';') + '#' + t.unreduced.join('.');
// sharding and unreduced are sets, so key layouts order-insensitively
const layoutKey = (t: Pick<TensorType, 'sharding' | 'unreduced'>) =>
  t.sharding.map((a) => [...a].sort().join(',')).join(';') +
  '#' +
  [...t.unreduced].sort().join(',');

// The cheapest sequence of concrete collectives realizing src -> dst:
// Dijkstra over layout space, where a node is a layout, an edge is one
// rewrite, which is priced by the backend's collective pricer. Instruction
// selection is by shortest path, so "clever" outcomes, e.g. an a2a vanishing
// because an all-reduce lands on the target directly, need no special cases.
export function planReshard(
  src: TensorType,
  dst: TensorType,
  names: MeshAxis[],
  elemBytes: number,
  pricer: BoundBackend['priceCollective'],
  // cache namespace, omit to plan uncached (direct/test invocations)
  hash?: string,
): { time: number; collectives: PlannedCollective[] } {
  // rewrites preserve shape, so layouts are keyed by sharding alone
  // and a shape change could otherwise "succeed" without being realized
  if (src.shape.length !== dst.shape.length || !src.shape.every((s, i) => dimEq(s, dst.shape[i])))
    throw new Error(`reshard changes the tensor shape`);

  let cache = hash === undefined ? undefined : planCache.get(hash);
  if (hash !== undefined && !cache)
    planCache.set(hash, (cache = new BoundedCache(PLANS_PER_NAMESPACE)));
  const memoKey = `${src.shape.join(',')}|${layoutStr(src)}|${layoutStr(dst)}|${names.join('.')}|${elemBytes}`;
  const hit = cache?.get(memoKey);
  if (hit) return hit;

  const target = layoutKey(dst);

  // the layouts this reshard can pass through, shared by every shape
  const ctxKey = `${layoutStr(dst)}|${names.join('.')}`;
  const graph = layoutGraph(ctxKey);
  const nodeOf = (t: TensorType): LayoutNode => {
    const k = layoutStr(t);
    let n = graph.nodes.get(k);
    if (!n)
      addLayout(
        graph,
        k,
        (n = { sharding: t.sharding, unreduced: t.unreduced, key: layoutKey(t) }),
      );
    return n;
  };
  // this shape's tensor at each layout the search settles
  const types = new Map<LayoutNode, TensorType>();
  const typeOf = (n: LayoutNode): TensorType => {
    let t = types.get(n);
    if (!t) types.set(n, (t = tt(src.shape, n.sharding, n.unreduced)));
    return t;
  };
  const start = nodeOf(src);
  types.set(start, src);

  // the cheapest known way to reach each layout seen so far
  type Step = { from: LayoutNode; to: LayoutNode; collective: CollectiveDesc };
  // `order` is when the layout was first reached: ties settle first-come
  type Node = { time: number; steps: Step[]; node: LayoutNode; order: number };
  const best = new Map<string, Node>();
  const done = new Set<string>();
  const frontier = new Frontier<Node>();
  const reach = (key: string, n: Omit<Node, 'order'>) => {
    const node = { ...n, order: best.get(key)?.order ?? best.size };
    best.set(key, node);
    frontier.push(node);
  };
  reach(start.key, { time: 0, steps: [], node: start });

  for (;;) {
    // settle the cheapest unvisited layout, skipping entries a cheaper
    // way to the same layout has since replaced
    let cur: Node | undefined;
    while ((cur = frontier.pop()) && (done.has(cur.node.key) || best.get(cur.node.key) !== cur));

    // every reachable layout is settled, so dst was never reachable
    if (!cur) throw new Error(`reshard has no legal path`);
    const curKey = cur.node.key;

    // success!
    if (curKey === target) {
      const plan = {
        time: cur.time,
        collectives: cur.steps.map(({ from, to, collective }): PlannedCollective => ({
          kind: 'collective',
          ...collective,
          x: typeOf(from),
          out: typeOf(to),
        })),
      };
      cache?.set(memoKey, plan);
      return plan;
    }

    done.add(curKey);

    // relax: keep whichever way to each neighboring layout is cheaper.
    // A layout's rewrites depend on its sharding alone, so they are
    // enumerated once per layout and only priced per shape
    const t = typeOf(cur.node);
    if (!cur.node.edges) {
      const edges: NonNullable<LayoutNode['edges']> = { collective: [], next: [] };
      for (const rewrite of rewrites(t, dst, names)) {
        const isCollective = 'kind' in rewrite;
        edges.collective.push(isCollective ? descOf(rewrite) : null);
        edges.next.push(nodeOf(isCollective ? rewrite.out : rewrite));
      }
      cur.node.edges = edges;
    }
    const { collective: collectives, next: nexts } = cur.node.edges;
    for (let e = 0; e < nexts.length; e++) {
      const [collective, next] = [collectives[e], nexts[e]];
      const nextKey = next.key;

      const cost = collective ? pricer(collective.variant, collective.axes, t, elemBytes) : 0;
      const time = cur.time + cost;

      if (best.has(nextKey) && best.get(nextKey)!.time <= time) continue;

      // a slice moves no data, so it adds no step: the next collective
      // simply starts from the sliced layout
      const steps = collective
        ? [...cur.steps, { from: cur.node, to: next, collective }]
        : cur.steps;
      reach(nextKey, { time, node: next, steps });
    }
  }
}

// A binary min-heap on (time, order): the cheapest layout, and among
// equally cheap ones the earliest reached.
class Frontier<T extends { time: number; order: number }> {
  private heap: T[] = [];
  private static less(a: { time: number; order: number }, b: { time: number; order: number }) {
    return a.time < b.time || (a.time === b.time && a.order < b.order);
  }
  push(x: T) {
    const h = this.heap;
    let i = h.push(x) - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!Frontier.less(h[i], h[p])) break;
      [h[i], h[p]] = [h[p], h[i]];
      i = p;
    }
  }
  pop(): T | undefined {
    const h = this.heap;
    const top = h[0];
    const last = h.pop();
    if (h.length && last) {
      h[0] = last;
      for (let i = 0; ;) {
        const [l, r] = [2 * i + 1, 2 * i + 2];
        let m = i;
        if (l < h.length && Frontier.less(h[l], h[m])) m = l;
        if (r < h.length && Frontier.less(h[r], h[m])) m = r;
        if (m === i) break;
        [h[i], h[m]] = [h[m], h[i]];
        i = m;
      }
    }
    return top;
  }
}

// Every legal single-collective rewrite from layout t, minus moves that
// provably cannot be on the cheapest path to dst. The prunes rest on one
// fact: a collective's price depends on payload sizes and mesh axes
// only, never on which tensor dim an axis shards. So (1) every scatter,
// route, and slice places its axis directly where dst wants it, because
// any other placement prices the same now and needs a paid fix-up later,
// (2) axes already in place are never touched, because gathering one
// inflates every later payload and it would have to be re-sliced anyway,
// and (3) partials dst keeps are never reduced, because no rewrite
// recreates a partial, so dst would become unreachable. Axes dst leaves
// idle do stay in play: slicing over one is free and shrinks a following
// reduction, and that can win even with the gather back afterwards (the
// 2D all-reduce trick). A rewrite missing from this list can only make a
// plan pricier, never impossibly cheap.
function rewrites(
  t: TensorType,
  dst: TensorType,
  names: MeshAxis[],
): (PlannedCollective | TensorType)[] {
  const out: (PlannedCollective | TensorType)[] = [];
  const used = new Set([...t.sharding.flat(), ...t.unreduced]);

  // the tensor dim dst shards each axis over, absent for gathered axes
  const dstDim = new Map<MeshAxis, number>();
  dst.sharding.forEach((axes, i) => axes.forEach((a) => dstDim.set(a, i)));

  // t with the given tensor dims resharded, and optionally fewer partials
  const edit = (dims: Record<number, MeshAxis[]>, unreduced = t.unreduced) =>
    tt(
      t.shape,
      t.sharding.map((axes, i) => dims[i] ?? axes),
      unreduced,
    );

  const op = (
    variant: CollectiveOp['variant'],
    over: MeshAxis[],
    next: TensorType,
  ): PlannedCollective => ({
    kind: 'collective',
    variant,
    label: `${variant} ${over.join('')}`,
    axes: over,
    x: t,
    out: next,
  });

  const reducible = t.unreduced.filter((a) => !dst.unreduced.includes(a));
  for (const u of subsets(reducible)) {
    if (u.length === 0) continue;
    const rest = t.unreduced.filter((a) => !u.includes(a));

    // all-reduce: partial sums become replicated values
    out.push(op('all-reduce', u, edit({}, rest)));

    // fused reduce-scatter: each partial lands on its destination dim
    const targets: Record<number, MeshAxis[]> = {};
    for (const a of u) {
      const j = dstDim.get(a) ?? 0;
      (targets[j] ??= [...t.sharding[j]]).push(a);
    }
    out.push(op('reduce-scatter', u, edit(targets, rest)));
  }

  // every sharded axis with the dim holding it
  const sharded: [MeshAxis, number][] = [];
  t.sharding.forEach((axes, i) => axes.forEach((a) => sharded.push([a, i])));

  // fused all-gather: any subset becomes replication in one op, spanning
  // tensor dims so the grid concurrency is expressible. Axes already at
  // their destination stay eligible on purpose: pulling one into the
  // group adds its links to the grid ingress and the re-slice back is
  // free, which can beat gathering the others alone (fuzz-found).
  for (const s of subsets(sharded)) {
    if (s.length === 0) continue;
    const strip: Record<number, MeshAxis[]> = {};
    for (const [a, i] of s) strip[i] = (strip[i] ?? t.sharding[i]).filter((x) => x !== a);
    out.push(
      op(
        'all-gather',
        s.map(([a]) => a),
        edit(strip),
      ),
    );
  }

  // all-to-all: each misplaced axis routes straight to the tensor dim
  // dst wants it on, and any subset of them can share one fused
  // exchange. The fusion exists for split axes: routing X0 and then X1
  // in separate ops pays for two exchanges, but if both move at once
  // the pricer sees all of X moving and charges a single whole-axis
  // exchange (the twin fuzz caught this overcharge). Axes already on
  // their destination dim stay out: unlike the fused all-gather above,
  // an extra axis brings no bandwidth to an a2a, it would just be
  // moved off its correct dim and paid for again on the way back
  const misplaced = sharded.filter(([a, i]) => (dstDim.get(a) ?? i) !== i);
  for (const s of subsets(misplaced)) {
    if (s.length === 0) continue;
    const routed: Record<number, MeshAxis[]> = {};
    const at = (k: number) => routed[k] ?? t.sharding[k];
    for (const [a, i] of s) {
      const j = dstDim.get(a)!;
      routed[i] = at(i).filter((x) => x !== a);
      routed[j] = [...at(j), a];
    }

    out.push(
      op(
        'all-to-all',
        s.map(([a]) => a),
        edit(routed),
      ),
    );
  }

  // slicing a replicated tensor over an idle axis moves no data, every
  // chip just keeps its own piece (spare axes canonically onto dim 0)
  for (const name of names)
    if (!used.has(name) && !dst.unreduced.includes(name)) {
      const j = dstDim.get(name) ?? 0;
      out.push(edit({ [j]: [...t.sharding[j], name] }));
    }

  return out;
}
