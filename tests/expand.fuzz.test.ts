import { expect } from 'vitest';
import { fuzzTest } from './fuzz';
import { randAxes, randPlacement, sizesOf } from './gen';
import { planReshard } from '../src/core/engine/sim/lowering/expand';
import { collectiveCost } from '../src/core/engine/sim/cost/helpers/collectives';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { CostBackend } from '../src/core/engine/sim/cost/types';
import { CollectiveKind } from '../src/core/engine/sim/ir/ops';
import { MeshAxis, sameAxes, TensorType, tt } from '../src/core/engine/sim/ir/tensors';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { Deployment } from '../src/core/engine/surface/deploy';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { MODEL_PRESETS } from '../src/core/model/models';
import { MeshDim, PhysAxis } from '../src/core/hardware/topology';

const hashed = makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 });
// same physics, but every bind mints a unique hash so plans never share:
// the ground truth the shared-cache path must bit-match
let bindSeq = 0;
const unshared: CostBackend = (ctx) => ({
  ...hashed(ctx),
  priceCollectiveHash: `unshared-${bindSeq++}`,
});

// planReshard prunes its rewrites by dominance arguments that claim to
// never change the argmin. This fuzz drives it against an UNPRUNED
// reference search (every scatter assignment, every detour, every axis
// touchable) over random cases: any wrong prune shows up as the
// reference finding a cheaper plan.

// the reference planner prices at ELEM_BYTES throughout, so its calls omit
// the width the real pricer takes
const ELEM_BYTES = 2;
type Price = (
  kind: Exclude<CollectiveKind, 'p2p'>,
  over: MeshAxis[],
  input: TensorType,
  elemBytes?: number,
) => number;

function subsets<T>(xs: T[]): T[][] {
  return xs
    .reduce<T[][]>((acc, x) => [...acc, ...acc.map((s) => [...s, x])], [[]])
    .filter((s) => s.length > 0);
}

// every function from u's axes to tensor dims
function assignments(count: number, rank: number): number[][] {
  return Array.from({ length: count }).reduce<number[][]>(
    (acc) => acc.flatMap((a) => Array.from({ length: rank }, (_, j) => [...a, j])),
    [[]],
  );
}

// every legal move, target-blind and unfused with nothing withheld
function allMoves(
  t: TensorType,
  names: MeshAxis[],
  price: Price,
): { cost: number; next: TensorType }[] {
  const out: { cost: number; next: TensorType }[] = [];
  const edit = (dims: Record<number, MeshAxis[]>, unreduced = t.unreduced) =>
    tt(
      t.shape,
      t.sharding.map((axes, i) => dims[i] ?? axes),
      unreduced,
    );

  for (const u of subsets(t.unreduced)) {
    const rest = t.unreduced.filter((a) => !u.includes(a));
    out.push({ cost: price('all-reduce', u, t), next: edit({}, rest) });
    for (const assign of assignments(u.length, t.shape.length)) {
      const target: Record<number, MeshAxis[]> = {};
      assign.forEach((j, idx) => (target[j] = [...(target[j] ?? t.sharding[j]), u[idx]]));
      out.push({ cost: price('reduce-scatter', u, t), next: edit(target, rest) });
    }
  }

  const pairs = t.sharding.flatMap((axes, i) => axes.map((a): [MeshAxis, number] => [a, i]));
  for (const s of subsets(pairs)) {
    const strip: Record<number, MeshAxis[]> = {};
    for (const [a, i] of s) strip[i] = (strip[i] ?? t.sharding[i]).filter((x) => x !== a);
    out.push({
      cost: price(
        'all-gather',
        s.map(([a]) => a),
        t,
      ),
      next: edit(strip),
    });
  }

  // every subset of sharded axes in one exchange, each axis to every
  // possible destination dim (fused a2as compose split factors, so the
  // reference must offer them too or the pruned search would beat it)
  for (const s of subsets(pairs)) {
    const choices = s.map(([, i]) =>
      Array.from({ length: t.shape.length }, (_, j) => j).filter((j) => j !== i),
    );
    for (const dsts of choices.reduce<number[][]>(
      (acc, js) => acc.flatMap((d) => js.map((j) => [...d, j])),
      [[]],
    )) {
      const routed: Record<number, MeshAxis[]> = {};
      const at = (k: number) => routed[k] ?? t.sharding[k];
      s.forEach(([a, i], idx) => {
        routed[i] = at(i).filter((x) => x !== a);
        routed[dsts[idx]] = [...at(dsts[idx]), a];
      });
      out.push({
        cost: price(
          'all-to-all',
          s.map(([a]) => a),
          t,
        ),
        next: edit(routed),
      });
    }
  }

  const used = new Set([...t.sharding.flat(), ...t.unreduced]);
  for (const name of names)
    if (!used.has(name))
      for (let i = 0; i < t.shape.length; i++)
        out.push({ cost: 0, next: edit({ [i]: [...t.sharding[i], name] }) });

  return out;
}

// Dijkstra with a lazy-deletion binary heap, the unpruned move set
// makes the state space too big for a linear scan
function referenceTime(src: TensorType, dst: TensorType, names: MeshAxis[], price: Price): number {
  const key = (t: TensorType) =>
    JSON.stringify([t.sharding.map((a) => [...a].sort()), [...t.unreduced].sort()]);
  const target = key(dst);
  const best = new Map<string, { time: number; type: TensorType }>([
    [key(src), { time: 0, type: src }],
  ]);
  const done = new Set<string>();

  const heap: [number, string][] = [[0, key(src)]];
  const push = (e: [number, string]) => {
    heap.push(e);
    for (let i = heap.length - 1; i > 0;) {
      const p = (i - 1) >> 1;
      if (heap[p][0] <= heap[i][0]) break;
      [heap[p], heap[i]] = [heap[i], heap[p]];
      i = p;
    }
  };
  const pop = (): [number, string] => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      for (let i = 0; ;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
        if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i], heap[m]];
        i = m;
      }
    }
    return top;
  };

  while (heap.length) {
    const [time, curKey] = pop();
    if (done.has(curKey)) continue;
    if (curKey === target) return time;
    done.add(curKey);
    for (const { cost, next } of allMoves(best.get(curKey)!.type, names, price)) {
      const k = key(next);
      const t = time + cost;
      if (best.has(k) && best.get(k)!.time <= t) continue;
      best.set(k, { time: t, type: next });
      push([t, k]);
    }
  }
  throw new Error(`unreachable`);
}

// Semantic oracle: replay the plan as layout transitions, verifying each
// collective is legal against its declared input and that consecutive
// layouts differ only by free slices. Catches right-cost-wrong-ops bugs
// the time comparison cannot.
type Planned = ReturnType<typeof planReshard>['collectives'][number];

const usedAxes = (t: TensorType) => new Set([...t.sharding.flat(), ...t.unreduced]);

// from -> to using free slices only: sharding grows by idle axes
function expectSliceReachable(from: TensorType, to: TensorType) {
  expect(sameAxes(from.unreduced, to.unreduced)).toBe(true);
  const inUse = usedAxes(from);
  to.sharding.forEach((axes, i) => {
    for (const a of from.sharding[i]) expect(axes).toContain(a);
    for (const a of axes) if (!from.sharding[i].includes(a)) expect(inUse.has(a)).toBe(false);
  });
}

function expectLegal(c: Planned) {
  const { x, out } = c;
  if (c.variant === 'all-reduce' || c.variant === 'reduce-scatter') {
    for (const a of c.axes) expect(x.unreduced).toContain(a);
    expect(
      sameAxes(
        out.unreduced,
        x.unreduced.filter((a) => !c.axes.includes(a)),
      ),
    ).toBe(true);
    const gained = out.sharding.flatMap((axes, i) =>
      axes.filter((a) => !x.sharding[i].includes(a)),
    );
    expect(sameAxes(gained, c.variant === 'reduce-scatter' ? c.axes : [])).toBe(true);
    out.sharding.forEach((axes, i) => {
      for (const a of x.sharding[i]) expect(axes).toContain(a);
    });
  } else if (c.variant === 'all-gather') {
    for (const a of c.axes) expect(x.sharding.flat()).toContain(a);
    expect(sameAxes(out.unreduced, x.unreduced)).toBe(true);
    out.sharding.forEach((axes, i) =>
      expect(
        sameAxes(
          axes,
          x.sharding[i].filter((a) => !c.axes.includes(a)),
        ),
      ).toBe(true),
    );
  } else {
    // all-to-all: same axes and partials, only the routed axes move dims
    expect(sameAxes(out.unreduced, x.unreduced)).toBe(true);
    expect(sameAxes(out.sharding.flat(), x.sharding.flat())).toBe(true);
    x.sharding.forEach((axes, i) => {
      for (const a of axes)
        if (c.axes.includes(a)) expect(out.sharding[i]).not.toContain(a);
        else expect(out.sharding[i]).toContain(a);
    });
  }
}

fuzzTest('pruned search matches the unpruned reference on random cases', (rand) => {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];

  const rank = pick([2, 2, 3, 3, 4]);
  const shape = Array.from({ length: rank }, () => pick([64, 256, 1024]));

  // meshes with split factors (contiguous + strided siblings), size-1
  // degenerates, switches, and wide bandwidth asymmetry
  const dims: MeshDim[] = [];
  let letter = 0;
  while (dims.length < 2 || (dims.length < 3 && rand() < 0.5)) {
    const name = 'ABCD'[letter++];
    const bw = pick([5e9, 2e10, 5e10, 1e11, 4e11]);
    if (rand() < 0.2) {
      const axis: PhysAxis = {
        name,
        size: 4,
        kind: 'ring',
        bandwidth: bw,
        latency: 0,
        wrap: rand() < 0.5,
      };
      dims.push({ name: `${name}0`, size: 2, stride: 1, axis });
      dims.push({ name: `${name}1`, size: 2, stride: 2, axis });
    } else if (rand() < 0.1) {
      dims.push({
        name,
        size: 1,
        stride: 1,
        axis: { name, size: 1, kind: 'ring', bandwidth: bw, latency: 0, wrap: false },
      });
    } else {
      const ring = rand() < 0.6;
      const size = pick([2, 4]);
      dims.push({
        name,
        size,
        stride: 1,
        axis: {
          name,
          size,
          kind: ring ? 'ring' : 'switch',
          bandwidth: bw,
          latency: 0,
          wrap: ring && rand() < 0.5,
        },
      });
    }
  }
  const names = dims.map((d) => d.name);

  // a placement per axis: one of the tensor dims, unreduced, or unused
  // (dst may only keep partials src already has)
  const layout = (allowUnreduced: (a: MeshAxis) => boolean) => {
    const sharding: MeshAxis[][] = shape.map(() => []);
    const unreduced: MeshAxis[] = [];
    for (const name of names) {
      const roll = Math.floor(rand() * (rank + 2));
      if (roll < rank) sharding[roll].push(name);
      else if (roll === rank && allowUnreduced(name)) unreduced.push(name);
    }
    return tt(shape, sharding, unreduced);
  };
  const src = layout(() => true);
  const dst = layout((a) => src.unreduced.includes(a));

  // fresh pricer per case so the plan cache cannot leak between meshes
  const price: Price = (kind, over, input, elemBytes = ELEM_BYTES) =>
    collectiveCost(kind, over, input, elemBytes, dims);

  const plan = planReshard(src, dst, names, ELEM_BYTES, price);

  // the unpruned reference explores every assignment and detour, which
  // grows too fast past a few axes, so optimality is asserted where
  // that is tractable and the semantic replay covers every case
  if ((rank + 2) ** names.length <= 2000) {
    const ref = referenceTime(src, dst, names, price);
    if (ref === 0) expect(plan.time).toBe(0);
    else expect(plan.time / ref).toBeCloseTo(1, 9);
  }

  // The width a reshard carries multiplies every candidate by the same
  // factor, so it moves what a plan costs but never what a plan is worth
  // relative to its rivals. A fixed per-collective term would break that,
  // and the planner would start choosing width-blind with nothing else
  // noticing. Only the price is pinned, not the chain: equal-cost chains
  // exist (gathering two axes separately ties with fusing them), and which
  // of them Dijkstra keeps turns on rounding that does not scale.
  const wider = planReshard(src, dst, names, 3 * ELEM_BYTES, price);
  expect(wider.time).toBeCloseTo(plan.time * 3, 9);

  // semantic replay: the emitted chain must actually realize the reshard
  let cur = src;
  for (const c of plan.collectives) {
    expectSliceReachable(cur, c.x);
    expectLegal(c);
    cur = c.out;
  }
  expectSliceReachable(cur, dst);
});

// The plan cache is shared between binds via priceCollectiveHash, a
// backend-DECLARED promise that the hash covers everything its
// collective pricing reads. This guard hunts for a stale promise: prices
// through the shared cache (polluted by an unrelated evaluation) must
// bit-match a backend whose binds can never share plans. If it
// fires, a context field started influencing prices without joining the
// hash.
fuzzTest('the shared plan cache never changes a price', (rand) => {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];

  const axes = randAxes(rand, pick, [2, 3, 4], [1, 2]);
  const other = randPlacement(axes, rand);
  const mesh = randPlacement(axes, rand);
  const model = pick(MODEL_PRESETS);
  const dispatch = pick(['ring-of-experts', 'expanded-a2a', 'coalesced-a2a'] as const);

  const price = (costBackend: CostBackend) => {
    const d: Deployment = { chip: CHIPS_BY_ID['h100-sxm'], mesh, moeDispatch: dispatch };
    const r = evaluatePrefill(
      { model, deployment: d, workload: { prefillLen: 512, generateLen: 0 } },
      sizesOf(mesh).DPA,
      'throughput',
      { costBackend },
    );
    return r.ok ? r.stepTime : null;
  };

  // pollute the shared cache with an unrelated same-axes evaluation
  evaluatePrefill(
    {
      model: pick(MODEL_PRESETS),
      deployment: { chip: CHIPS_BY_ID['h100-sxm'], mesh: other, moeDispatch: dispatch },
      workload: { prefillLen: 512, generateLen: 0 },
    },
    sizesOf(other).DPA,
    'throughput',
    { costBackend: hashed },
  );

  expect(price(hashed)).toBe(price(unshared));
});
