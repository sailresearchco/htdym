import { expect, test } from 'vitest';
import { expandReshards, planReshard } from '../src/core/engine/sim/lowering/expand';
import { BoundBackend } from '../src/core/engine/sim/cost/types';
import { collectiveCost } from '../src/core/engine/sim/cost/helpers/collectives';
import { OpId, ReshardOp } from '../src/core/engine/sim/ir/ops';
import { tt } from '../src/core/engine/sim/ir/tensors';
import { MeshDim, PhysAxis } from '../src/core/hardware/topology';

const W = 100e9;

function ringAxis(name: string, size: number, wrap: boolean): PhysAxis {
  return { name, size, kind: 'ring', bandwidth: W, latency: 0, wrap };
}
function whole(axis: PhysAxis): MeshDim {
  return { name: axis.name, size: axis.size, axis, stride: 1 };
}

// the roofline oracle over the given mesh, as runPipeline would build it
function pricer(dims: MeshDim[]): { names: string[]; price: BoundBackend['priceCollective'] } {
  return {
    names: dims.map((d) => d.name),
    price: (kind, over, input, elemBytes) => collectiveCost(kind, over, input, elemBytes, dims),
  };
}

test('a bare partial expands to one all-reduce', () => {
  const dims = [whole(ringAxis('X', 4, true))];
  const { names, price } = pricer(dims);
  const { time, collectives } = planReshard(tt([1e6], [[]], ['X']), tt([1e6]), names, 2, price);
  expect(collectives.map((s) => [s.variant, s.axes])).toEqual([['all-reduce', ['X']]]);
  expect(time).toBe((0.75 * 2e6) / W); // 2 * (F/4 * 3) / 2W
});

test('slicing a replicated tensor is free', () => {
  const dims = [whole(ringAxis('X', 4, true))];
  const { names, price } = pricer(dims);
  const { time, collectives } = planReshard(tt([1e6]), tt([1e6], [['X']]), names, 2, price);
  expect(collectives).toEqual([]);
  expect(time).toBe(0);
});

test('moving an axis between dims is one all-to-all', () => {
  const sw: PhysAxis = {
    name: 'D',
    size: 8,
    kind: 'switch',
    bandwidth: W,
    latency: 0,
    wrap: false,
  };
  const dims = [whole(sw)];
  const { names, price } = pricer(dims);
  const src = tt([8e5, 64], [['D'], []]);
  const { time, collectives } = planReshard(src, tt([8e5, 64], [[], ['D']]), names, 2, price);
  expect(collectives.map((s) => s.variant)).toEqual(['all-to-all']);
  expect(time).toBe(collectiveCost('all-to-all', ['D'], src, 2, dims));
});

test('the running example boundary picks the cheapest path', () => {
  const dims = [
    whole({ name: 'X', size: 4, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true }),
    whole({ name: 'Y', size: 2, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true }),
    whole({ name: 'Z', size: 2, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true }),
  ];
  // attention out: [B_Z, D] with a TP partial over XY, MoE wants it replicated
  const src = tt([256, 7168], [['Z'], []], ['X', 'Y']);
  const dst = tt([256, 7168]);
  const { names, price } = pricer(dims);
  const { time, collectives } = planReshard(src, dst, names, 2, price);

  // the winner must price no worse than the hand-written candidates from
  // the design doc: all-reduce then gather, and scatter then big gather
  const ar = collectiveCost('all-reduce', ['X', 'Y'], src, 2, dims);
  const agZ = collectiveCost('all-gather', ['Z'], tt([256, 7168], [['Z'], []]), 2, dims);
  const rs = collectiveCost('reduce-scatter', ['X', 'Y'], src, 2, dims);
  const agAll = collectiveCost(
    'all-gather',
    ['X', 'Y', 'Z'],
    tt([256, 7168], [['Z', 'X', 'Y'], []]),
    2,
    dims,
  );
  expect(time).toBeLessThanOrEqual(ar + agZ);
  expect(time).toBeLessThanOrEqual(rs + agAll);

  // and the reported time must be exactly the sum of its own collectives
  const recomputed = collectives.reduce(
    (t, s) => t + collectiveCost(s.variant, s.axes, s.x, 2, dims),
    0,
  );
  expect(time).toBe(recomputed);
  expect(collectives.length).toBeGreaterThan(0);
});

test('partials the target keeps are not reduced', () => {
  const dims = [whole(ringAxis('X', 4, true)), whole(ringAxis('Y', 2, true))];
  const { names, price } = pricer(dims);
  const src = tt([1e6], [[]], ['X', 'Y']);
  const { collectives } = planReshard(src, tt([1e6], [[]], ['Y']), names, 2, price);
  expect(collectives.map((c) => [c.variant, c.axes])).toEqual([['all-reduce', ['X']]]);
});

test('a shape-changing reshard throws', () => {
  const dims = [whole(ringAxis('X', 4, true))];
  const { names, price } = pricer(dims);
  expect(() => planReshard(tt([1e6]), tt([2e6]), names, 2, price)).toThrow();
});

test('expandReshards wires the chain in place of the reshard', () => {
  const dims = [
    whole({ name: 'X', size: 4, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true }),
    whole({ name: 'Y', size: 2, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true }),
    whole({ name: 'Z', size: 2, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true }),
  ];
  const { price } = pricer(dims);
  const mesh = { dims, roles: { DPA: [], TP: [], EP: [], ETP: [], PP: [] } };
  const dst = tt([256, 7168]);
  const op: ReshardOp = {
    kind: 'reshard',
    dtype: 'bf16',
    id: 'reshard#5' as OpId,
    label: 'reshard',
    deps: ['o-proj#4' as OpId],
    x: tt([256, 7168], [['Z'], []], ['X', 'Y']),
    out: dst,
  };
  const out = expandReshards([{ label: 'boundary', ops: [op], repeat: 1 }], mesh, {
    priceCollective: price,
    priceCollectiveHash: 'expand-test',
  })[0].ops;

  // the running example boundary needs more than one collective
  expect(out.length).toBeGreaterThan(1);
  // first op takes over the reshard's deps, the rest chain in order
  expect(out[0].deps).toEqual(['o-proj#4']);
  for (let i = 1; i < out.length; i++) expect(out[i].deps).toEqual([out[i - 1].id]);
  // the last op answers to the reshard's id and declared output
  expect(out[out.length - 1].id).toBe('reshard#5');
  expect(out[out.length - 1].out).toEqual(dst);
});

test('gathers spanning tensor dims fuse into one collective', () => {
  const dims = [whole(ringAxis('X', 4, true)), whole(ringAxis('Y', 4, true))];
  const { names, price } = pricer(dims);
  const src = tt([4e3, 4e3], [['X'], ['Y']]);
  const { time, collectives } = planReshard(src, tt([4e3, 4e3]), names, 2, price);
  // one grid-concurrent gather at summed ingress, not two staged hops
  expect(collectives.map((c) => [c.variant, c.axes])).toEqual([['all-gather', ['X', 'Y']]]);
  expect(time).toBe((2e6 * 15) / (4 * W)); // shard*(n-1)/(2 * 2W)
});

test('spare axes parallelize a reduction (2D all-reduce)', () => {
  const dims = [whole(ringAxis('X', 4, true)), whole(ringAxis('Y', 4, true))];
  const { names, price } = pricer(dims);
  const src = tt([4e6], [[]], ['X']);
  const { time, collectives } = planReshard(src, tt([4e6]), names, 2, price);
  // slicing over idle Y shrinks the reduction payload, then gathers back
  expect(time).toBeLessThan(collectiveCost('all-reduce', ['X'], src, 2, dims));
  const recomputed = collectives.reduce(
    (t, c) => t + collectiveCost(c.variant, c.axes, c.x, 2, dims),
    0,
  );
  expect(time).toBe(recomputed);
});
