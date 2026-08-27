import { expect, test } from 'vitest';
import { collectiveCost } from '../src/core/engine/sim/cost/helpers/collectives';
import { tt } from '../src/core/engine/sim/ir/tensors';
import { MeshDim, PhysAxis } from '../src/core/hardware/topology';

const W = 100e9;

function ringAxis(name: string, size: number, wrap: boolean): PhysAxis {
  return { name, size, kind: 'ring', bandwidth: W, latency: 0, wrap };
}
function whole(axis: PhysAxis): MeshDim {
  return { name: axis.name, size: axis.size, axis, stride: 1 };
}

test('all-reduce on a wrapped ring, and 2x on an open segment', () => {
  const partial = tt([1e9], [[]], ['X']); // full 1 GB local tensor, partial over X
  // rs + ag over a closed 4-ring: 2 * (F/4 * 3) / 2W = 0.75 F/W
  expect(collectiveCost('all-reduce', ['X'], partial, 1, [whole(ringAxis('X', 4, true))])).toBe(
    (0.75 * 1e9) / W,
  );
  expect(collectiveCost('all-reduce', ['X'], partial, 1, [whole(ringAxis('X', 4, false))])).toBe(
    (1.5 * 1e9) / W,
  );
});

test('an even open mesh snakes into a closed ring', () => {
  const dims = [whole(ringAxis('X', 2, false)), whole(ringAxis('Y', 4, false))];
  const partial = tt([8e8], [[]], ['X', 'Y']);
  // snaked 8-ring closes without wraparound: 2 * (F/8 * 7) / 2W, half the
  // dimension-wise staging over two open axes
  expect(collectiveCost('all-reduce', ['X', 'Y'], partial, 1, dims)).toBe((0.875 * 8e8) / W);
});

test('a strided factor of an open parent pays for shared links', () => {
  const parent = ringAxis('X', 4, false);
  const dims: MeshDim[] = [
    { name: 'X0', size: 2, axis: parent, stride: 1 },
    { name: 'X1', size: 2, axis: parent, stride: 2 },
  ];
  const contiguous = collectiveCost('all-gather', ['X0'], tt([1e8], [['X0']]), 1, dims);
  const strided = collectiveCost('all-gather', ['X1'], tt([1e8], [['X1']]), 1, dims);
  expect(strided).toBe(2 * contiguous);
});

test('all-to-all is one hop on a switch, bisection-bound on a mesh', () => {
  const sw: PhysAxis = {
    name: 'D',
    size: 8,
    kind: 'switch',
    bandwidth: W,
    latency: 0,
    wrap: false,
  };
  const grid = [whole(ringAxis('X', 2, false)), whole(ringAxis('Y', 4, false))];
  // 1e8 local payload on each of 8 chips either way
  const swCost = collectiveCost('all-to-all', ['D'], tt([8e8], [['D']]), 1, [whole(sw)]);
  const gridCost = collectiveCost('all-to-all', ['X', 'Y'], tt([8e8], [['X', 'Y']]), 1, grid);
  expect(swCost).toBe((1e8 * 7) / (8 * W)); // F(n-1)/n through egress
  // both dims run concurrently, so the slowest dim's bisection is the
  // whole bill: max(F*2/4W, F*4/4W)
  expect(gridCost).toBe((1e8 * 4) / (4 * W));
});

test('a wrapped 3D torus uses all six link directions at once', () => {
  const dims = ['X', 'Y', 'Z'].map((name) => whole(ringAxis(name, 4, true)));
  const partial = tt([1.92e9], [[]], ['X', 'Y', 'Z']);
  // rs + ag with summed ingress 6W: 2 * (F/64 * 63) / 6W, 3x the
  // single-axis bandwidth
  const expected = (2 * (1.92e9 / 64) * 63) / (6 * W);
  expect(collectiveCost('all-reduce', ['X', 'Y', 'Z'], partial, 1, dims) / expected).toBeCloseTo(
    1,
    12,
  );
});

test('a whole-axis x contiguous-factor window snakes too', () => {
  const y = ringAxis('Y', 4, false);
  const dims: MeshDim[] = [
    whole(ringAxis('X', 4, false)),
    { name: 'Y0', size: 2, axis: y, stride: 1 },
  ];
  // the 4x2 window of an open mesh closes an 8-ring just like a 2x4 slice
  const partial = tt([8e8], [[]], ['X', 'Y0']);
  expect(collectiveCost('all-reduce', ['X', 'Y0'], partial, 1, dims)).toBe((0.875 * 8e8) / W);
});

test('factors of one axis compose back into the whole ring', () => {
  const parent = ringAxis('X', 4, true);
  const dims: MeshDim[] = [
    { name: 'X0', size: 2, axis: parent, stride: 1 },
    { name: 'X1', size: 2, axis: parent, stride: 2 },
  ];
  // the pair covers all 4 chips: one closed wrapped ring, shard*(n-1)/2W,
  // not two staged sub-rings
  expect(collectiveCost('all-gather', ['X0', 'X1'], tt([1e8], [['X0', 'X1']]), 1, dims)).toBe(
    (2.5e7 * 3) / (2 * W),
  );
});

test('the grid cannot beat a slow dim through its cut', () => {
  const big = ringAxis('B', 16, true);
  const dims: MeshDim[] = [
    whole(ringAxis('X', 4, true)),
    { name: 'B1', size: 2, axis: big, stride: 8 },
  ];
  // summed ingress would claim ~3.1 shard/W, but every shard must cross
  // between the B1 slices over W/8 links: agTime(shard, B1) = 4 shard/W
  const shard = tt([1e8], [['X', 'B1']]); // 1.25e7 local
  expect(collectiveCost('all-gather', ['X', 'B1'], shard, 1, dims)).toBe((4 * 1.25e7) / W);
});

test('hierarchical all-reduce shrinks before crossing the slow tier', () => {
  const d: PhysAxis = {
    name: 'D',
    size: 8,
    kind: 'switch',
    bandwidth: 4e11,
    latency: 0,
    wrap: false,
  };
  const n: PhysAxis = {
    name: 'N',
    size: 2,
    kind: 'switch',
    bandwidth: 5e10,
    latency: 0,
    wrap: false,
  };
  const partial = tt([6.4e8], [[]], ['D', 'N']);
  // reduce-scatter over the fast domain first so only 1/8 of the data
  // crosses the NIC tier, then gather back out
  expect(collectiveCost('all-reduce', ['D', 'N'], partial, 1, [whole(d), whole(n)])).toBe(
    2 * ((8e7 * 7) / 4e11 + 4e7 / 5e10),
  );
});

test('no candidate beats any dim cut bound', () => {
  const big = ringAxis('B', 16, true);
  const group: MeshDim[] = [
    whole(ringAxis('X', 4, true)),
    whole(ringAxis('Y', 3, false)),
    { name: 'B1', size: 2, axis: big, stride: 8 },
  ];
  const shard = 2.4e9 / 24;
  const cost = collectiveCost(
    'all-gather',
    ['X', 'Y', 'B1'],
    tt([2.4e9], [['X', 'Y', 'B1']]),
    1,
    group,
  );
  // every shard must cross into each neighboring slice of each dim
  const floors = [
    (shard * 3) / (2 * W), // X, closed
    (shard * 2) / W, // Y, open path
    shard / (2 * (W / 8)), // B1, closed but stride-derated
  ];
  for (const floor of floors) expect(cost).toBeGreaterThanOrEqual(floor);
});

test('mismatched input types throw', () => {
  const dims = [whole(ringAxis('X', 4, true))];
  // all-gather over a dim the input is not sharded over
  expect(() => collectiveCost('all-gather', ['X'], tt([1e9]), 1, dims)).toThrow();
  // all-reduce over a dim the input does not carry unreduced
  expect(() => collectiveCost('all-reduce', ['X'], tt([1e9], [['X']]), 1, dims)).toThrow();
});

test('degenerate groups cost nothing', () => {
  expect(collectiveCost('all-reduce', [], tt([1e9]), 2, [])).toBe(0);
  expect(
    collectiveCost('all-gather', ['X'], tt([1e9], [['X']]), 2, [whole(ringAxis('X', 1, false))]),
  ).toBe(0);
});

test('latency charges per pipelined step', () => {
  const lat = 2e-6;
  const sw: PhysAxis = {
    name: 'D',
    size: 8,
    kind: 'switch',
    bandwidth: W,
    latency: lat,
    wrap: false,
  };
  // switch all-gather: n-1 steps through the switch on top of the wire time
  expect(collectiveCost('all-gather', ['D'], tt([8e8], [['D']]), 1, [whole(sw)])).toBe(
    (1e8 * 7) / W + 7 * lat,
  );
  // switch a2a is a single concurrent exchange round: one traversal
  expect(collectiveCost('all-to-all', ['D'], tt([8e8], [['D']]), 1, [whole(sw)])).toBe(
    (1e8 * 7) / (8 * W) + lat,
  );

  // closed ring all-reduce: rs + ag, each ceil((n-1)/2) = 2 hops deep
  const ring: PhysAxis = {
    name: 'X',
    size: 4,
    kind: 'ring',
    bandwidth: W,
    latency: lat,
    wrap: true,
  };
  expect(
    collectiveCost('all-reduce', ['X'], tt([1e9], [[]], ['X']), 1, [whole(ring)]) /
      ((0.75 * 1e9) / W + 4 * lat),
  ).toBeCloseTo(1, 12);
});

test('a strided hop pays its physical links in series', () => {
  const lat = 1e-6;
  const parent: PhysAxis = {
    name: 'X',
    size: 4,
    kind: 'ring',
    bandwidth: W,
    latency: lat,
    wrap: false,
  };
  const dims: MeshDim[] = [
    { name: 'X0', size: 2, axis: parent, stride: 1 },
    { name: 'X1', size: 2, axis: parent, stride: 2 },
  ];
  // one open-path step each, but the stride-2 hop crosses 2 links
  expect(collectiveCost('all-gather', ['X0'], tt([1e8], [['X0']]), 1, dims)).toBe(5e7 / W + lat);
  expect(collectiveCost('all-gather', ['X1'], tt([1e8], [['X1']]), 1, dims)).toBe(
    5e7 / (W / 2) + 2 * lat,
  );
});

test('p2p pays the slowest crossed tier once', () => {
  const d: PhysAxis = {
    name: 'D',
    size: 8,
    kind: 'switch',
    bandwidth: 4e11,
    latency: 2e-6,
    wrap: false,
  };
  const n: PhysAxis = {
    name: 'N',
    size: 2,
    kind: 'switch',
    bandwidth: 5e10,
    latency: 5e-6,
    wrap: false,
  };
  expect(collectiveCost('p2p', ['D', 'N'], tt([1e8]), 1, [whole(d), whole(n)])).toBe(
    1e8 / 5e10 + 5e-6,
  );
});

test('hierarchical all-reduce latency follows the staged steps', () => {
  const d: PhysAxis = {
    name: 'D',
    size: 8,
    kind: 'switch',
    bandwidth: 4e11,
    latency: 2e-6,
    wrap: false,
  };
  const n: PhysAxis = {
    name: 'N',
    size: 2,
    kind: 'switch',
    bandwidth: 5e10,
    latency: 5e-6,
    wrap: false,
  };
  const partial = tt([6.4e8], [[]], ['D', 'N']);
  // same staging as the bandwidth-only sibling test: 7 fast steps and 1
  // slow step each way
  expect(
    collectiveCost('all-reduce', ['D', 'N'], partial, 1, [whole(d), whole(n)]) /
      (2 * ((8e7 * 7) / 4e11 + 7 * 2e-6 + 4e7 / 5e10 + 1 * 5e-6)),
  ).toBeCloseTo(1, 12);
});

test('a grown gather group can win, but never beats a member cut', () => {
  // growing a group CAN price cheaper (a fat axis's links carry the slow
  // axes' data, which the reshard planner exploits), but the cut floors
  // keep the claim physical: every shard still crosses each member dim
  const dims = [whole(ringAxis('N', 2, true)), whole(ringAxis('B', 2, true))];
  dims[0].axis.bandwidth = 5e9;
  const input = tt([1e6], [['N', 'B']]);
  const grown = collectiveCost('all-gather', ['N', 'B'], input, 1, dims);
  const shard = 2.5e5 * 1; // localElems * bytesPerElem
  expect(grown).toBeGreaterThanOrEqual(shard / (2 * 5e9)); // N cut
  expect(grown).toBeGreaterThanOrEqual((shard * 1) / (2 * W)); // B cut
});
