import { expect, test } from 'vitest';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import { roleSize } from '../src/core/engine/surface/deploy';
import { sameAxes } from '../src/core/engine/sim/ir/tensors';
import { PhysAxis } from '../src/core/hardware/topology';

// the running example machine: (X=4, Y=2, Z=2)
const AXES: PhysAxis[] = [
  { name: 'X', size: 4, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true },
  { name: 'Y', size: 2, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true },
  { name: 'Z', size: 2, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true },
];

test('the running example config enumerates its known placement', () => {
  const meshes = enumeratePlacements(AXES, { DPA: 2, TP: 8, EP: 8, ETP: 2, PP: 1 });
  // 4 prime slots, one to DPA and independently one to ETP: 4 x 4 = 16,
  // quotiented by the Y<->Z relabeling (identical axes): 4 pairs on
  // {X0, X1} are fixed, the other 12 pair up, 4 + 6 = 10
  expect(meshes.length).toBe(10);
  // the known placement, up to the free Y<->Z relabeling
  expect(
    meshes.some((m) =>
      [
        ['Z', 'Y'],
        ['Y', 'Z'],
      ].some(
        ([a, b]) =>
          sameAxes(m.roles.DPA, [a]) &&
          sameAxes(m.roles.TP, ['X', b]) &&
          sameAxes(m.roles.EP, ['X', b]) &&
          sameAxes(m.roles.ETP, [a]),
      ),
    ),
  ).toBe(true);
  for (const m of meshes) {
    expect(roleSize(m, 'DPA')).toBe(2);
    expect(roleSize(m, 'TP')).toBe(8);
    expect(roleSize(m, 'EP')).toBe(8);
    expect(roleSize(m, 'ETP')).toBe(2);
  }
});

test('PP occupies shared slots and both phases split the remainder', () => {
  const meshes = enumeratePlacements(AXES, { DPA: 1, TP: 8, EP: 4, ETP: 2, PP: 2 });
  for (const m of meshes) {
    expect(roleSize(m, 'PP')).toBe(2);
    expect(roleSize(m, 'TP')).toBe(8);
    expect(roleSize(m, 'EP')).toBe(4);
  }
  // PP on one whole size-2 axis with TP spanning X and the other must
  // be among them (whichever Y/Z labeling survived the quotient)
  expect(
    meshes.some(
      (m) =>
        (sameAxes(m.roles.PP, ['Z']) && sameAxes(m.roles.TP, ['X', 'Y'])) ||
        (sameAxes(m.roles.PP, ['Y']) && sameAxes(m.roles.TP, ['X', 'Z'])),
    ),
  ).toBe(true);
});

test('a mixed-prime axis yields both stride structures', () => {
  const axes: PhysAxis[] = [
    { name: 'X', size: 6, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true },
  ];
  const meshes = enumeratePlacements(axes, { DPA: 2, TP: 3, EP: 2, ETP: 3, PP: 1 });
  // DPA can take the contiguous or the strided factor of 2, aligned with
  // EP or crossed against it: [2,3] and [3,2] splits x 1 assignment each
  // of the aligned case, and the crossed cases collapse (a 6-ring has one
  // 2x3 crossing per split order)
  const strides = meshes.map((m) => m.dims.find((d) => m.roles.DPA.includes(d.name))!.stride);
  expect(strides).toContain(1);
  expect(strides).toContain(3);
});

test('mismatched sizes throw before enumerating', () => {
  expect(() => enumeratePlacements(AXES, { DPA: 2, TP: 4, EP: 8, ETP: 2, PP: 1 })).toThrow();
  expect(() => enumeratePlacements(AXES, { DPA: 2, TP: 8, EP: 3, ETP: 2, PP: 1 })).toThrow();
});
