import { expect, test } from 'vitest';
import { deployedAxes, factorAxes } from '../src/core/hardware/topology';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';

const v5p = CHIPS_BY_ID['tpu-v5p'].interconnect;
const v5e = CHIPS_BY_ID['tpu-v5e'].interconnect;
const h100 = CHIPS_BY_ID['h100-sxm'].interconnect;

test('torus slice resolves to per-axis rings', () => {
  const slice = v5p.topologies!.find((t) => t.name === '4x4x8')!;
  expect(deployedAxes(v5p, slice)).toEqual([
    { name: 'X', size: 4, kind: 'ring', bandwidth: 90e9, latency: 1e-6, wrap: true },
    { name: 'Y', size: 4, kind: 'ring', bandwidth: 90e9, latency: 1e-6, wrap: true },
    { name: 'Z', size: 8, kind: 'ring', bandwidth: 90e9, latency: 1e-6, wrap: true },
  ]);
});

test('cylinder wraps only the long axis', () => {
  const slice = v5e.topologies!.find((t) => t.name === '8x16')!;
  const axes = deployedAxes(v5e, slice);
  expect(axes.map((a) => a.wrap)).toEqual([false, true]);
  expect(axes[0].bandwidth).toBe(45e9);
});

test('switched board plus scale-out tier', () => {
  expect(deployedAxes(h100, { domain: 8, nodes: 2 })).toEqual([
    { name: 'D', size: 8, kind: 'switch', bandwidth: 450e9, latency: 2e-6, wrap: false },
    { name: 'N', size: 2, kind: 'switch', bandwidth: 50e9, latency: 5e-6, wrap: false },
  ]);
  expect(deployedAxes(h100, { domain: 4, nodes: 1 })).toEqual([
    { name: 'D', size: 4, kind: 'switch', bandwidth: 450e9, latency: 2e-6, wrap: false },
  ]);
  expect(() => deployedAxes(h100, { domain: 16, nodes: 1 })).toThrow();
  expect(() => deployedAxes(h100, { domain: 8, nodes: 16 })).toThrow();
});

test('the fabric family and the use variant must agree', () => {
  expect(() => deployedAxes(v5p, { domain: 4, nodes: 1 })).toThrow();
  expect(() => deployedAxes(h100, v5p.topologies![0])).toThrow();
});

test('factorAxes splits with inner-to-outer strides', () => {
  const axes = deployedAxes(
    v5p,
    v5p.topologies!.find((t) => t.name === '2x4x4')!,
  );
  expect(
    factorAxes(axes, { Y: [2, 2] }).map((d) => [d.name, d.size, d.axis.name, d.stride]),
  ).toEqual([
    ['X', 2, 'X', 1],
    ['Y0', 2, 'Y', 1],
    ['Y1', 2, 'Y', 2],
    ['Z', 4, 'Z', 1],
  ]);
  expect(() => factorAxes(axes, { Y: [3, 2] })).toThrow();
  expect(() => factorAxes(axes, { Q: [2] })).toThrow();
});
