import { expect, test } from 'vitest';
import { makeMesh, roleSize, validateSizes } from '../src/core/engine/surface/deploy';
import { MODEL_PRESETS } from '../src/core/model/models';
import { PhysAxis } from '../src/core/hardware/topology';

// the running example: (X=4, Y=2, Z=2), attention DPA=Z TP=XY, MoE EP=XY ETP=Z
const AXES: PhysAxis[] = [
  { name: 'X', size: 4, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true },
  { name: 'Y', size: 2, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true },
  { name: 'Z', size: 2, kind: 'ring', bandwidth: 90e9, latency: 0, wrap: true },
];
const ROLES = { DPA: ['Z'], TP: ['X', 'Y'], EP: ['X', 'Y'], ETP: ['Z'], PP: [] };

test('the running example builds with derived role sizes', () => {
  const m = makeMesh(AXES, ROLES);
  expect(roleSize(m, 'TP')).toBe(8);
  expect(roleSize(m, 'DPA')).toBe(2);
  expect(roleSize(m, 'ETP')).toBe(2);
  expect(roleSize(m, 'PP')).toBe(1);
});

test('split mesh dims carry different roles per phase', () => {
  const m = makeMesh(
    AXES.slice(0, 2).map((ax) => ({ ...ax, size: ax.name === 'Y' ? 4 : 2 })),
    { DPA: ['X'], TP: ['Y0', 'Y1'], EP: ['X', 'Y0'], ETP: ['Y1'], PP: [] },
    { Y: [2, 2] },
  );
  expect(roleSize(m, 'TP')).toBe(4);
  expect(roleSize(m, 'EP')).toBe(4);
  expect(m.dims.map((d) => [d.name, d.size, d.axis.name, d.stride])).toEqual([
    ['X', 2, 'X', 1],
    ['Y0', 2, 'Y', 1],
    ['Y1', 2, 'Y', 2],
  ]);
});

test('PP carves the machine and is excluded from both phases', () => {
  const m = makeMesh(
    AXES,
    { DPA: ['Y'], TP: ['Z'], EP: ['Y', 'Z'], ETP: [], PP: ['X0', 'X1'] },
    { X: [2, 2] },
  );
  expect(roleSize(m, 'PP')).toBe(4);
});

test('malformed placements throw', () => {
  // splits do not multiply to the axis size
  expect(() => makeMesh(AXES, ROLES, { X: [3, 2] })).toThrow();
  // unknown mesh dim
  expect(() =>
    makeMesh(AXES, { DPA: ['Z'], TP: ['X', 'Y'], EP: ['X', 'Y'], ETP: ['Q'], PP: [] }),
  ).toThrow();
  // Z reused within the attention phase
  expect(() =>
    makeMesh(AXES, { DPA: ['Z'], TP: ['X', 'Z'], EP: ['X', 'Y'], ETP: ['Z'], PP: [] }),
  ).toThrow();
  // Y owned by no attention-phase role
  expect(() =>
    makeMesh(AXES, { DPA: ['Z'], TP: ['X'], EP: ['X', 'Y'], ETP: ['Z'], PP: [] }),
  ).toThrow();
  // a phase role reuses PP's dim
  expect(() =>
    makeMesh(
      AXES,
      { DPA: ['Z'], TP: ['X0', 'Y'], EP: ['X0', 'Y'], ETP: ['Z'], PP: ['X0'] },
      { X: [2, 2] },
    ),
  ).toThrow();
});

test('validateSizes flags model-dependent feasibility', () => {
  const llama = MODEL_PRESETS.find((m) => m.name === 'LLaMA 3 8B')!; // dense, N=32 K=8 L=32
  const oss = MODEL_PRESETS.find((m) => m.name === 'gpt-oss-20b MXFP4/BF16')!; // 32 experts
  const kimi = MODEL_PRESETS.find((m) => m.name.startsWith('Kimi K2.6'))!; // MLA, 384 experts
  const check = (
    m: (typeof MODEL_PRESETS)[number],
    s: Partial<Record<'TP' | 'PP' | 'EP', number>>,
  ) => validateSizes(m, { TP: 1, PP: 1, EP: 1, ...s });
  const codes = (...args: Parameters<typeof check>) => check(...args).map((x) => x.code);

  expect(codes(llama, { TP: 2 })).toEqual([]);

  // TP past the query heads is an error, past only the KV heads a warning
  expect(codes(llama, { TP: 64 })).toContain('tp-exceeds-heads');
  const kv = check(llama, { TP: 16 });
  expect(kv).toEqual([expect.objectContaining({ severity: 'warning', code: 'kv-replicated' })]);

  expect(codes(llama, { PP: 64 })).toEqual(['pp-exceeds-layers']);

  // EP against the expert count: absent, exceeded, and non-divisor
  const dense = check(llama, { EP: 4 });
  expect(dense).toEqual([expect.objectContaining({ severity: 'warning', code: 'ep-without-moe' })]);
  expect(codes(oss, { EP: 64 })).toEqual(['ep-exceeds-experts']);
  expect(codes(kimi, { EP: 5 })).toEqual(['ep-not-divisor']);

  expect(codes(kimi, { TP: 2 })).toEqual(['mla-kv-replicated']);
});
