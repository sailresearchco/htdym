import { expect } from 'vitest';
import { fuzzTest } from './fuzz';
import { collectiveCost } from '../src/core/engine/sim/cost/helpers/collectives';
import { CollectiveKind } from '../src/core/engine/sim/ir/ops';
import { localElems, MeshAxis, tt } from '../src/core/engine/sim/ir/tensors';
import { MeshDim, PhysAxis } from '../src/core/hardware/topology';

// Physics properties collectiveCost must satisfy regardless of which
// candidate realizations it enumerates: linearity in payload, per-dim
// cut bounds (data crossing between slices of a dim cannot beat that
// dim's links), monotonicity in bandwidth, and wraparound never hurting.
// The bounds are derived here from first principles, so a candidate that
// overclaims (the failure direction that matters) breaks them.

type Kind = Exclude<CollectiveKind, 'p2p'>;

fuzzTest('collectiveCost respects physics over random channels', (rand) => {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];

  // random mesh, same families as the reshard fuzz
  const dims: MeshDim[] = [];
  let letter = 0;
  while (dims.length < 1 || (dims.length < 3 && rand() < 0.6)) {
    const name = 'ABCD'[letter++];
    const bw = pick([5e9, 2e10, 5e10, 1e11, 4e11]);
    const latency = pick([0, 0, 1e-6, 5e-6]);
    if (rand() < 0.2) {
      const axis: PhysAxis = {
        name,
        size: 4,
        kind: 'ring',
        bandwidth: bw,
        latency,
        wrap: rand() < 0.5,
      };
      dims.push({ name: `${name}0`, size: 2, stride: 1, axis });
      dims.push({ name: `${name}1`, size: 2, stride: 2, axis });
    } else {
      const ring = rand() < 0.6;
      const size = pick([2, 4, 8]);
      dims.push({
        name,
        size,
        stride: 1,
        axis: {
          name,
          size,
          kind: ring ? 'ring' : 'switch',
          bandwidth: bw,
          latency,
          wrap: ring && rand() < 0.5,
        },
      });
    }
  }

  const kind: Kind = pick(['all-reduce', 'all-gather', 'reduce-scatter', 'all-to-all']);
  const group = dims.filter(() => rand() < 0.6);
  if (!group.length) group.push(pick(dims));
  const over = group.map((d) => d.name);
  const rest = dims.filter((d) => !group.includes(d));

  // input honoring the payload convention, extra dims placed randomly
  const shape = [pick([256, 1024]), pick([64, 512])];
  const sharding: MeshAxis[][] = shape.map(() => []);
  const unreduced: MeshAxis[] = [];
  const place = (d: MeshDim, shardable: boolean) => {
    const roll = rand();
    if (roll < 0.4 && shardable) sharding[Math.floor(rand() * shape.length)].push(d.name);
    else if (roll < 0.6) unreduced.push(d.name);
  };
  if (kind === 'all-reduce' || kind === 'reduce-scatter') unreduced.push(...over);
  else group.forEach((d) => sharding[Math.floor(rand() * shape.length)].push(d.name));
  rest.forEach((d) => place(d, true));
  const input = tt(shape, sharding, unreduced);

  const bpe = pick([1, 2]);
  const price = collectiveCost(kind, over, input, bpe, dims);
  const n = group.reduce((p, d) => p * d.size, 1);
  if (n === 1) {
    expect(price).toBe(0);
    return;
  }

  // affine in payload: the wire part doubles, the latency part rides
  // along, so doubling sits between 1x and 2x (exactly 2x when every
  // latency is zero)
  const doubled = tt([shape[0] * 2, shape[1]], sharding, unreduced);
  const twice = collectiveCost(kind, over, doubled, bpe, dims);
  if (group.every((d) => !d.axis.latency)) expect(twice / (2 * price)).toBeCloseTo(1, 9);
  else {
    expect(twice).toBeLessThanOrEqual(2 * price * (1 + 1e-9));
    expect(twice).toBeGreaterThanOrEqual(price * (1 - 1e-9));
  }

  // per-dim cut bounds: restricted to any one dim, the collective's
  // traffic must cross that dim's links regardless of how routing is
  // staged, so no realization may price below any dim's own time
  {
    const local = localElems(input, dims) * bpe;
    const shard = kind === 'all-gather' ? local : local / n;
    const passes = kind === 'all-reduce' ? 2 : 1;
    for (const d of group) {
      if (d.size === 1) continue;
      const closed = d.axis.kind === 'ring' && d.axis.wrap && d.size * d.stride === d.axis.size;
      const linkBw = d.axis.kind === 'switch' ? d.axis.bandwidth : d.axis.bandwidth / d.stride;
      // a2a keeps the full payload on every dim (each chip owes local/n
      // to every peer, so local/n_d to each slice of dim d)
      const cut =
        kind === 'all-to-all'
          ? d.axis.kind === 'switch'
            ? (local * (d.size - 1)) / (d.size * linkBw)
            : (local * d.size) / ((closed ? 8 : 4) * linkBw)
          : (passes * (shard * (d.size - 1))) / ((closed ? 2 : 1) * linkBw);
      expect(price).toBeGreaterThanOrEqual(cut * (1 - 1e-9));
    }
  }

  // more bandwidth never costs more
  const faster = dims.map((d) => ({ ...d, axis: { ...d.axis, bandwidth: d.axis.bandwidth * 2 } }));
  // preserve shared parents for split factors
  for (const d of faster)
    for (const e of faster) if (e !== d && e.axis.name === d.axis.name) e.axis = d.axis;
  expect(collectiveCost(kind, over, input, bpe, faster)).toBeLessThanOrEqual(price * (1 + 1e-9));

  // closing a ring never costs more
  const wrapped = dims.map((d) => ({ ...d, axis: { ...d.axis, wrap: d.axis.kind === 'ring' } }));
  for (const d of wrapped)
    for (const e of wrapped) if (e !== d && e.axis.name === d.axis.name) e.axis = d.axis;
  expect(collectiveCost(kind, over, input, bpe, wrapped)).toBeLessThanOrEqual(price * (1 + 1e-9));

  // zeroing latency never costs more
  const instant = dims.map((d) => ({ ...d, axis: { ...d.axis, latency: 0 } }));
  for (const d of instant)
    for (const e of instant) if (e !== d && e.axis.name === d.axis.name) e.axis = d.axis;
  expect(collectiveCost(kind, over, input, bpe, instant)).toBeLessThanOrEqual(price * (1 + 1e-9));
});

fuzzTest('a fully split axis prices no worse than the whole axis', (rand) => {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rand() * xs.length)];

  const bw = pick([5e9, 5e10, 1e11]);
  const wrap = rand() < 0.5;
  const parent: PhysAxis = {
    name: 'P',
    size: 4,
    kind: 'ring',
    bandwidth: bw,
    latency: pick([0, 2e-6]),
    wrap,
  };
  const split: MeshDim[] = [
    { name: 'P0', size: 2, stride: 1, axis: parent },
    { name: 'P1', size: 2, stride: 2, axis: parent },
  ];
  const whole: MeshDim[] = [{ name: 'P', size: 4, stride: 1, axis: parent }];
  const kind: Kind = pick(['all-reduce', 'all-gather', 'reduce-scatter', 'all-to-all']);
  const reduces = kind === 'all-reduce' || kind === 'reduce-scatter';

  const asSplit = reduces ? tt([1024, 64], [], ['P0', 'P1']) : tt([1024, 64], [['P0', 'P1']]);
  const asWhole = reduces ? tt([1024, 64], [], ['P']) : tt([1024, 64], [['P']]);
  // composeFactors should recover the whole ring, wraparound included
  expect(collectiveCost(kind, ['P0', 'P1'], asSplit, 2, split)).toBeLessThanOrEqual(
    collectiveCost(kind, ['P'], asWhole, 2, whole) * (1 + 1e-9),
  );
});
