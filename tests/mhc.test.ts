import { expect, test } from 'vitest';
import { MODEL_PRESETS, ModelSpec } from '../src/core/model/models';
import { lowerStage } from '../src/core/engine/sim/lowering/lower';
import { partitionIntoStages } from '../src/core/engine/sim/lowering/stages';
import { makeMesh } from '../src/core/engine/surface/deploy';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { PhysAxis } from '../src/core/hardware/topology';
import { totalParams } from '../src/core/model/utils';

// Two chips on one axis, both owned by PP, so every stage but the last
// emits a boundary send and nothing else is sharded.
const AXES: PhysAxis[] = [
  { name: 'X', size: 2, kind: 'switch', bandwidth: 4e11, latency: 0, wrap: false },
];
const mesh = () => makeMesh(AXES, { DPA: [], TP: [], EP: [], ETP: [], PP: ['X'] });

// Width of the p2p payload stage 0 hands to stage 1, in elements per token.
function boundaryWidth(m: ModelSpec): number {
  const stages = partitionIntoStages(m, 2);
  const segs = lowerStage(
    {
      model: m,
      deployment: { chip: CHIPS_BY_ID['h100-sxm'], mesh: mesh(), moeDispatch: 'ring-of-experts' },
      phase: 'decode',
      tokensTotal: 1,
      tokensPerGroup: 1,
      seqsPerGroup: 1,
      ctx: 1024,
      includeBoundary: true,
    },
    stages[0],
  );
  const p2p = segs.flatMap((s) => s.ops).filter((o) => o.kind === 'p2p');
  expect(p2p).toHaveLength(1);
  return p2p[0].out.shape[p2p[0].out.shape.length - 1];
}

test('hyper-connections widen only the stage-boundary send', () => {
  const v4 = MODEL_PRESETS.find((x) => x.name.startsWith('DeepSeek V4 Flash'))!;
  expect(v4.residualStreams).toBe(4);

  // mHC hands the next stage 4 streams, not 1
  expect(boundaryWidth(v4)).toBe(4 * v4.modelDim);

  // and it is only the send: the mixing vectors are not modeled, so the
  // parameter count must not move with the expansion rate
  const single = { ...v4, residualStreams: 1 };
  expect(totalParams(single)).toBe(totalParams(v4));
  expect(boundaryWidth(single)).toBe(v4.modelDim);

  // every other preset keeps a single stream
  for (const m of MODEL_PRESETS)
    if (m.residualStreams === undefined) expect(boundaryWidth(m)).toBe(m.modelDim);
});

test('the DSA indexer carries its own weights', () => {
  const glm = MODEL_PRESETS.find((x) => x.name.startsWith('GLM 5.2'))!;
  const b = glm.blocks[0].pattern[0].block;
  if (b.attn.kind !== 'mla' || !b.attn.dsa) throw new Error('expected an MLA block with DSA');
  const { dsa, dqc } = b.attn;

  // per-head queries off the q latent, the keys they score against, and the
  // per-head score projection, shared across shareEvery layers
  const perLayer =
    (dqc! * dsa.indexHeads * dsa.indexHeadDim +
      glm.modelDim * dsa.indexHeadDim +
      glm.modelDim * dsa.indexHeads) /
    dsa.shareEvery;
  expect(perLayer).toBe(2_342_912);

  // 78 layers of it, which is what this branch used to drop on the floor
  const withoutIndexer = { ...b.attn, dsa: undefined } as const;
  const stripped: ModelSpec = {
    ...glm,
    blocks: glm.blocks.map((g) => ({
      ...g,
      pattern: g.pattern.map((r) => ({ ...r, block: { ...r.block, attn: withoutIndexer } })),
    })),
  };
  expect(totalParams(glm) - totalParams(stripped)).toBe(78 * perLayer);
});
