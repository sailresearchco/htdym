import { expect, test } from 'vitest';
import { BlockSpec } from '../src/core/model/block';
import { MODEL_PRESETS } from '../src/core/model/models';
import { DTYPE_BYTES } from '../src/core/model/dtype';
import {
  activeParams,
  flopsPerDecodeToken,
  kvBytesPerSeq,
  layerCount,
  totalParams,
  weightBytesTotal,
} from '../src/core/model/utils';

// Published counts are the one oracle a model spec has that is independent
// of this codebase: a transposed or dropped dimension moves them.

test('DeepSeek V4 Flash reproduces its hybrid architecture and published scale', () => {
  const m = MODEL_PRESETS.find((x) => x.name === 'DeepSeek V4 Flash MXFP4/FP8')!;
  const count = (p: (b: BlockSpec) => boolean) =>
    m.blocks.reduce(
      (s, g) => s + g.repeat * g.pattern.reduce((t, r) => t + (p(r.block) ? r.count : 0), 0),
      0,
    );

  expect(layerCount(m)).toBe(43);
  expect(count((b) => b.attn.kind === 'mqa')).toBe(2);
  expect(count((b) => b.attn.kind === 'csa')).toBe(21);
  expect(count((b) => b.attn.kind === 'hca')).toBe(20);
  expect(count((b) => b.mlp.kind === 'moe')).toBe(43);

  // The model card reports 284B total and 13B activated. Both are rounded;
  // norms, mHC's small mixing projections, and the MTP draft are not part of
  // the simulated decoder stack.
  expect(totalParams(m) / 284e9).toBeCloseTo(1, 2);
  expect(activeParams(m) / 13e9).toBeCloseTo(1, 1);

  // At 1M context: every layer retains 128 raw mixed FP8/BF16 entries,
  // CSA keeps one core + MXFP4 index entry per 4 tokens, and HCA keeps one
  // core entry per 128. This lands on the report's roughly 3.5 GB cache.
  const len = 1 << 20;
  const coreEntry = (512 - 64) * DTYPE_BYTES.fp8 + 64 * DTYPE_BYTES.bf16;
  const indexEntry = 128 * DTYPE_BYTES.mxfp4;
  const expectedKv =
    43 * 128 * coreEntry + 21 * (len / 4) * (coreEntry + indexEntry) + 20 * (len / 128) * coreEntry;
  expect(kvBytesPerSeq(m, DTYPE_BYTES[m.precision.kv], len)).toBe(expectedKv);
  expect(expectedKv / 1e9).toBeCloseTo(3.64, 2);

  // Every CSA indexer scores all len/4 compressed entries with 64x128
  // heads. Keep that work in its native FP4 bucket, separate from FP8 core
  // attention and projections.
  expect(flopsPerDecodeToken(m, len).get('fp4')).toBe(21 * 2 * (len / 4) * 64 * 128);
});

test('Kimi K3 reproduces its published architecture and parameter counts', () => {
  const m = MODEL_PRESETS.find((x) => x.name === 'Kimi K3 MXFP4/MXFP8')!;
  const count = (p: (b: BlockSpec) => boolean) =>
    m.blocks.reduce(
      (s, g) => s + g.repeat * g.pattern.reduce((t, r) => t + (p(r.block) ? r.count : 0), 0),
      0,
    );

  expect(layerCount(m)).toBe(93);
  expect(count((b) => b.attn.kind === 'linear')).toBe(69);
  expect(count((b) => b.attn.kind === 'mla')).toBe(24);
  expect(count((b) => b.mlp.kind === 'dense')).toBe(1);

  // 2.78T total and 104.2B activated, from the tech report's table 1. Norms,
  // short-conv taps and the router bias are not modeled, so a spec that is
  // otherwise right lands a fraction of a percent under.
  expect(totalParams(m) / 2.7799e12).toBeCloseTo(1, 2);
  expect(activeParams(m) / 104.19e9).toBeCloseTo(1, 2);

  // and the checkpoint's own total_size, which prices every category
  // separately: MXFP4 routed experts against everything else in bf16
  expect(weightBytesTotal(m) / 1_560_860_324_864).toBeCloseTo(1, 2);

  // the KDA layers hold a recurrent state that does not grow with context,
  // so a longer sequence only costs more in the 24 MLA layers' latent cache
  const kv = (len: number) => kvBytesPerSeq(m, DTYPE_BYTES[m.precision.kv], len);
  expect(kv(2 * 131072) - kv(131072)).toBe(24 * (512 + 64) * DTYPE_BYTES[m.precision.kv] * 131072);
});

test('Muse Glimmer 30B reproduces its hybrid attention stack and decoder scale', () => {
  const m = MODEL_PRESETS.find((x) => x.name === 'Muse Glimmer 30B')!;
  const count = (p: (b: BlockSpec) => boolean) =>
    m.blocks.reduce(
      (s, g) => s + g.repeat * g.pattern.reduce((t, r) => t + (p(r.block) ? r.count : 0), 0),
      0,
    );

  // config.json's layer_types is 13 repeats of sliding/sliding/sliding/full
  expect(layerCount(m)).toBe(52);
  expect(count((b) => b.attn.kind === 'gqa' && b.attn.window === 2048)).toBe(39);
  expect(count((b) => b.attn.kind === 'gqa' && b.attn.window === null)).toBe(13);

  // The checkpoint is an exact oracle here. Its safetensors index totals
  // 59,553,253,376 bytes across language_model + lm_head + vision_tower +
  // projector; peeling off the 1.853B vision tower, the 69M projector and
  // the 1.39M of 1-D norm weights the simulator omits leaves exactly the
  // decoder stack this preset describes. (The blog's "2B vision + 28B text"
  // split is that same number rounded.)
  expect(totalParams(m)).toBe(27_853_389_824);
  expect(weightBytesTotal(m)).toBe(55_706_779_648);

  // Every layer sigmoid-gates its attention output on a projection of the
  // layer input -- self_attn.gate_proj, [4096, 6656] in all 52 layers, so
  // D x N*H and not a narrower gate. Dropping it would cost 1.4B params.
  expect(count((b) => b.attn.kind === 'gqa' && b.attn.outputGate === true)).toBe(52);

  // 32:2 heads at headDim 128, so each entry is 2 heads x 128 x (K and V).
  // The 39 sliding layers stop growing once the 2048-token window fills.
  const kv = (len: number) => kvBytesPerSeq(m, DTYPE_BYTES[m.precision.kv], len);
  const perTokenPerLayer = 2 * 2 * 128 * DTYPE_BYTES.bf16;
  expect(kv(1024)).toBe(52 * 1024 * perTokenPerLayer);
  expect(kv(131072)).toBe((39 * 2048 + 13 * 131072) * perTokenPerLayer);
});

test('Qwen3 4B FP8 reproduces its published architecture and scale', () => {
  const m = MODEL_PRESETS.find((x) => x.name === 'Qwen3 4B FP8')!;
  const [block] = m.blocks[0].pattern;

  expect(layerCount(m)).toBe(36);
  expect(block.block.attn).toMatchObject({
    kind: 'gqa',
    queryHeads: 32,
    kvHeads: 8,
    headDim: 128,
    window: null,
  });
  expect(block.block.mlp).toMatchObject({ kind: 'dense', ffDim: 9728, variant: 'gated' });

  // The model card reports 4.0B total and 3.6B non-embedding parameters.
  // Norms are omitted by the simulator, so both published rounded counts
  // should agree with the modeled projection and embedding matrices.
  const embeddingParams = m.vocab * m.modelDim;
  expect(totalParams(m) / 4e9).toBeCloseTo(1, 1);
  expect((totalParams(m) - embeddingParams) / 3.6e9).toBeCloseTo(1, 1);

  expect(m.precision.weights.attention).toBe('fp8');
  expect(m.precision.weights.denseMlp).toBe('fp8');
  expect(m.precision.weights.embeddings).toBe('bf16');
  expect(m.precision.kv).toBe('bf16');
});

test('Qwen3 4B uses the same architecture at BF16 throughout', () => {
  const bf16 = MODEL_PRESETS.find((x) => x.name === 'Qwen3 4B BF16')!;
  const fp8 = MODEL_PRESETS.find((x) => x.name === 'Qwen3 4B FP8')!;
  const { name: _bf16Name, precision: _bf16Precision, ...bf16Architecture } = bf16;
  const { name: _fp8Name, precision: _fp8Precision, ...fp8Architecture } = fp8;

  expect(bf16Architecture).toEqual(fp8Architecture);
  expect(new Set(Object.values(bf16.precision.weights))).toEqual(new Set(['bf16']));
  expect(new Set(Object.values(bf16.precision.activations))).toEqual(new Set(['bf16']));
  expect(bf16.precision.residual).toBe('bf16');
  expect(bf16.precision.kv).toBe('bf16');
});

test('Qwen3.8 27B reproduces its hybrid linear/full stack and decoder scale', () => {
  const m = MODEL_PRESETS.find((x) => x.name === 'Qwen3.8 27B')!;
  const count = (p: (b: BlockSpec) => boolean) =>
    m.blocks.reduce(
      (s, g) => s + g.repeat * g.pattern.reduce((t, r) => t + (p(r.block) ? r.count : 0), 0),
      0,
    );

  // config.json's layer_types is 16 repeats of linear/linear/linear/full
  expect(layerCount(m)).toBe(64);
  expect(count((b) => b.attn.kind === 'linear')).toBe(48);
  expect(count((b) => b.attn.kind === 'gqa')).toBe(16);

  // The safetensors index totals 55,562,855,904 bytes across the decoder,
  // embeddings, lm_head, a 461M ViT and a 425M MTP draft head. Drop the two
  // the simulator does not run, plus the 2.6M of norms, conv taps and dt/A
  // params it omits by design, and exactly this stack is what remains.
  expect(totalParams(m)).toBe(26_893_352_960);
  expect(weightBytesTotal(m)).toBe(53_786_705_920);

  // attn_output_gate fuses the gate into q_proj, so the full-attention layers
  // ship [12288, 5120] there -- twice 24 heads x 256. The linear layers spell
  // the same gate separately as in_proj_z over their 48 value heads.
  expect(count((b) => b.attn.kind === 'gqa' && b.attn.outputGate === true)).toBe(16);

  // Only the 16 full-attention layers cache per token (4 kv heads x 256, K
  // and V); the 48 gated-deltanet layers hold a fixed-size recurrent state.
  const kv = (len: number) => kvBytesPerSeq(m, DTYPE_BYTES[m.precision.kv], len);
  expect(kv(2 * 131072) - kv(131072)).toBe(16 * 2 * 4 * 256 * DTYPE_BYTES.bf16 * 131072);
});

test('DeepSeek V4 Pro matches the released decoder layout and scale', () => {
  const m = MODEL_PRESETS.find((x) => x.name === 'DeepSeek V4 Pro MXFP4/FP8')!;
  const layers = m.blocks.flatMap((g) =>
    Array.from({ length: g.repeat }, () =>
      g.pattern.flatMap((r) => Array.from({ length: r.count }, () => r.block)),
    ).flat(),
  );
  // config.json has 61 decoder layers plus a trailing MTP entry. Pro's
  // first two layers compress at 128, unlike Flash's sliding-only prefix.
  expect(layerCount(m)).toBe(61);
  expect(layers.filter((b) => b.attn.kind === 'csa')).toHaveLength(30);
  expect(layers.filter((b) => b.attn.kind === 'hca')).toHaveLength(31);
  expect(layers.slice(0, 3).map((b) => b.attn.kind)).toEqual(['hca', 'hca', 'csa']);
  expect(layers.at(-1)?.attn.kind).toBe('csa');
  expect(layers[2].attn).toMatchObject({
    queryHeads: 128,
    qRank: 1536,
    outputGroups: 16,
    indexer: { topK: 1024 },
  });
  expect(layers.every((b) => b.mlp.kind === 'moe')).toBe(true);
  expect(m.residualStreams).toBe(4);
  // Match the model card at its published precision (one decimal trillion).
  // The simulated decoder omits the MTP draft and small norm/mHC weights.
  expect(totalParams(m) / 1e12).toBeCloseTo(1.6, 1);
  expect(activeParams(m) / 49e9).toBeCloseTo(1, 1);
  expect(m.precision.weights.routedExperts).toBe('mxfp4');
  expect(m.precision.activations.routedExperts).toBe('fp8');
  expect(m.precision.kv).toBe('fp8');
  const len = 1 << 20;
  const coreEntry = (512 - 64) * DTYPE_BYTES.fp8 + 64 * DTYPE_BYTES.bf16;
  const indexEntry = 128 * DTYPE_BYTES.mxfp4;
  expect(kvBytesPerSeq(m, DTYPE_BYTES[m.precision.kv], len)).toBe(
    61 * 128 * coreEntry + 30 * (len / 4) * (coreEntry + indexEntry) + 31 * (len / 128) * coreEntry,
  );
  expect(flopsPerDecodeToken(m, len).get('fp4')).toBe(30 * 2 * (len / 4) * 64 * 128);
});

test('GLM 5.3 FP8 retains the published GLM 5.2 architecture and serving precision', () => {
  const m = MODEL_PRESETS.find((x) => x.name === 'GLM 5.3 FP8')!;
  expect(MODEL_PRESETS.some((x) => x.name.startsWith('GLM 5.2'))).toBe(false);
  expect(layerCount(m)).toBe(78);
  expect(m.modelDim).toBe(6144);
  expect(m.vocab).toBe(154880);
  expect(m.blocks[0].pattern[0].count).toBe(3);
  expect(m.blocks[0].pattern[0].block.mlp).toMatchObject({ kind: 'dense', ffDim: 12288 });
  expect(m.blocks[1].pattern[0].count).toBe(75);
  expect(m.blocks[1].pattern[0].block.mlp).toMatchObject({
    kind: 'moe',
    experts: 256,
    topK: 8,
    expertDim: 2048,
    sharedExperts: 1,
  });
  expect(m.blocks[0].pattern[0].block.attn).toMatchObject({
    kind: 'mla',
    queryHeads: 64,
    headDim: 192,
    dc: 512,
    dqc: 2048,
    dRope: 64,
    valueHeadDim: 256,
    dsa: { topk: 2048, indexHeads: 32, indexHeadDim: 128, shareEvery: 4 },
  });
  expect(totalParams(m) / 744e9).toBeCloseTo(1, 2);
  expect(activeParams(m) / 40e9).toBeCloseTo(1, 1);
  expect(m.precision.weights.routedExperts).toBe('fp8');
  expect(m.precision.activations.routedExperts).toBe('fp8');
  expect(m.precision.weights.embeddings).toBe('bf16');
  expect(m.precision.residual).toBe('bf16');
  expect(m.precision.kv).toBe('fp8');
});
