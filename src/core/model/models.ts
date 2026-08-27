import type { BlockGroup } from './block';
import { ALL_BF16, BF16_ALL, FP8_ALL, type PrecisionSpec } from './dtype';
import { compressedMqa, gqa, linearAttn, mla } from './block/attn';
import { denseMlp, moeMlp } from './block/mlp';
import { denseThenMoe, repeat, swaStack } from './block/stack';

export interface ModelSpec {
  name: string;
  // d_model (residual stream width)
  modelDim: number;
  // vocabulary size
  vocab: number;
  // tied input/output embeddings (V*D once instead of twice)
  tiedEmbeddings: boolean;
  // native weight/activation precision of the release
  precision: PrecisionSpec;
  // ordered layer-pattern groups; L = sum of repeat * pattern length
  blocks: BlockGroup[];
  // Hyper-connection expansion rate (DeepSeek V4's mHC): the state carried
  // between blocks is this many residual streams wide, while each block
  // still reads and writes one stream's worth. Only the pipeline
  // stage-boundary send sees the multiple. The mixing weights themselves
  // are a handful of vectors per layer and are omitted like the conv taps.
  // Absent = 1, an ordinary single residual stream.
  residualStreams?: number;
}

// Kimi K3 interleaves its two attention variants across several block
// groups, so its pieces are named once here rather than at every use.
const k3Kda = linearAttn({ N: 96, H: 128, valueHeads: 96, valueHeadDim: 128 });
const k3Mla = mla({
  N: 96,
  H: 128,
  dc: 512,
  dqc: 1536,
  dRope: 64,
  valueHeadDim: 128,
  outputGate: true,
});
const k3Moe = moeMlp({
  experts: 896,
  topK: 16,
  expertDim: 3072,
  sharedExperts: 2,
  latentDim: 3584,
});

const v4Attn = {
  N: 64,
  H: 512,
  qRank: 1024,
  outputGroups: 8,
  outputRank: 1024,
  window: 128,
  ropeDim: 64,
};
const v4Mqa = compressedMqa({ ...v4Attn, kind: 'mqa' });
const v4Csa = compressedMqa({
  ...v4Attn,
  kind: 'csa',
  compressionRate: 4,
  indexer: { heads: 64, headDim: 128, topK: 512, cacheDtype: 'mxfp4' },
});
const v4Hca = compressedMqa({ ...v4Attn, kind: 'hca', compressionRate: 128 });
const v4Moe = moeMlp({ experts: 256, topK: 6, expertDim: 2048, sharedExperts: 1 });

export const MODEL_PRESETS: ModelSpec[] = [
  {
    name: 'Gemma 4 31B',
    modelDim: 5376,
    vocab: 262144,
    tiedEmbeddings: true,
    precision: ALL_BF16,
    blocks: swaStack({
      layers: 60,
      localPerPattern: 5,
      globalPerPattern: 1,
      local: gqa({ N: 32, H: 256, kvHeads: 16, window: 1024 }),
      global: gqa({ N: 32, H: 512, kvHeads: 4, kEqV: true }),
      mlp: denseMlp(21504),
    }),
  },
  {
    name: 'Gemma 4 12B',
    modelDim: 3840,
    vocab: 262144,
    tiedEmbeddings: true,
    precision: ALL_BF16,
    blocks: swaStack({
      layers: 48,
      localPerPattern: 5,
      globalPerPattern: 1,
      local: gqa({ N: 16, H: 256, kvHeads: 8, window: 1024 }),
      global: gqa({ N: 16, H: 512, kvHeads: 1, kEqV: true }),
      mlp: denseMlp(15360),
    }),
  },
  {
    name: 'Gemma 4 26B A4B',
    modelDim: 2816,
    vocab: 262144,
    tiedEmbeddings: true,
    precision: ALL_BF16,
    blocks: swaStack({
      layers: 30,
      localPerPattern: 5,
      globalPerPattern: 1,
      local: gqa({ N: 16, H: 256, kvHeads: 8, window: 1024 }),
      global: gqa({ N: 16, H: 512, kvHeads: 2, kEqV: true }),
      mlp: moeMlp({ experts: 128, topK: 8, expertDim: 704, sharedExperts: 3 }),
    }),
  },
  {
    name: 'DeepSeek V4 Flash MXFP4/FP8',
    modelDim: 4096,
    vocab: 129280,
    tiedEmbeddings: false,
    precision: {
      // The native checkpoint keeps routed experts in E2M1 with one E8M0
      // scale per 32 values. Most remaining matrices are FP8; embeddings
      // and the router ship BF16.
      weights: { ...FP8_ALL, routedExperts: 'mxfp4', router: 'bf16', embeddings: 'bf16' },
      activations: { ...FP8_ALL, router: 'bf16', embeddings: 'bf16' },
      residual: 'bf16',
      // Core KV uses FP8 except for its 64 BF16 RoPE dimensions (declared
      // on the attention config); the Lightning Indexer cache is MXFP4.
      kv: 'fp8',
      indexer: 'fp4',
    },
    // mHC carries 4 residual streams between blocks (hc_mult in the config)
    residualStreams: 4,
    // Two sliding-only bootstrap layers, then CSA/HCA interleaved starting
    // and ending with CSA: 21 CSA + 20 HCA. Every layer is MoE; the first
    // three use hash routing, which has the same inference dimensions.
    blocks: [
      ...repeat(2, v4Mqa, v4Moe),
      ...swaStack({
        layers: 41,
        localPerPattern: 1,
        globalPerPattern: 1,
        local: v4Csa,
        global: v4Hca,
        mlp: v4Moe,
      }),
    ],
  },
  {
    name: 'GLM 5.2 FP8',
    modelDim: 6144,
    vocab: 154880,
    tiedEmbeddings: false,
    precision: {
      weights: { ...FP8_ALL, router: 'bf16', embeddings: 'bf16' },
      activations: { ...FP8_ALL, router: 'bf16', embeddings: 'bf16' },
      // the DeepSeek-style FP8 recipe casts matmul
      // inputs only, the stream between blocks stays bf16
      residual: 'bf16',
      kv: 'fp8',
    },
    blocks: denseThenMoe(
      78,
      3,
      mla({
        N: 64,
        H: 192,
        dc: 512,
        dqc: 2048,
        dRope: 64,
        valueHeadDim: 256,
        dsa: { topk: 2048, indexHeads: 32, indexHeadDim: 128, shareEvery: 4 },
      }),
      denseMlp(12288),
      moeMlp({ experts: 256, topK: 8, expertDim: 2048, sharedExperts: 1 }),
    ),
  },
  {
    name: 'Kimi K2.6 INT4/BF16',
    modelDim: 7168,
    vocab: 163840,
    tiedEmbeddings: false,
    precision: {
      weights: { ...BF16_ALL, routedExperts: 'int4' },
      activations: BF16_ALL,
      residual: 'bf16',
      kv: 'bf16',
    },
    blocks: denseThenMoe(
      61,
      1,
      mla({ N: 64, H: 128, dc: 512, dqc: 1536, dRope: 64, valueHeadDim: 128 }),
      denseMlp(18432),
      moeMlp({ experts: 384, topK: 8, expertDim: 2048, sharedExperts: 1 }),
    ),
  },
  {
    name: 'Kimi K3 MXFP4/MXFP8',
    modelDim: 7168,
    vocab: 163840,
    tiedEmbeddings: false,
    precision: {
      weights: { ...BF16_ALL, routedExperts: 'mxfp4' },
      activations: { ...BF16_ALL, routedExperts: 'mxfp8' },
      residual: 'bf16',
      // vLLM's benchmark recipe runs the latent cache fp8
      kv: 'fp8',
    },
    // 93 layers: gated MLA on every 4th and on the
    // last, KDA on the rest, dense MLP on the first
    blocks: [
      ...repeat(1, k3Kda, denseMlp(33792)),
      ...repeat(2, k3Kda, k3Moe),
      ...repeat(1, k3Mla, k3Moe),
      ...swaStack({
        layers: 88,
        localPerPattern: 3,
        globalPerPattern: 1,
        local: k3Kda,
        global: k3Mla,
        mlp: k3Moe,
      }),
      ...repeat(1, k3Mla, k3Moe),
    ],
  },
  {
    name: 'Muse Glimmer 30B',
    modelDim: 6656,
    vocab: 202048,
    tiedEmbeddings: false,
    precision: ALL_BF16,
    // The 30B release is a 2B ViT perception encoder in front of a 28B text
    // decoder; only the decoder stack is simulated. 13 repeats of three RoPE
    // sliding layers then one NoPE global -- NoPE and the qk-norm / logit
    // softcapping scalars cost nothing here, but the gate on every layer's
    // attention output is a second D x N*H projection that does.
    blocks: swaStack({
      layers: 52,
      localPerPattern: 3,
      globalPerPattern: 1,
      local: gqa({ N: 32, H: 128, kvHeads: 2, window: 2048, outputGate: true }),
      global: gqa({ N: 32, H: 128, kvHeads: 2, outputGate: true }),
      mlp: denseMlp(19968),
    }),
  },
  {
    name: 'Qwen3 4B BF16',
    modelDim: 2560,
    vocab: 151936,
    tiedEmbeddings: true,
    precision: ALL_BF16,
    blocks: repeat(36, gqa({ N: 32, H: 128, kvHeads: 8 }), denseMlp(9728)),
  },
  {
    name: 'Qwen3 4B FP8',
    modelDim: 2560,
    vocab: 151936,
    tiedEmbeddings: true,
    precision: {
      // The checkpoint uses fine-grained E4M3 FP8 (128x128 weight blocks)
      // with dynamically quantized activations for linear layers. The tied
      // embedding / LM-head weights, residual stream, and KV cache stay BF16.
      weights: { ...FP8_ALL, embeddings: 'bf16' },
      activations: { ...FP8_ALL, embeddings: 'bf16' },
      residual: 'bf16',
      kv: 'bf16',
    },
    blocks: repeat(36, gqa({ N: 32, H: 128, kvHeads: 8 }), denseMlp(9728)),
  },
  {
    name: 'Qwen3.8 27B', // Also covers Qwen3.6 27B: they have the same architecture
    modelDim: 5120,
    vocab: 248320,
    tiedEmbeddings: false,
    precision: ALL_BF16,
    // Dense sibling of the 35B A3B, and multimodal: a 461M ViT and the 425M
    // MTP draft head sit outside the simulated decoder. 16 repeats of three
    // gated-deltanet layers then one full-attention layer. attn_output_gate
    // fuses the gate into q_proj, which ships at 2x the query width.
    blocks: swaStack({
      layers: 64,
      localPerPattern: 3,
      globalPerPattern: 1,
      local: linearAttn({ N: 16, H: 128, valueHeads: 48, valueHeadDim: 128 }),
      global: gqa({ N: 24, H: 256, kvHeads: 4, outputGate: true }),
      mlp: denseMlp(17408),
    }),
  },
  {
    name: 'Qwen3.6 35B A3B',
    modelDim: 2048,
    vocab: 248320,
    tiedEmbeddings: false,
    precision: ALL_BF16,
    blocks: swaStack({
      layers: 40,
      localPerPattern: 3,
      globalPerPattern: 1,
      local: linearAttn({ N: 16, H: 128, valueHeads: 32, valueHeadDim: 128 }),
      global: gqa({ N: 16, H: 256, kvHeads: 2, outputGate: true }),
      mlp: moeMlp({ experts: 256, topK: 8, expertDim: 512, sharedExperts: 1 }),
    }),
  },
  {
    name: 'gpt-oss-120b MXFP4/BF16',
    modelDim: 2880,
    vocab: 201088,
    tiedEmbeddings: false,
    precision: {
      weights: { ...BF16_ALL, routedExperts: 'mxfp4' },
      activations: BF16_ALL,
      residual: 'bf16',
      kv: 'bf16',
    },
    blocks: swaStack({
      layers: 36,
      localPerPattern: 1,
      globalPerPattern: 1,
      local: gqa({ N: 64, H: 64, kvHeads: 8, window: 128 }),
      global: gqa({ N: 64, H: 64, kvHeads: 8 }),
      mlp: moeMlp({ experts: 128, topK: 4, expertDim: 2880 }),
    }),
  },
  {
    name: 'gpt-oss-20b MXFP4/BF16',
    modelDim: 2880,
    vocab: 201088,
    tiedEmbeddings: false,
    precision: {
      weights: { ...BF16_ALL, routedExperts: 'mxfp4' },
      activations: BF16_ALL,
      residual: 'bf16',
      kv: 'bf16',
    },
    blocks: swaStack({
      layers: 24,
      localPerPattern: 1,
      globalPerPattern: 1,
      local: gqa({ N: 64, H: 64, kvHeads: 8, window: 128 }),
      global: gqa({ N: 64, H: 64, kvHeads: 8 }),
      mlp: moeMlp({ experts: 32, topK: 4, expertDim: 2880 }),
    }),
  },
  {
    name: 'LLaMA 3 8B',
    modelDim: 4096,
    vocab: 128256,
    tiedEmbeddings: false,
    precision: ALL_BF16,
    blocks: repeat(32, gqa({ N: 32, H: 128, kvHeads: 8 }), denseMlp(14336)),
  },
  {
    name: 'LLaMA 3 70B',
    modelDim: 8192,
    vocab: 128256,
    tiedEmbeddings: false,
    precision: ALL_BF16,
    blocks: repeat(80, gqa({ N: 64, H: 128, kvHeads: 8 }), denseMlp(28672)),
  },
  {
    name: 'LLaMA 3.1 405B',
    modelDim: 16384,
    vocab: 128256,
    tiedEmbeddings: false,
    precision: ALL_BF16,
    blocks: repeat(126, gqa({ N: 128, H: 128, kvHeads: 8 }), denseMlp(53248)),
  },
];
