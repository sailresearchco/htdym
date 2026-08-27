import type { ModelSpec } from '../models';
import { DTYPE_BYTES } from '../dtype';
import type { AttnConfig } from './attn';
import type { MlpConfig } from './mlp';

// One transformer block in the model's layer stack.
export interface BlockSpec {
  attn: AttnConfig;
  mlp: MlpConfig;
}

// Consecutive copies of one block inside a group's pattern.
export interface BlockRun {
  block: BlockSpec;
  count: number;
}

// The layer pattern repeated `repeat` times. The unit of lowering and
// pricing (the design doc's segment): a 60-layer MoE stack is one group,
// Gemma's 5-local-1-global stack is two runs, and everything downstream
// costs each distinct block once, scaled by repeat times count.
export interface BlockGroup {
  pattern: BlockRun[];
  repeat: number;
}

// The layers [start, end) of one pattern copy, runs split as needed.
export function sliceRuns(runs: BlockRun[], start: number, end: number): BlockRun[] {
  const out: BlockRun[] = [];
  let at = 0;
  for (const r of runs) {
    const lo = Math.max(start, at);
    const hi = Math.min(end, at + r.count);
    if (lo < hi) out.push({ block: r.block, count: hi - lo });
    at += r.count;
  }
  return out;
}

// Attention weight params for one block, split into the fused QKV (or
// MLA q/kv) projections and the output projection.
export function blockAttnParams(
  m: ModelSpec,
  b: BlockSpec,
): { total: number; qkv: number; o: number } {
  const { queryHeads: N, headDim: H } = b.attn;
  if (b.attn.kind === 'mla') {
    const { dc, dqc, dRope, valueHeadDim: hV, dsa } = b.attn;
    const q = dqc ? m.modelDim * dqc + dqc * N * (H + dRope) : m.modelDim * N * (H + dRope);
    const kv = m.modelDim * (dc + dRope) + dc * N * H + dc * N * hV;
    const o = N * hV * m.modelDim;
    // the output gate reads the residual, so it rides with the q/kv projections
    const gate = b.attn.outputGate ? m.modelDim * N * hV : 0;
    // The sparse indexer's own weights: per-head queries off the q latent
    // where there is one, the keys it scores them against, and the per-head
    // score projection. One indexer serves shareEvery layers, so each
    // carries its fraction. The csa/hca kinds below count the equivalent
    // set; this branch was simply missing them.
    const index = dsa
      ? ((dqc ?? m.modelDim) * dsa.indexHeads * dsa.indexHeadDim +
          m.modelDim * dsa.indexHeadDim +
          m.modelDim * dsa.indexHeads) /
        dsa.shareEvery
      : 0;
    return { total: q + kv + gate + index + o, qkv: q + kv + gate + index, o };
  }

  if (b.attn.kind === 'linear') {
    // in-proj: q,k over the query/key heads, v and the output gate over the
    // value heads, plus a decay pair per value head; conv taps and dt/A
    // params are sub-0.1% of the layer and omitted
    const { valueHeads: Hv, valueHeadDim: hV } = b.attn;
    const qkv = m.modelDim * (2 * N * H + 2 * Hv * hV + 2 * Hv);
    const o = Hv * hV * m.modelDim;
    return { total: qkv + o, qkv, o };
  }

  if (b.attn.kind === 'gqa') {
    // the output gate reads the residual, so it rides with the q/kv projections
    const gate = b.attn.outputGate ? m.modelDim * N * H : 0;
    const qkv = m.modelDim * N * H + b.attn.kvTensors * m.modelDim * b.attn.kvHeads * H + gate;
    const o = N * H * m.modelDim;
    return { total: qkv + o, qkv, o };
  }

  const a = b.attn;
  // The query down-projection is shared by core attention and the CSA
  // indexer. Every layer also emits one uncompressed, shared K=V head.
  const q = m.modelDim * a.qRank + a.qRank * N * H;
  const kv = m.modelDim * H;

  let compressor = 0;
  if (a.compressionRate !== null) {
    // KV and gate projections. CSA emits two overlapping series; HCA one.
    compressor += (a.kind === 'csa' ? 4 : 2) * m.modelDim * H;
  }
  if (a.indexer) {
    const i = a.indexer;
    // Two-series indexer KV + gate projections, query up-projection from
    // the shared latent, and the per-head score weighting projection.
    compressor += 4 * m.modelDim * i.headDim + a.qRank * i.heads * i.headDim + m.modelDim * i.heads;
  }

  // Per-group N*H/g -> outputRank, followed by g*outputRank -> D.
  const o = N * H * a.outputRank + a.outputGroups * a.outputRank * m.modelDim;
  const qkv = q + kv + compressor;
  return { total: qkv + o, qkv, o };
}

// Dense MLP parameters of one block (zero for an MoE block).
export function blockDenseMlpParams(m: ModelSpec, b: BlockSpec): number {
  if (b.mlp.kind !== 'dense') return 0;
  return (b.mlp.variant === 'gated' ? 3 : 2) * m.modelDim * b.mlp.ffDim;
}

// Always-on shared-expert parameters of one MoE block.
export function blockSharedExpertParams(m: ModelSpec, b: BlockSpec): number {
  if (b.mlp.kind !== 'moe') return 0;
  return b.mlp.sharedExperts * (b.mlp.variant === 'gated' ? 3 : 2) * m.modelDim * b.mlp.expertDim;
}

// Router projection parameters ([modelDim, experts]) of one MoE block.
export function blockRouterParams(m: ModelSpec, b: BlockSpec): number {
  if (b.mlp.kind !== 'moe') return 0;
  return m.modelDim * b.mlp.experts;
}

// Routed-expert parameters of one MoE block, total and active per token.
export function blockRoutedExpertParams(
  m: ModelSpec,
  b: BlockSpec,
): { total: number; active: number } {
  if (b.mlp.kind !== 'moe') return { total: 0, active: 0 };
  const per =
    (b.mlp.variant === 'gated' ? 3 : 2) * (b.mlp.latentDim || m.modelDim) * b.mlp.expertDim;
  return { total: b.mlp.experts * per, active: b.mlp.topK * per };
}

// The always-active down/up projection pair bracketing a latent MoE's
// routed experts (zero when they already run at the residual width).
export function blockLatentProjParams(m: ModelSpec, b: BlockSpec): number {
  if (b.mlp.kind !== 'moe' || !b.mlp.latentDim) return 0;
  return 2 * m.modelDim * b.mlp.latentDim;
}

export interface AttnFlopParts {
  core: number;
  indexer: number;
}

// Dot-product attention FLOPs for one token of one block at context ctx,
// split because sparse indexers can run narrower than core attention.
export function blockAttnFlopsDecodeParts(b: BlockSpec, ctx: number): AttnFlopParts {
  const { queryHeads: N } = b.attn;
  if (b.attn.kind === 'mla') {
    const { dc, dRope, dsa } = b.attn;
    const span = dsa ? Math.min(ctx, dsa.topk) : ctx;
    const indexer = dsa ? (2 * ctx * dsa.indexHeads * dsa.indexHeadDim) / dsa.shareEvery : 0;
    return { core: 2 * span * N * (dc + dRope + dc), indexer };
  }
  if (b.attn.kind === 'linear') {
    // three matvecs against each value head's [H, hV] state, ctx-independent:
    // read it with the key, rank-1 update, read out with the query. The
    // channel-wise decay scales the state elementwise and is not matmul work.
    return { core: 6 * b.attn.valueHeads * b.attn.headDim * b.attn.valueHeadDim, indexer: 0 };
  }
  if (b.attn.kind === 'gqa') {
    const span = b.attn.window === null ? ctx : Math.min(ctx, b.attn.window);
    return { core: 2 * span * N * (b.attn.headDim + b.attn.headDim), indexer: 0 };
  }

  const a = b.attn;
  const compressed = a.compressionRate === null ? 0 : ctx / a.compressionRate;
  const longSpan = a.indexer ? Math.min(compressed, a.indexer.topK) : compressed;
  const coreSpan = Math.min(ctx, a.window) + longSpan;
  return {
    core: 2 * coreSpan * N * (a.headDim + a.headDim),
    indexer: a.indexer ? 2 * compressed * a.indexer.heads * a.indexer.headDim : 0,
  };
}

export function blockAttnFlopsDecode(b: BlockSpec, ctx: number): number {
  const p = blockAttnFlopsDecodeParts(b, ctx);
  return p.core + p.indexer;
}

function causalWindowPairs(T: number, window: number): number {
  const w = Math.min(T, window);
  return (w * w) / 2 + (T - w) * w;
}

// Query-to-compressed-entry pairs. Before a sparse cap is reached, one new
// entry appears per `rate` query positions; after that every query sees cap.
function compressedPairs(T: number, rate: number, cap?: number): number {
  if (cap === undefined) return (T * T) / (2 * rate);
  const ramp = Math.min(T, cap * rate);
  return (ramp * ramp) / (2 * rate) + (T - ramp) * cap;
}

export function blockAttnFlopsPrefillParts(b: BlockSpec, T: number): AttnFlopParts {
  const { queryHeads: N, headDim: H } = b.attn;
  if (b.attn.kind === 'mla') {
    const { dRope, valueHeadDim: hV, dsa } = b.attn;
    const w = dsa ? Math.min(T, dsa.topk) : T;
    const pairs = causalWindowPairs(T, w);
    const indexer = dsa
      ? (2 * ((T * T) / 2) * dsa.indexHeads * dsa.indexHeadDim * T) / (dsa.shareEvery * T)
      : 0;
    return { core: 2 * pairs * N * (H + dRope + hV), indexer };
  }
  // chunked linear-attention prefill is the same per-token arithmetic reorganized
  if (b.attn.kind === 'linear') return { core: T * blockAttnFlopsDecode(b, T), indexer: 0 };
  if (b.attn.kind === 'gqa') {
    const pairs = causalWindowPairs(T, b.attn.window === null ? T : b.attn.window);
    return { core: 2 * pairs * N * (H + H), indexer: 0 };
  }

  const a = b.attn;
  const localPairs = causalWindowPairs(T, a.window);
  const longPairs =
    a.compressionRate === null ? 0 : compressedPairs(T, a.compressionRate, a.indexer?.topK);
  const indexPairs =
    a.compressionRate !== null && a.indexer ? compressedPairs(T, a.compressionRate) : 0;
  return {
    core: 2 * (localPairs + longPairs) * N * (H + H),
    indexer: a.indexer ? 2 * indexPairs * a.indexer.heads * a.indexer.headDim : 0,
  };
}

export function blockAttnFlopsPrefill(b: BlockSpec, T: number): number {
  const p = blockAttnFlopsPrefillParts(b, T);
  return p.core + p.indexer;
}

// KV bytes one sequence stores for one block ('store') or one decode step
// reads ('read', differs under DSA and sliding windows cap both). Linear
// layers hold a fixed-size recurrent state instead of a growing cache, in
// the state dtype rather than the model's kv dtype.
export interface KvByteParts {
  core: number;
  indexer: number;
}

export function blockKvBytesParts(
  b: BlockSpec,
  kvBytes: number,
  len: number,
  access: 'store' | 'read',
): KvByteParts {
  if (b.attn.kind === 'linear')
    return {
      core: b.attn.valueHeads * b.attn.headDim * b.attn.valueHeadDim * b.attn.stateBytes,
      indexer: 0,
    };
  if (b.attn.kind === 'mla') {
    const { dc, dRope, dsa } = b.attn;
    const latentLen = access === 'read' && dsa ? Math.min(len, dsa.topk) : len;
    const indexer = dsa ? (dsa.indexHeadDim / dsa.shareEvery) * len * kvBytes : 0;
    return { core: (dc + dRope) * latentLen * kvBytes, indexer };
  }
  if (b.attn.kind === 'gqa') {
    const span = b.attn.window === null ? len : Math.min(len, b.attn.window);
    return {
      core: b.attn.kvTensors * b.attn.kvHeads * b.attn.headDim * span * kvBytes,
      indexer: 0,
    };
  }

  const a = b.attn;
  const entryBytes = (a.headDim - a.ropeDim) * kvBytes + a.ropeDim * DTYPE_BYTES[a.ropeDtype];
  const localEntries = Math.min(len, a.window);
  const compressedEntries = a.compressionRate === null ? 0 : len / a.compressionRate;
  const longEntries =
    access === 'read' && a.indexer
      ? Math.min(compressedEntries, a.indexer.topK)
      : compressedEntries;
  return {
    core: (localEntries + longEntries) * entryBytes,
    indexer: a.indexer
      ? compressedEntries * a.indexer.headDim * DTYPE_BYTES[a.indexer.cacheDtype]
      : 0,
  };
}

export function blockKvBytes(
  b: BlockSpec,
  kvBytes: number,
  len: number,
  access: 'store' | 'read',
): number {
  const p = blockKvBytesParts(b, kvBytes, len, access);
  return p.core + p.indexer;
}

// The KV head count that bounds TP for this block's cache (1 = the
// slice is shared by all heads and replicates under TP). Linear state
// shards by value head.
export function blockKvHeads(b: BlockSpec): number {
  if (b.attn.kind === 'mla') return 1;
  if (b.attn.kind === 'linear') return b.attn.valueHeads;
  return b.attn.kind === 'gqa' ? b.attn.kvHeads : 1;
}
