import type { Dtype } from '../dtype';

// Per-layer attention config (GQA-family, MLA, compressed MQA, or
// gated-deltanet linear).
export type AttnConfig =
  | {
      kind: 'gqa';
      // query heads
      queryHeads: number;
      // query / KV / output head dimension
      headDim: number;
      kvHeads: number;
      // 1 when K==V unified, else 2
      kvTensors: number;
      // sigmoid gate on the attention output, a second D x N*H projection
      outputGate?: boolean;
      // sliding-window cap, or null for full attention
      window: number | null;
    }
  | {
      kind: 'mla';
      // query heads
      queryHeads: number;
      // qk_nope head dim (absorbed Q/K latent width per head)
      headDim: number;
      dc: number;
      dqc?: number;
      dRope: number;
      valueHeadDim: number;
      // sigmoid gate on the attention output, a second D x N*hV projection
      outputGate?: boolean;
      dsa?: { topk: number; indexHeads: number; indexHeadDim: number; shareEvery: number };
    }
  | {
      kind: 'linear';
      // query/key heads
      queryHeads: number;
      // query/key head dimension
      headDim: number;
      valueHeads: number;
      valueHeadDim: number;
      // recurrent state bytes per element (fp32 in shipped kernels)
      stateBytes: number;
    }
  | {
      // Shared-KV MQA with a low-rank query and grouped low-rank output.
      // CSA adds overlapped compression plus a sparse indexer; HCA uses a
      // much larger non-overlapping compression rate and no indexer.
      kind: 'mqa' | 'csa' | 'hca';
      queryHeads: number;
      headDim: number;
      qRank: number;
      outputGroups: number;
      outputRank: number;
      window: number;
      // trailing dimensions kept wider than the rest of each KV entry
      ropeDim: number;
      ropeDtype: Dtype;
      // null for the sliding-only MQA bootstrap layers
      compressionRate: number | null;
      indexer?: CompressedIndexer;
      // Cross-layer sharing (DeepSeek V4.1): the layer reads an earlier
      // layer's compressed cache instead of compressing its own, so it
      // stores only its window and carries no compressor weights.
      sharedCache?: boolean;
      // Sparse top-k over the compressed entries for a layer with no
      // indexer of its own (it reuses the last indexer's picks).
      // Absent = the indexer's topK, or every entry without one.
      topK?: number;
      // Stored bytes per KV entry when the kernel packs it differently from
      // dims x dtype (FlashMLA's 528 B fp8 entry on SM100). Absent = derived.
      entryBytes?: number;
      // compressor projections over the residual (KV and gate per series);
      // absent = V4's 4 for overlapping CSA, 2 for HCA
      compressorProjections?: number;
    };

export interface CompressedIndexer {
  heads: number;
  headDim: number;
  topK: number;
  cacheDtype: Dtype;
  // entries the scorer can reach at most (V4.1's candidate blocks from an
  // earlier dense pass); absent = every compressed entry
  scoreSpan?: number;
  // scores against an earlier layer's indexer keys: no key cache or key
  // projection of its own
  sharedCache?: boolean;
  // keys projected from the compressor's KV latent (V4.1) rather than the
  // residual with V4's two-series compressor
  keysFromLatent?: boolean;
}

export function gqa(opts: {
  N: number;
  H: number;
  kvHeads: number;
  window?: number | null;
  kEqV?: boolean;
  outputGate?: boolean;
}): AttnConfig {
  return {
    kind: 'gqa',
    queryHeads: opts.N,
    headDim: opts.H,
    kvHeads: opts.kvHeads,
    kvTensors: opts.kEqV ? 1 : 2,
    outputGate: opts.outputGate,
    window: opts.window === undefined ? null : opts.window,
  };
}

export function mla(opts: {
  N: number;
  H: number;
  dc: number;
  dRope: number;
  valueHeadDim: number;
  dqc?: number;
  outputGate?: boolean;
  dsa?: { topk: number; indexHeads: number; indexHeadDim: number; shareEvery?: number };
}): AttnConfig {
  return {
    kind: 'mla',
    queryHeads: opts.N,
    headDim: opts.H,
    dc: opts.dc,
    dqc: opts.dqc,
    dRope: opts.dRope,
    valueHeadDim: opts.valueHeadDim,
    outputGate: opts.outputGate,
    dsa: opts.dsa
      ? {
          topk: opts.dsa.topk,
          indexHeads: opts.dsa.indexHeads,
          indexHeadDim: opts.dsa.indexHeadDim,
          shareEvery: opts.dsa.shareEvery ?? 1,
        }
      : undefined,
  };
}

export function linearAttn(opts: {
  N: number;
  H: number;
  valueHeads: number;
  valueHeadDim: number;
  stateBytes?: number;
}): AttnConfig {
  return {
    kind: 'linear',
    queryHeads: opts.N,
    headDim: opts.H,
    valueHeads: opts.valueHeads,
    valueHeadDim: opts.valueHeadDim,
    stateBytes: opts.stateBytes ?? 4,
  };
}

export function compressedMqa(opts: {
  kind: 'mqa' | 'csa' | 'hca';
  N: number;
  H: number;
  qRank: number;
  outputGroups: number;
  outputRank: number;
  window: number;
  ropeDim: number;
  ropeDtype?: Dtype;
  compressionRate?: number;
  indexer?: CompressedIndexer;
  sharedCache?: boolean;
  topK?: number;
  entryBytes?: number;
  compressorProjections?: number;
}): AttnConfig {
  return {
    kind: opts.kind,
    queryHeads: opts.N,
    headDim: opts.H,
    qRank: opts.qRank,
    outputGroups: opts.outputGroups,
    outputRank: opts.outputRank,
    window: opts.window,
    ropeDim: opts.ropeDim,
    ropeDtype: opts.ropeDtype ?? 'bf16',
    compressionRate: opts.compressionRate ?? null,
    indexer: opts.indexer,
    sharedCache: opts.sharedCache,
    topK: opts.topK,
    entryBytes: opts.entryBytes,
    compressorProjections: opts.compressorProjections,
  };
}
