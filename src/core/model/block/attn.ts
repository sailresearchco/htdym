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
      indexer?: { heads: number; headDim: number; topK: number; cacheDtype: Dtype };
    };

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
  indexer?: { heads: number; headDim: number; topK: number; cacheDtype: Dtype };
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
  };
}
