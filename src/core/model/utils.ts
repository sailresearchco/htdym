import {
  BlockSpec,
  blockAttnFlopsDecodeParts,
  blockAttnFlopsPrefillParts,
  blockAttnParams,
  blockEngramParams,
  blockDenseMlpParams,
  blockKvBytes,
  blockKvHeads,
  blockLatentProjParams,
  blockRouterParams,
  blockRoutedExpertParams,
  blockSharedExpertParams,
} from './block';
import { ModelSpec } from './models';
import { DTYPE_BYTES, type Dtype } from './dtype';

// Per-layer quantity summed over the whole stack, repeats and counts applied.
function sumBlocks(m: ModelSpec, f: (b: BlockSpec) => number): number {
  return m.blocks.reduce(
    (s, g) => s + g.repeat * g.pattern.reduce((t, r) => t + r.count * f(r.block), 0),
    0,
  );
}

// The distinct block structures, in order (for predicates and extrema).
function uniqueBlocks(m: ModelSpec): BlockSpec[] {
  return m.blocks.flatMap((g) => g.pattern.map((r) => r.block));
}

export function layerCount(m: ModelSpec): number {
  return m.blocks.reduce((s, g) => s + g.repeat * g.pattern.reduce((t, r) => t + r.count, 0), 0);
}

// Total weight bytes at the model's native per-component precision. This is
// the checkpoint footprint modulo block-scale overhead, so published sizes
// anchor it directly.
export function weightBytesTotal(m: ModelSpec): number {
  const w = m.precision.weights;
  const B = (d: Dtype) => DTYPE_BYTES[d];
  const blockBytes = sumBlocks(
    m,
    (b) =>
      blockAttnParams(m, b).total * B(w.attention) +
      (blockDenseMlpParams(m, b) + blockLatentProjParams(m, b)) * B(w.denseMlp) +
      blockSharedExpertParams(m, b) * B(w.sharedExperts) +
      blockRoutedExpertParams(m, b).total * B(w.routedExperts) +
      blockRouterParams(m, b) * B(w.router) +
      blockEngramParams(b) * B(w.attention),
  );
  const embeddingCopies = m.tiedEmbeddings ? 1 : 2;
  return blockBytes + embeddingCopies * m.vocab * m.modelDim * B(w.embeddings);
}

function totalMlpParams(m: ModelSpec): number {
  return sumBlocks(
    m,
    (b) =>
      blockDenseMlpParams(m, b) +
      blockSharedExpertParams(m, b) +
      blockRouterParams(m, b) +
      blockLatentProjParams(m, b) +
      blockRoutedExpertParams(m, b).total,
  );
}

function totalAttnParams(m: ModelSpec): number {
  return sumBlocks(m, (b) => blockAttnParams(m, b).total);
}

export function totalParams(m: ModelSpec): number {
  return totalAttnParams(m) + totalMlpParams(m) + (m.tiedEmbeddings ? 1 : 2) * m.vocab * m.modelDim;
}

// MLP parameters one token touches: top-k + shared experts and the router.
function activeMlpParams(m: ModelSpec): number {
  return sumBlocks(
    m,
    (b) =>
      blockDenseMlpParams(m, b) +
      blockSharedExpertParams(m, b) +
      blockRouterParams(m, b) +
      blockLatentProjParams(m, b) +
      blockRoutedExpertParams(m, b).active,
  );
}

// Parameters one token touches: attention + active MLP + the unembedding
// matmul, but NOT the embedding gather, the convention of the gpt-oss model
// card ("unembedding parameters are counted towards active, but not
// embeddings"). Equals totalParams minus embeddings for dense models,
// and matmulFlopsPerToken = 2 x this by construction.
export function activeParams(m: ModelSpec): number {
  return totalAttnParams(m) + activeMlpParams(m) + m.vocab * m.modelDim;
}

// FLOPs split by the input width they run at. A release can quantize its
// routed experts below its attention, and the two then land on different
// peaks, so a single FLOP total has no one rate to be divided by.
export type FlopsByWidth = Map<Dtype, number>;

function addFlops(out: FlopsByWidth, d: Dtype, flops: number): void {
  if (flops > 0) out.set(d, (out.get(d) ?? 0) + flops);
}

// Matmul FLOPs one token touches, bucketed by each category's activation
// width. Sums to 2 x activeParams when a model runs one width throughout.
// `share` is the fraction of tokens each block sees (prefill tails).
function matmulFlopsByWidth(m: ModelSpec, share: (b: BlockSpec) => number = () => 1): FlopsByWidth {
  const a = m.precision.activations;
  const out: FlopsByWidth = new Map();
  const add = (d: Dtype, params: number) => addFlops(out, d, 2 * params);
  const sum = (f: (b: BlockSpec) => number) => sumBlocks(m, (b) => share(b) * f(b));

  add(
    a.attention,
    sum((b) => blockAttnParams(m, b).total + blockEngramParams(b)),
  );
  add(
    a.denseMlp,
    sum((b) => blockDenseMlpParams(m, b) + blockLatentProjParams(m, b)),
  );
  add(
    a.sharedExperts,
    sum((b) => blockSharedExpertParams(m, b)),
  );
  add(
    a.router,
    sum((b) => blockRouterParams(m, b)),
  );
  add(
    a.routedExperts,
    sum((b) => blockRoutedExpertParams(m, b).active),
  );
  add(a.embeddings, m.vocab * m.modelDim);
  return out;
}

// Whole-model FLOPs to decode one token at context ctx (matmuls + attention).
export function flopsPerDecodeToken(m: ModelSpec, ctx: number): FlopsByWidth {
  const out = matmulFlopsByWidth(m);
  const attn = m.precision.activations.attention;
  const core = sumBlocks(m, (b) => blockAttnFlopsDecodeParts(b, ctx).core);
  const indexer = sumBlocks(m, (b) => blockAttnFlopsDecodeParts(b, ctx).indexer);
  addFlops(out, attn, core);
  addFlops(out, m.precision.indexer ?? attn, indexer);
  return out;
}

// Average per-token FLOPs to prefill a sequence of length T (matmuls + attention).
// A block with a prefill tail runs only each prompt's last tokens, which
// still attend over everything before them.
export function flopsPerPrefillToken(m: ModelSpec, T: number): FlopsByWidth {
  const start = (b: BlockSpec) =>
    b.prefillTail === undefined ? 0 : Math.max(0, T - b.prefillTail);
  const out = matmulFlopsByWidth(m, (b) => (T - start(b)) / T);
  const attn = m.precision.activations.attention;
  const parts = (b: BlockSpec) => {
    const full = blockAttnFlopsPrefillParts(b, T);
    if (start(b) === 0) return full;
    const skipped = blockAttnFlopsPrefillParts(b, start(b));
    return { core: full.core - skipped.core, indexer: full.indexer - skipped.indexer };
  };
  const core = sumBlocks(m, (b) => parts(b).core) / T;
  const indexer = sumBlocks(m, (b) => parts(b).indexer) / T;
  addFlops(out, attn, core);
  addFlops(out, m.precision.indexer ?? attn, indexer);
  return out;
}

// Total KV cache bytes for one sequence of length len, unsharded.
// access: bytes resident in HBM ('store') or bytes one decode step reads
// ('read'). They differ only under DSA, which caps latent reads at topk
// while storage stays full-length.
export function kvBytesPerSeq(
  m: ModelSpec,
  kvBytes: number,
  len: number,
  access: 'store' | 'read' = 'store',
): number {
  return sumBlocks(m, (b) => blockKvBytes(b, kvBytes, len, access));
}

export function hasMoeLayers(m: ModelSpec): boolean {
  return uniqueBlocks(m).some((b) => b.mlp.kind === 'moe');
}

// Max routed-expert count across MoE blocks (0 if dense).
export function moeExperts(m: ModelSpec): number {
  let e = 0;
  for (const b of uniqueBlocks(m)) if (b.mlp.kind === 'moe') e = Math.max(e, b.mlp.experts);
  return e;
}

export function hasMlaLayers(m: ModelSpec): boolean {
  return uniqueBlocks(m).some((b) => b.attn.kind === 'mla');
}

export function minQueryHeads(m: ModelSpec): number {
  const blocks = uniqueBlocks(m);
  if (blocks.length === 0) return 0;
  return Math.min(...blocks.map((b) => b.attn.queryHeads));
}

export function minKvHeads(m: ModelSpec): number {
  const blocks = uniqueBlocks(m);
  return blocks.length ? Math.min(...blocks.map(blockKvHeads)) : 0;
}

// Whether an even pp-way layer split (partitionIntoStages' rule) puts a
// layer on a different stage from the earlier layer whose compressed KV
// or indexer keys it reads (V4.1's cross-layer sharing). Such a stage
// would read a cache it neither holds nor receives.
export function ppSplitsSharedCache(m: ModelSpec, pp: number): boolean {
  if (pp <= 1) return false;
  const layers = m.blocks
    .flatMap((g) =>
      Array.from({ length: g.repeat }, () =>
        g.pattern.flatMap((r) => Array(r.count).fill(r.block)),
      ),
    )
    .flat() as BlockSpec[];
  const base = Math.floor(layers.length / pp);
  const extra = layers.length % pp;
  const stageOf: number[] = [];
  for (let s = 0; s < pp; s++) for (let i = 0; i < base + (s < extra ? 1 : 0); i++) stageOf.push(s);

  let kvSource = -1;
  let keySource = -1;
  for (const [i, b] of layers.entries()) {
    const a = b.attn;
    if (a.kind !== 'mqa' && a.kind !== 'csa' && a.kind !== 'hca') continue;
    if (a.compressionRate !== null) {
      if (!a.sharedCache) kvSource = i;
      else if (kvSource < 0 || stageOf[kvSource] !== stageOf[i]) return true;
    }
    if (a.indexer) {
      if (!a.indexer.sharedCache) keySource = i;
      else if (keySource < 0 || stageOf[keySource] !== stageOf[i]) return true;
    }
  }
  return false;
}
