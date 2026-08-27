import {
  BlockSpec,
  blockAttnFlopsDecode,
  blockAttnFlopsDecodeParts,
  blockAttnFlopsPrefill,
  blockAttnFlopsPrefillParts,
  blockAttnParams,
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

// Parameters of the ROUTED experts across all MoE layers: the weight bytes
// whose per-step read traffic depends on the batch (only the experts the
// batch routes to are touched). Shared experts and routers are always read,
// so they count with the dense weights instead.
export function routedExpertParams(m: ModelSpec): number {
  return sumBlocks(m, (b) => blockRoutedExpertParams(m, b).total);
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
      blockRouterParams(m, b) * B(w.router),
  );
  const embeddingCopies = m.tiedEmbeddings ? 1 : 2;
  return blockBytes + embeddingCopies * m.vocab * m.modelDim * B(w.embeddings);
}

// Expected fraction of routed-expert weight bytes read while processing
// tokens tokens in one step: each token picks topK of E experts, so an
// expert is untouched w.p. (1 - k/E)^tokens under uniform routing. 1 for
// dense models.
export function expertReadFraction(m: ModelSpec, tokens: number): number {
  for (const b of uniqueBlocks(m)) {
    if (b.mlp.kind === 'moe') return 1 - Math.pow(1 - b.mlp.topK / b.mlp.experts, tokens);
  }
  return 1;
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

export function totalAttnParams(m: ModelSpec): number {
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

export function totalFlops(byWidth: FlopsByWidth): number {
  return [...byWidth.values()].reduce((a, b) => a + b, 0);
}

function addFlops(out: FlopsByWidth, d: Dtype, flops: number): void {
  if (flops > 0) out.set(d, (out.get(d) ?? 0) + flops);
}

// Matmul FLOPs one token touches, bucketed by each category's activation
// width. Sums to 2 x activeParams when a model runs one width throughout.
function matmulFlopsByWidth(m: ModelSpec): FlopsByWidth {
  const a = m.precision.activations;
  const out: FlopsByWidth = new Map();
  const add = (d: Dtype, params: number) => addFlops(out, d, 2 * params);

  add(a.attention, totalAttnParams(m));
  add(
    a.denseMlp,
    sumBlocks(m, (b) => blockDenseMlpParams(m, b) + blockLatentProjParams(m, b)),
  );
  add(
    a.sharedExperts,
    sumBlocks(m, (b) => blockSharedExpertParams(m, b)),
  );
  add(
    a.router,
    sumBlocks(m, (b) => blockRouterParams(m, b)),
  );
  add(
    a.routedExperts,
    sumBlocks(m, (b) => blockRoutedExpertParams(m, b).active),
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
export function flopsPerPrefillToken(m: ModelSpec, T: number): FlopsByWidth {
  const out = matmulFlopsByWidth(m);
  const attn = m.precision.activations.attention;
  const core = sumBlocks(m, (b) => blockAttnFlopsPrefillParts(b, T).core) / T;
  const indexer = sumBlocks(m, (b) => blockAttnFlopsPrefillParts(b, T).indexer) / T;
  addFlops(out, attn, core);
  addFlops(out, m.precision.indexer ?? attn, indexer);
  return out;
}

// Dot-product attention FLOPs to prefill one sequence of T tokens, whole
// model. Per layer, attending over p (query, key) pairs costs
// 2*p*N*(H_qk + H_v). Causal full attention gives p = T^2/2. A sliding window
// caps each query's span at W, so p = W^2/2 + (T-W)*W for T > W. DSA caps the
// main attention's span at topk the same way, but its indexer still scores
// every prior token (quadratic).
export function attnFlopsPrefill(m: ModelSpec, T: number): number {
  return sumBlocks(m, (b) => blockAttnFlopsPrefill(b, T));
}

// Dot-product attention FLOPs to decode ONE token of ONE sequence with ctx
// tokens of context, whole model. Local layers only attend over the window.
// MLA uses the "absorbed" decode form: attention runs directly in the latent,
// so every query head attends over (d_c + d_rope)-dim keys and d_c-dim
// values. DSA caps the main attention at topk tokens, but its indexer still
// scores all ctx.
export function attnFlopsPerDecodeToken(m: ModelSpec, ctx: number): number {
  return sumBlocks(m, (b) => blockAttnFlopsDecode(b, ctx));
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

// Representative topK from the first MoE block (uniform MoE stacks).
export function moeTopK(m: ModelSpec): number {
  for (const b of uniqueBlocks(m)) if (b.mlp.kind === 'moe') return b.mlp.topK;
  return 0;
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
