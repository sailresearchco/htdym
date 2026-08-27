export type Dtype = 'bf16' | 'fp8' | 'mxfp8' | 'fp4' | 'mxfp4' | 'nvfp4' | 'int8' | 'int4';

// Bytes per element in memory, block scales included: an MX format carries
// one E8M0 per group of 32, NVFP4 an E4M3 per 16, and compressed-tensors int4
// a bf16 per 32. Published checkpoint sizes count the scales, so these do too.
// int8 is the exception: it is scaled  per channel or per token, which is one
// value per row rather than per group, and rounds away at this precision.
export const DTYPE_BYTES: Record<Dtype, number> = {
  bf16: 2,
  fp8: 1,
  mxfp8: 1 + 1 / 32,
  fp4: 0.5,
  mxfp4: 0.5 + 1 / 32,
  nvfp4: 0.5 + 1 / 16,
  int8: 1,
  int4: 0.5 + 2 / 32,
};

// every format, in the order above (widest first, floats before ints)
export const DTYPES = Object.keys(DTYPE_BYTES) as Dtype[];

// The format of each category of weight, and of the activations fed to it.
// Releases quantize categories independently, e.g. Kimi K3 runs its routed
// experts in MXFP4 under MXFP8 activations while attention stays bf16.
export interface CategoryDtypes {
  attention: Dtype;
  denseMlp: Dtype;
  sharedExperts: Dtype;
  routedExperts: Dtype;
  router: Dtype;
  embeddings: Dtype;
}

export interface PrecisionSpec {
  // what the checkpoint stores
  weights: CategoryDtypes;
  // what each matmul takes its inputs in, which picks the chip's unit for it
  // and sizes the activations and collective payloads it produces
  activations: CategoryDtypes;
  // width of the residual stream between blocks. prices the pipeline
  // stage-boundary sends and the reductions that sum into the stream (the
  // TP all-reduces after attention and MLP, the MoE combine). Unlike an
  // activation format it never decides which matmul unit runs, each
  // projection reading the stream declares its own input format above.
  // Every current release keeps this bf16 even when its matmul inputs are
  // fp8/mxfp8: stacks cast right at each GEMM's input and still reduce in
  // bf16 (DeepEP combines in bf16 under fp8 dispatch), because summing
  // narrow compounds rounding error at every layer.
  residual: Dtype;
  kv: Dtype;
  // Optional compute format for a sparse-attention indexer. Its cache
  // storage format belongs to the attention config and can differ again.
  indexer?: Dtype;
}

export const BF16_ALL: CategoryDtypes = {
  attention: 'bf16',
  denseMlp: 'bf16',
  sharedExperts: 'bf16',
  routedExperts: 'bf16',
  router: 'bf16',
  embeddings: 'bf16',
};
export const FP8_ALL: CategoryDtypes = {
  attention: 'fp8',
  denseMlp: 'fp8',
  sharedExperts: 'fp8',
  routedExperts: 'fp8',
  router: 'fp8',
  embeddings: 'fp8',
};
export const ALL_BF16: PrecisionSpec = {
  weights: BF16_ALL,
  activations: BF16_ALL,
  residual: 'bf16',
  kv: 'bf16',
};
