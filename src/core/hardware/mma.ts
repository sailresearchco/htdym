import type { Dtype } from '../model/dtype';

// Small GEMMs can use smaller tiles, sometimes at reduced throughput. Padding
// every dimension to 128 can therefore overestimate their compute time.
//
// For C[M,N] = A[M,K] × B[K,N], each dtype lists alternative instruction shapes
// or effective throughput tiles, including transposed paths. Dense and grouped
// GEMMs round M/N/K up for each candidate, divide padded work by its relative
// rate, and choose the cheapest. realizableFlopsFrac applies separately.
// The grouped-row approximation is documented in adjustedFlops in naiveOpCost.ts.
//
// H100, B200 and B300 floating-point rates come from instruction microbenchmarks.
// Other entries use documented shapes or throughput approximations, with sources
// and assumptions beside the tables. Uncharacterized paths use the 128³ fallback.
// Only arithmetic is estimated here: layout conversion, unpacking and kernel
// launch overhead are omitted. HBM traffic uses unpadded tensor sizes separately.
export interface MmaShape {
  // A dimension of 1 skips padding on that axis.
  m: number;
  n: number;
  k: number;
  // Fraction of the resolved dtype's peak, before realizableFlopsFrac.
  // A rate of .5 doubles the compute time for the same padded work.
  rate: number;
}

// ChipSpec.formats determines dtype support; these tables only price arithmetic.
export type MmaShapes = Record<Dtype, readonly MmaShape[]>;

// Fallback: pad each axis to 128 at full rate until a better estimate is available.
// This is a cost approximation, not a hardware instruction shape.
const APPROXIMATE_CUBE: readonly MmaShape[] = [{ m: 128, n: 128, k: 128, rate: 1 }];
export const APPROXIMATE_MMA: MmaShapes = {
  bf16: APPROXIMATE_CUBE,
  fp8: APPROXIMATE_CUBE,
  fp4: APPROXIMATE_CUBE,
  mxfp8: APPROXIMATE_CUBE,
  mxfp4: APPROXIMATE_CUBE,
  nvfp4: APPROXIMATE_CUBE,
  int8: APPROXIMATE_CUBE,
  int4: APPROXIMATE_CUBE,
};

// C = AB or Cᵀ = BᵀAᵀ; swapped paths also exchange operand scale tensors.
// Loading/repacking costs are outside this arithmetic model.
const orientations = (m: number, n: number, k: number, rate = 1): MmaShape[] => [
  { m, n, k, rate },
  { m: n, n: m, k, rate },
];

// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
const warp = (k: number, rate = 1): MmaShape[] => orientations(16, 8, k, rate);

export const AMPERE_MMA: MmaShapes = {
  ...APPROXIMATE_MMA,
  bf16: warp(16),
  int8: warp(32),
  int4: warp(64),
};
export const ADA_MMA: MmaShapes = { ...AMPERE_MMA, fp8: warp(32) };

// H100 instruction measurements (CUDA 12.8, 2026-09-06): register-A/shared-B
// WGMMA at N32 reaches ~97-99% of wide throughput, rounded to 1; N16 ~.6.
// BF16 warp MMA reaches ~.67. Operand movement/conversion costs are omitted.
const hopper = (k: number): MmaShape[] => orientations(64, 32, k);
// INT8 retains the H800 paper's estimate; it was not remeasured here.
// https://arxiv.org/html/2501.12084v1 (Tables VII-XI)
export const HOPPER_MMA: MmaShapes = {
  ...APPROXIMATE_MMA,
  bf16: [...warp(16, 0.67), ...hopper(16)],
  fp8: [
    ...warp(16, 0.335), // Explicit BF16 widening: .67 × .5 of FP8 peak.
    ...orientations(64, 16, 32, 0.6),
    ...hopper(32),
  ],
  int8: [...warp(32, 0.67), { m: 64, n: 64, k: 32, rate: 1 }],
};

// B200/B300 instruction microbenchmarks, CUDA 12.8/13.1, CUTLASS 4.2.1 (2026-09-06).
// Rounded rates relative to wide tcgen05 throughput. One-CTA block-scaled
// instructions require M=128; BF16/FP8 also support M=64. These are native
// dimensions, before adding transposed paths. N<32 paths are omitted because
// they add little benefit to this padding model.
// FP4 uses block scaling, with unit scales for the unscaled format.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape
const blackwellM128 = (k: number, rate = 1): MmaShape[] => [
  { m: 128, n: 128, k, rate },
  ...orientations(128, 64, k, (2 / 3) * rate),
  ...orientations(128, 32, k, 0.4 * rate),
];
const blackwellM64 = (k: number): MmaShape[] => [
  { m: 64, n: 64, k, rate: 0.5 },
  ...orientations(64, 32, k, 1 / 3),
];
export const BLACKWELL_MMA: MmaShapes = {
  ...APPROXIMATE_MMA,
  bf16: [...warp(16, 0.25), ...blackwellM128(16), ...blackwellM64(16)],
  // FP8 explicitly widens to BF16 for the warp path: .25 × .5 of FP8 peak.
  // Conversion cost is omitted; this is not a native FP8 warp instruction rate.
  fp8: [...warp(16, 0.125), ...blackwellM128(32), ...blackwellM64(32)],
  int8: [{ m: 128, n: 128, k: 32, rate: 1 }], // Unmeasured wide tcgen05 estimate.
  mxfp8: blackwellM128(32),
  fp4: blackwellM128(64),
  mxfp4: blackwellM128(64),
  nvfp4: blackwellM128(64),
};
// B300 reaches its higher FP4 peak with a two-CTA 256x256x96 instruction.
// Smaller tiles retain B200 throughput (2/3 of B300's FP4 peak); the
// 128x256x96 path reaches 8/9. Keep K64 as well to avoid unnecessary K padding.
const blackwellUltraFp4: MmaShape[] = [
  ...blackwellM128(64, 2 / 3),
  ...blackwellM128(96, 2 / 3),
  ...orientations(128, 256, 96, 8 / 9),
  { m: 256, n: 256, k: 96, rate: 1 },
];
export const BLACKWELL_ULTRA_MMA: MmaShapes = {
  ...APPROXIMATE_MMA,
  bf16: BLACKWELL_MMA.bf16,
  fp8: BLACKWELL_MMA.fp8,
  mxfp8: BLACKWELL_MMA.mxfp8,
  fp4: blackwellUltraFp4,
  mxfp4: blackwellUltraFp4,
  nvfp4: blackwellUltraFp4,
};

// Rubin SM107: documented wide instruction shapes, with an unmeasured full-rate
// assumption. Smaller and older paths are omitted until their rates are measured.
// FP8 K64 requires M128 for one CTA; FP4 uses unit scales for the unscaled format.
// https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/examples/python/CuTeDSL/cute/rubin/kernel/dense_gemm/dense_gemm_persistent.py
// https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/include/cute/atom/mma_traits_sm107.hpp
export const RUBIN_MMA: MmaShapes = {
  ...APPROXIMATE_MMA,
  bf16: [{ m: 128, n: 128, k: 16, rate: 1 }],
  fp8: [{ m: 128, n: 128, k: 64, rate: 1 }],
  mxfp8: [{ m: 128, n: 128, k: 64, rate: 1 }],
  fp4: [{ m: 128, n: 128, k: 128, rate: 1 }],
  mxfp4: [{ m: 128, n: 128, k: 128, rate: 1 }],
  nvfp4: [{ m: 128, n: 128, k: 128, rate: 1 }],
};

// SM120 uses warp MMA, including K32/K64 block-scaled forms (unit scales for FP4).
// https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html
export const SM120_MMA: MmaShapes = {
  ...APPROXIMATE_MMA,
  bf16: warp(16),
  fp8: warp(32),
  int8: warp(32),
  mxfp8: warp(32),
  fp4: warp(64),
  mxfp4: warp(64),
  nvfp4: warp(64),
};

// Equal-throughput MFMA alternatives; wider output has half the K.
// CDNA4's older instructions remain available at half its new peak.
// https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
const mfma = (k: number, rate = 1): MmaShape[] => [
  { m: 16, n: 16, k, rate },
  { m: 32, n: 32, k: k / 2, rate },
];
export const CDNA3_MMA: MmaShapes = {
  ...APPROXIMATE_MMA,
  bf16: mfma(16),
  fp8: mfma(32),
  int8: mfma(32),
};
export const CDNA4_MMA: MmaShapes = {
  ...APPROXIMATE_MMA,
  bf16: [...mfma(32), ...mfma(16, 0.5)],
  fp8: [...mfma(128), ...mfma(32, 0.5)],
  int8: [...mfma(64), ...mfma(32, 0.5)],
  mxfp8: mfma(128),
  fp4: mfma(128),
  mxfp4: mfma(128),
};

// Gaudi's 256x256 array holds outputs while K streams through; k=1 skips K
// padding. Startup/drain and Gaudi2's configurable geometries are unmodeled.
// https://cdrdv2-public.intel.com/839363/Intel-Gaudi2-AI-Accelerators-whitepaper.pdf
// https://cdrdv2-public.intel.com/845118/gaudi-3-ai-accelerator-30-3-30.pdf
const gaudi: MmaShape[] = [{ m: 256, n: 256, k: 1, rate: 1 }];
export const GAUDI_MMA: MmaShapes = { ...APPROXIMATE_MMA, bf16: gaudi, fp8: gaudi };

// Effective pipelined tiles: wide stationary inputs have a 64-cycle issue
// floor. Either operand can be stationary. Trn2's double-FP8 mode doubles K.
// These approximate throughput, not minimum legal instruction dimensions.
// https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/api/generated/nki.isa.nc_matmul.html
// https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium2_arch.html#double-fp8-matmul-performance
const neuron = (k: number): MmaShape[] => orientations(64, 128, k);
// NeuronCore-v2 has native INT8, but the NKI timing model above covers only
// floating point. INT8 retains the legacy 128³ estimate; its geometry is unknown.
// https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/about-neuron/arch/neuron-hardware/neuron-core-v2.html
export const NEURON_V2_MMA: MmaShapes = { ...APPROXIMATE_MMA, bf16: neuron(128), fp8: neuron(128) };
export const NEURON_V3_MMA: MmaShapes = { ...NEURON_V2_MMA, fp8: neuron(256) };

// Streamed MXU work: v5p-era BF16 issues 8x128 @ 128x128 every 8 cycles.
// Later arrays are 256x256; retaining 8 rows is an approximation. Extending
// BF16 geometry to packed formats is also inferred. Startup is unmodeled.
// Swapping operands covers thin N; layout conversion costs are omitted.
// https://jax-ml.github.io/scaling-book/tpus/#appendix-b-how-does-a-systolic-array-work
// https://docs.cloud.google.com/tpu/docs/performance-guide
const tpu = (width: number): MmaShape[] => orientations(8, width, width);
export const TPU_V5P_MMA: MmaShapes = {
  ...APPROXIMATE_MMA,
  bf16: tpu(128),
  int8: tpu(128),
  int4: tpu(128),
};
export const TPU_V6E_MMA: MmaShapes = {
  ...APPROXIMATE_MMA,
  bf16: tpu(256),
  int8: tpu(256),
  int4: tpu(256),
};
export const TPU_V7X_MMA: MmaShapes = { ...APPROXIMATE_MMA, bf16: tpu(256), fp8: tpu(256) };
