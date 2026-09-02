import type { Dtype } from '../model/dtype';
import { InterconnectSpec, SliceTopology } from './topology';

export interface ChipSpec {
  id: string;
  name: string;
  vendor: string;

  // What a chip supports for a give format: a number is a peak dense matmul rate,
  // the chip has a unit and multiplies the format as stored, and 'kernel-widened'
  // means there is a known kernel which can unpack a tensor of this format
  // on its way from HBM into the matmul units into a wider format, meaning
  // it can be used to speed up weight loads, just not matmul flops/sec.
  // If there is no entry for a dtype, we still assume a kernel could exist
  // and price accordingly, but we'll throw a warning that it needs to be written.
  // We assume these kernels don't add any overhead, which is not a perfect assumption,
  // but probably close enough in practice, it shouldn't be too hard to pipeline.
  //
  // bf16 must be a rate: it is the floor every substitution ladder
  // ends at, which is what lets runsAs always land somewhere.
  formats: Partial<Record<Dtype, number | 'kernel-widened'>> & { bf16: number };
  // HBM capacity in bytes
  hbmCapacity: number;
  // HBM bandwidth in bytes/s
  hbmBandwidth: number;
  interconnect: InterconnectSpec;
  // Rental $/chip-hour, rounded so each price ÷ the H100's 1.8 lands on a
  // clean public multiplier (the UI quotes prices relative to the H100).
  costPerHour?: number;
  // Thermal design power per chip in watts, the per-kW efficiency view's
  // denominator (quoted relative to the H100 the way prices are). Vendor
  // datasheet figures where published, credible third-party estimates
  // (Epoch AI, SemiAnalysis) where not; omitted entirely when neither
  // exists — the chip then shows no per-kW efficiency.
  tdp?: number;
  // Fraction of the datasheet matmul rate a well-tuned, well-shaped GEMM
  // sustains (sustained clocks, kernel quality). Derates compute pricing
  // only; shape-dependent padding is priced separately via matmulSatRows.
  realizableFlopsFrac: number;
  // Fraction of hbmBandwidth a well-tuned streaming kernel sustains;
  // derates every memory price.
  realizableHbmBwFrac: number;
  // Edge of the matmul array's tile: every GEMM dim (M, N and K) pads up
  // to it, so thin shards run at padded-volume utilization (a 64-deep
  // contraction on a 128x128 MXU idles half the array).
  // DEFAULT_MATMUL_SAT_ROWS = 128, since TPU MXUs are 128x128 and
  // H100-class GEMMs want >=128-row tiles.
  matmulSatRows?: number;
  // Row (M) shapes of the chip's matmul instructions and the fraction of
  // peak each sustains. A grouped GEMM (MoE experts) pads every activated
  // group to the cheapest one rather than to a saturating matmulSatRows
  // tile: a 2-token expert costs one m16 mma.sync on Hopper, not 128 rows.
  // Omitted = a single full-rate matmulSatRows shape.
  mmaShapes?: { m: number; rate: number }[];
}

// Fallback ChipSpec.matmulSatRows, see that field's doc.
export const DEFAULT_MATMUL_SAT_ROWS = 128;

// Full-rate row equivalents one activated group of a grouped GEMM pays.
export function groupPadRows(chip: ChipSpec): number {
  const shapes = chip.mmaShapes ?? [{ m: chip.matmulSatRows ?? DEFAULT_MATMUL_SAT_ROWS, rate: 1 }];
  return Math.min(...shapes.map((s) => s.m / s.rate));
}

// What a format degrades to when a chip has no matmul unit for it: each names
// the next format its data can be unpacked into, and following the chain gives
// the substitution ladder. A rung must be at least as expressive as the format
// above it, never narrower: that is why mxfp8, whose scales extend its range
// past unscaled fp8, widens straight to bf16.
const DEGRADES_TO: Record<Dtype, Dtype | null> = {
  bf16: null,
  fp4: 'fp8',
  fp8: 'bf16',
  mxfp4: 'mxfp8',
  mxfp8: 'bf16',
  nvfp4: 'mxfp8',
  int8: 'bf16',
  int4: 'bf16',
};

// The format a matmul asking for `want` actually runs in on this
// chip (the first rung of its ladder this chip has a unit for).
export function runsAs(chip: ChipSpec, want: Dtype): Dtype {
  for (let d: Dtype | null = want; d !== null; d = DEGRADES_TO[d]) if (peakFlops(chip, d)) return d;
  return 'bf16'; // unreachable: the loop already returned at the bf16 rung
}

// prettier-ignore
const V6E_SLICES = slices([
  '1x1:m', '2x2:m', '2x4:m', '4x4:m', '4x8:m', '8x8:m', '8x16:c', '16x16:t',
]);

// prettier-ignore
const V5P_SLICES = slices([
  '1x1x1:m', '1x2x1:m', '2x2x1:m', '2x2x2:m', '2x2x4:m', '2x4x4:m',
  '4x4x4:t', '4x4x8:t', '4x4x12:t', '4x4x16:t', '4x4x20:t', '4x4x24:t',
  '4x8x8:t', '4x8x12:t', '4x8x16:t', '4x8x20:t', '4x8x24:t',
  '4x12x12:t', '4x12x16:t', '4x12x20:t', '4x12x24:t',
  '4x16x16:t', '4x16x20:t', '4x16x24:t',
  '8x8x8:t', '8x8x12:t', '8x8x16:t', '8x8x20:t', '8x8x24:t',
  '8x12x12:t', '8x12x16:t', '8x12x20:t', '8x12x24:t',
  '8x16x16:t', '8x16x20:t', '8x16x24:t',
  '12x12x12:t', '12x12x16:t', '12x12x20:t', '12x12x24:t',
  '12x16x16:t', '12x16x20:t', '12x16x24:t',
  '16x16x16:t', '16x16x20:t', '16x16x24:t',
]);

// prettier-ignore
const V7X_SLICES = slices([
  '1x1x1:m', '1x2x1:m', '2x2x1:m', '2x2x2:m', '2x2x4:m', '2x4x4:m',
  '4x4x4:t', '4x4x8:t', '4x4x12:t', '4x4x16:t', '4x4x20:t', '4x4x24:t', '4x4x32:t',
  '4x8x8:t', '4x8x12:t', '4x8x16:t', '4x8x20:t', '4x8x24:t', '4x8x32:t',
  '4x12x12:t', '4x12x16:t', '4x12x20:t', '4x12x24:t',
  '4x16x16:t', '4x16x20:t', '4x16x24:t', '4x16x32:t',
  '8x8x8:t', '8x8x12:t', '8x8x16:t', '8x8x20:t', '8x8x24:t', '8x8x32:t',
  '8x12x12:t', '8x12x16:t', '8x12x20:t', '8x12x24:t',
  '8x16x16:t', '8x16x20:t', '8x16x24:t', '8x16x32:t',
  '12x12x12:t', '12x12x16:t', '12x12x20:t', '12x12x24:t',
  '12x16x16:t', '12x16x20:t', '12x16x24:t', '12x24x24:t',
  '16x16x16:t', '16x16x20:t', '16x16x24:t', '16x16x32:t',
]);

export const CHIPS: ChipSpec[] = [
  {
    id: 'a100-sxm',
    name: 'A100 SXM',
    vendor: 'NVIDIA',
    formats: {
      bf16: 312e12,
      int8: 624e12,
      int4: 1248e12,
      fp8: 'kernel-widened',
      mxfp8: 'kernel-widened',
      mxfp4: 'kernel-widened',
      nvfp4: 'kernel-widened',
    },
    hbmCapacity: 80e9,
    hbmBandwidth: 2.039e12,
    // NVLink 3: 600 GB/s bidirectional per GPU -> 300 GB/s one-way, HGX/DGX board of 8.
    // Scale-out: DGX A100 reference, 8x ConnectX-6 200 Gb/s HDR IB (one per
    // GPU) -> 25 GB/s one-way, 8 nodes.
    interconnect: {
      bandwidthPerChip: 300e9,
      latency: 2e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 25e9, latency: 5e-6, maxNodes: 8 },
    },
    // measured: MAMF 271/312 TF bf16; BabelStream 1.73/2.04 TB/s
    realizableFlopsFrac: 0.85,
    realizableHbmBwFrac: 0.85,
    costPerHour: 0.81,
    tdp: 400,
  },
  {
    id: 'h100-sxm',
    name: 'H100 SXM',
    vendor: 'NVIDIA',
    // int8 shares the fp8 unit and rate. No int4 unit: Hopper dropped it.
    formats: {
      bf16: 989e12,
      fp8: 1979e12,
      int8: 1979e12,
      int4: 'kernel-widened',
      mxfp8: 'kernel-widened',
      mxfp4: 'kernel-widened',
      nvfp4: 'kernel-widened',
    },
    hbmCapacity: 80e9,
    hbmBandwidth: 3.35e12,
    // NVLink 4: 900 GB/s bidirectional per GPU -> 450 GB/s one-way, HGX board of 8.
    // Scale-out: DGX/HGX H100 reference, one ConnectX-7 400 Gb/s NDR IB NIC
    // per GPU -> 50 GB/s one-way, 8-node cluster cap for the calculator.
    interconnect: {
      bandwidthPerChip: 450e9,
      latency: 2e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 50e9, latency: 5e-6, maxNodes: 8 },
    },
    // measured: sustained GEMM ~720-794/989 TF bf16, fp8 fraction lower
    // (SemiAnalysis/MAMF); BabelStream copy/triad ~0.9
    // wgmma m64 at full rate; legacy mma.sync m16 sustains ~2/3 of peak
    // (Luo et al. 2024, arXiv:2402.13499, measured ~63-67% on H800)
    mmaShapes: [
      { m: 64, rate: 1 },
      { m: 16, rate: 0.67 },
    ],
    realizableFlopsFrac: 0.75,
    realizableHbmBwFrac: 0.9,
    costPerHour: 1.8,
    tdp: 700,
  },
  {
    id: 'h200-sxm',
    name: 'H200 SXM',
    vendor: 'NVIDIA',
    // same GH100 die: units and kernel story as H100 above
    formats: {
      bf16: 989e12,
      fp8: 1979e12,
      int8: 1979e12,
      int4: 'kernel-widened',
      mxfp8: 'kernel-widened',
      mxfp4: 'kernel-widened',
      nvfp4: 'kernel-widened',
    },
    hbmCapacity: 141e9,
    hbmBandwidth: 4.8e12,
    // NVLink 4 board of 8, scale-out one CX-7 400 Gb NDR NIC per GPU (DGX
    // H200 reference) -> 50 GB/s one-way, 8 nodes.
    interconnect: {
      bandwidthPerChip: 450e9,
      latency: 2e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 50e9, latency: 5e-6, maxNodes: 8 },
    },
    // same GH100 silicon: sustained GEMM as H100; copy kernels ~0.85-0.9
    mmaShapes: [
      { m: 64, rate: 1 },
      { m: 16, rate: 0.67 },
    ],
    realizableFlopsFrac: 0.75,
    realizableHbmBwFrac: 0.9,
    costPerHour: 2.25,
    tdp: 700,
  },
  {
    id: 'b200',
    name: 'B200 SXM',
    vendor: 'NVIDIA',
    formats: {
      bf16: 2250e12,
      fp8: 4500e12,
      mxfp8: 4500e12,
      fp4: 9000e12,
      mxfp4: 9000e12,
      nvfp4: 9000e12,
      int8: 4500e12,
      int4: 'kernel-widened',
    },
    hbmCapacity: 180e9,
    hbmBandwidth: 7.7e12,
    // NVLink 5: 1.8 TB/s bidirectional -> 900 GB/s one-way, HGX board of 8.
    // Scale-out: DGX B200, one CX-7 400 Gb/s NIC per GPU -> 50 GB/s one-way
    interconnect: {
      bandwidthPerChip: 900e9,
      latency: 2e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 50e9, latency: 5e-6, maxNodes: 8 },
    },
    // measured: MAMF 1745/2250 TF bf16, fp8 ~0.76; copy kernels 6.6-7 TB/s
    // and improving with drivers
    // tcgen05.mma M=128 fills the datapath; M=64 drives half of it (PTX ISA,
    // tcgen05 data-path layouts), so a small group pays a full 128 either way
    mmaShapes: [
      { m: 128, rate: 1 },
      { m: 64, rate: 0.5 },
    ],
    realizableFlopsFrac: 0.75,
    realizableHbmBwFrac: 0.85,
    costPerHour: 3.6,
    tdp: 1000,
  },
  {
    id: 'gb200-nvl72',
    name: 'GB200 NVL72',
    vendor: 'NVIDIA',
    // same units and int4 kernel story as the B200 entry above
    formats: {
      bf16: 2500e12,
      fp8: 5000e12,
      mxfp8: 5000e12,
      fp4: 10000e12,
      mxfp4: 10000e12,
      nvfp4: 10000e12,
      int8: 5000e12,
      int4: 'kernel-widened',
    },
    hbmCapacity: 186e9,
    hbmBandwidth: 8e12,
    // Fifth-generation NVLink: 1.8 TB/s bidirectional per GPU across one
    // NVLink domain -> 900 GB/s one-way. The rack holds 72 GPUs; 64 is what
    // gets deployed, since the remaining factor of 9 is one no expert or
    // head count divides by, leaving those chips with no role to fill.
    interconnect: { bandwidthPerChip: 900e9, latency: 2e-6, domainSize: 64 },
    // B200 silicon at a higher datasheet clock: MAMF 1822/2500, same HBM
    mmaShapes: [
      { m: 128, rate: 1 },
      { m: 64, rate: 0.5 },
    ],
    realizableFlopsFrac: 0.75,
    realizableHbmBwFrac: 0.85,
    costPerHour: 4.5,
    // the 2.7 kW superchip minus ~300 W of Grace, over its 2 GPUs
    tdp: 1200,
  },
  {
    id: 'b300',
    name: 'B300 SXM',
    vendor: 'NVIDIA',
    // Blackwell Ultra's uplift is FP4-only. NVIDIA quotes DGX B300 at 108
    // PFLOPS dense FP4 over its 8 GPUs, so 13.5 each against the B200
    // board's 9, and 72 PFLOPS sparse FP8, whose dense half is the same 4.5
    // the B200 entry above already carries. BF16 is half of FP8 and is
    // unchanged too. Same int4 kernel story as B200.
    formats: {
      bf16: 2250e12,
      fp8: 4500e12,
      mxfp8: 4500e12,
      fp4: 13500e12,
      mxfp4: 13500e12,
      nvfp4: 13500e12,
      int8: 'kernel-widened',
      int4: 'kernel-widened',
    },
    // 288 GB of HBM3e, 270 usable, which is the 2.1 TB NVIDIA quotes for
    // the 8-GPU system
    hbmCapacity: 270e9,
    hbmBandwidth: 8e12,
    // NVLink 5: the system's 14.4 TB/s aggregate over 8 GPUs is 1.8 TB/s
    // bidirectional each -> 900 GB/s one-way, as on B200.
    // Scale-out: one 800 Gb/s ConnectX-8 per GPU -> 100 GB/s one-way.
    interconnect: {
      bandwidthPerChip: 900e9,
      latency: 2e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 100e9, latency: 5e-6, maxNodes: 8 },
    },
    mmaShapes: [
      { m: 128, rate: 1 },
      { m: 64, rate: 0.5 },
    ],
    realizableFlopsFrac: 0.75,
    realizableHbmBwFrac: 0.85,
    costPerHour: 5.4,
    tdp: 1400,
  },
  {
    id: 'vr100-nvl72',
    name: 'VR100 NVL72',
    vendor: 'NVIDIA',
    formats: {
      bf16: 4000e12,
      fp8: 17500e12,
      mxfp8: 17500e12,
      fp4: 35000e12,
      mxfp4: 35000e12,
      nvfp4: 35000e12,
      int4: 'kernel-widened',
    },
    // 288 GB HBM4 at 22 TB/s, the rack's 20.7 TB / 1580 TB/s over 72 GPUs.
    hbmCapacity: 288e9,
    hbmBandwidth: 22e12,
    // Sixth-generation NVLink: 3.6 TB/s bidirectional per GPU -> 1.8 TB/s
    // one-way. The rack's 260 TB/s is switch aggregate, not per-GPU (72 x
    // 3.6). Domain is 64 of the 72 for the same reason as GB200 above.
    interconnect: { bandwidthPerChip: 1.8e12, latency: 2e-6, domainSize: 64 },
    // unannounced: extrapolated from Blackwell-generation sustained fractions
    realizableFlopsFrac: 0.7,
    realizableHbmBwFrac: 0.8,
    // unannounced part, no public price
    tdp: 2300, // semianalysis rumor (update on release!)
  },
  {
    id: 'rtx-pro-6000',
    name: 'RTX PRO 6000',
    vendor: 'NVIDIA',
    formats: {
      bf16: 503.8e12,
      fp8: 1007.6e12,
      mxfp8: 1007.6e12,
      fp4: 2015.2e12,
      mxfp4: 2015.2e12,
      nvfp4: 2015.2e12,
      int8: 1007.6e12,
      int4: 'kernel-widened',
    },
    // 96 GB GDDR7 (not HBM) on a 512-bit bus.
    hbmCapacity: 96e9,
    hbmBandwidth: 1.792e12,
    // No NVLink: PCIe 5.0 x16 only, 64 GB/s one-way. Modeled as a switched
    // 8-GPU PCIe server.
    // Scale-out: no vendor fabric; modeled as one 100 GbE NIC per GPU ->
    // 12.5 GB/s one-way, 8 nodes.
    interconnect: {
      bandwidthPerChip: 64e9,
      latency: 5e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 12.5e9, latency: 10e-6, maxNodes: 8 },
    },
    // measured: cuBLAS 386/504 TF bf16 (ungated part); GDDR7 fraction inferred
    // from the 5090's subsystem
    realizableFlopsFrac: 0.75,
    realizableHbmBwFrac: 0.9,
    costPerHour: 1.26,
    tdp: 600,
  },
  {
    id: 'rtx-5090',
    name: 'RTX 5090',
    vendor: 'NVIDIA',
    formats: {
      bf16: 209.5e12,
      fp8: 419e12,
      mxfp8: 419e12,
      fp4: 1676e12,

      mxfp4: 1676e12,
      nvfp4: 1676e12,
      int8: 838e12,
      int4: 'kernel-widened',
    },
    // 32 GB GDDR7 (not HBM) on a 512-bit bus.
    hbmCapacity: 32e9,
    hbmBandwidth: 1.792e12,
    // No NVLink: PCIe 5.0 x16 only, 64 GB/s one-way. Modeled as a switched
    // 8-GPU PCIe server, as RTX PRO 6000 above.
    // Scale-out: no vendor fabric; one 100 GbE NIC per GPU -> 12.5 GB/s
    // one-way, 8 nodes.
    interconnect: {
      bandwidthPerChip: 64e9,
      latency: 5e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 12.5e9, latency: 10e-6, maxNodes: 8 },
    },
    // GeForce datasheet tensor rates already carry the fp32-accumulate gate, so
    // tuned GEMMs land at ~1.0x datasheet; streaming reads 1.67/1.79 TB/s
    realizableFlopsFrac: 0.95,
    realizableHbmBwFrac: 0.95,
    costPerHour: 0.54,
    tdp: 575,
  },
  {
    id: 'rtx-4090',
    name: 'RTX 4090',
    vendor: 'NVIDIA',
    formats: {
      bf16: 165.2e12,
      fp8: 330.4e12,
      int8: 660.6e12,
      int4: 1321.2e12,
      mxfp8: 'kernel-widened',
      mxfp4: 'kernel-widened',
      nvfp4: 'kernel-widened',
    },
    // 24 GB GDDR6X.
    hbmCapacity: 24e9,
    hbmBandwidth: 1.008e12,
    // No NVLink: PCIe 4.0 x16 only, ~32 GB/s one-way. Modeled as a switched
    // 8-GPU PCIe server, as RTX PRO 6000 above.
    // Scale-out: no vendor fabric; one 100 GbE NIC per GPU -> 12.5 GB/s
    // one-way, 8 nodes.
    interconnect: {
      bandwidthPerChip: 32e9,
      latency: 5e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 12.5e9, latency: 10e-6, maxNodes: 8 },
    },
    // gated datasheet as above: fp32-acc GEMMs ~1.0x datasheet; streaming ~0.9
    realizableFlopsFrac: 0.95,
    realizableHbmBwFrac: 0.9,
    costPerHour: 0.36,
    tdp: 450,
  },
  {
    id: 'mi300x',
    name: 'MI300X',
    vendor: 'AMD',
    formats: {
      bf16: 1307.4e12,
      fp8: 2614.9e12,
      int8: 2614.9e12,
      int4: 'kernel-widened',
      mxfp4: 'kernel-widened',
      mxfp8: 'kernel-widened',
    },
    hbmCapacity: 192e9,
    hbmBandwidth: 5.3e12,
    // 8-GPU UBB platform, fully connected: 7 Infinity Fabric links x 128 GB/s
    // bidirectional each = 896 GB/s bidi -> 448 GB/s one-way egress.
    // Scale-out: one 400 Gb/s NIC per GPU (CX-7 IB on the Azure ND MI300X
    // v5 reference) -> 50 GB/s one-way, 8 nodes.
    interconnect: {
      bandwidthPerChip: 448e9,
      latency: 2e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 50e9, latency: 5e-6, maxNodes: 8 },
    },
    // sustained clock ~1.2 of the 2.1 GHz boost the datasheet assumes: tuned
    // GEMMs 620-708/1307 TF bf16 (SemiAnalysis/AMD MAF); BabelStream
    // 3.9-4.3/5.3 TB/s
    // MFMA 16x16xK is a full-rate instruction on CDNA3/CDNA4
    mmaShapes: [{ m: 16, rate: 1 }],
    realizableFlopsFrac: 0.5,
    realizableHbmBwFrac: 0.8,
    costPerHour: 2.16,
    tdp: 750,
  },
  {
    id: 'mi325x',
    name: 'MI325X',
    vendor: 'AMD',
    formats: {
      bf16: 1307.4e12,
      fp8: 2614.9e12,
      int8: 2614.9e12,
      int4: 'kernel-widened',
      mxfp4: 'kernel-widened',
      mxfp8: 'kernel-widened',
    },
    hbmCapacity: 256e9,
    hbmBandwidth: 6e12,
    // Same UBB fabric as MI300X, scale-out one 400 GbE RoCE NIC per GPU, 8 nodes.
    interconnect: {
      bandwidthPerChip: 448e9,
      latency: 2e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 50e9, latency: 5e-6, maxNodes: 8 },
    },
    // same silicon at 1000 W sustains higher clocks: AMD MAF 843/1307 TF; HBM
    // fraction inferred from MI300X
    mmaShapes: [{ m: 16, rate: 1 }],
    realizableFlopsFrac: 0.6,
    realizableHbmBwFrac: 0.8,
    costPerHour: 2.7,
    tdp: 1000,
  },
  {
    id: 'mi355x',
    name: 'MI355X',
    vendor: 'AMD',
    formats: {
      bf16: 2500e12,
      fp8: 5000e12,
      mxfp8: 5000e12,
      fp4: 10100e12,
      mxfp4: 10100e12,
      int8: 5000e12,
      int4: 'kernel-widened',
    },
    hbmCapacity: 288e9,
    hbmBandwidth: 8e12,
    // 8-GPU UBB, fully connected: 7 IF4 links x 153.6 GB/s bidirectional
    // = 1075 GB/s bidi -> 537 GB/s one-way egress.
    // Scale-out: one 400 GbE RoCE NIC per GPU, 8 nodes.
    interconnect: {
      bandwidthPerChip: 537e9,
      latency: 2e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 50e9, latency: 5e-6, maxNodes: 8 },
    },
    // measured: HipKittens ~1610/2500 TF bf16, fp8 ~0.65; streaming reads
    // ~5.8/8 TB/s
    mmaShapes: [{ m: 16, rate: 1 }],
    realizableFlopsFrac: 0.65,
    realizableHbmBwFrac: 0.7,
    costPerHour: 3.6,
    tdp: 1400,
  },
  {
    id: 'gaudi2',
    name: 'Gaudi 2',
    vendor: 'Intel',
    formats: { bf16: 432e12, fp8: 865e12, int8: 'kernel-widened' },
    hbmCapacity: 96e9,
    hbmBandwidth: 2.45e12,
    // HLBA-225 baseboard: each OAM uses 21 x 100GbE links for all-to-all
    // OAM-to-OAM traffic, and 3 more ports are routed out for scale-out
    // -> 3 x 100 GbE = 37.5 GB/s one-way, 8 nodes.
    interconnect: {
      bandwidthPerChip: 262.5e9,
      latency: 2e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 37.5e9, latency: 5e-6, maxNodes: 8 },
    },
    realizableFlopsFrac: 0.3, // penalty for software stack being horrible sorry
    realizableHbmBwFrac: 0.8,
    costPerHour: 0.54,
    tdp: 600,
  },
  {
    id: 'gaudi3',
    name: 'Gaudi 3',
    vendor: 'Intel',
    formats: { bf16: 1835e12, fp8: 1835e12, int8: 'kernel-widened' },
    hbmCapacity: 128e9,
    hbmBandwidth: 3.7e12,
    // HLB-325 baseboard follows the same 8-OAM all-to-all pattern as Gaudi2
    // with 21 internal 200GbE links per OAM, and 3 more ports are routed out
    // -> 3 x 200 GbE = 75 GB/s one-way, 8 nodes.
    interconnect: {
      bandwidthPerChip: 525e9,
      latency: 2e-6,
      domainSize: 8,
      scaleOut: { bandwidthPerChip: 75e9, latency: 5e-6, maxNodes: 8 },
    },
    realizableFlopsFrac: 0.3, // penalty for software stack being horrible sorry
    realizableHbmBwFrac: 0.75,
    costPerHour: 1.26,
    // liquid-cooled rating; the air-cooled OAM runs 900 W
    tdp: 1200,
  },
  {
    id: 'inferentia2',
    name: 'Inferentia2',
    vendor: 'AWS',
    formats: { bf16: 190e12, fp8: 190e12, int8: 380e12 },
    hbmCapacity: 32e9,
    hbmBandwidth: 8.8e11,
    // NeuronLink-v2 ring, 192 GB/s per chip -> 96 GB/s one-way per our
    // halve-bidirectional convention, inf2.48xlarge = 12 chips.
    // The 6-chip 24xlarge is an open segment, only the full 12 closes a ring.
    interconnect: {
      bandwidthPerChip: 96e9,
      latency: 1e-6,
      domainSize: 12,
      topologies: slices(['1:m', '6:m', '12:t']),
      chipsPerHost: 12,
    },
    // NKI kernel suites average ~0.5-0.6 of peak; single-chip decode streams
    // ~0.57 of HBM
    realizableFlopsFrac: 0.55,
    realizableHbmBwFrac: 0.6,
    costPerHour: 0.36,
    // no tdp: AWS publishes none and no third party has estimated one, so
    // the chip sits out the per-kW view rather than carry a guess
  },
  {
    id: 'trainium1',
    name: 'Trainium1',
    vendor: 'AWS',
    formats: { bf16: 190e12, fp8: 190e12, int8: 380e12 },
    hbmCapacity: 32e9,
    hbmBandwidth: 8.8e11,
    // NeuronLink-v2 2D torus in trn1.32xlarge (16 chips), 768 GB/s per chip
    // -> 384 GB/s one-way per our halve-bidirectional convention
    interconnect: {
      bandwidthPerChip: 384e9,
      latency: 1e-6,
      domainSize: 16,
      topologies: slices(['1x1:m', '4x4:t']),
      chipsPerHost: 16,
    },
    // NKI kernel suites average ~0.5-0.6 of peak (AccelOpt); BW fraction
    // inferred from Inferentia2 twin silicon
    realizableFlopsFrac: 0.55,
    realizableHbmBwFrac: 0.6,
    costPerHour: 0.45,
    // no tdp: AWS publishes none and no third party has estimated one, so
    // the chip sits out the per-kW view rather than carry a guess (AWS's
    // Trainium2 launch claim — 4x training perf at 2x perf/W — would put it
    // near half Trainium2's ~500 W if an estimate is ever wanted)
  },
  {
    id: 'trainium2',
    name: 'Trainium2',
    vendor: 'AWS',
    formats: { bf16: 668.75e12, fp8: 1300e12, int8: 'kernel-widened' },
    hbmCapacity: 96e9,
    hbmBandwidth: 2.9e12,
    // NeuronLink-v3 4x4 2D torus in trn2.48xlarge (16 chips), 1024 GB/s per
    // chip -> 512 GB/s one-way per our halve-bidirectional convention
    // The 64-chip Trn2 UltraServer is NOT modeled: its inter-instance links
    // run at 256 GB/s/chip, so the fabric isn't one uniform domain.
    interconnect: {
      bandwidthPerChip: 512e9,
      latency: 1e-6,
      domainSize: 16,
      topologies: slices(['4x4:t']),
      chipsPerHost: 16,
    },
    // AccelOpt baseline 0.45 -> tuned ~0.6; SemiAnalysis calls the spec-sheet
    // FLOPs inflated; built low-intensity for BW-bound work
    realizableFlopsFrac: 0.55,
    realizableHbmBwFrac: 0.7,
    costPerHour: 0.9,
    tdp: 500, // semianalysis rumor
  },
  {
    id: 'tpu-v5p',
    name: 'TPU v5p',
    vendor: 'Google',
    formats: { bf16: 459e12, int8: 918e12, int4: 1836e12, fp8: 'kernel-widened' },
    hbmCapacity: 95e9,
    hbmBandwidth: 2.765e12,
    // 3D torus, 6 ICI links x 90 GB/s one-way each = 540 GB/s egress.
    // Slices below the 4x4x4 cube are OPEN MESHES. From the cube up the OCS
    // provides wraparound on every axis. Largest acquirable slice is
    // 16x16x24 = 6144 chips (v5p-12288), the fast domain, though the
    // physical pod is larger.
    interconnect: {
      bandwidthPerChip: 540e9,
      latency: 1e-6,
      domainSize: 6144,
      topologies: V5P_SLICES,
      chipsPerHost: 4,
    },
    realizableFlopsFrac: 0.9,
    realizableHbmBwFrac: 0.85,
    costPerHour: 0.81,
    tdp: 540, // Epoch AI derivation from Google's perf/W disclosures
  },
  {
    id: 'tpu-v6e',
    name: 'TPU v6e',
    vendor: 'Google',
    formats: { bf16: 918e12, int8: 1836e12, int4: 3672e12, fp8: 'kernel-widened' },
    hbmCapacity: 32e9,
    hbmBandwidth: 1.64e12,
    // 2D torus, 4 ICI links x 90 GB/s one-way each = 360 GB/s egress, 16x16
    // slice. No OCS: smaller slices are open meshes, 8x16 a cylinder (wrap
    // on the 16-axis only), 16x16 the full torus. Note no 2-chip slice.
    interconnect: {
      bandwidthPerChip: 360e9,
      latency: 1e-6,
      domainSize: 256,
      topologies: V6E_SLICES,
      chipsPerHost: 8,
    },
    realizableFlopsFrac: 0.9,
    realizableHbmBwFrac: 0.85,
    costPerHour: 0.594,
    tdp: 380, // Epoch AI derivation from Google's perf/W disclosures, SemiAnalysis estimates ~300 W
  },
  {
    id: 'tpu-v7x',
    name: 'TPU v7x',
    vendor: 'Google',
    formats: {
      bf16: 2307e12,
      fp8: 4614e12,
      int8: 'kernel-widened',
      mxfp4: 'kernel-widened',
      nvfp4: 'kernel-widened',
    },
    hbmCapacity: 192e9,
    hbmBandwidth: 7.37e12,
    // 3D torus, 6 ICI links x 90 GB/s one-way each = 540 GB/s egress.
    // Slice topologies mirror v5p (meshes below the 4x4x4 cube, OCS torus
    // from it up) plus z-extents to 32. Domain is the 9,216-chip pod.
    interconnect: {
      bandwidthPerChip: 540e9,
      latency: 1e-6,
      domainSize: 9216,
      topologies: V7X_SLICES,
      chipsPerHost: 4,
    },
    realizableFlopsFrac: 0.85,
    realizableHbmBwFrac: 0.85,
    costPerHour: 4.5,
    // Epoch AI derivation from Google's perf/W disclosures; the
    // 9216-chip-pod ≈ 10 MW claim lands ~1.08 kW/chip with overhead
    tdp: 960,
  },
];

export const CHIPS_BY_ID: Record<string, ChipSpec> = Object.fromEntries(
  CHIPS.map((c) => [c.id, c]),
);

// Parse slice-topology specs ("4x8x12:t") into named SliceTopology entries.
// Types follow the TPU Topology Visualizer
// (https://tpu-visualizer.uc.r.appspot.com/): m = mesh (no wraparound),
// c = cylinder (wraparound on the last axis only), t = torus (all axes).
// Twisted-torus variants share dims and count with their plain torus and
// are not modeled separately (the twist reroutes bisection bandwidth, not
// per-chip egress).
function slices(specs: string[]): SliceTopology[] {
  return specs.map((spec) => {
    const [name, type] = spec.split(':');
    const dims = name.split('x').map(Number);
    return {
      name,
      dims,
      wrapped: dims.map((_, i) => (type === 't' ? true : type === 'c' && i === dims.length - 1)),
      count: dims.reduce((p, v) => p * v, 1),
    };
  });
}

// This chip's peak rate for `d`, or undefined if it has no unit for it.
export function peakFlops(chip: ChipSpec, d: Dtype): number | undefined {
  const f = chip.formats[d];
  return typeof f === 'number' ? f : undefined;
}
