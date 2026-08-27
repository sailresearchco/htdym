import {
  activeParams,
  FlopsByWidth,
  flopsPerDecodeToken,
  flopsPerPrefillToken,
  kvBytesPerSeq,
  totalParams,
  weightBytesTotal,
} from '../model/utils';
import { ModelSpec } from '../model/models';
import { DTYPE_BYTES } from '../model/dtype';
import { ChipSpec, peakFlops } from '../hardware/chips';
import { SimInput } from './surface/api';
import { runnableOn } from './sim/run/validate';

export interface Roofline {
  paramsTotal: number;
  paramsActive: number;
  weightBytesTotal: number;
  minChipsForWeights: number;
  kvBytesPerSeq: number;
  // tokens per stage-batch for prefill compute to cover the weight load
  critTokens: number;
  // compute-roofline prefill tok/s/chip ceiling
  prefillCeiling: number;
  // Decode tok/s/chip ceiling band at B->infinity, ideal sharding. The fixed
  // serial -> full-overlap bracket the UI draws hatched, the one place the
  // bracket survives. The overlap knobs cannot move it.
  decodeCeilingOverlapped: number;
  decodeCeilingSerial: number;
}

// Seconds of matmul for one token's FLOP buckets, each at the chip's peak
// for that width. Null when the chip has no unit for one of them.
export function matmulSeconds(
  byWidth: FlopsByWidth,
  chip: ChipSpec,
  flopsFrac: number,
): number | null {
  let seconds = 0;
  for (const [d, flops] of byWidth) {
    const peak = peakFlops(chip, d);
    if (peak === undefined) return null;
    seconds += flops / (peak * flopsFrac);
  }
  return seconds;
}

export function roofline(
  m: ModelSpec,
  chip: ChipSpec,
  workload: SimInput['workload'],
  flopsFrac = 1,
): Roofline | null {
  m = runnableOn(m, chip);
  const BW = chip.hbmBandwidth * chip.realizableHbmBwFrac;
  const T = workload.prefillLen;
  const ctxAvg = T + workload.generateLen / 2;
  const wTotal = weightBytesTotal(m);
  const perSeqCompute = matmulSeconds(flopsPerDecodeToken(m, ctxAvg), chip, flopsFrac);
  const prefillCompute = matmulSeconds(flopsPerPrefillToken(m, T), chip, flopsFrac);
  if (perSeqCompute === null || prefillCompute === null) return null;
  const perSeqMemory = kvBytesPerSeq(m, DTYPE_BYTES[m.precision.kv], ctxAvg, 'read') / BW;
  return {
    paramsTotal: totalParams(m),
    paramsActive: activeParams(m),
    weightBytesTotal: wTotal,
    minChipsForWeights: Math.ceil(wTotal / chip.hbmCapacity),
    kvBytesPerSeq: kvBytesPerSeq(m, DTYPE_BYTES[m.precision.kv], T + workload.generateLen),
    critTokens: wTotal / (BW * prefillCompute),
    prefillCeiling: 1 / prefillCompute,
    decodeCeilingOverlapped: 1 / Math.max(perSeqCompute, perSeqMemory),
    decodeCeilingSerial: 1 / (perSeqCompute + perSeqMemory),
  };
}
