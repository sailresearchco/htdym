/**
 * Metric registry shared by the results table (columns) and the tradeoff
 * chart (axes). `value` returns the sortable/plottable scalar: the predicted
 * value at the Memory/Comms overlap you set.
 */
import { UiResult } from './results';
import { fmtBytes, fmtInt, fmtMult, fmtPct, fmtSI, fmtTime } from './format';

export type MetricGroup = 'memory' | 'prefill' | 'decode';

export interface MetricDef {
  /** shown behind an instant info icon where the metric is displayed */
  info?: string;
  key: string;
  group: MetricGroup;
  label: string;
  desc: string;
  value: (r: UiResult) => number | null;
  format: (r: UiResult) => string;
  /** offer as a chart axis */
  chartable: boolean;
  /** ranking direction: true when smaller values win (latencies) */
  lowerIsBetter?: boolean;
}

const na = '—';

export const METRICS: MetricDef[] = [
  {
    key: 'prefillTokPerChip',
    group: 'prefill',
    label: 'Prefill tok/s/chip',
    desc: 'Prefill tokens per second per chip, with the pipeline saturated.',
    value: (r) => r.prefill?.tokPerSecPerChip ?? null,
    format: (r) => (r.prefill ? fmtSI(r.prefill.tokPerSecPerChip) : na),
    chartable: true,
  },
  {
    key: 'decodeTokPerChip',
    group: 'decode',
    label: 'Decode tok/s/chip',
    desc: 'Decode tokens per second per chip at the operating batch.',
    value: (r) => r.decode?.tokPerSecPerChip ?? null,
    format: (r) => (r.decode ? fmtSI(r.decode.tokPerSecPerChip) : na),
    chartable: true,
  },
  {
    key: 'prefillTokPerChipRel',
    group: 'prefill',
    label: 'Prefill vs HMVP',
    desc: 'Prefill speed per chip, quoted as a multiple vs the smallest H100 machine that can serve this model — price not included.',
    value: (r) => r.prefill?.relRate ?? null,
    format: (r) => (r.prefill?.relRate !== undefined ? fmtMult(r.prefill.relRate) : na),
    chartable: true,
  },
  {
    key: 'decodeTokPerChipRel',
    group: 'decode',
    label: 'Decode vs HMVP',
    desc: 'Decode speed per chip, quoted as a multiple vs the smallest H100 machine that can serve this model — price not included.',
    value: (r) => r.decode?.relRate ?? null,
    format: (r) => (r.decode?.relRate !== undefined ? fmtMult(r.decode.relRate) : na),
    chartable: true,
  },
  {
    key: 'requestEff',
    group: 'decode',
    label: 'Requests/$',
    desc: 'Requests served per $ of compute, quoted as a multiple vs the smallest H100 machine that can serve this model.',
    value: (r) => r.requestEff ?? null,
    format: (r) => (r.requestEff !== undefined ? fmtMult(r.requestEff) : na),
    chartable: true,
  },
  {
    key: 'prefillEff',
    group: 'prefill',
    label: 'Prefill tok/chip/$',
    desc: 'Prefill tokens per $ of compute, quoted as a multiple vs the smallest H100 machine that can serve this model.',
    value: (r) => r.prefill?.eff ?? null,
    format: (r) => (r.prefill?.eff !== undefined ? fmtMult(r.prefill.eff) : na),
    chartable: true,
  },
  {
    key: 'decodeEff',
    group: 'decode',
    label: 'Decode tok/chip/$',
    desc: 'Decode tokens per $ of compute, quoted as a multiple vs the smallest H100 machine that can serve this model.',
    value: (r) => r.decode?.eff ?? null,
    format: (r) => (r.decode?.eff !== undefined ? fmtMult(r.decode.eff) : na),
    chartable: true,
  },
  {
    key: 'ttft',
    group: 'prefill',
    label: 'TTFT',
    desc: 'Time to first token for a single sequence — TP reduces it, PP does not.',
    value: (r) => r.prefill?.ttft ?? null,
    format: (r) => (r.prefill ? fmtTime(r.prefill.ttft) : na),
    chartable: true,
    lowerIsBetter: true,
  },
  {
    key: 'tokPerUser',
    group: 'decode',
    label: 'Tok/s/user',
    desc: 'Tokens per second a single user sees while decoding (1/TPOT).',
    value: (r) => r.decode?.tokPerSecPerUser ?? null,
    format: (r) => (r.decode ? fmtSI(r.decode.tokPerSecPerUser) : na),
    chartable: true,
  },
  {
    key: 'rps',
    group: 'decode',
    label: 'Req/s',
    desc: 'Whole requests per second the machine sustains at steady state for this workload.',
    value: (r) => machineRps(r),
    format: (r) => {
      const v = machineRps(r);
      return v === null ? na : fmtSI(v);
    },
    chartable: true,
  },
  {
    key: 'prefillFracOfCeiling',
    group: 'prefill',
    label: '% of prefill ceiling',
    desc: "Share of this hardware's best possible prefill rate that this sharding achieves.",
    value: (r) => r.prefill?.fracOfCeiling ?? null,
    format: (r) => (r.prefill ? fmtPct(r.prefill.fracOfCeiling) : na),
    chartable: true,
  },
  {
    key: 'fracOfCeiling',
    group: 'decode',
    label: '% of decode ceiling',
    desc: "Share of this hardware's best possible decode rate that this sharding achieves.",
    value: (r) => r.decode?.fracOfCeiling ?? null,
    format: (r) => (r.decode ? fmtPct(r.decode.fracOfCeiling) : na),
    chartable: true,
  },
  {
    key: 'batchSaturation',
    group: 'decode',
    label: 'Batch saturation',
    desc: "Operating throughput as a share of this sharding's own infinite-batch rate — low means KV room, not the sharding, caps it.",
    value: (r) => r.decode?.batchSaturation ?? null,
    format: (r) =>
      r.decode?.batchSaturation !== undefined ? fmtPct(r.decode.batchSaturation) : na,
    chartable: true,
  },
  {
    key: 'residentAtOp',
    group: 'decode',
    label: 'Resident seqs',
    desc: 'Sequences held resident at the operating point.',
    value: (r) => r.decode?.residentSeqs ?? null,
    format: (r) => (r.decode ? fmtInt(r.decode.residentSeqs) : na),
    chartable: true,
  },
  {
    key: 'stepTime',
    group: 'decode',
    label: 'Decode step time',
    desc: 'Time for one decode step of one microbatch.',
    value: (r) => r.decode?.stepTime ?? null,
    format: (r) => (r.decode ? fmtTime(r.decode.stepTime) : na),
    chartable: true,
    lowerIsBetter: true,
  },
  {
    key: 'weightsPerChip',
    group: 'memory',
    label: 'Weights/chip',
    desc: 'Parameter bytes resident per chip after sharding.',
    value: (r) => r.memory?.weightBytesPerChip ?? null,
    format: (r) => (r.memory ? fmtBytes(r.memory.weightBytesPerChip) : na),
    chartable: true,
  },
  {
    key: 'kvSpacePerChip',
    group: 'memory',
    label: 'KV space/chip',
    desc: 'HBM left for KV cache per chip once the weights are resident.',
    info: 'Chip HBM minus resident weight bytes. This is what caps the batch: divide by KV bytes per sequence to get resident sequences. Sharding that spreads weights wider (higher EP, PP) buys KV space; replicating them (higher DPA) spends it.',
    value: (r) => r.memory?.kvSpaceBytesPerChip ?? null,
    format: (r) => (r.memory ? fmtBytes(r.memory.kvSpaceBytesPerChip) : na),
    chartable: true,
  },
  {
    key: 'prefillMfu',
    group: 'prefill',
    label: 'Prefill MFU',
    desc: 'Model FLOPs utilization during prefill.',
    value: (r) => r.prefill?.mfu ?? null,
    format: (r) => (r.prefill ? fmtPct(r.prefill.mfu) : na),
    chartable: true,
  },
  {
    key: 'decodeMfu',
    group: 'decode',
    label: 'Decode MFU',
    desc: 'Model FLOPs utilization during decode.',
    value: (r) => r.decode?.mfu ?? null,
    format: (r) => (r.decode ? fmtPct(r.decode.mfu) : na),
    chartable: true,
  },
  {
    key: 'maxResidentSeqs',
    group: 'memory',
    label: 'Max resident seqs',
    desc: 'Maximum sequences whose KV cache fits in HBM alongside the weights, machine-wide.',
    value: (r) => r.memory?.maxResidentSeqs ?? null,
    format: (r) => (r.memory ? fmtInt(r.memory.maxResidentSeqs) : na),
    chartable: true,
  },

  // ---- table / details only (not chart axes) ----
  {
    key: 'prefillBound',
    group: 'prefill',
    label: 'Bound',
    desc: 'Which resource limits prefill: compute, HBM memory, or interconnect comms.',
    value: () => null,
    format: (r) => r.prefill?.boundBy ?? na,
    chartable: false,
  },
  {
    key: 'decodeBound',
    group: 'decode',
    label: 'Bound',
    desc: 'Which resource limits decode at the operating batch.',
    value: () => null,
    format: (r) => r.decode?.boundBy ?? na,
    chartable: false,
  },
];

/**
 * Machine-level steady-state requests per second: one request costs T prefill
 * tokens plus S decode tokens, served at the machine's aggregate per-phase
 * rates. Null until both phases are evaluated.
 */
export function machineRps(r: UiResult): number | null {
  const pf = r.prefill;
  const dec = r.decode;
  if (!pf || !dec) return null;
  const { prefillLen: T, generateLen: S } = r.workload;
  const secsPerReq = T / (pf.tokPerSecPerChip * r.nChips) + S / (dec.tokPerSecPerChip * r.nChips);
  return secsPerReq > 0 ? 1 / secsPerReq : null;
}

export const CHART_METRICS = METRICS.filter((m) => m.chartable);

/** The cost metric filling each role — columns, SLA filters, and the
 * details panel consume it. */
export const COST_KEYS = { request: 'requestEff', prefill: 'prefillEff', decode: 'decodeEff' };

/** Leaderboard overview columns: the request cost metric, then prefill, then
 * decode. Slot 0 is also the leaderboard's initial sort. */
export const OVERVIEW_METRIC_KEYS: readonly string[] = [
  COST_KEYS.request,
  COST_KEYS.prefill,
  'prefillTokPerChip',
  'ttft',
  COST_KEYS.decode,
  'decodeTokPerChip',
  'tokPerUser',
];

/** Expanded sweep columns: the overview set plus request rate and batch saturation. */
export const SWEEP_METRIC_KEYS: readonly string[] = [
  ...OVERVIEW_METRIC_KEYS,
  'rps',
  'batchSaturation',
];

export function metricByKey(key: string): MetricDef {
  const m = METRICS.find((x) => x.key === key);
  if (!m) throw new Error(`unknown metric ${key}`);
  return m;
}

export function fmtMetricValue(m: MetricDef, v: number): string {
  // concise scalar formatting for chart tooltips/axes
  switch (m.key) {
    case 'ttft':
    case 'stepTime':
      return fmtTime(v);
    case 'weightsPerChip':
    case 'kvSpacePerChip':
      return fmtBytes(v);
    case 'prefillMfu':
    case 'decodeMfu':
    case 'fracOfCeiling':
    case 'prefillFracOfCeiling':
    case 'batchSaturation':
      return fmtPct(v);
    case 'requestEff':
    case 'prefillEff':
    case 'decodeEff':
    case 'prefillTokPerChipRel':
    case 'decodeTokPerChipRel':
      return fmtMult(v);
    default:
      return fmtSI(v);
  }
}
