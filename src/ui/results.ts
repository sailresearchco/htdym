import type { Roofline } from '../core/engine/roofline';
import type { ShardingRole } from '../core/engine/sim/ir/sharding/roles';
import type { Diagnostic, Mesh, MoeDispatch } from '../core/engine/surface/deploy';
import type { ChipSpec } from '../core/hardware/chips';

export type { Diagnostic };

// A chip as the UI holds it: the editable spec plus the machine the user
// picked for it (a ring-fabric slice name or a switched-fabric node count),
// both defaulting in machineOf when unset.
export interface UiChip extends ChipSpec {
  slice?: string;
  nodes?: number;
}

export interface ComponentTimes {
  compute: number;
  memory: number;
  comms: number;
}

export type Boundedness = keyof ComponentTimes;

export interface UiWorkload {
  prefillLen: number;
  generateLen: number;
  // 0 = unconstrained sentinel while the field is edited (as the old UI)
  sloTokPerSecPerUser?: number;
  batching?: 'max' | 'b1';
}

export interface UiOverlap {
  memoryOverlap: number;
  commsOverlap: number;
}

// One streamed configuration row: a role-size tuple's best placement and
// dispatch, evaluated for both phases. Mostly display-ready scalars, plus
// the winning mesh so the details view can re-lower the op trace.
export interface UiResult {
  id: string;
  chipId: string;
  sizes: Partial<Record<ShardingRole, number>>;
  placement: string;
  // how many of the TP ranks hold a sequence slice instead of a head slice
  decodeContextParallel: number;
  dispatch: MoeDispatch;
  // the placement's resolved mesh, for re-lowering the trace in the details DAG
  mesh: Mesh;
  nChips: number;
  workload: { prefillLen: number; generateLen: number };
  diagnostics: Diagnostic[];
  memory?: {
    weightBytesPerChip: number;
    // HBM left for KV after the weights land: what actually caps batch
    kvSpaceBytesPerChip: number;
    kvBytesPerSeqPerChip: number;
    // machine total (per-chip residency x DPA groups)
    maxResidentSeqs: number;
  };
  // cost-efficiency for the whole workload vs the HMVP baseline; filled at
  // render time from the live prices and baseline, never by the worker
  requestEff?: number;
  prefill?: {
    tokPerSecPerChip: number;
    ttft: number;
    batchSeqs: number;
    mfu: number;
    fracOfCeiling: number;
    boundBy: Boundedness;
    components: ComponentTimes;
    // (rate ÷ relative price) over the HMVP's rate; filled at render time
    // from the live prices and baseline, never by the worker
    eff?: number;
    // rate over the HMVP's rate — pure speed, price not included
    relRate?: number;
  };
  decode?: {
    tokPerSecPerChip: number;
    tokPerSecPerUser: number;
    tpot: number;
    stepTime: number;
    batchPerStage: number;
    residentSeqs: number;
    mfu: number;
    fracOfCeiling: number;
    // operating tok/s/chip over this config's own B -> inf rate (KV gate
    // off); low = throughput is KV-room-starved, not sharding-limited
    batchSaturation?: number;
    boundBy: Boundedness;
    components: ComponentTimes;
    // (rate ÷ relative price) over the HMVP's rate; filled at render time
    // from the live prices and baseline, never by the worker
    eff?: number;
    // rate over the HMVP's rate — pure speed, price not included
    relRate?: number;
  };
}

// One chip's slot in the leaderboard: its fixed machine, the hardware
// rooflines, and the streamed configs (best-first arrival order).
export interface UiGroup {
  key: string;
  chip: ChipSpec;
  machineLabel: string;
  nChips: number;
  hardware: Roofline | null;
  configs: UiResult[];
  // streaming progress for this chip's search
  done: number;
  total: number;
  // the whole chip failed to evaluate (e.g. unsupported dtype)
  error?: string;
}

// glyph and color class for a diagnostic line, by severity
export const DIAG_GLYPH: Record<Diagnostic['severity'], string> = {
  error: '✕',
  warning: '⚠',
  info: 'ⓘ',
};
export const DIAG_CLASS: Record<Diagnostic['severity'], string> = {
  error: 'diag-err',
  warning: 'diag-warn',
  info: 'diag-info',
};

export function hasError(r: UiResult): boolean {
  return r.diagnostics.some((d) => d.severity === 'error');
}

export function hasWarning(r: UiResult): boolean {
  return r.diagnostics.some((d) => d.severity === 'warning');
}

// Traffic-light ramp for attainment gauges (was FabricView's).
export function gaugeColor(frac: number): string {
  return frac >= 0.9 ? 'var(--status-ok)' : frac >= 0.5 ? 'var(--bar-warn)' : 'var(--red)';
}

// Progress-ring tone: red early, yellow past 30% priced, green when complete.
export function ringTone(frac: number): string {
  return frac >= 1 ? 'var(--green)' : frac >= 0.3 ? 'var(--bar-warn)' : 'var(--red)';
}
