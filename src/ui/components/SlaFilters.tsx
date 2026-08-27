/**
 * SLA filter bar: log-scale sliders over serving targets (TTFT, per-phase
 * throughput, $/M token, interactivity, request rate). Each active slider
 * hides every sharding config that misses — or cannot prove — its target.
 */
import { useMemo } from 'react';
import { COST_KEYS, fmtMetricValue, metricByKey } from '../metrics';
import { CostBasis } from '../pricing';
import { UiGroup, UiResult, hasError } from '../results';

export interface SlaFilterDef {
  /** metric supplying the value, label, and formatting */
  metricKey: string;
  /** 'max': keep values ≤ threshold (latency, cost); 'min': keep ≥ (rates) */
  dir: 'min' | 'max';
  tip: string;
}

export const SLA_FILTERS: SlaFilterDef[] = [
  {
    metricKey: 'ttft',
    dir: 'max',
    tip: 'Hide configs whose single-sequence time to first token exceeds this.',
  },
  {
    metricKey: 'prefillTokPerChip',
    dir: 'min',
    tip: 'Hide configs below this aggregate prefill throughput per chip.',
  },
  {
    metricKey: 'decodeTokPerChip',
    dir: 'min',
    tip: 'Hide configs below this aggregate decode throughput per chip.',
  },
  {
    metricKey: 'tokPerUser',
    dir: 'min',
    tip: 'Hide configs below this per-user decode interactivity (1/TPOT).',
  },
  {
    metricKey: COST_KEYS.prefill,
    dir: 'min',
    tip: 'Hide configs below this prefill cost-efficiency vs the Hopper MVP baseline. Chips without a price/power figure are hidden while this is set.',
  },
  {
    metricKey: COST_KEYS.decode,
    dir: 'min',
    tip: 'Hide configs below this decode cost-efficiency vs the Hopper MVP baseline. Chips without a price/power figure are hidden while this is set.',
  },
  {
    metricKey: 'kvSpacePerChip',
    dir: 'min',
    tip: 'Hide configs leaving less than this much HBM per chip for KV cache once the weights are resident.',
  },
  {
    metricKey: 'rps',
    dir: 'min',
    tip: 'Hide configs below this machine-level steady-state request rate for the current workload.',
  },
];

/** Active thresholds by metric key; a missing key means unconstrained. */
export type SlaThresholds = Record<string, number>;

/** A config passes when every active target is met; a config missing the
 * metric (no price, phase not evaluated) cannot prove the SLA and fails. */
export function passesSla(r: UiResult, thresholds: SlaThresholds): boolean {
  for (const f of SLA_FILTERS) {
    const limit = thresholds[f.metricKey];
    if (limit === undefined) continue;
    const v = metricByKey(f.metricKey).value(r);
    if (v === null || !Number.isFinite(v)) return false;
    if (f.dir === 'max' ? v > limit : v < limit) return false;
  }
  return true;
}

const STEPS = 400;

interface PanelProps {
  /** unfiltered groups: slider ranges must not shrink as filters bite */
  groups: UiGroup[];
  /** active cost basis, for the eff filters' $ ↔ kW labels */
  basis: CostBasis;
  thresholds: SlaThresholds;
  onChange: (metricKey: string, value: number | undefined) => void;
  onReset: () => void;
}

export function SlaFilterPanel({ groups, basis, thresholds, onChange, onReset }: PanelProps) {
  // data-driven slider range per filter: the spread of that metric across
  // every feasible config currently priced/streamed
  const bounds = useMemo(() => {
    const rows = groups.flatMap((g) => g.configs).filter((r) => !hasError(r));
    const map: Record<string, [number, number]> = {};
    for (const f of SLA_FILTERS) {
      const m = metricByKey(f.metricKey);
      let lo = Infinity;
      let hi = 0;
      for (const r of rows) {
        const v = m.value(r);
        if (v === null || !Number.isFinite(v) || v <= 0) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (hi > 0) map[f.metricKey] = [lo, hi];
    }
    return map;
  }, [groups]);

  const nActive = Object.keys(thresholds).length;

  return (
    <div className="sla-panel">
      <div className="sla-head">
        <span className="sla-title">
          SLA targets
          <span
            className="info-i info-i--below"
            data-tip="Each slider hides sharding configs that miss its target. Drag a slider off its loose end to release it. Ranges span the values seen across the current sweep."
            aria-label="explanation"
          >
            ⓘ
          </span>
        </span>
        <button className="sla-reset" onClick={onReset} disabled={nActive === 0}>
          Reset all
        </button>
      </div>
      <div className="sla-grid">
        {SLA_FILTERS.map((f) => (
          <SlaSlider
            key={f.metricKey}
            def={f}
            basis={basis}
            bounds={bounds[f.metricKey]}
            value={thresholds[f.metricKey]}
            onChange={(v) => onChange(f.metricKey, v)}
          />
        ))}
      </div>
    </div>
  );
}

function SlaSlider({
  def,
  basis,
  bounds,
  value,
  onChange,
}: {
  def: SlaFilterDef;
  basis: CostBasis;
  bounds?: [number, number];
  value?: number;
  onChange: (v: number | undefined) => void;
}) {
  const m = metricByKey(def.metricKey, basis);
  const [lo, hi] = bounds ?? [1, 1];
  const llo = Math.log(lo);
  const lhi = Math.log(hi);
  const span = lhi - llo;
  const toFrac = (v: number) =>
    span > 0 ? (Math.log(Math.min(hi, Math.max(lo, v))) - llo) / span : 0.5;
  // unset rests at the loose end: right for ≤ targets, left for ≥ targets
  const frac = value !== undefined ? toFrac(value) : def.dir === 'max' ? 1 : 0;
  const active = value !== undefined;

  return (
    <div className={`sla-row ${active ? 'sla-on' : ''}`} title={def.tip}>
      <span className="sla-label">
        {m.label} {def.dir === 'max' ? '≤' : '≥'}
      </span>
      <input
        type="range"
        min={0}
        max={STEPS}
        step={1}
        disabled={!bounds}
        value={Math.round(frac * STEPS)}
        onChange={(e) => {
          const t = e.target.valueAsNumber / STEPS;
          // the loose extreme releases the constraint entirely
          if (def.dir === 'max' ? t >= 0.9975 : t <= 0.0025) onChange(undefined);
          else onChange(Math.exp(llo + t * span));
        }}
        aria-label={`${m.label} SLA target`}
      />
      <span className="sla-val mono">{active ? fmtMetricValue(m, value) : 'off'}</span>
      <button
        className="sla-clear"
        aria-label={`clear ${m.label} target`}
        disabled={!active}
        onClick={() => onChange(undefined)}
      >
        ×
      </button>
    </div>
  );
}
