/**
 * Details slide-over for one configuration: memory breakdown, roofline
 * component bars, MFU, diagnostics — shown against the group's hardware
 * rooflines.
 */
import type { ModelSpec } from '../../core/model/models';
import type { ShardingRole } from '../../core/engine/sim/ir/sharding/roles';
import { naiveOverlapBreakdown } from '../../core/engine/sim/cost/helpers/naiveOverlap';
import { fmtBytes, fmtInt, fmtPct, fmtSI, fmtTime } from '../format';
import { COST_KEYS, metricByKey } from '../metrics';
import {
  Boundedness,
  ComponentTimes,
  DIAG_CLASS,
  DIAG_GLYPH,
  UiGroup,
  UiOverlap,
  UiResult,
  gaugeColor,
} from '../results';
import { ShardingPill } from './Leaderboard';
import { OpDagView } from './OpDagView';
import { FabricView } from './FabricView';

interface Props {
  result: UiResult;
  group: UiGroup;
  model: ModelSpec;
  overlap: UiOverlap;
  onClose: () => void;
}

export function DetailsPanel({ result: r, group, model, overlap, onClose }: Props) {
  const chip = group.chip;
  const mem = r.memory;
  const dec = r.decode;
  const pf = r.prefill;
  const hw = group.hardware;
  const pp = r.sizes.PP ?? 1;

  return (
    <aside className="details">
      <div className="details-head">
        <h3>
          {chip.name}
          {group.machineLabel && <span className="pill">{group.machineLabel}</span>}
          <ShardingPill r={r} />
        </h3>
        <button className="details-close" onClick={onClose} aria-label="Close details">
          ✕
        </button>
      </div>

      <FabricView mesh={r.mesh} />

      {mem && (
        <>
          <dl className="kv">
            <dt>KV per sequence (per chip)</dt>
            <dd>{fmtBytes(mem.kvBytesPerSeqPerChip)}</dd>
            <dt>Max resident sequences (total)</dt>
            <dd>{fmtInt(mem.maxResidentSeqs)}</dd>
          </dl>
          <HbmBar r={r} cap={chip.hbmCapacity} />
        </>
      )}

      {pf && (
        <>
          <div className="details-sect">Prefill</div>
          <dl className="kv">
            <CostRow role="prefill" r={r} />
          </dl>
          <GaugeRow
            tip="Saturated prefill rate per chip at the Memory/Comms overlap you set. The gauge's full width is the compute-roofline ceiling — the best any sharding of this model could prefill on this hardware at its Realizable FLOPs setting, so the gap is sharding losses, not kernel efficiency."
            frac={pf.fracOfCeiling}
            text={fmtSI(pf.tokPerSecPerChip)}
          />
          <dl className="kv">
            <dt>Prefill MFU</dt>
            <dd>{fmtPct(pf.mfu)}</dd>
            <dt>TTFT</dt>
            <dd>{fmtTime(pf.ttft)}</dd>
            {hw && (
              <>
                <dt>Critical tokens / batch</dt>
                <dd>{fmtInt(hw.critTokens)}</dd>
              </>
            )}
            <dt>{pp > 1 ? 'Batched prompts / stage' : 'Batched prompts'}</dt>
            <dd>{fmtInt(pf.batchSeqs)}</dd>
          </dl>
          <div className="time-card">
            <ComponentBars c={pf.components} bound={pf.boundBy} />
            <StepTimeBar c={pf.components} a={overlap} per={pp > 1 ? 'stage' : 'batch'} />
          </div>
        </>
      )}

      {dec && (
        <>
          <div className="details-sect">Decode</div>
          <dl className="kv">
            <CostRow role="decode" r={r} />
          </dl>
          <GaugeRow
            tip="Decode rate per chip at the operating batch and the Memory/Comms overlap you set. The gauge's full width is the B = ∞ ideal-sharding ceiling — the best any sharding of this model could decode on this hardware at its Realizable FLOPs / HBM BW settings. The tick marks this sharding's own B = ∞ saturation rate: the bar can grow to the tick with more KV room, but never past it."
            frac={dec.fracOfCeiling}
            text={fmtSI(dec.tokPerSecPerChip)}
            mark={
              dec.batchSaturation !== undefined
                ? dec.fracOfCeiling / dec.batchSaturation
                : undefined
            }
          />
          {dec.batchSaturation !== undefined && (
            <GaugeRow
              label="Batch saturation"
              tip="The operating throughput as a share of this exact sharding's B = ∞ rate (KV-capacity gate off). Low means the config is batch-starved — HBM room for KV, not the sharding, is what caps throughput."
              frac={dec.batchSaturation}
              text={fmtPct(dec.batchSaturation, 0)}
            />
          )}
          <dl className="kv">
            <dt>Tok/s/user (1/TPOT)</dt>
            <dd>{fmtSI(dec.tokPerSecPerUser)}</dd>
            <dt>Decode MFU</dt>
            <dd>{fmtPct(dec.mfu)}</dd>
            <dt>{pp > 1 ? 'Batch / microbatch' : 'Batch'}</dt>
            <dd>{fmtInt(dec.batchPerStage)}</dd>
            <dt>Resident sequences</dt>
            <dd>{fmtInt(dec.residentSeqs)}</dd>
          </dl>
          <div className="time-card">
            <ComponentBars c={dec.components} bound={dec.boundBy} />
            <StepTimeBar c={dec.components} a={overlap} per="step" />
          </div>
        </>
      )}

      <WorkloadCost r={r} />

      {(dec || pf) && (
        <>
          <div className="details-sect">Execution trace</div>
          <OpDagView result={r} model={model} chip={chip} overlap={overlap} />
        </>
      )}

      <div className="details-sect">Details</div>
      <dl className="kv kv-labelfit">
        <dt>Axes</dt>
        <dd>
          <AxesValue r={r} />
        </dd>
        {/* with EP = 1 there is nothing to route, so the dispatch is noise */}
        {r.sizes.EP != null && (
          <>
            <dt>MoE dispatch</dt>
            <dd>{r.dispatch}</dd>
          </>
        )}
      </dl>
      {r.diagnostics.map((d, i) => (
        <div key={i} className="diag-line">
          <span className={`diag-glyph ${DIAG_CLASS[d.severity]}`}>{DIAG_GLYPH[d.severity]}</span>
          <span>{d.message}</span>
        </div>
      ))}
    </aside>
  );
}

/** One phase's ×HMVP cost line; empty when it can't be computed (no price). */
function CostRow({ role, r }: { role: 'prefill' | 'decode'; r: UiResult }) {
  const m = metricByKey(COST_KEYS[role]);
  if (m.value(r) === null) return null;
  return (
    <>
      <dt>{m.label}</dt>
      <dd>{m.format(r)}</dd>
    </>
  );
}

/** The whole-request ×HMVP section. */
function WorkloadCost({ r }: { r: UiResult }) {
  const m = metricByKey(COST_KEYS.request);
  if (m.value(r) === null) return null;
  const { prefillLen, generateLen } = r.workload;
  return (
    <>
      <div className="details-sect">Workload efficiency</div>
      <dl className="kv">
        <dt>
          {m.label} ({prefillLen} in + {generateLen} out)
        </dt>
        <dd>{m.format(r)}</dd>
      </dl>
    </>
  );
}

// role -> physical axes; axes in the accent/sans style of the op trace
function AxesValue({ r }: { r: UiResult }) {
  const roles = Object.keys(r.sizes) as ShardingRole[];
  return (
    <>
      {roles.map((role, i) => (
        <span key={role}>
          {i > 0 ? ' ' : ''}
          {role}[
          {(r.mesh.roles[role] ?? []).map((a, j) => (
            <span key={a}>
              {j > 0 ? ',' : ''}
              <span className="op-flow-axis">{a}</span>
            </span>
          ))}
          ]
        </span>
      ))}
    </>
  );
}

/**
 * One chip's HBM as a stacked bar: weights, active KV (the operating batch),
 * idle KV (room for more residents).
 */
function HbmBar({ r, cap }: { r: UiResult; cap: number }) {
  const mem = r.memory!;
  const dpa = r.sizes.DPA ?? 1;
  const residentPerChip = (r.decode?.residentSeqs ?? mem.maxResidentSeqs) / dpa;
  const maxPerChip = mem.maxResidentSeqs / dpa;
  const w = mem.weightBytesPerChip;
  const kv = residentPerChip * mem.kvBytesPerSeqPerChip;
  const idleSeqs = Math.max(0, maxPerChip - residentPerChip);
  const idleKv = idleSeqs * mem.kvBytesPerSeqPerChip;
  const pct = (v: number) => `${(v / cap) * 100}%`;
  return (
    <div
      title={`${fmtBytes(cap)} HBM = ${fmtBytes(w)} weights + ${fmtBytes(kv)} active KV + ${fmtBytes(idleKv)} idle KV`}
    >
      <div className="hbm-plot">
        <div className="hbm-bar">
          <i className="hbm-w" style={{ width: pct(w) }} />
          <i className="hbm-kv" style={{ width: pct(kv) }} />
          {idleSeqs > 0 && <i className="hbm-headroom" style={{ width: pct(idleKv) }} />}
        </div>
      </div>
      <div className="rrg-legend bar-legend">
        <span>
          <span className="sw hbm-w" />
          Weights/chip <b>{fmtBytes(w)}</b>
        </span>
        <span>
          <span className="sw hbm-kv" />
          Active KV/chip <b>{fmtBytes(kv)}</b>
          <span
            className="info-i info-i--end"
            data-tip={`KV cache the operating decode batch keeps resident on each chip: ${fmtInt(residentPerChip)} sequences x ${fmtBytes(mem.kvBytesPerSeqPerChip)} per sequence.`}
            aria-label="explanation"
          >
            ⓘ
          </span>
        </span>
        <span className="hbm-free">
          {idleSeqs > 0 && <span className="sw hbm-headroom" />}
          Free/chip <b>{fmtBytes(idleKv)}</b>
        </span>
      </div>
    </div>
  );
}

const COMPONENT_LABEL = { compute: 'Compute', memory: 'Memory', comms: 'Comms' } as const;

/**
 * Phase time as a stacked bar: each solid segment is the visible time a
 * component contributes under the tuned overlap, the pool winner first.
 * Work hidden behind the pool is hatched inside the pool's span —
 * concurrent, adding no wall-clock time.
 */
function StepTimeBar({ c, a, per }: { c: ComponentTimes; a: UiOverlap; per: string }) {
  const { parts, pool, hidden } = naiveOverlapBreakdown(c, a);
  const total = parts.compute + parts.memory + parts.comms;
  if (total <= 0) return null;
  const order = [
    pool.by,
    ...(['compute', 'memory', 'comms'] as const).filter((k) => k !== pool.by),
  ];
  const hiddenKeys = order.filter((k) => hidden[k] > 0);
  const hiddenDesc = hiddenKeys
    .map((k) => `${COMPONENT_LABEL[k]} ${fmtTime(hidden[k])}`)
    .join(', ');
  const firstSolid = order.slice(1).find((k) => parts[k] > 0);
  const abutKey = firstSolid !== undefined && hidden[firstSolid] > 0 ? firstSolid : null;
  const hiddenSum = hiddenKeys.reduce((s, k) => s + hidden[k], 0);
  const fit = hiddenSum > pool.time ? pool.time / hiddenSum : 1;
  const poolColorEnd = parts[pool.by] / total;
  const hiddenSegs: { k: keyof ComponentTimes; left: number; width: number }[] = [];
  let x = 0;
  for (const k of hiddenKeys.filter((k) => k !== abutKey)) {
    const width = (hidden[k] * fit) / total;
    hiddenSegs.push({ k, left: x, width });
    x += width;
  }
  if (abutKey) {
    const width = (hidden[abutKey] * fit) / total;
    hiddenSegs.push({ k: abutKey, left: poolColorEnd - width, width });
  }
  return (
    <div
      title={
        `Wall-clock time for one ${per}: the longest component sets the pace; hatched work runs in parallel and doesn't add latency.` +
        (hiddenDesc ? ` Hatched: ${hiddenDesc}.` : '')
      }
    >
      <div className="hbm-bar">
        {order.map(
          (k) =>
            parts[k] > 0 && (
              <i key={k} className={`stp-${k}`} style={{ width: `${(parts[k] / total) * 100}%` }} />
            ),
        )}
        {hiddenSegs.map(({ k, left, width }) => (
          <span
            key={k}
            className={`stp-hid stp-hid-${k}`}
            style={{ left: `${left * 100}%`, width: `${width * 100}%` }}
          />
        ))}
      </div>
      <div className="rrg-legend bar-legend">
        {order.map(
          (k) =>
            parts[k] > 0 && (
              <span key={k}>
                <span className={`sw stp-${k}`} />
                {COMPONENT_LABEL[k]} <b>{fmtTime(parts[k])}</b>
              </span>
            ),
        )}
        <span className="hbm-free">
          <b>{fmtTime(total)}</b>/{per}
        </span>
      </div>
    </div>
  );
}

/** Attainment gauge on the shared gauge-row grid: value vs its ceiling. */
function GaugeRow({
  tip,
  frac,
  text,
  label = 'Tok/s/chip',
  mark,
}: {
  tip: string;
  frac: number;
  text: string;
  label?: string;
  // tick position on the same 0..1 scale as frac
  mark?: number;
}) {
  return (
    <div className="gauge-row">
      <span>
        {label}
        <span className="info-i" data-tip={tip} aria-label="explanation">
          ⓘ
        </span>
      </span>
      <span className="bar">
        <i
          style={{
            width: `${Math.max(2, Math.min(1, frac) * 100)}%`,
            background: gaugeColor(frac),
          }}
        />
        {mark !== undefined && (
          <span className="tick" style={{ left: `${Math.min(1, mark) * 100}%` }} />
        )}
      </span>
      <span>{text}</span>
    </div>
  );
}

/** The three roofline components as bars, scaled to the largest, the
 * binding one marked by a colored dot + emphasized label. */
function ComponentBars({ c, bound }: { c: ComponentTimes; bound: Boundedness }) {
  const max = Math.max(c.compute, c.memory, c.comms, 1e-12);
  const rows: [string, Boundedness, number][] = [
    ['Memory', 'memory', c.memory],
    ['Compute', 'compute', c.compute],
    ['Comms', 'comms', c.comms],
  ];
  return (
    <div className="comps">
      {rows.map(([label, b, v]) => (
        <div key={label} className={b === bound ? 'comp comp-bind' : 'comp'}>
          <span className="muted">
            {label}
            {b === bound && <span className={`bind-dot bind-dot-${b}`} />}
          </span>
          <span className="bar">
            <i className={`stp-${b}`} style={{ width: `${Math.max(1, (v / max) * 100)}%` }} />
          </span>
          <span className="mono comp-v">{fmtTime(v)}</span>
        </div>
      ))}
    </div>
  );
}
