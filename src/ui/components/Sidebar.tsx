/** Left-hand configuration panel: model, workload, hardware, overlap. */
import { ChangeEvent, useState } from 'react';
import { CHIPS_BY_ID } from '../../core/hardware/chips';
import { kvBytesPerSeq } from '../../core/model/utils';
import { MODEL_PRESETS, ModelSpec } from '../../core/model/models';
import { DTYPE_BYTES } from '../../core/model/dtype';
import { fmtBytes } from '../format';
import { machineAtNodes, machineLabel, machineOf, maxNodesOf, slicesOf } from '../machines';
import { CostBasis, H100_ID, relPriceOf, relPriceToDollars } from '../pricing';
import { UiChip, UiWorkload } from '../results';
import { VendorLogo } from './VendorLogo';

export interface SweepControls {
  chipIds: string[];
  /** which machine each chip deploys on: rank every one it offers, honor its
   * own settings pick, or force the machine nearest a global node count */
  machine: 'sweep' | 'per-chip' | number;
  /** fraction of HBM traffic hidden behind compute */
  memoryOverlap: number;
  /** fraction of collective traffic hidden behind compute */
  commsOverlap: number;
}

interface Props {
  model: ModelSpec;
  onModel: (m: ModelSpec) => void;
  workload: UiWorkload;
  onWorkload: (w: UiWorkload) => void;
  sweep: SweepControls;
  onSweep: (s: SweepControls) => void;
  /** the app's editable copy of the chip database */
  chips: UiChip[];
  onChips: (chips: UiChip[]) => void;
  /** active cost basis: the chip list's cost column shows price or TDP */
  basis: CostBasis;
}

export function Sidebar(p: Props) {
  const { model, workload, sweep, chips, basis } = p;
  const vendors = [...new Set(chips.map((c) => c.vendor))];
  const patchChip = (id: string, patch: Partial<UiChip>) =>
    p.onChips(chips.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  const [openSettings, setOpenSettings] = useState<Set<string>>(new Set());
  const toggleSettings = (id: string) =>
    setOpenSettings((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const onPreset = (e: ChangeEvent<HTMLSelectElement>) => {
    const preset = MODEL_PRESETS.find((m) => m.name === e.target.value);
    if (preset) p.onModel(preset);
  };

  const toggle = <T,>(list: T[], v: T): T[] =>
    list.includes(v) ? list.filter((x) => x !== v) : [...list, v];

  // double-click a chip: solo it; double-click the soloed chip: everything
  // but it. (The double-click's own pair of change events toggles the box
  // twice — a net no-op — before this fires.)
  const soloChip = (id: string) =>
    p.onSweep({
      ...sweep,
      chipIds:
        sweep.chipIds.length === 1 && sweep.chipIds[0] === id
          ? chips.map((c) => c.id).filter((x) => x !== id)
          : [id],
    });

  // one control for the decode batching policy: an SLO value present (NaN
  // while the field is cleared mid-edit) means SLO mode, else the policy
  const batchMode: 'max' | 'b1' | 'slo' =
    workload.sloTokPerSecPerUser !== undefined ? 'slo' : (workload.batching ?? 'max');
  const onBatchMode = (e: ChangeEvent<HTMLSelectElement>) => {
    const v = e.target.value as 'max' | 'b1' | 'slo';
    if (v === 'slo') p.onWorkload({ ...workload, batching: 'max', sloTokPerSecPerUser: 0 });
    else p.onWorkload({ ...workload, batching: v, sloTokPerSecPerUser: undefined });
  };

  return (
    <aside className="sidebar">
      <section className="side-section">
        <h3 className="cs-spec-head">
          Model
          <SpecLink file="src/core/model/models.ts" />
        </h3>
        <label className="field">
          <select value={model.name} onChange={onPreset}>
            {MODEL_PRESETS.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="side-section">
        <h3>Workload</h3>
        <div className="field-grid">
          <NumField
            label="Prefill tokens"
            value={workload.prefillLen}
            onChange={(e) => p.onWorkload({ ...workload, prefillLen: e.target.valueAsNumber })}
          />
          <NumField
            label="Decode tokens"
            value={workload.generateLen}
            onChange={(e) => p.onWorkload({ ...workload, generateLen: e.target.valueAsNumber })}
          />
          <label className="field">
            <span>
              Decode batching
              <span
                className="info-i"
                data-tip="How deep decode batches. Maximum: fill KV capacity — best aggregate throughput, slowest per user. SLO-based: largest batch that still meets Minimum tok/s/user. Minimum (B=1): fastest per user, worst throughput."
                aria-label="explanation"
              >
                ⓘ
              </span>
            </span>
            <select className="sel-compact" value={batchMode} onChange={onBatchMode}>
              <option value="max">Maximum</option>
              <option value="slo">SLO-based</option>
              <option value="b1">Minimum (B=1)</option>
            </select>
          </label>
          <NumField
            label="Minimum tok/s/user"
            value={workload.sloTokPerSecPerUser || NaN}
            placeholder="Unconstrained"
            disabled={batchMode !== 'slo'}
            onChange={(e) =>
              p.onWorkload({
                ...workload,
                // 0 = unconstrained; stays in SLO mode while cleared mid-edit
                sloTokPerSecPerUser:
                  Number.isFinite(e.target.valueAsNumber) && e.target.valueAsNumber > 0
                    ? e.target.valueAsNumber
                    : 0,
              })
            }
          />
        </div>
        <div className="hint hint-spread">
          KV cache per sequence:{' '}
          <b>
            {fmtBytes(
              kvBytesPerSeq(
                model,
                DTYPE_BYTES[model.precision.kv],
                workload.prefillLen + workload.generateLen,
              ),
            )}
          </b>
        </div>
      </section>

      <section className="side-section">
        <h3>Hardware</h3>
        <label className="cs-row nodes-row">
          <span>
            Nodes
            <span
              className="info-i"
              data-tip="How many hosts each chip deploys across. N nodes: N scale-out nodes on switched fabrics (GPUs), the slice nearest N hosts' worth of chips on ring fabrics (TPUs, Trainium); fixed-size machines (NVL72) keep their one size. Per-chip: each chip's own settings pick (default 1 node). Sweep all sizes: rank every slice / node count a chip offers as its own row."
              aria-label="explanation"
            >
              ⓘ
            </span>
          </span>
          <select
            className="sel-compact"
            value={sweep.machine}
            onChange={(e) =>
              p.onSweep({
                ...sweep,
                machine:
                  e.target.value === 'sweep' || e.target.value === 'per-chip'
                    ? e.target.value
                    : Number(e.target.value),
              })
            }
          >
            {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
              <option key={n} value={n}>
                {n} node{n > 1 ? 's' : ''}
              </option>
            ))}
            <option value="per-chip">Per-chip</option>
            <option value="sweep">Sweep all sizes</option>
          </select>
        </label>
        {vendors.map((vendor, vi) => (
          <div key={vendor} className={`vendor-group ${vi > 0 ? 'vendor-group-later' : ''}`}>
            <div className="vendor-label">
              <VendorLogo vendor={vendor} size={12} />
              {vendor}
              {vi === 0 && (
                <span className="price-col-label">
                  {basis === 'power' ? 'TDP' : 'Price vs H100'}
                  <span
                    className="info-i info-i--end"
                    data-tip={
                      basis === 'power'
                        ? "Thermal design power per chip. The per-kW efficiency columns divide by it, relative to the H100's 700 W."
                        : 'Hourly rental price per chip, as a multiple of one H100 (H100 = 1).'
                    }
                    aria-label="explanation"
                  >
                    ⓘ
                  </span>
                </span>
              )}
            </div>
            {chips
              .filter((c) => c.vendor === vendor)
              // sort by the chips.ts default price (not the live-edited one,
              // so rows don't reorder mid-typing in the price field)
              .sort(
                (a, b) =>
                  (CHIPS_BY_ID[a.id]?.costPerHour ?? Infinity) -
                  (CHIPS_BY_ID[b.id]?.costPerHour ?? Infinity),
              )
              .map((c) => (
                <div key={c.id}>
                  <div className="chip-row">
                    <label className="check">
                      <input
                        type="checkbox"
                        checked={sweep.chipIds.includes(c.id)}
                        onChange={() =>
                          p.onSweep({ ...sweep, chipIds: toggle(sweep.chipIds, c.id) })
                        }
                        onDoubleClick={() => soloChip(c.id)}
                        title="double-click: only this chip / all but this chip"
                      />
                      <span>{c.name}</span>
                    </label>
                    {basis === 'power' ? (
                      <span
                        className="cost-input"
                        title={
                          c.tdp !== undefined
                            ? `${c.name} thermal design power`
                            : `${c.vendor} publishes no TDP for the ${c.name}, so it shows no per-kW efficiency`
                        }
                      >
                        {c.tdp !== undefined ? (
                          <span className="mono">{c.tdp} W</span>
                        ) : (
                          <span className="muted">unknown</span>
                        )}
                      </span>
                    ) : (
                      <span
                        className="cost-input"
                        title={
                          c.id === H100_ID
                            ? 'The reference chip: every price is quoted relative to one H100 chip-hour'
                            : `Rental price per ${c.name} chip-hour, relative to one H100 chip-hour`
                        }
                      >
                        <input
                          type="number"
                          step={0.05}
                          min={0}
                          disabled={c.id === H100_ID}
                          value={relDisplay(c)}
                          onChange={(e) =>
                            patchChip(c.id, {
                              costPerHour: Number.isFinite(e.target.valueAsNumber)
                                ? relPriceToDollars(e.target.valueAsNumber)
                                : undefined,
                            })
                          }
                        />
                        × H100
                      </span>
                    )}
                    <button
                      className={`gear ${openSettings.has(c.id) ? 'on' : ''}`}
                      aria-label={`${c.name} settings`}
                      onClick={() => toggleSettings(c.id)}
                    >
                      <svg
                        viewBox="0 0 24 24"
                        width="16"
                        height="16"
                        fill="currentColor"
                        aria-hidden="true"
                      >
                        <path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 0 0 .12-.61l-1.92-3.32a.488.488 0 0 0-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 0 0-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96a.488.488 0 0 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58a.49.49 0 0 0-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z" />
                      </svg>
                    </button>
                  </div>
                  {openSettings.has(c.id) && (
                    <div className="chip-settings">
                      <div className="cs-row">
                        <span title="Fraction of the datasheet matmul rate a well-tuned, well-shaped GEMM sustains on this chip (sustained clocks, kernel quality). Derates compute pricing only — HBM, collectives, and tile-padding losses are each priced separately.">
                          Realizable FLOPs
                        </span>
                        <input
                          type="range"
                          min={0.05}
                          max={1}
                          step={0.05}
                          value={Number.isFinite(c.realizableFlopsFrac) ? c.realizableFlopsFrac : 1}
                          onChange={(e) =>
                            patchChip(c.id, { realizableFlopsFrac: e.target.valueAsNumber })
                          }
                          aria-label={`${c.name} realizable FLOPs`}
                        />
                        <span className="mono mfu-val">
                          {Math.round(
                            (Number.isFinite(c.realizableFlopsFrac) ? c.realizableFlopsFrac : 1) *
                              100,
                          )}
                          %
                        </span>
                      </div>
                      <div className="cs-row">
                        <span title="Fraction of peak HBM bandwidth a well-tuned streaming kernel sustains on this chip. Derates every memory price: weight loads, KV reads, activation streams.">
                          Realizable HBM BW
                        </span>
                        <input
                          type="range"
                          min={0.05}
                          max={1}
                          step={0.05}
                          value={Number.isFinite(c.realizableHbmBwFrac) ? c.realizableHbmBwFrac : 1}
                          onChange={(e) =>
                            patchChip(c.id, { realizableHbmBwFrac: e.target.valueAsNumber })
                          }
                          aria-label={`${c.name} realizable HBM bandwidth`}
                        />
                        <span className="mono mfu-val">
                          {Math.round(
                            (Number.isFinite(c.realizableHbmBwFrac) ? c.realizableHbmBwFrac : 1) *
                              100,
                          )}
                          %
                        </span>
                      </div>
                      <div className="cs-label cs-spec-head" title="Datasheet spec, read-only">
                        Chip spec
                        <SpecLink file="src/core/hardware/chips.ts" />
                      </div>
                      <pre className="cs-json">{specJson(c)}</pre>
                      <MachinePicker
                        chip={c}
                        swept={sweep.machine === 'sweep'}
                        globalNodes={typeof sweep.machine === 'number' ? sweep.machine : undefined}
                        onPatch={(patch) => patchChip(c.id, patch)}
                      />
                    </div>
                  )}
                </div>
              ))}
          </div>
        ))}
      </section>

      <section className="side-section">
        <h3>Overlap</h3>
        <OverlapRow
          label="Memory"
          tip="Fraction of HBM traffic your kernels hide behind compute (and comms); the rest is paid serially. At 100%, HBM waits fully behind compute; ~90% is typical for well-fused kernels."
          value={sweep.memoryOverlap}
          onChange={(v) => p.onSweep({ ...sweep, memoryOverlap: v })}
        />
        <OverlapRow
          label="Comms"
          tip="Fraction of interconnect traffic hidden behind compute. Needs async collectives and careful scheduling. At 100%, interconnect waits fully behind compute; 50% is a realistic default."
          value={sweep.commsOverlap}
          onChange={(v) => p.onSweep({ ...sweep, commsOverlap: v })}
        />
      </section>
    </aside>
  );
}

/** A chip's relative price for the ×H100 input, shown to two decimals.
 * Display-only rounding: the stored price moves only when the user types. */
function relDisplay(c: UiChip): number | '' {
  const rel = relPriceOf(c);
  return rel !== undefined && Number.isFinite(rel) ? Number(rel.toFixed(2)) : '';
}

/** External-link icon to a definitions file on GitHub. */
function SpecLink(props: { file: string }) {
  const name = props.file.split('/').pop();
  return (
    <a
      className="cs-spec-link"
      href={`https://github.com/sailresearchco/htdym/blob/main/${props.file}`}
      target="_blank"
      rel="noreferrer"
      title={`${name} on GitHub`}
      aria-label={`${name} on GitHub`}
    >
      <svg
        viewBox="0 0 24 24"
        width="11"
        height="11"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
        <path d="M15 3h6v6" />
        <path d="M10 14L21 3" />
      </svg>
    </a>
  );
}

/** Overlap-fraction slider row with an instant explanation tooltip. */
function OverlapRow(props: {
  label: string;
  tip: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="cs-row ov-row">
      <span className="ov-label">
        {props.label}
        <span className="info-i" data-tip={props.tip} aria-label="explanation">
          ⓘ
        </span>
      </span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={props.value}
        onChange={(e) => props.onChange(e.target.valueAsNumber)}
        aria-label={`${props.label} overlap fraction`}
      />
      <span className="mono mfu-val">{Math.round(props.value * 100)}%</span>
    </div>
  );
}

// Machine selector for the settings card: a ring-fabric slice, or a
// scale-out node count on switched fabrics that have one. Inert while the
// global Nodes control is sweeping every size or forcing a node count.
function MachinePicker({
  chip,
  swept,
  globalNodes,
  onPatch,
}: {
  chip: UiChip;
  swept: boolean;
  /** the sidebar-wide node count currently forced on every chip */
  globalNodes?: number;
  onPatch: (patch: Partial<UiChip>) => void;
}) {
  const title =
    globalNodes !== undefined
      ? 'Set by the global Nodes control above'
      : swept
        ? 'Sweeping every machine size'
        : undefined;
  const slices = slicesOf(chip);
  if (slices)
    return (
      <label className="cs-row cs-machine" title={title}>
        <span title="Ring-fabric slice this chip is deployed on">Machine</span>
        <select
          className="sel-compact"
          disabled={swept || globalNodes !== undefined}
          value={machineLabel(globalNodes !== undefined ? machineAtNodes(chip, globalNodes) : chip)}
          onChange={(e) => onPatch({ slice: e.target.value })}
        >
          {slices.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name} · {s.count} chip{s.count > 1 ? 's' : ''}
            </option>
          ))}
        </select>
      </label>
    );

  const maxNodes = maxNodesOf(chip);
  const m = machineOf(chip);
  const nodes =
    globalNodes !== undefined ? Math.min(globalNodes, maxNodes) : 'nodes' in m ? m.nodes : maxNodes;
  return (
    <label className="cs-row cs-machine" title={title}>
      <span title="Scale-out nodes of one domain joined over the switched network">Nodes</span>
      <select
        className="sel-compact"
        disabled={swept || globalNodes !== undefined}
        value={nodes}
        onChange={(e) => onPatch({ nodes: Number(e.target.value) })}
      >
        {Array.from({ length: maxNodes }, (_, i) => i + 1).map((n) => (
          <option key={n} value={n}>
            {n} node{n > 1 ? 's' : ''} · {chip.interconnect.domainSize * n} chips
          </option>
        ))}
      </select>
    </label>
  );
}

/** The immutable part of a chip spec as display JSON: everything but the
 * knobs edited above (realizable derates, machine, price) and the row's
 * own identity. Large magnitudes render in e-notation like the source. */
function specJson(c: UiChip): string {
  const spec = {
    formats: c.formats,
    hbmCapacity: c.hbmCapacity,
    hbmBandwidth: c.hbmBandwidth,
    interconnect: c.interconnect,
    ...(c.tdp !== undefined && { tdp: c.tdp }),
    ...(c.matmulSatRows !== undefined && { matmulSatRows: c.matmulSatRows }),
  };
  return JSON.stringify(spec, null, 2).replace(/-?\d[\d.]*(e[+-]?\d+)?/g, (s) => {
    const v = Number(s);
    if (v === 0 || (Math.abs(v) < 1e5 && Math.abs(v) >= 1e-3)) return s;
    return v.toExponential().replace('e+', 'e');
  });
}

function NumField(props: {
  label: string;
  value: number;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
  step?: number;
  disabled?: boolean;
  placeholder?: string;
}) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input
        type="number"
        value={Number.isFinite(props.value) ? props.value : ''}
        step={props.step ?? 1}
        min={0}
        disabled={props.disabled}
        placeholder={props.placeholder}
        onChange={props.onChange}
      />
    </label>
  );
}
