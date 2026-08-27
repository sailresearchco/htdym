/**
 * Hardware leaderboard: one row per chip (on its fixed machine), ranked by
 * the active metric and showing that chip's best sharding inline. Expanding
 * a row reveals the streamed sharding sweep; clicking a config row opens
 * the details slide-over.
 */
import { MouseEvent, useMemo, useState } from 'react';
import {
  Diagnostic,
  DIAG_CLASS,
  DIAG_GLYPH,
  UiGroup,
  UiResult,
  gaugeColor,
  hasError,
  hasWarning,
  ringTone,
} from '../results';
import { fmtBytes, fmtInt, fmtMult, fmtParams, fmtPct, fmtSI } from '../format';
import {
  COST_KEYS,
  OVERVIEW_METRIC_KEYS,
  SWEEP_METRIC_KEYS,
  MetricDef,
  fmtMetricValue,
  metricByKey,
} from '../metrics';
import { CostBasis, HmvpBaseline, hmvpEff, relCostOf } from '../pricing';
import { ProgressRing } from './ProgressRing';
import { VendorLogo } from './VendorLogo';

interface Props {
  groups: UiGroup[];
  /** the ×HMVP reference, for the group stats' efficiency-ceiling column */
  baseline: HmvpBaseline | null;
  /** what the eff columns divide by: relative price or relative power */
  basis: CostBasis;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
}

interface SortState {
  key: string;
  dir: 1 | -1;
}

interface TipState {
  x: number;
  y: number;
  above: boolean;
  diagnostics: Diagnostic[];
}

export function Leaderboard({
  groups,
  baseline,
  basis,
  selectedId,
  onSelect,
  hoveredId,
  onHover,
}: Props) {
  // default ranking: the lead cost column — Requests/$ in the dollar view,
  // ×HMVP efficiency publicly (unpriced chips sink to the bottom)
  const [sort, setSort] = useState<SortState>({ key: OVERVIEW_METRIC_KEYS[0], dir: -1 });
  // expansion overrides; default: all collapsed
  const [expanded, setExpanded] = useState<Map<string, boolean>>(new Map());
  const [tip, setTip] = useState<TipState | null>(null);

  // column defs follow the basis: same keys, cost labels swap $ ↔ kW
  const overviewCols = useMemo(
    () => OVERVIEW_METRIC_KEYS.map((k) => metricByKey(k, basis)),
    [basis],
  );
  const sweepCols = useMemo(() => SWEEP_METRIC_KEYS.map((k) => metricByKey(k, basis)), [basis]);

  const rankMetric = metricByKey(sort.key, basis);

  const ranked = useMemo(() => {
    const val = (r: UiResult) => rankMetric.value(r);
    const cmp = (a: UiResult, b: UiResult): number => {
      if (hasError(a) !== hasError(b)) return hasError(a) ? 1 : -1;
      const va = val(a);
      const vb = val(b);
      if (va === null && vb === null) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      return sort.dir * (va - vb);
    };
    const sorted = groups.map((g) => ({ ...g, configs: [...g.configs].sort(cmp) }));
    sorted.sort((a, b) => {
      if (!a.configs.length || !b.configs.length) return a.configs.length ? -1 : 1;
      return cmp(a.configs[0], b.configs[0]);
    });
    return sorted;
  }, [groups, rankMetric, sort.dir]);

  // Clicking a header ranks by it best-first; clicking again flips.
  const clickSort = (m: MetricDef) =>
    setSort((s) => {
      const bestFirst: 1 | -1 = m.lowerIsBetter ? 1 : -1;
      return s.key === m.key
        ? { key: m.key, dir: s.dir === 1 ? -1 : 1 }
        : { key: m.key, dir: bestFirst };
    });
  const arrow = (key: string) => (sort.key === key ? (sort.dir === 1 ? ' ↑' : ' ↓') : '');

  const isOpen = (key: string) => expanded.get(key) ?? false;
  const toggle = (key: string) => setExpanded((m) => new Map(m).set(key, !isOpen(key)));

  const showTip = (e: MouseEvent<HTMLElement>, diagnostics: Diagnostic[]) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const above = rect.bottom > window.innerHeight - 180;
    setTip({ x: rect.right + 8, y: above ? rect.top - 6 : rect.bottom + 6, above, diagnostics });
  };
  const hideTip = () => setTip(null);

  return (
    <div className="table-wrap">
      <table className="lb">
        <thead>
          <tr>
            <th className="lb-rank">#</th>
            <th className="lb-left" title="Shows the best config under the current sort.">
              Hardware
            </th>
            <th className="lb-left">
              Best sharding <span className="by-metric">by {rankMetric.label}</span>
            </th>
            {overviewCols.map((m) => (
              <th
                key={m.key}
                className="num sortable th-tip"
                data-tip={m.desc}
                onClick={() => clickSort(m)}
              >
                {m.label}
                {arrow(m.key)}
              </th>
            ))}
            <th className="num">Configs</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((g, gi) => (
            <GroupRows
              key={g.key}
              group={g}
              baseline={baseline}
              basis={basis}
              overviewCols={overviewCols}
              sweepCols={sweepCols}
              rank={gi + 1}
              open={isOpen(g.key)}
              onToggle={() => toggle(g.key)}
              rankKey={sort.key}
              sortDir={sort.dir}
              onSort={clickSort}
              selectedId={selectedId}
              onSelect={onSelect}
              hoveredId={hoveredId}
              onHover={onHover}
              showTip={showTip}
              hideTip={hideTip}
            />
          ))}
        </tbody>
      </table>
      {ranked.length === 0 && (
        <div className="empty">No configurations — select at least one chip.</div>
      )}
      {tip && (
        <div
          className="diag-tooltip"
          style={{
            left: tip.x,
            top: tip.y,
            transform: tip.above ? 'translateY(-100%)' : undefined,
          }}
        >
          {tip.diagnostics.length === 0 ? (
            <div className="diag-line">
              <span className="diag-glyph diag-ok">✓</span>
              <span>Feasible — no warnings flagged.</span>
            </div>
          ) : (
            tip.diagnostics.map((d, i) => (
              <div key={i} className="diag-line">
                <span className={`diag-glyph ${DIAG_CLASS[d.severity]}`}>
                  {DIAG_GLYPH[d.severity]}
                </span>
                <span>{d.message}</span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

// Which of the HMVP machine's per-metric reference configs a row is, by the
// anchor ids the baseline recorded while folding its maxima.
const ANCHOR_TIPS: Record<string, string> = {
  Req: 'This sharding is the request reference: the cheapest whole-request config on the HMVP machine, so the request-efficiency column compares every config against it.',
  Prefill:
    'This sharding is the prefill reference: the fastest prefill config on the HMVP machine, so the prefill ×HMVP columns compare every config against it.',
  Decode:
    'This sharding is the decode reference: the fastest decode config on the HMVP machine, so the decode ×HMVP columns compare every config against it.',
};

function AnchorBadges({ r, baseline }: { r: UiResult; baseline: HmvpBaseline }) {
  const anchors = [
    ...(r.id === baseline.requestAnchorId ? ['Req'] : []),
    ...(r.id === baseline.prefillAnchorId ? ['Prefill'] : []),
    ...(r.id === baseline.decodeAnchorId ? ['Decode'] : []),
  ];
  return (
    <>
      {anchors.map((a) => (
        <span key={a} className="hmvp-badge" data-tip={ANCHOR_TIPS[a]}>
          {a} HMVP
        </span>
      ))}
    </>
  );
}

export function ShardingPill({ r }: { r: UiResult }) {
  const parts = Object.entries(r.sizes);
  return (
    <span className="pill" title={r.placement}>
      {parts.length === 0 ? (
        <span className="pill-lab">single chip</span>
      ) : (
        parts.map(([role, size], i) => (
          <span key={role}>
            {i > 0 && <span className="pill-lab"> × </span>}
            <span className="pill-lab">{role}</span> {size}
          </span>
        ))
      )}
    </span>
  );
}

function GroupRows({
  group,
  baseline,
  basis,
  overviewCols,
  sweepCols,
  rank,
  open,
  onToggle,
  rankKey,
  sortDir,
  onSort,
  selectedId,
  onSelect,
  hoveredId,
  onHover,
  showTip,
  hideTip,
}: {
  group: UiGroup;
  baseline: HmvpBaseline | null;
  basis: CostBasis;
  overviewCols: MetricDef[];
  sweepCols: MetricDef[];
  rank: number;
  open: boolean;
  onToggle: () => void;
  rankKey: string;
  sortDir: 1 | -1;
  onSort: (m: MetricDef) => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  showTip: (e: MouseEvent<HTMLElement>, diagnostics: Diagnostic[]) => void;
  hideTip: () => void;
}) {
  const arrow = (key: string) => (rankKey === key ? (sortDir === 1 ? ' ↑' : ' ↓') : '');
  // on the power basis a chip with no published TDP can't fill the cost
  // columns — name the reason instead of a bare dash
  const noTdp = basis === 'power' && group.chip.tdp === undefined;
  // configs arrive best-first with errors sunk, so [0] is the group's winner
  const best = group.configs[0];
  const bestUsable = best && !hasError(best) ? best : null;
  const nErr = group.configs.filter(hasError).length;
  const nWarn = group.configs.filter((r) => !hasError(r) && hasWarning(r)).length;
  const nOk = group.configs.length - nErr - nWarn;
  const totalCols = 4 + overviewCols.length;
  const tone = ringTone(group.total ? group.done / group.total : 0);
  const canExpand = !!bestUsable;
  // this row is the ×HMVP reference machine itself: same identity check the
  // app uses to seat it (machine keys match; baseline is null in dollar mode)
  const isHmvp = baseline !== null && !baseline.exhausted && group.key === baseline.key;

  return (
    <>
      <tr
        className={`lb-lead ${canExpand ? 'expandable' : ''} ${open && canExpand ? 'open' : ''} ${isHmvp ? 'lb-hmvp' : ''}`}
        onClick={canExpand ? onToggle : undefined}
        onMouseEnter={() => bestUsable && onHover(bestUsable.id)}
        onMouseLeave={() => onHover(null)}
      >
        <td className="lb-rank num">{rank}</td>
        <td className="lb-left">
          <span className="row-line">
            <span className="chev" aria-hidden="true">
              {canExpand && (
                <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                  <path d="M16.59 8.59 12 13.17 7.41 8.59 6 10l6 6 6-6z" />
                </svg>
              )}
            </span>
            <VendorLogo vendor={group.chip.vendor} />
            <span className="chip-name">{group.chip.name}</span>
            {group.machineLabel && <span className="pill pill-dt">{group.machineLabel}</span>}
            {isHmvp && (
              <span
                className="hmvp-badge"
                data-tip={`The Hopper MVP: the smallest H100 machine (${baseline!.nChips} chips) that can serve this model and workload, and the 1× mark for every ×HMVP column. Each column compares against a different sharding of this machine — Prefill vs its fastest-prefill config, Decode vs its fastest-decode config, Request eff. vs its cheapest whole-request config (expand this row to see them badged). That's why this row never beats 1×, and why whichever eff column you sort by shows exactly 1× here.`}
                onClick={(e) => e.stopPropagation()}
              >
                HMVP
              </span>
            )}
          </span>
        </td>
        <td className="lb-left">
          {group.error ? (
            <span className="c-err">{group.error}</span>
          ) : group.done === group.total && !best ? (
            <span className="muted">No viable configurations</span>
          ) : (
            <span className="row-line">
              {best && <StatusIcon r={bestUsable ?? best} showTip={showTip} hideTip={hideTip} />}
              {bestUsable ? <ShardingPill r={bestUsable} /> : '—'}
            </span>
          )}
        </td>
        {overviewCols.map((m) => (
          <td key={m.key} className={`num mono ${m.key === rankKey ? 'lb-best' : ''}`}>
            {noTdp && COST_KEY_SET.has(m.key) ? (
              <UnknownTdp vendor={group.chip.vendor} />
            ) : bestUsable ? (
              <HeadlineCell m={m} r={bestUsable} />
            ) : (
              '—'
            )}
          </td>
        ))}
        <td className="num">
          <span className="cfg-counts">
            {!group.error && (
              <span className="ring-slot">
                <ProgressRing
                  frac={group.total ? group.done / group.total : 0}
                  size={14}
                  color={tone}
                />
              </span>
            )}
            <span className="c-ok">{nOk}✓</span>
            {nWarn > 0 && <span className="c-warn">{nWarn}⚠</span>}
            {nErr > 0 && <span className="c-err">{nErr}✕</span>}
          </span>
        </td>
      </tr>
      {open && canExpand && (
        <tr className="lb-inner">
          <td colSpan={totalCols}>
            <GroupStats
              group={group}
              baseline={baseline}
              basis={basis}
              best={bestUsable}
              selectedId={selectedId}
              onSelect={onSelect}
              hoveredId={hoveredId}
              onHover={onHover}
            />
            <div className="sweep-card">
              <table className="lb-sweep">
                <thead>
                  <tr>
                    <th className="lb-left">Sharding</th>
                    <th className="num">Chips</th>
                    {sweepCols.map((m) => (
                      <th
                        key={m.key}
                        className={`num sortable ${m.key === rankKey ? 'lb-best' : ''} ${m.info ? '' : 'th-tip'}`}
                        data-tip={m.info ? undefined : m.desc}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSort(m);
                        }}
                      >
                        {m.label}
                        {arrow(m.key)}
                        {m.info && (
                          <span
                            className="info-i info-i--below info-i--end"
                            data-tip={m.info}
                            aria-label="explanation"
                            onClick={(e) => e.stopPropagation()}
                          >
                            ⓘ
                          </span>
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {group.configs.map((r) => (
                    <tr
                      key={r.id}
                      className={[
                        hasError(r) ? 'row-error' : '',
                        r.id === selectedId ? 'row-sel' : '',
                        r.id === hoveredId ? 'row-hover' : '',
                      ].join(' ')}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelect(r.id === selectedId ? null : r.id);
                      }}
                      onMouseEnter={() => onHover(r.id)}
                      onMouseLeave={() => onHover(null)}
                    >
                      <td className="lb-left">
                        <span className="row-line">
                          <StatusIcon r={r} showTip={showTip} hideTip={hideTip} />
                          <ShardingPill r={r} />
                          {isHmvp && <AnchorBadges r={r} baseline={baseline!} />}
                        </span>
                      </td>
                      <td className="num mono">{r.nChips}</td>
                      {sweepCols.map((m) => (
                        <td
                          key={m.key}
                          className={`num mono ${m.key === rankKey ? 'lb-best' : ''}`}
                        >
                          {noTdp && COST_KEY_SET.has(m.key) ? (
                            <UnknownTdp vendor={group.chip.vendor} />
                          ) : hasError(r) ? (
                            '—'
                          ) : (
                            <HeadlineCell m={m} r={r} />
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

/** Traffic-light tone for a %-of-ceiling attainment value. */
const attainTone = (frac: number) => (frac >= 0.9 ? 'c-ok' : frac >= 0.5 ? 'c-warn' : 'c-err');

/**
 * Roofline gauge table: two phase rows × labeled numeric columns (Ceiling,
 * Best, % of ceiling, cost Floor), with the roofline chart as one bounded
 * column. Every feasible config is a point at its tuned-overlap estimate,
 * colored by the binding component, the best config ring-marked.
 */
function GroupStats({
  group,
  baseline,
  basis,
  best,
  selectedId,
  onSelect,
  hoveredId,
  onHover,
}: {
  group: UiGroup;
  baseline: HmvpBaseline | null;
  basis: CostBasis;
  best: UiResult | null;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
}) {
  const hw = group.hardware;
  if (!hw) return null;
  const feasible = group.configs.filter((r) => !hasError(r));
  const dec = best?.decode;
  const pf = best?.prefill;
  const decScale = Math.max(
    hw.decodeCeilingOverlapped,
    ...feasible.map((r) => r.decode?.tokPerSecPerChip ?? 0),
  );
  const pfScale = Math.max(
    hw.prefillCeiling,
    ...feasible.map((r) => r.prefill?.tokPerSecPerChip ?? 0),
  );
  const pct = (v: number, scale: number) => `${Math.min(100, Math.max(0, (v / scale) * 100))}%`;
  const chip = group.chip;
  // the cost column's bound: best possible ×HMVP efficiency, at ceiling
  const costBound = (ceilingTokPerSec: number, baseRate: number | undefined) => {
    const rel = relCostOf(chip, basis);
    if (rel === undefined && basis === 'power' && chip.tdp === undefined)
      return <UnknownTdp vendor={chip.vendor} />;
    return rel !== undefined && baseRate
      ? `≤ ${fmtMult(hmvpEff(ceilingTokPerSec, rel, baseRate))}`
      : '—';
  };

  const points = (phase: (r: UiResult) => { value: number; bound: string } | null, scale: number) =>
    feasible.map((r) => {
      const point = phase(r);
      if (!point) return null;
      const sizes = Object.entries(r.sizes)
        .map(([role, size]) => `${role} ${size}`)
        .join(' × ');
      const label = `${sizes || 'single chip'} · ${fmtSI(point.value)}/s · ${point.bound}-bound`;
      const isBest = best !== null && r.id === best.id;
      return (
        <span
          key={r.id}
          className={[
            'rrg-pt',
            `rrg-pt-${point.bound}`,
            isBest ? 'rrg-pt-best' : '',
            r.id === selectedId ? 'rrg-pt-sel' : '',
            hoveredId === r.id ? 'rrg-pt-hover' : '',
          ].join(' ')}
          style={{ left: pct(point.value, scale) }}
          title={label}
          onClick={() => onSelect(r.id === selectedId ? null : r.id)}
          onMouseEnter={() => onHover(r.id)}
          onMouseLeave={() => onHover(null)}
        />
      );
    });

  return (
    <div className="gstats">
      <span className="gstats-h" />
      <span className="gstats-h" />
      <span className="gstats-h">
        Ceiling
        <span
          className="info-i info-i--below"
          data-tip="The best tok/s/chip any sharding of this model could reach on this hardware, at the chip's Realizable FLOPs / HBM BW settings (not datasheet peaks) — so the gap to 100% is sharding losses, never kernel efficiency. Prefill: the compute roofline. Decode: batch → ∞, where only per-sequence compute and KV reads remain. Sharding points carry your tuned overlap and can sit BELOW the band: finite batches still pay weight loads, collectives, and KV replication that the B = ∞ ideal does not."
          aria-label="explanation"
        >
          ⓘ
        </span>
      </span>
      <span
        className="gstats-h"
        title="Best feasible config: predicted per-chip rate and its share of the ceiling"
      >
        Best
      </span>
      <span
        className="gstats-h gstats-h-c"
        title="Best config's throughput as a share of this hardware's ceiling"
      >
        % of ceiling
      </span>
      <span
        title={`Best possible cost efficiency on this chip: ceiling rate ÷ ${basis === 'power' ? 'power' : 'price'} relative to H100, as a multiple of the Hopper MVP baseline.`}
        className="gstats-h"
      >
        Eff. ceiling
      </span>
      <span className="gstats-rule" />

      <div className="gstats-cell gstats-phase">
        Prefill<small>tok/s/chip</small>
      </div>
      <div className="gstats-cell">
        <div className="rrg-track">
          {points(
            (r) =>
              r.prefill ? { value: r.prefill.tokPerSecPerChip, bound: r.prefill.boundBy } : null,
            pfScale,
          )}
        </div>
        <div className="rrg-nums">
          <span>0 tok/s</span>
          <span>{fmtSI(pfScale)}/s</span>
        </div>
      </div>
      <div className="gstats-cell gstats-num">
        <span className="v">{fmtSI(hw.prefillCeiling)}/s</span>
        <span className="s">{fmtSI(hw.critTokens)}+ tokens</span>
      </div>
      <div className="gstats-cell gstats-num">
        {pf ? (
          <>
            <span className="v">{fmtSI(pf.tokPerSecPerChip)}/s</span>
            <span className="s">{pf.boundBy}-bound</span>
          </>
        ) : (
          '—'
        )}
      </div>
      <div className="gstats-cell gstats-pct">
        {pf ? (
          <span className={`v ${attainTone(pf.fracOfCeiling)}`}>{fmtPct(pf.fracOfCeiling, 0)}</span>
        ) : (
          '—'
        )}
      </div>
      <div className="gstats-cell gstats-num">
        <span className="v">{costBound(hw.prefillCeiling, baseline?.prefillTokPerSecPerChip)}</span>
        <span className="s">at ceiling</span>
      </div>

      <div className="gstats-cell gstats-phase">
        Decode<small>tok/s/chip</small>
      </div>
      <div className="gstats-cell">
        <div className="rrg-track">
          <span
            className="rrg-ceil"
            title={`Decode ceiling: ${fmtSI(hw.decodeCeilingSerial)}/s at zero overlap (per-token compute and KV read fully serialized) → ${fmtSI(hw.decodeCeilingOverlapped)}/s at 100% overlap`}
            style={{
              left: pct(hw.decodeCeilingSerial, decScale),
              width: pct(hw.decodeCeilingOverlapped - hw.decodeCeilingSerial, decScale),
            }}
          />
          {points(
            (r) =>
              r.decode ? { value: r.decode.tokPerSecPerChip, bound: r.decode.boundBy } : null,
            decScale,
          )}
        </div>
        <div className="rrg-nums">
          <span>0 tok/s</span>
          {hw.decodeCeilingSerial < 0.8 * decScale && (
            <span className="rrg-mark" style={{ left: pct(hw.decodeCeilingSerial, decScale) }}>
              {fmtSI(hw.decodeCeilingSerial)}/s
            </span>
          )}
          <span>{fmtSI(decScale)}/s</span>
        </div>
      </div>
      <div className="gstats-cell gstats-num">
        <span className="v">{fmtSI(hw.decodeCeilingOverlapped)}/s</span>
        <span className="s">B = ∞</span>
      </div>
      <div className="gstats-cell gstats-num">
        {dec ? (
          <>
            <span className="v">{fmtSI(dec.tokPerSecPerChip)}/s</span>
            <span className="s">B = {fmtInt(dec.batchPerStage)}</span>
          </>
        ) : (
          '—'
        )}
      </div>
      <div className="gstats-cell gstats-pct">
        {dec ? (
          <span className={`v ${attainTone(dec.fracOfCeiling)}`}>
            {fmtPct(dec.fracOfCeiling, 0)}
          </span>
        ) : (
          '—'
        )}
      </div>
      <div className="gstats-cell gstats-num">
        <span className="v">
          {costBound(hw.decodeCeilingOverlapped, baseline?.decodeTokPerSecPerChip)}
        </span>
        <span className="s">at ceiling</span>
      </div>

      <div className="gstats-foot">
        <div
          className="rrg-legend"
          title="Each point is one sharding at the Memory/Comms overlap you set"
        >
          <span>
            <span className="sw sw-ceil" />
            ceiling (0% → 100% overlap)
          </span>
          <span>
            <span className="sw sw-dot rrg-pt-compute" />
            compute-bound
          </span>
          <span>
            <span className="sw sw-dot rrg-pt-memory" />
            memory-bound
          </span>
          <span>
            <span className="sw sw-dot rrg-pt-comms" />
            comms-bound
          </span>
          <span>
            <span className="sw sw-dot sw-dot-best" />
            best
          </span>
        </div>
        <div className="gstats-mem">
          <span title="Total model parameter count">
            Params <b>{fmtParams(hw.paramsTotal)}</b>
          </span>
          <span title="Minimum chips whose HBM holds the weights alone (KV needs more)">
            Weights fit in{' '}
            <b>
              ≥ {fmtInt(hw.minChipsForWeights)} chip{hw.minChipsForWeights > 1 ? 's' : ''}
            </b>
          </span>
          <span title="KV cache bytes for one full sequence (prefill + generate), whole model">
            KV / seq <b>{fmtBytes(hw.kvBytesPerSeq)}</b>
          </span>
        </div>
      </div>
    </div>
  );
}

/** Predicted value at the tuned overlap; % of ceiling renders as a small attainment bar. */
// the columns that divide by the cost basis, and so have nothing to show
// for a chip with no published TDP on the power basis
const COST_KEY_SET = new Set<string>(Object.values(COST_KEYS));

function UnknownTdp({ vendor }: { vendor: string }) {
  return (
    <span
      className="muted"
      data-tip={`${vendor} does not publish this chip's TDP, so per-kW efficiency can't be computed.`}
    >
      Unknown TDP
    </span>
  );
}

function HeadlineCell({ m, r }: { m: MetricDef; r: UiResult }) {
  const v = m.value(r);
  if (v === null || !Number.isFinite(v)) return <>—</>;
  if (m.key === 'fracOfCeiling' || m.key === 'batchSaturation') {
    return (
      <span className="frac-cell">
        <span className={`bar ${v < 0.5 ? 'warn' : ''}`}>
          <i style={{ width: `${Math.round(v * 100)}%`, background: gaugeColor(v) }} />
        </span>
        <span className="frac-pct">{fmtPct(v, 0)}</span>
      </span>
    );
  }
  if (m.key === 'tokPerUser') return <>{fmtInt(v)}</>;
  return <>{fmtMetricValue(m, v)}</>;
}

function StatusIcon({
  r,
  showTip,
  hideTip,
}: {
  r: UiResult;
  showTip: (e: MouseEvent<HTMLElement>, diagnostics: Diagnostic[]) => void;
  hideTip: () => void;
}) {
  const cls = hasError(r) ? 'status-error' : hasWarning(r) ? 'status-warn' : 'status-ok';
  const glyph = hasError(r) ? '✕' : hasWarning(r) ? '!' : '✓';
  return (
    <span
      className={`status ${cls}`}
      onMouseEnter={(e) => showTip(e, r.diagnostics)}
      onMouseLeave={hideTip}
    >
      {glyph}
    </span>
  );
}
