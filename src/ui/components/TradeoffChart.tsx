/** Scatter chart of the configuration tradeoff space with selectable axes. */
import { useMemo, useRef, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from 'recharts';
import { ChipSpec } from '../../core/hardware/chips';
import { UiResult, hasError, hasWarning } from '../results';
import { CHART_METRICS, fmtMetricValue, metricByKey } from '../metrics';
import { VENDOR_COLORS } from './VendorLogo';

interface Props {
  results: UiResult[];
  chipsById: Record<string, ChipSpec>;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  /** click a point to open its details slide-over */
  onSelect: (id: string | null) => void;
}

/** Fallback when a chip's vendor isn't in VENDOR_COLORS. */
const UNKNOWN_VENDOR = '#5f6368';

/** Marker shapes cycled within a vendor so same-brand chips stay distinct. */
type MarkerShape = 'circle' | 'square' | 'diamond' | 'triangle' | 'triangleDown' | 'cross' | 'star';

const MARKER_SHAPES: MarkerShape[] = [
  'circle',
  'square',
  'diamond',
  'triangle',
  'triangleDown',
  'cross',
  'star',
];

interface Point {
  x: number;
  y: number;
  id: string;
  label: string;
  warn: boolean;
}

interface ChipSeries {
  vendor: string;
  color: string;
  shape: MarkerShape;
  pts: Point[];
}

// Mantissa patterns per decade, finest first; logTicks picks the finest that fits.
const LOG_TICK_MANTISSAS = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 5], [1, 3], [1]];

/**
 * Explicit ticks for a log axis over [min, max]. Recharts' built-in tick
 * generator emits duplicate values on log scales (duplicate React keys), so
 * we hand it a deduplicated set instead.
 */
function logTicks(min: number, max: number): number[] | undefined {
  if (!(min > 0) || !(max > 0)) return undefined;
  if (min === max) return [min];
  const loExp = Math.floor(Math.log10(min));
  const hiExp = Math.ceil(Math.log10(max));
  for (const mantissas of LOG_TICK_MANTISSAS) {
    const ticks: number[] = [];
    for (let exp = loExp; exp <= hiExp; exp++) {
      for (const m of mantissas) {
        const t = m * 10 ** exp;
        if (t >= min * 0.999 && t <= max * 1.001) ticks.push(t);
      }
    }
    if (ticks.length >= 2 && ticks.length <= 8) return ticks;
  }
  // Range too narrow for any round tick to land inside: label the endpoints.
  return [min, max];
}

/** Nice linear step multipliers (× 10^k). Prefer fewer, rounder ticks. */
const LINEAR_STEPS = [1, 2, 2.5, 5, 10];

/**
 * Explicit ticks for a linear axis. Recharts' default for scatter charts can
 * emit a tick per distinct data value, which overlaps badly on dense axes.
 */
function linearTicks(min: number, max: number): number[] | undefined {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return undefined;
  if (min === max) return [min];
  const lo = Math.min(min, max);
  const hi = Math.max(min, max);
  const span = hi - lo;
  const rough = span / 5; // aim for ~5–6 ticks
  const exp = Math.floor(Math.log10(rough));
  const base = 10 ** exp;
  let step = LINEAR_STEPS[0] * base;
  for (const m of LINEAR_STEPS) {
    const s = m * base;
    if (span / s <= 7) {
      step = s;
      break;
    }
  }
  // If even 10×base is still too dense, bump another decade.
  if (span / step > 7) step = 10 * base;
  const start = Math.ceil(lo / step) * step;
  const ticks: number[] = [];
  // Guard float drift (e.g. 0.1 + 0.2) with a small epsilon past hi.
  for (let t = start; t <= hi + step * 1e-9; t += step) {
    const v = Math.abs(t) < step * 1e-12 ? 0 : t;
    ticks.push(v);
  }
  if (ticks.length < 2) return [lo, hi];
  return ticks;
}

function axisTicks(min: number, max: number, log: boolean): number[] | undefined {
  return log ? logTicks(min, max) : linearTicks(min, max);
}

/** Solve Ax = b for a dense square matrix (Gaussian elimination w/ partial pivot). */
function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    if (piv !== col) [M[col], M[piv]] = [M[piv], M[col]];
    const div = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row) => row[n]);
}

/**
 * Least-squares polynomial fit of degree `deg` (y ≈ Σ c_k x^k).
 * Returns coefficients [c0, c1, …] or null if underdetermined / singular.
 */
function polyFit(xs: number[], ys: number[], deg: number): number[] | null {
  const n = xs.length;
  if (n < deg + 1) return null;
  const m = deg + 1;
  const A: number[][] = Array.from({ length: m }, () => Array(m).fill(0));
  const b: number[] = Array(m).fill(0);
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    const y = ys[i];
    let xp = 1;
    const powers: number[] = [];
    for (let k = 0; k < m; k++) {
      powers.push(xp);
      xp *= x;
    }
    for (let r = 0; r < m; r++) {
      b[r] += powers[r] * y;
      for (let c = 0; c < m; c++) A[r][c] += powers[r] * powers[c];
    }
  }
  return solveLinear(A, b);
}

function polyEval(coeffs: number[], x: number): number {
  let y = 0;
  let xp = 1;
  for (const c of coeffs) {
    y += c * xp;
    xp *= x;
  }
  return y;
}

function rSquared(xs: number[], ys: number[], coeffs: number[]): number {
  const mean = ys.reduce((s, y) => s + y, 0) / ys.length;
  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < xs.length; i++) {
    const pred = polyEval(coeffs, xs[i]);
    ssRes += (ys[i] - pred) ** 2;
    ssTot += (ys[i] - mean) ** 2;
  }
  if (ssTot < 1e-18) return 1;
  return 1 - ssRes / ssTot;
}

export function TradeoffChart({ results, chipsById, hoveredId, onHover, onSelect }: Props) {
  const [xKey, setXKey] = useState('prefillEff');
  const [yKey, setYKey] = useState('decodeEff');
  const [logX, setLogX] = useState(false);
  const [logY, setLogY] = useState(false);
  const [showPareto, setShowPareto] = useState(false);
  const [showFit, setShowFit] = useState(false);
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'error'>('idle');
  const chartBoxRef = useRef<HTMLDivElement>(null);

  const xm = metricByKey(xKey);
  const ym = metricByKey(yKey);

  const byChip = useMemo(() => {
    const groups = new Map<string, ChipSeries>();
    for (const r of results) {
      if (hasError(r)) continue;
      const x = xm.value(r);
      const y = ym.value(r);
      if (x === null || y === null || !Number.isFinite(x) || !Number.isFinite(y)) continue;
      if ((logX && x <= 0) || (logY && y <= 0)) continue;
      const chip = chipsById[r.chipId];
      if (!chip) continue;
      const sizes = Object.entries(r.sizes)
        .map(([role, size]) => `${role}${size}`)
        .join(' ');
      const pt: Point = {
        x,
        y,
        id: r.id,
        label: `${chip.name} · ${sizes || 'single chip'}${r.sizes.EP != null ? ` · ${r.dispatch}` : ''}`,
        warn: hasWarning(r),
      };
      const series = groups.get(chip.name) ?? {
        vendor: chip.vendor,
        color: VENDOR_COLORS[chip.vendor] ?? UNKNOWN_VENDOR,
        shape: 'circle',
        pts: [],
      };
      series.pts.push(pt);
      groups.set(chip.name, series);
    }
    // Distinct marker per chip within a vendor; color stays the brand.
    const byVendor = new Map<string, string[]>();
    for (const [name, s] of groups) {
      const list = byVendor.get(s.vendor) ?? [];
      list.push(name);
      byVendor.set(s.vendor, list);
    }
    for (const names of byVendor.values()) {
      names.forEach((name, i) => {
        groups.get(name)!.shape = MARKER_SHAPES[i % MARKER_SHAPES.length];
      });
    }
    return groups;
  }, [results, chipsById, xm, ym, logX, logY]);

  // Ids of points not dominated on the chosen axes (better-or-equal on both,
  // strictly better on one). Direction comes from each metric's lowerIsBetter.
  const paretoIds = useMemo(() => {
    const pts = [...byChip.values()].flatMap((s) => s.pts);
    const sx = xm.lowerIsBetter ? (p: Point) => -p.x : (p: Point) => p.x;
    const sy = ym.lowerIsBetter ? (p: Point) => -p.y : (p: Point) => p.y;
    const ids = new Set<string>();
    for (const p of pts) {
      const dominated = pts.some(
        (q) => sx(q) >= sx(p) && sy(q) >= sy(p) && (sx(q) > sx(p) || sy(q) > sy(p)),
      );
      if (!dominated) ids.add(p.id);
    }
    return ids;
  }, [byChip, xm, ym]);

  const { xTicks, yTicks } = useMemo(() => {
    const pts = [...byChip.values()].flatMap((s) => s.pts);
    if (pts.length === 0) return { xTicks: undefined, yTicks: undefined };
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    return {
      xTicks: axisTicks(Math.min(...xs), Math.max(...xs), logX),
      yTicks: axisTicks(Math.min(...ys), Math.max(...ys), logY),
    };
  }, [byChip, logX, logY]);

  // Quadratic (or linear if few points) least-squares fit; in log-space when
  // that axis is log so the curve tracks the visual scale.
  const fit = useMemo(() => {
    if (!showFit) return null;
    const pts = [...byChip.values()].flatMap((s) => s.pts);
    if (pts.length < 3) return null;
    const xs = pts.map((p) => (logX ? Math.log10(p.x) : p.x));
    const ys = pts.map((p) => (logY ? Math.log10(p.y) : p.y));
    if (xs.some((v) => !Number.isFinite(v)) || ys.some((v) => !Number.isFinite(v))) return null;
    const deg = pts.length >= 5 ? 2 : 1;
    const coeffs = polyFit(xs, ys, deg);
    if (!coeffs) return null;
    const r2 = rSquared(xs, ys, coeffs);
    const xMin = Math.min(...pts.map((p) => p.x));
    const xMax = Math.max(...pts.map((p) => p.x));
    const n = 48;
    const curve: { x: number; y: number }[] = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const x = logX
        ? 10 ** (Math.log10(xMin) + t * (Math.log10(xMax) - Math.log10(xMin)))
        : xMin + t * (xMax - xMin);
      const xFit = logX ? Math.log10(x) : x;
      const yFit = polyEval(coeffs, xFit);
      const y = logY ? 10 ** yFit : yFit;
      if (!Number.isFinite(x) || !Number.isFinite(y) || (y <= 0 && logY)) continue;
      curve.push({ x, y });
    }
    if (curve.length < 2) return null;
    return { curve, r2, deg };
  }, [byChip, showFit, logX, logY]);

  const copyScreenshot = async () => {
    const root = chartBoxRef.current;
    if (!root) return;
    try {
      await copyChartToClipboard(root);
      setCopyState('copied');
      window.setTimeout(() => setCopyState('idle'), 1600);
    } catch {
      setCopyState('error');
      window.setTimeout(() => setCopyState('idle'), 2200);
    }
  };

  return (
    <div className="chart-panel">
      <div className="axis-controls">
        <AxisPicker label="X" value={xKey} onChange={setXKey} log={logX} onLog={setLogX} />
        <AxisPicker label="Y" value={yKey} onChange={setYKey} log={logY} onLog={setLogY} />
        <label className="check">
          <input
            type="checkbox"
            checked={showPareto}
            onChange={(e) => setShowPareto(e.target.checked)}
          />
          <span>Highlight Pareto frontier</span>
        </label>
        <label
          className="check"
          title="Least-squares polynomial overlay (quadratic when ≥5 points)"
        >
          <input type="checkbox" checked={showFit} onChange={(e) => setShowFit(e.target.checked)} />
          <span>
            Fit curve
            {fit && (
              <span className="fit-r2">
                {' '}
                (R² {fit.r2.toFixed(2)}, deg {fit.deg})
              </span>
            )}
          </span>
        </label>
        <button
          type="button"
          className="chart-copy-btn"
          onClick={copyScreenshot}
          title="Copy chart screenshot to clipboard"
        >
          {copyState === 'copied'
            ? 'Copied'
            : copyState === 'error'
              ? 'Copy failed'
              : 'Copy screenshot'}
        </button>
      </div>
      <div className="chart-box" ref={chartBoxRef}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 16, bottom: 40, left: 16 }}>
            <CartesianGrid stroke="var(--border-light)" />
            <XAxis
              type="number"
              dataKey="x"
              scale={logX ? 'log' : 'linear'}
              domain={['auto', 'auto']}
              ticks={xTicks}
              name={xm.label}
              tickFormatter={(v: number) => fmtMetricValue(xm, v)}
              tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
              stroke="var(--border)"
              label={{
                value: xm.label,
                position: 'bottom',
                offset: 12,
                fill: 'var(--text-secondary)',
                fontSize: 12,
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              scale={logY ? 'log' : 'linear'}
              domain={['auto', 'auto']}
              ticks={yTicks}
              name={ym.label}
              tickFormatter={(v: number) => fmtMetricValue(ym, v)}
              tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
              stroke="var(--border)"
              width={62}
              label={{
                value: ym.label,
                angle: -90,
                position: 'insideLeft',
                offset: 4,
                fill: 'var(--text-secondary)',
                fontSize: 12,
                style: { textAnchor: 'middle' },
              }}
            />
            <Tooltip
              cursor={{ strokeDasharray: '3 3', stroke: 'var(--text-secondary)' }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const pt = payload[0].payload as Point;
                if (!pt?.id) return null;
                return (
                  <div className="chart-tooltip">
                    <div className="tt-title">{pt.label}</div>
                    <div>
                      {xm.label}: <b>{fmtMetricValue(xm, pt.x)}</b>
                    </div>
                    <div>
                      {ym.label}: <b>{fmtMetricValue(ym, pt.y)}</b>
                    </div>
                    {pt.warn && <div className="tt-warn">⚠ has warnings — click for details</div>}
                  </div>
                );
              }}
            />
            <Legend
              verticalAlign="top"
              align="right"
              wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }}
              content={() => (
                <ul className="chart-legend">
                  {[...byChip.entries()].map(([chipName, { color, shape }]) => (
                    <li key={chipName}>
                      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                        <Marker shape={shape} cx={6} cy={6} r={4} fill={color} stroke={color} />
                      </svg>
                      <span>{chipName}</span>
                    </li>
                  ))}
                </ul>
              )}
            />
            {fit && (
              <Scatter
                name="_fit"
                data={fit.curve}
                line={{ stroke: 'var(--text-secondary)', strokeWidth: 1.5, strokeDasharray: '5 4' }}
                lineType="joint"
                shape={() => <g />}
                legendType="none"
                tooltipType="none"
                isAnimationActive={false}
              />
            )}
            {[...byChip.entries()].map(([chipName, { color, shape, pts }]) => (
              <Scatter
                key={chipName}
                name={chipName}
                data={
                  // draw dimmed points first so the frontier stays on top
                  showPareto
                    ? [...pts].sort(
                        (a, b) => Number(paretoIds.has(a.id)) - Number(paretoIds.has(b.id)),
                      )
                    : pts
                }
                fill={color}
                shape={(props: unknown) => (
                  <Dot
                    {...(props as DotProps)}
                    shape={shape}
                    hoveredId={hoveredId}
                    dimIds={showPareto ? paretoIds : null}
                  />
                )}
                onMouseEnter={(pt: Point) => onHover(pt.id)}
                onMouseLeave={() => onHover(null)}
                onClick={(pt: Point) => onSelect(pt.id)}
                isAnimationActive={false}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

interface DotProps {
  cx?: number;
  cy?: number;
  fill?: string;
  payload?: Point;
  shape: MarkerShape;
  hoveredId: string | null;
  /** when set, points NOT in this set render faded (Pareto-frontier mode) */
  dimIds?: Set<string> | null;
}

function Dot({ cx, cy, fill, payload, shape, hoveredId, dimIds }: DotProps) {
  if (cx === undefined || cy === undefined || !payload) return null;
  const hovered = hoveredId === payload.id;
  const dimmed = dimIds != null && !dimIds.has(payload.id);
  const color = fill ?? UNKNOWN_VENDOR;
  return (
    <Marker
      shape={shape}
      cx={cx}
      cy={cy}
      r={hovered ? 7 : 4.5}
      fill={payload.warn ? 'var(--surface)' : color}
      stroke={color}
      strokeWidth={hovered ? 2.5 : 1.5}
      opacity={dimmed && !hovered ? 0.18 : 1}
    />
  );
}

function Marker(props: {
  shape: MarkerShape;
  cx: number;
  cy: number;
  r: number;
  fill: string;
  stroke: string;
  strokeWidth?: number;
  opacity?: number;
}) {
  const { shape, cx, cy, r, fill, stroke, strokeWidth = 1.5, opacity = 1 } = props;
  const common = { fill, stroke, strokeWidth, opacity };
  if (shape === 'circle') return <circle cx={cx} cy={cy} r={r} {...common} />;
  if (shape === 'square') {
    const s = r * 1.6;
    return <rect x={cx - s / 2} y={cy - s / 2} width={s} height={s} {...common} />;
  }
  if (shape === 'diamond') {
    return (
      <polygon
        points={`${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`}
        {...common}
      />
    );
  }
  if (shape === 'triangle') {
    return (
      <polygon
        points={`${cx},${cy - r} ${cx + r * 0.95},${cy + r * 0.75} ${cx - r * 0.95},${cy + r * 0.75}`}
        {...common}
      />
    );
  }
  if (shape === 'triangleDown') {
    return (
      <polygon
        points={`${cx},${cy + r} ${cx + r * 0.95},${cy - r * 0.75} ${cx - r * 0.95},${cy - r * 0.75}`}
        {...common}
      />
    );
  }
  if (shape === 'cross') {
    const t = r * 0.35;
    return (
      <path
        d={`M${cx - t} ${cy - r} H${cx + t} V${cy - t} H${cx + r} V${cy + t} H${cx + t} V${cy + r} H${cx - t} V${cy + t} H${cx - r} V${cy - t} H${cx - t} Z`}
        {...common}
      />
    );
  }
  // 5-point star
  const pts: string[] = [];
  for (let i = 0; i < 5; i++) {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
    const b = a + Math.PI / 5;
    pts.push(`${cx + Math.cos(a) * r},${cy + Math.sin(a) * r}`);
    pts.push(`${cx + Math.cos(b) * r * 0.45},${cy + Math.sin(b) * r * 0.45}`);
  }
  return <polygon points={pts.join(' ')} {...common} />;
}

function AxisPicker(props: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  log: boolean;
  onLog: (v: boolean) => void;
}) {
  return (
    <div className="axis-picker">
      <span className="axis-label">{props.label}</span>
      <select value={props.value} onChange={(e) => props.onChange(e.target.value)}>
        {CHART_METRICS.map((m) => (
          <option key={m.key} value={m.key}>
            {m.label}
          </option>
        ))}
      </select>
      <label className="check">
        <input
          type="checkbox"
          checked={props.log}
          onChange={(e) => props.onLog(e.target.checked)}
        />
        <span>log</span>
      </label>
    </div>
  );
}

/** Resolve CSS-var paints on a live SVG onto a clone so the PNG export is self-contained. */
function inlineSvgComputedPaints(source: SVGSVGElement, clone: SVGSVGElement) {
  const srcEls = [source, ...source.querySelectorAll<SVGElement>('*')];
  const dstEls = [clone, ...clone.querySelectorAll<SVGElement>('*')];
  for (let i = 0; i < srcEls.length; i++) {
    const src = srcEls[i];
    const dst = dstEls[i];
    if (!dst) continue;
    const cs = getComputedStyle(src);
    if (cs.fill && cs.fill !== 'none') dst.setAttribute('fill', cs.fill);
    if (cs.stroke && cs.stroke !== 'none') dst.setAttribute('stroke', cs.stroke);
    if (cs.opacity && cs.opacity !== '1') dst.setAttribute('opacity', cs.opacity);
    if (cs.strokeWidth) dst.setAttribute('stroke-width', cs.strokeWidth);
    if (src.tagName === 'text' || src.tagName === 'tspan') {
      dst.setAttribute('font-size', cs.fontSize);
      dst.setAttribute('font-family', cs.fontFamily);
      dst.setAttribute('font-weight', cs.fontWeight);
    }
  }
}

function loadSvgImage(svg: SVGSVGElement): Promise<HTMLImageElement> {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  inlineSvgComputedPaints(svg, clone);
  const rect = svg.getBoundingClientRect();
  const w = Math.max(1, Math.ceil(rect.width));
  const h = Math.max(1, Math.ceil(rect.height));
  clone.setAttribute('width', String(w));
  clone.setAttribute('height', String(h));
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const xml = new XMLSerializer().serializeToString(clone);
  const url = URL.createObjectURL(new Blob([xml], { type: 'image/svg+xml;charset=utf-8' }));
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('svg image load failed'));
    };
    img.src = url;
  });
}

async function copyChartToClipboard(root: HTMLElement): Promise<void> {
  const svg = root.querySelector<SVGSVGElement>('svg.recharts-surface');
  if (!svg) throw new Error('chart svg missing');
  if (!navigator.clipboard || typeof ClipboardItem === 'undefined') {
    throw new Error('clipboard image write unsupported');
  }

  const scale = Math.min(2, window.devicePixelRatio || 1);
  const rootRect = root.getBoundingClientRect();
  const svgRect = svg.getBoundingClientRect();
  const w = Math.max(1, Math.ceil(rootRect.width));
  const h = Math.max(1, Math.ceil(rootRect.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('canvas unsupported');
  ctx.scale(scale, scale);

  const surface =
    getComputedStyle(document.documentElement).getPropertyValue('--surface').trim() || '#fff';
  ctx.fillStyle = surface;
  ctx.fillRect(0, 0, w, h);

  const img = await loadSvgImage(svg);
  ctx.drawImage(
    img,
    svgRect.left - rootRect.left,
    svgRect.top - rootRect.top,
    svgRect.width,
    svgRect.height,
  );

  // Legend is HTML (not in the SVG); paint it from the live DOM so markers match.
  const legend = root.querySelector<HTMLElement>('.chart-legend');
  if (legend) {
    const items = [...legend.querySelectorAll<HTMLElement>('li')];
    for (const item of items) {
      const mark = item.querySelector('svg');
      const label = item.querySelector('span');
      if (mark) {
        const markRect = mark.getBoundingClientRect();
        const markImg = await loadSvgImage(mark as unknown as SVGSVGElement);
        ctx.drawImage(
          markImg,
          markRect.left - rootRect.left,
          markRect.top - rootRect.top,
          markRect.width,
          markRect.height,
        );
      }
      if (label) {
        const labelRect = label.getBoundingClientRect();
        const cs = getComputedStyle(label);
        ctx.fillStyle = cs.color;
        ctx.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
        ctx.textBaseline = 'middle';
        ctx.fillText(
          label.textContent ?? '',
          labelRect.left - rootRect.left,
          labelRect.top - rootRect.top + labelRect.height / 2,
        );
      }
    }
  }

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('png encode failed');
  await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
}
