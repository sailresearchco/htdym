/** Number formatting helpers for table cells and chart labels. */

const SI = [
  { div: 1e12, suffix: 'T' },
  { div: 1e9, suffix: 'G' },
  { div: 1e6, suffix: 'M' },
  { div: 1e3, suffix: 'k' },
];

export function fmtSI(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  for (const { div, suffix } of SI) {
    if (a >= div) return `${toPrecision(v / div, digits)}${suffix}`;
  }
  return toPrecision(v, digits);
}

const PARAMS = [
  { div: 1e12, suffix: 'T' },
  { div: 1e9, suffix: 'B' },
  { div: 1e6, suffix: 'M' },
  { div: 1e3, suffix: 'k' },
];

export function fmtParams(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  for (const { div, suffix } of PARAMS) {
    if (a >= div) return `${toPrecision(v / div, digits)}${suffix}`;
  }
  return toPrecision(v, digits);
}

export function fmtBytes(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e12) return `${toPrecision(v / 1e12, digits)} TB`;
  if (a >= 1e9) return `${toPrecision(v / 1e9, digits)} GB`;
  if (a >= 1e6) return `${toPrecision(v / 1e6, digits)} MB`;
  if (a >= 1e3) return `${toPrecision(v / 1e3, digits)} kB`;
  return `${Math.round(v)} B`;
}

export function fmtTime(s: number, digits = 3): string {
  if (!Number.isFinite(s)) return '—';
  if (s >= 1) return `${toPrecision(s, digits)} s`;
  if (s >= 1e-3) return `${toPrecision(s * 1e3, digits)} ms`;
  return `${toPrecision(s * 1e6, digits)} µs`;
}

export function fmtPct(frac: number, digits = 1): string {
  if (!Number.isFinite(frac)) return '—';
  return `${(frac * 100).toFixed(digits)}%`;
}

export function fmtInt(v: number): string {
  if (!Number.isFinite(v)) return '—';
  return Math.round(v).toLocaleString('en-US');
}

/** Efficiency multipliers ("1.6×"), the public build's only cost quote. */
export function fmtMult(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  let s = toPrecision(v, digits);
  // "1×" reads as the HMVP anchor itself, so a value that merely rounds to
  // 1 keeps an extra digit (1.004×) instead of displaying as exact
  if (s === '1' && v !== 1) s = toPrecision(v, digits + 1);
  return `${s}×`;
}

function toPrecision(v: number, digits: number): string {
  const s = v.toPrecision(digits);
  // strip trailing zeros after a decimal point ("14.0" -> "14")
  return s.includes('.') && !s.includes('e') ? s.replace(/\.?0+$/, '') : s;
}
