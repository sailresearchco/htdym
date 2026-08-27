import { useEffect, useMemo, useRef, useState } from 'react';
import { CHIPS, ChipSpec } from '../core/hardware/chips';
import { MODEL_PRESETS, ModelSpec } from '../core/model/models';
import { DetailsPanel } from './components/DetailsPanel';
import { Leaderboard } from './components/Leaderboard';
import { PriceNotice } from './components/PriceNotice';
import { ProgressRing } from './components/ProgressRing';
import { Sidebar, SweepControls } from './components/Sidebar';
import { SlaFilterPanel, SlaThresholds, passesSla } from './components/SlaFilters';
import { TradeoffChart } from './components/TradeoffChart';
import { UiChip, UiGroup, UiResult, UiWorkload, hasError, ringTone } from './results';
import { H100_ID, hmvpEff, relPriceOf, requestChipSecs, useHmvpBaseline } from './pricing';
import { makeSearchClient } from './searchClient';
import {
  machineAtNodes,
  machineChips,
  machineKey,
  machineLabel,
  machineVariants,
} from './machines';
import './app.css';

function SailResearchLogo() {
  return (
    <svg
      className="sail-logo"
      viewBox="30 22 400 155"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Sail Research"
    >
      <path d="M406.81 171.925V79.5708H423.154V171.925H406.81Z" fill="currentColor" />
      <path
        d="M378.723 96.3035V80.9977H394.807V96.3035H378.723ZM378.593 171.925V104.994H394.937V171.925H378.593Z"
        fill="currentColor"
      />
      <path
        d="M370.534 171.925H354.19C353.153 170.109 352.634 166.996 352.374 163.753C347.964 169.979 340.7 173.481 330.842 173.481C316.574 173.481 306.716 166.347 306.716 153.376C306.716 142.221 313.461 134.308 332.399 132.493L342.516 131.585C348.743 130.806 351.985 128.861 351.985 123.932C351.985 118.743 349.261 115.63 339.793 115.63C330.453 115.63 326.562 118.095 325.913 126.266H309.829C310.737 112.128 319.039 103.437 339.922 103.437C359.898 103.437 367.81 111.479 367.81 123.543V159.213C367.81 164.012 368.718 169.33 370.534 171.925ZM334.734 161.807C343.424 161.807 351.985 157.138 351.985 145.464V139.367C350.299 140.924 347.835 141.702 344.332 142.091L335.512 143.129C326.432 144.166 323.449 147.409 323.449 152.727C323.449 158.305 327.211 161.807 334.734 161.807Z"
        fill="currentColor"
      />
      <path
        d="M263.976 174C239.331 174 224.544 162.715 224.414 141.443H241.666C241.925 155.97 251.653 159.862 264.495 159.862C276.558 159.862 283.173 155.192 283.173 146.501C283.173 139.238 278.374 135.995 264.235 133.141L256.453 131.585C238.553 128.212 226.749 120.3 226.749 103.567C226.749 88.2613 238.812 77.4954 260.992 77.4954C286.935 77.4954 297.96 89.6882 298.608 107.069H281.616C280.968 97.2114 275.909 91.6338 261.252 91.6338C250.097 91.6338 244.649 96.044 244.649 103.048C244.649 110.442 248.929 113.814 262.679 116.668L270.721 118.224C292.512 122.505 301.332 130.936 301.332 145.593C301.332 163.883 287.064 174 263.976 174Z"
        fill="currentColor"
      />
      <path
        d="M199.371 25.2675C192.86 38.3429 153.423 114.947 95.2881 173.046C94.9724 173.361 95.1953 173.902 95.6416 173.902H199.824C200.1 173.902 200.324 173.678 200.324 173.402V25.5279C200.324 24.9959 199.608 24.7912 199.371 25.2675Z"
        fill="currentColor"
      />
      <path
        d="M121.497 74.7074C115.025 83.6604 74.1574 139.303 34.1783 173.022C33.8209 173.323 34.0372 173.902 34.5047 173.902H121.933C122.209 173.902 122.383 173.679 122.383 173.402L122.383 75.0412C122.383 74.5542 121.782 74.3127 121.497 74.7074Z"
        fill="currentColor"
      />
    </svg>
  );
}

type ThemeMode = 'light' | 'dark';

const THEME_STORAGE_KEY = 'htdym-theme';

function initialTheme(): ThemeMode {
  if (typeof window === 'undefined') return 'light';
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  const theme =
    stored === 'light' || stored === 'dark'
      ? stored
      : window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
  document.documentElement.dataset.theme = theme;
  return theme;
}

export function App() {
  const [theme, setTheme] = useState<ThemeMode>(initialTheme);
  const [model, setModel] = useState<ModelSpec>(MODEL_PRESETS[0]);
  const [workload, setWorkload] = useState<UiWorkload>({
    prefillLen: 4096,
    generateLen: 1024,
    batching: 'max',
  });
  const [sweep, setSweep] = useState<SweepControls>({
    // unpriced chips (unannounced parts) start deselected
    chipIds: CHIPS.filter((c) => c.costPerHour !== undefined).map((c) => c.id),
    machine: 1,
    // realistic serving defaults: HBM traffic overlaps well, comms partially
    memoryOverlap: 0.9,
    commsOverlap: 0.65,
  });
  // Editable copy of the chip database: every spec field (FLOPs, HBM,
  // interconnect, MFU, price) can be overridden from the Hardware panel.
  const [chips, setChips] = useState<UiChip[]>(() =>
    CHIPS.map((c) => ({ ...c, formats: { ...c.formats }, interconnect: { ...c.interconnect } })),
  );
  const [filterText, setFilterText] = useState('');
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<'table' | 'chart'>('table');
  // active SLA targets by metric key; the panel toggles from the toolbar
  const [sla, setSla] = useState<SlaThresholds>({});
  const [slaOpen, setSlaOpen] = useState(false);
  // the ×HMVP reference: a hidden search of the smallest viable H100
  // machine, always running off the live-edited H100 spec
  const chipsById = useMemo(() => Object.fromEntries(chips.map((c) => [c.id, c])), [chips]);
  // stable identity so the baseline hook's change-detection key isn't
  // recomputed on unrelated renders
  const overlap = useMemo(
    () => ({ memoryOverlap: sweep.memoryOverlap, commsOverlap: sweep.commsOverlap }),
    [sweep.memoryOverlap, sweep.commsOverlap],
  );
  const baseline = useHmvpBaseline(model, workload, overlap, chipsById[H100_ID]);

  // streamed search state: one group per enabled chip, rows arriving live.
  // Each sweep supersedes the previous accepted request.
  const [groups, setGroups] = useState<Map<string, UiGroup>>(new Map());
  const activeReq = useRef(0);
  const clientRef = useRef<ReturnType<typeof makeSearchClient> | null>(null);

  useEffect(() => {
    const client = makeSearchClient((u) => {
      if (activeReq.current !== u.id) return;
      setGroups((gs) => {
        const next = new Map(gs);
        if (u.kind === 'group') {
          const prev = next.get(u.key);
          next.set(u.key, {
            key: u.key,
            chip: chipsRef.current.find((c) => c.id === u.chipId)!,
            machineLabel: u.machineLabel,
            nChips: u.nChips,
            hardware: u.hardware,
            configs: prev?.configs ?? [],
            done: prev?.done ?? 0,
            total: u.total,
            error: u.error,
          });
        } else {
          const g = next.get(u.key);
          if (!g) return gs;
          const row = u.row;
          next.set(u.key, {
            ...g,
            done: Math.max(g.done, u.done),
            configs: row ? [...g.configs, row] : g.configs,
          });
        }
        return next;
      });
    });
    clientRef.current = client;
    return () => client.dispose();
  }, []);

  // the worker needs the current edited chips at message time
  const chipsRef = useRef(chips);
  chipsRef.current = chips;
  const groupsRef = useRef(groups);
  groupsRef.current = groups;
  // what the last request searched: shared inputs + per-chip spec keys
  const lastSearched = useRef({ shared: '', specs: new Map<string, string>() });

  // Debounced incremental re-search (300ms after the last keystroke): only
  // chips whose engine-visible spec changed go back to the worker, plus any
  // whose stream the new request supersedes mid-flight. Finished groups are
  // kept. Price edits never search ($/hr is applied at render time).
  useEffect(() => {
    const t = setTimeout(() => {
      if (![workload.prefillLen, workload.generateLen].every((v) => Number.isFinite(v) && v > 0))
        return;
      const enabled = chips
        .filter(
          (c) =>
            sweep.chipIds.includes(c.id) &&
            [
              c.hbmCapacity,
              c.hbmBandwidth,
              c.interconnect.bandwidthPerChip,
              c.interconnect.domainSize,
              c.realizableFlopsFrac,
              c.realizableHbmBwFrac,
            ].every(Number.isFinite),
        )
        .flatMap((c) =>
          sweep.machine === 'sweep'
            ? machineVariants(c)
            : // a global node count overrides every chip's own machine pick
              [typeof sweep.machine === 'number' ? machineAtNodes(c, sweep.machine) : c],
        );
      const specOf = ({ costPerHour: _, ...spec }: UiChip) => JSON.stringify(spec);
      const shared = JSON.stringify([
        model.name,
        workload,
        sweep.memoryOverlap,
        sweep.commsOverlap,
      ]);
      const prev = lastSearched.current;
      const stale = enabled.filter((c) => {
        const key = machineKey(c);
        if (shared !== prev.shared || specOf(c) !== prev.specs.get(key)) return true;
        const g = groupsRef.current.get(key);
        return !g || (!g.error && g.done < g.total);
      });
      lastSearched.current = {
        shared,
        specs: new Map(enabled.map((c) => [machineKey(c), specOf(c)])),
      };
      setGroups((gs) => {
        const staleKeys = new Set(stale.map(machineKey));
        return new Map(
          enabled.map((c) => {
            const key = machineKey(c);
            const current = gs.get(key);
            if (current && !staleKeys.has(key)) return [key, current];
            return [
              key,
              {
                key,
                chip: c,
                machineLabel: machineLabel(c),
                nChips: machineChips(c),
                hardware: null,
                configs: [],
                done: 0,
                total: 0,
              },
            ];
          }),
        );
      });
      if (stale.length) {
        const id = clientRef.current!.search({
          modelName: model.name,
          chips: stale,
          workload,
          overlap,
        });
        activeReq.current = id;
      }
    }, 300);
    return () => clearTimeout(t);
  }, [model, workload, sweep, chips]);

  // ×HMVP efficiencies are derived here from the live prices and the baseline
  // (not in the worker), so price edits reprice existing rows without a search
  const groupList = useMemo(() => {
    const order = new Map(chips.map((c, i) => [c.id, i]));
    const basePf = baseline?.prefillTokPerSecPerChip;
    const baseDec = baseline?.decodeTokPerSecPerChip;
    const priced = <S extends { tokPerSecPerChip: number }>(
      s: S | undefined,
      chip: ChipSpec,
      baseRate: number | undefined,
    ) => {
      const rel = relPriceOf(chip);
      return (
        s && {
          ...s,
          eff:
            rel !== undefined && baseRate ? hmvpEff(s.tokPerSecPerChip, rel, baseRate) : undefined,
          relRate: baseRate ? s.tokPerSecPerChip / baseRate : undefined,
        }
      );
    };
    // relative cost of one request (T prefill + S decode tokens), HMVP's
    // cheapest-request config ÷ this config — its own reference, separate
    // from the per-phase maxima the phase columns normalize by
    const baseReq = baseline?.requestChipSeconds;
    const requestEff = (r: UiResult, chip: ChipSpec): number | undefined => {
      const rel = relPriceOf(chip);
      if (rel === undefined || baseReq === undefined || !r.prefill || !r.decode) return undefined;
      const { prefillLen: T, generateLen: S } = r.workload;
      const secs = requestChipSecs(T, S, r.prefill.tokPerSecPerChip, r.decode.tokPerSecPerChip);
      return baseReq / (rel * secs);
    };
    // the reference machine always sits on the board: when the sweep doesn't
    // already include the H100 at the MVP size, seat it from the hidden
    // baseline search (its key comes from machineKey, so it collides — and
    // dedupes — exactly with the real group's)
    const h100 = chipsById[H100_ID];
    const seated: UiGroup[] =
      h100 &&
      baseline &&
      !baseline.exhausted &&
      (baseline.configs?.length ?? 0) > 0 &&
      !groups.has(baseline.key)
        ? [
            {
              key: baseline.key,
              chip: h100,
              machineLabel: baseline.machineLabel,
              nChips: baseline.nChips,
              hardware: baseline.hardware ?? null,
              configs: baseline.configs!,
              done: baseline.done,
              total: baseline.total,
            },
          ]
        : [];
    return [...groups.values(), ...seated]
      .map((g) => {
        const chip = chipsById[g.chip.id] ?? g.chip;
        return {
          ...g,
          chip,
          configs: g.configs.map((r) => ({
            ...r,
            prefill: priced(r.prefill, chip, basePf),
            decode: priced(r.decode, chip, baseDec),
            requestEff: requestEff(r, chip),
          })),
        };
      })
      .sort(
        (a, b) => (order.get(a.chip.id) ?? 0) - (order.get(b.chip.id) ?? 0) || a.nChips - b.nChips,
      );
  }, [groups, chips, baseline]);

  const visibleGroups = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    const slaActive = Object.keys(sla).length > 0;
    return groupList
      .filter((g) => !q || g.chip.name.toLowerCase().includes(q))
      .map((g) => (slaActive ? { ...g, configs: g.configs.filter((r) => passesSla(r, sla)) } : g));
  }, [groupList, filterText, sla]);

  const setSlaThreshold = (metricKey: string, v: number | undefined) =>
    setSla((s) => {
      const next = { ...s };
      if (v === undefined) delete next[metricKey];
      else next[metricKey] = v;
      return next;
    });

  const allResults = useMemo(() => visibleGroups.flatMap((g) => g.configs), [visibleGroups]);

  const selected = useMemo(() => {
    if (!selectedId) return null;
    for (const g of visibleGroups) {
      const r = g.configs.find((c) => c.id === selectedId);
      if (r) return { result: r, group: g };
    }
    return null;
  }, [visibleGroups, selectedId]);

  // errored chips never price a tuple, so they'd peg the ring below 100%
  // forever — leave them out of the progress denominator entirely
  const progress = groupList.reduce(
    (acc, g) => (g.error ? acc : { done: acc.done + g.done, total: acc.total + g.total }),
    { done: 0, total: 0 },
  );
  const globalTone = ringTone(progress.total > 0 ? progress.done / progress.total : 0);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'));

  return (
    <div className="app">
      <PriceNotice />
      <header className="topbar">
        <SailResearchLogo />
        <div className="topbar-actions">
          <button
            className="theme-toggle"
            type="button"
            aria-label={theme === 'dark' ? 'Use light mode' : 'Use dark mode'}
            aria-pressed={theme === 'dark'}
            onClick={toggleTheme}
          >
            <span className="theme-toggle-track" aria-hidden="true">
              <span className="theme-toggle-thumb">
                {theme === 'dark' ? (
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                    <path d="M12.04 3.5a1 1 0 0 1 .95.69l.43 1.31 1.31.43a1 1 0 0 1 0 1.9l-1.31.43-.43 1.31a1 1 0 0 1-1.9 0l-.43-1.31-1.31-.43a1 1 0 0 1 0-1.9l1.31-.43.43-1.31a1 1 0 0 1 .95-.69Zm5.29 7.17a1 1 0 0 1 .95.69l.25.76.76.25a1 1 0 0 1 0 1.9l-.76.25-.25.76a1 1 0 0 1-1.9 0l-.25-.76-.76-.25a1 1 0 0 1 0-1.9l.76-.25.25-.76a1 1 0 0 1 .95-.69ZM8.44 12.9a6.55 6.55 0 0 0 7.66 7.66 8 8 0 1 1-7.66-7.66Z" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                    <path d="M12 4a1 1 0 0 1 1 1v1.2a1 1 0 1 1-2 0V5a1 1 0 0 1 1-1Zm0 13.8a1 1 0 0 1 1 1V20a1 1 0 1 1-2 0v-1.2a1 1 0 0 1 1-1ZM4 12a1 1 0 0 1 1-1h1.2a1 1 0 1 1 0 2H5a1 1 0 0 1-1-1Zm13.8 0a1 1 0 0 1 1-1H20a1 1 0 1 1 0 2h-1.2a1 1 0 0 1-1-1Zm-10.78-4.98a1 1 0 0 1 1.42 0l.85.85a1 1 0 0 1-1.42 1.42l-.85-.85a1 1 0 0 1 0-1.42Zm7.69 7.69a1 1 0 0 1 1.42 0l.85.85a1 1 0 0 1-1.42 1.42l-.85-.85a1 1 0 0 1 0-1.42Zm2.27-7.69a1 1 0 0 1 0 1.42l-.85.85a1 1 0 0 1-1.42-1.42l.85-.85a1 1 0 0 1 1.42 0ZM9.29 14.71a1 1 0 0 1 0 1.42l-.85.85a1 1 0 0 1-1.42-1.42l.85-.85a1 1 0 0 1 1.42 0ZM12 8.5a3.5 3.5 0 1 1 0 7 3.5 3.5 0 0 1 0-7Z" />
                  </svg>
                )}
              </span>
            </span>
          </button>
        </div>
      </header>
      <div className="body">
        <Sidebar
          model={model}
          onModel={setModel}
          workload={workload}
          onWorkload={setWorkload}
          sweep={sweep}
          onSweep={setSweep}
          chips={chips}
          onChips={setChips}
        />
        <main className="main">
          <div className="toolbar">
            <div className="seg" role="tablist" aria-label="View">
              <button
                role="tab"
                aria-selected={view === 'table'}
                className={view === 'table' ? 'on' : ''}
                onClick={() => setView('table')}
              >
                Table
              </button>
              <button
                role="tab"
                aria-selected={view === 'chart'}
                className={view === 'chart' ? 'on' : ''}
                onClick={() => {
                  setView('chart');
                  setSelectedId(null);
                }}
              >
                Chart
              </button>
            </div>
            <input
              className="search"
              type="search"
              placeholder="Filter by chip…"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
            />
            <button
              className={`sla-btn ${slaOpen || Object.keys(sla).length > 0 ? 'on' : ''}`}
              aria-expanded={slaOpen}
              onClick={() => setSlaOpen((o) => !o)}
            >
              SLA filters
              {Object.keys(sla).length > 0 && (
                <span className="sla-badge">{Object.keys(sla).length}</span>
              )}
            </button>
            <div className="progress">
              <span className="ring-slot">
                <ProgressRing
                  frac={progress.total > 0 ? progress.done / progress.total : 0}
                  color={globalTone}
                />
              </span>
              <span>
                {progress.done}/{progress.total}
              </span>
            </div>
          </div>
          {slaOpen && (
            <SlaFilterPanel
              groups={groupList}
              thresholds={sla}
              onChange={setSlaThreshold}
              onReset={() => setSla({})}
            />
          )}
          <div className="content">
            {view === 'table' ? (
              <Leaderboard
                groups={visibleGroups}
                baseline={baseline}
                selectedId={selectedId}
                onSelect={setSelectedId}
                hoveredId={hoveredId}
                onHover={setHoveredId}
              />
            ) : (
              <TradeoffChart
                results={allResults.filter((r) => !hasError(r))}
                chipsById={chipsById}
                hoveredId={hoveredId}
                onHover={setHoveredId}
                onSelect={setSelectedId}
              />
            )}
            {selected && (
              <DetailsPanel
                result={selected.result}
                group={selected.group}
                model={model}
                overlap={overlap}
                onClose={() => setSelectedId(null)}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
