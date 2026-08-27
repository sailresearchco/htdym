/**
 * Price policy and the ×HMVP reference baseline.
 *
 * The tool never shows a dollar: chip prices display relative to the H100,
 * and the cost metrics quote "×HMVP" efficiency — throughput per unit of
 * relative cost, as a multiple of the same figure for the Hopper MVP, the
 * smallest H100 machine that can serve the model. The cost a chip is charged
 * is either its rental price or its board power, both relative to the H100.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { roofline, Roofline } from '../core/engine/roofline';
import { CHIPS_BY_ID } from '../core/hardware/chips';
import { ModelSpec } from '../core/model/models';
import { machineChips, machineKey, machineLabel, maxNodesOf } from './machines';
import { hasError, UiChip, UiOverlap, UiResult, UiWorkload } from './results';
import { makeSearchClient } from './searchClient';

export const H100_ID = 'h100-sxm';

/** What the efficiency metrics divide throughput by: rental price or board
 * power, each relative to the H100. */
export type CostBasis = 'price' | 'power';

// The anchor every relative price divides by: the H100's default list $/hr.
// The live-edited H100 never moves it, so ratios stay stable under edits.
const H100_DOLLARS_PER_HOUR = CHIPS_BY_ID[H100_ID].costPerHour!;

// The per-kW view's anchor: the H100's board power.
const H100_TDP = CHIPS_BY_ID[H100_ID].tdp!;

/** A chip's hourly price as a multiple of the H100's — the only form of
 * price the public build displays. */
export function relPriceOf(chip: { costPerHour?: number }): number | undefined {
  return chip.costPerHour !== undefined ? chip.costPerHour / H100_DOLLARS_PER_HOUR : undefined;
}

/** A chip's relative cost on the active basis: ×H100 price or ×H100 power. */
export function relCostOf(
  chip: { costPerHour?: number; tdp?: number },
  basis: CostBasis,
): number | undefined {
  if (basis === 'power') return chip.tdp !== undefined ? chip.tdp / H100_TDP : undefined;
  return relPriceOf(chip);
}

/** Inverse of relPriceOf, for the sidebar's relative-price input. */
export function relPriceToDollars(rel: number): number {
  return rel * H100_DOLLARS_PER_HOUR;
}

/** The ×HMVP quote: throughput per unit of relative cost (price or power),
 * as a multiple of the baseline rate. The one definition every eff display
 * shares. */
export function hmvpEff(rate: number, relCost: number, baseRate: number): number {
  return rate / relCost / baseRate;
}

/** Chip-seconds to serve one request of T prefill + S decode tokens. */
export function requestChipSecs(T: number, S: number, pfRate: number, decRate: number): number {
  return T / pfRate + S / decRate;
}

// The ×HMVP reference, streamed from a hidden search of the smallest viable
// H100 machine. Each ×HMVP metric normalizes by the best the machine can do
// ON THAT METRIC — best prefill rate, best decode rate, cheapest whole
// request — which are typically three different configs. So the HMVP machine
// never displays above 1× and tops out at exactly 1× under any sort, at the
// price that one row's request eff is not exactly the blend of its phase
// effs (no single config achieves all three bests).
export interface HmvpBaseline {
  // the machine's identity and label, in the same form the main sweep's
  // groups use (machineKey/machineLabel), so app and leaderboard recognize
  // "the HMVP row" by comparing keys instead of re-deriving it
  key: string;
  machineLabel: string;
  nodes: number;
  nChips: number;
  // streaming progress of the hidden search
  done: number;
  total: number;
  // per-phase maxima across the machine's feasible configs, and which
  // config (by row id) set each — the ×HMVP references and their badges
  prefillTokPerSecPerChip?: number;
  decodeTokPerSecPerChip?: number;
  prefillAnchorId?: string;
  decodeAnchorId?: string;
  // chip-seconds for the cheapest whole request, minimized over configs —
  // the request-efficiency reference — and the config that set it
  requestChipSeconds?: number;
  requestAnchorId?: string;
  // the machine's streamed sweep, so the app can seat the HMVP on the
  // leaderboard as a real group even when the sweep doesn't include it
  hardware?: Roofline | null;
  configs?: UiResult[];
  // no H100 machine up to the node cap can serve this model/workload
  exhausted?: boolean;
}

/**
 * Search the Hopper MVP in the background: start at the smallest H100 node
 * count whose HBM could hold the weights plus one sequence of KV, and bump
 * up whenever a machine finishes with no feasible config. Runs regardless
 * of whether the H100 is in the sweep, off the app's live-edited H100 spec.
 */
export function useHmvpBaseline(
  model: ModelSpec,
  workload: UiWorkload,
  overlap: UiOverlap,
  h100: UiChip | undefined,
): HmvpBaseline | null {
  const [baseline, setBaseline] = useState<HmvpBaseline | null>(null);
  const clientRef = useRef<ReturnType<typeof makeSearchClient> | null>(null);
  // the running hidden search: request id, accumulating baseline, and how
  // many feasible rows it has produced (0 at completion = bump the size)
  const run = useRef<{ id: number; acc: HmvpBaseline | null; feasible: number }>({
    id: 0,
    acc: null,
    feasible: 0,
  });
  const inputs = useRef({ model, workload, overlap, h100 });
  inputs.current = { model, workload, overlap, h100 };

  // reads only refs and stable setters, so every render's instance is
  // interchangeable — safe to call from the mount effect's closure
  const searchAt = (nodes: number) => {
    const { model, workload, overlap, h100 } = inputs.current;
    if (!h100 || !clientRef.current) return;
    const chip = { ...h100, nodes };
    run.current = {
      id: clientRef.current.search({ modelName: model.name, chips: [chip], workload, overlap }),
      acc: {
        key: machineKey(chip),
        machineLabel: machineLabel(chip),
        nodes,
        nChips: machineChips(chip),
        done: 0,
        total: 0,
        hardware: null,
        configs: [],
      },
      feasible: 0,
    };
    setBaseline(run.current.acc);
  };

  useEffect(() => {
    const client = makeSearchClient((u) => {
      const r = run.current;
      if (u.id !== r.id || !r.acc) return;
      const acc = { ...r.acc };
      if (u.kind === 'group') {
        if (u.error) acc.exhausted = true;
        else {
          acc.total = u.total;
          acc.hardware = u.hardware;
        }
      } else {
        acc.done = Math.max(acc.done, u.done);
        if (u.row) acc.configs = [...(acc.configs ?? []), u.row];
        const row = u.row && !hasError(u.row) ? u.row : undefined;
        if (row) {
          r.feasible++;
          if (row.prefill && row.prefill.tokPerSecPerChip > (acc.prefillTokPerSecPerChip ?? 0)) {
            acc.prefillTokPerSecPerChip = row.prefill.tokPerSecPerChip;
            acc.prefillAnchorId = row.id;
          }
          if (row.decode && row.decode.tokPerSecPerChip > (acc.decodeTokPerSecPerChip ?? 0)) {
            acc.decodeTokPerSecPerChip = row.decode.tokPerSecPerChip;
            acc.decodeAnchorId = row.id;
          }
          if (row.prefill && row.decode) {
            const { prefillLen: T, generateLen: S } = inputs.current.workload;
            const secs = requestChipSecs(
              T,
              S,
              row.prefill.tokPerSecPerChip,
              row.decode.tokPerSecPerChip,
            );
            if (secs < (acc.requestChipSeconds ?? Infinity)) {
              acc.requestChipSeconds = secs;
              acc.requestAnchorId = row.id;
            }
          }
        }
        // finished with nothing feasible: this machine can't serve the
        // model, the MVP is the next size up (if the fabric offers one)
        if (acc.done >= acc.total && acc.total > 0 && r.feasible === 0) {
          const chip = inputs.current.h100;
          if (chip && acc.nodes < maxNodesOf(chip)) {
            searchAt(acc.nodes + 1);
            return;
          }
          acc.exhausted = true;
        }
      }
      r.acc = acc;
      setBaseline(acc);
    });
    clientRef.current = client;
    return () => client.dispose();
  }, []);

  // Re-search when anything the baseline depends on changes, on the same
  // debounce as the main sweep. Price and machine picks are excluded: price
  // is display-only, and the baseline always sizes its own machine.
  const key = useMemo(() => {
    if (!h100) return '';
    const { costPerHour: _p, slice: _s, nodes: _n, ...spec } = h100;
    return JSON.stringify([model.name, workload, overlap, spec]);
  }, [model, workload, overlap, h100]);

  useEffect(() => {
    if (!key) {
      run.current = { id: 0, acc: null, feasible: 0 };
      setBaseline(null);
      return;
    }
    const t = setTimeout(() => {
      const { model, workload, h100 } = inputs.current;
      if (![workload.prefillLen, workload.generateLen].every((v) => Number.isFinite(v) && v > 0))
        return;
      const hw = roofline(model, h100!, workload);
      if (!hw) {
        run.current = { id: 0, acc: null, feasible: 0 };
        setBaseline({
          key: '',
          machineLabel: '',
          nodes: 0,
          nChips: 0,
          done: 0,
          total: 0,
          exhausted: true,
        });
        return;
      }
      const chips = Math.ceil((hw.weightBytesTotal + hw.kvBytesPerSeq) / h100!.hbmCapacity);
      const perNode = h100!.interconnect.domainSize;
      searchAt(Math.min(maxNodesOf(h100!), Math.max(1, Math.ceil(chips / perNode))));
    }, 300);
    return () => clearTimeout(t);
  }, [key]);

  return baseline;
}
