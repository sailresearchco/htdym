import type { Roofline } from '../core/engine/roofline';
import type { UiChip, UiOverlap, UiResult, UiWorkload } from './results';

export interface SearchRequest {
  id: number;
  modelName: string;
  // the app's edited copies of the enabled chips, one entry per machine
  chips: UiChip[];
  workload: UiWorkload;
  overlap: UiOverlap;
}

// `key` is the chip-on-a-machine the update belongs to (machineKey)
export type SearchUpdate = { id: number; key: string } & (
  | {
      // a machine's slot, posted upfront with its search size (`total` tuples)
      kind: 'group';
      chipId: string;
      machineLabel: string;
      nChips: number;
      hardware: Roofline | null;
      total: number;
      error?: string;
    }
  | {
      // one priced tuple of a machine's search (row absent = infeasible tuple)
      kind: 'row';
      done: number;
      row?: UiResult;
    }
);

// Each chip gets its own pinned worker (spawned lazily), so searches run
// in parallel and a chip's warm plan caches stay with the worker that will
// be asked about it again. A request fans out one message per chip; each
// worker abandons its own older requests. Bursts from many workers are
// buffered and flushed in one task so React coalesces them into one render.
export function makeSearchClient(onUpdate: (u: SearchUpdate) => void) {
  const workers = new Map<string, Worker>();
  const buffer: SearchUpdate[] = [];
  let flush = 0;
  const onMessage = (e: MessageEvent<SearchUpdate>) => {
    buffer.push(e.data);
    if (!flush)
      flush = setTimeout(() => {
        flush = 0;
        for (const u of buffer.splice(0)) onUpdate(u);
      }, 50);
  };
  const workerFor = (chipId: string) => {
    let w = workers.get(chipId);
    if (!w) {
      w = new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' });
      w.onmessage = onMessage;
      workers.set(chipId, w);
    }
    return w;
  };
  let seq = 0;
  return {
    search(req: Omit<SearchRequest, 'id'>): number {
      const id = ++seq;
      // a chip's machines share its worker (and its warm plan caches), so
      // they go out as one message and are searched one after another
      const byChip = new Map<string, UiChip[]>();
      for (const chip of req.chips) byChip.set(chip.id, [...(byChip.get(chip.id) ?? []), chip]);
      for (const [chipId, chips] of byChip)
        workerFor(chipId).postMessage({ ...req, id, chips } satisfies SearchRequest);
      return id;
    },
    dispose: () => {
      clearTimeout(flush);
      workers.forEach((w) => w.terminate());
    },
  };
}
