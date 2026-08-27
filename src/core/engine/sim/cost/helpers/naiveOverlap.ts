import type { HardwareResource } from '../../../surface/api';

export interface NaiveOverlapBreakdown {
  // additive visible parts, they sum to the combined time
  parts: Record<HardwareResource, number>;
  // the overlapped pool: the stream credited with it and its wall-clock width
  pool: { by: HardwareResource; time: number };
  // work hidden behind the pool: concurrent, adds no wall-clock time,
  // and each entry is <= pool.time by construction
  hidden: Record<HardwareResource, number>;
}

export function naiveOverlapBreakdown(
  c: Record<HardwareResource, number>,
  a: { memoryOverlap: number; commsOverlap: number },
): NaiveOverlapBreakdown {
  const mem = a.memoryOverlap * c.memory;
  const comms = a.commsOverlap * c.comms;
  const parts: Record<HardwareResource, number> = {
    compute: 0,
    memory: (1 - a.memoryOverlap) * c.memory,
    comms: (1 - a.commsOverlap) * c.comms,
  };
  const by: HardwareResource =
    c.compute >= mem && c.compute >= comms ? 'compute' : mem >= comms ? 'memory' : 'comms';
  const pool = { by, time: Math.max(c.compute, mem, comms) };
  parts[by] += pool.time;
  const hidden: Record<HardwareResource, number> = {
    compute: by === 'compute' ? 0 : c.compute,
    memory: by === 'memory' ? 0 : mem,
    comms: by === 'comms' ? 0 : comms,
  };
  return { parts, pool, hidden };
}
