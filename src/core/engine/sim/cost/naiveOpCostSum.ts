import { collectiveCost } from './helpers/collectives';
import { naiveOpCost } from './helpers/naiveOpCost';
import type { Deployment } from '../../surface/deploy';
import { naiveOverlapBreakdown } from './helpers/naiveOverlap';

import type { OpId, Segment } from '../ir/ops';
import type { HardwareResource } from '../../surface/api';
import type { CostBackend, TraceCost } from './types';

export type { CostBackend, TraceCost } from './types';

export interface NaiveOpCostSumTraceCost extends TraceCost {
  // per-resource busy sums before overlap
  busy: Record<HardwareResource, number>;
  // per-resource cost sums after overlap
  parts: Record<HardwareResource, number>;
}

export function makeNaiveOpCostSumBackend(options: {
  memoryOverlap: number;
  commsOverlap: number;
}) {
  return ((deployment: Deployment) => {
    return {
      // everything priceCollective reads. two bound backend instances agreeing here price
      // collectives identically and thus we can share cached reshard plans between them
      priceCollectiveHash: JSON.stringify(['naive-op-cost-sum', deployment.mesh.dims]),
      priceCollective: (kind, over, input, elemBytes) =>
        collectiveCost(kind, over, input, elemBytes, deployment.mesh.dims),
      priceTrace: (trace: Segment[]): NaiveOpCostSumTraceCost => {
        const busy: Record<HardwareResource, number> = { compute: 0, memory: 0, comms: 0 };
        const busyPerOp = new Map<OpId, Record<HardwareResource, number>>();
        for (const s of trace)
          for (const op of s.ops) {
            const c = naiveOpCost(op, deployment);
            busy.compute += c.compute * s.repeat;
            busy.memory += c.memory * s.repeat;
            busy.comms += c.comms * s.repeat;
            busyPerOp.set(op.id, {
              compute: c.compute * s.repeat,
              memory: c.memory * s.repeat,
              comms: c.comms * s.repeat,
            });
          }

        const overlap = naiveOverlapBreakdown(busy, options);
        return {
          ...overlap,
          time: overlap.parts.compute + overlap.parts.memory + overlap.parts.comms,
          busy,
          busyPerOp,
        };
      },
    };
  }) satisfies CostBackend;
}
