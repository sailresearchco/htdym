import { collectiveCost } from './collectives';
import { ExpandedOp } from '../../ir/ops';
import { localElems, shardWays } from '../../ir/tensors';
import { DEFAULT_MATMUL_SAT_ROWS, peakFlops } from '../../../../hardware/chips';
import type { HardwareResource } from '../../../surface/api';
import type { Deployment } from '../../../surface/deploy';
import { DTYPE_BYTES, type Dtype } from '../../../../model/dtype';

export type OpCost = Record<HardwareResource, number>;

// Padded-tile utilization: the matmul array's tile is `tile` wide on
// every edge (a 128x128 MXU, 128-row tensor-core macro-tiles), so all
// three GEMM dims pad up to it. A thin N or K shard (deep TP/ETP
// slicing a projection) idles the array's columns or depth exactly like
// a short M idles its rows. Grouped GEMMs pad rows per activated group
// (multinomial token counts smooth the per-group ceiling, keep the
// floor: every activated group pads to one tile).
function tileUtil(m: number, n: number, k: number, groups: number, tile: number): number {
  // tile 1 = no tiling at all (fractional dims, e.g. MLA's equivalent
  // columns, must not round)
  if (tile <= 1 || m <= 0 || n <= 0 || k <= 0) return 1;
  const pad = (d: number) => tile * Math.ceil(d / tile);
  const rows = groups <= 1 ? pad(m) : Math.max(m, groups * tile);
  return (m * n * k) / (rows * pad(n) * pad(k));
}

export function naiveOpCost(op: ExpandedOp, deployment: Deployment): OpCost {
  const { chip, mesh } = deployment;
  const { dims } = mesh;

  const zero: OpCost = { compute: 0, memory: 0, comms: 0 };

  // the ! holds because runnableOn resolved every op dtype to a unit this chip has
  const rate = (d: Dtype) => peakFlops(chip, d)! * chip.realizableFlopsFrac;
  const hbm = chip.hbmBandwidth * chip.realizableHbmBwFrac;

  switch (op.kind) {
    case 'gemm': {
      const [m, k] = op.x.shape;
      const last = op.w.shape.length - 1;
      const mLocal = m / shardWays(op.x.sharding[0], dims);
      const kLocal = k / shardWays(op.x.sharding[1], dims);
      const nLocal = op.w.shape[last] / shardWays(op.w.sharding[last], dims);
      const util = tileUtil(
        mLocal,
        nLocal,
        kLocal,
        op.groups ?? 1,
        chip.matmulSatRows ?? DEFAULT_MATMUL_SAT_ROWS,
      );
      return {
        compute: (2 * mLocal * kLocal * nLocal) / (rate(op.dtype) * util),
        memory: (mLocal * (kLocal + nLocal) * DTYPE_BYTES[op.dtype]) / hbm,
        comms: 0,
      };
    }
    case 'attention':
      return {
        compute: op.flops / rate(op.dtype),
        memory: (op.kvReadBytes + op.kvWriteBytes) / hbm,
        comms: 0,
      };
    case 'weight-load':
      return {
        compute: 0,
        memory: (localElems(op.out, dims) * DTYPE_BYTES[op.dtype] * op.loadFraction) / hbm,
        comms: 0,
      };
    case 'join':
      return zero;
    case 'collective':
      return {
        compute: 0,
        memory: 0,
        comms: collectiveCost(op.variant, op.axes, op.x, DTYPE_BYTES[op.dtype], dims),
      };
    case 'p2p': {
      if (!mesh.roles['PP'].length) throw new Error(`pipeline send without PP dims`);

      return {
        compute: 0,
        memory: 0,
        comms: collectiveCost('p2p', mesh.roles['PP'], op.out, DTYPE_BYTES[op.dtype], dims),
      };
    }
  }
}
