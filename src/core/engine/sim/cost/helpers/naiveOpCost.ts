import { collectiveCost } from './collectives';
import { ExpandedOp } from '../../ir/ops';
import { localElems, shardWays } from '../../ir/tensors';
import { peakFlops } from '../../../../hardware/chips';
import type { MmaShape } from '../../../../hardware/mma';
import type { HardwareResource } from '../../../surface/api';
import type { Deployment } from '../../../surface/deploy';
import { DTYPE_BYTES, type Dtype } from '../../../../model/dtype';

export type OpCost = Record<HardwareResource, number>;

// Padded work divided by the path's relative rate, expressed as equivalent FLOPs
// at full rate. The caller divides by the chip's sustained FLOP/s to get seconds.
function adjustedFlops(
  shapes: readonly MmaShape[],
  m: number,
  n: number,
  k: number,
  groups = 1,
): number {
  if (m <= 0 || n <= 0 || k <= 0) return 0;
  // Unit tiles disable padding, including for fractional analytical shapes.
  const pad = (d: number, tile: number) => (tile <= 1 ? d : tile * Math.ceil(d / tile));
  return Math.min(
    ...shapes.map((s) => {
      // M counts total rows across groups. Round that total, then require at least
      // one tile per active group. Individual group sizes are unknown, so this
      // can miss partial-tile waste. One group is identical to a dense GEMM.
      const rows = s.m <= 1 ? m : Math.max(pad(m, s.m), groups * s.m);
      return (2 * rows * pad(n, s.n) * pad(k, s.k)) / s.rate;
    }),
  );
}

export function naiveOpCost(op: ExpandedOp, deployment: Deployment): OpCost {
  const { chip, mesh } = deployment;
  const { dims } = mesh;

  const zero: OpCost = { compute: 0, memory: 0, comms: 0 };

  // The ! holds because runnableOn resolved every op dtype to a unit this chip has.
  const flopsPerSecond = (d: Dtype) => peakFlops(chip, d)! * chip.realizableFlopsFrac;
  const hbm = chip.hbmBandwidth * chip.realizableHbmBwFrac;

  switch (op.kind) {
    case 'gemm': {
      const [m, k] = op.x.shape;
      const last = op.w.shape.length - 1;
      const mLocal = m / shardWays(op.x.sharding[0], dims);
      const kLocal = k / shardWays(op.x.sharding[1], dims);
      const nLocal = op.w.shape[last] / shardWays(op.w.sharding[last], dims);
      return {
        compute:
          adjustedFlops(chip.mmaShapes[op.dtype], mLocal, nLocal, kLocal, op.groups) /
          flopsPerSecond(op.dtype),
        memory: (mLocal * (kLocal + nLocal) * DTYPE_BYTES[op.dtype]) / hbm,
        comms: 0,
      };
    }
    case 'attention':
      return {
        compute: op.flops / flopsPerSecond(op.dtype),
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
