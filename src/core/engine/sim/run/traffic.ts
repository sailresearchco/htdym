import type { MeshDim } from '../../../hardware/topology';
import { localElems } from '../ir/tensors';
import type { Segment } from '../ir/ops';
import { DTYPE_BYTES } from '../../../model/dtype';

// HBM bytes one chip streams per step for the model's state: the weights it
// loads and the KV cache it reads and appends. Activation streams (a GEMM's
// input and output rows) are deliberately left out, so this is the numerator
// of MBU in its usual sense: (weights + KV) / TPOT over peak bandwidth.
export interface HbmTraffic {
  weightBytes: number;
  kvBytes: number;
}

export function hbmTraffic(trace: Segment[], dims: MeshDim[]): HbmTraffic {
  const t: HbmTraffic = { weightBytes: 0, kvBytes: 0 };
  for (const s of trace)
    for (const op of s.ops) {
      if (op.kind === 'weight-load')
        t.weightBytes +=
          s.repeat * localElems(op.out, dims) * DTYPE_BYTES[op.dtype] * op.loadFraction;
      else if (op.kind === 'attention') t.kvBytes += s.repeat * (op.kvReadBytes + op.kvWriteBytes);
    }
  return t;
}
