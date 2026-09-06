import type { MmaShape, MmaShapes } from '../src/core/hardware/mma';
import { DTYPES } from '../src/core/model/dtype';

// Conservation tests count useful FLOPs without padding or instruction-rate losses.
export const IDEAL_MMA = Object.fromEntries<readonly MmaShape[]>(
  DTYPES.map((dtype) => [dtype, [{ m: 1, n: 1, k: 1, rate: 1 }]]),
) as MmaShapes;
