import type { HardwareResource } from '../../surface/api';
import type { Deployment } from '../../surface/deploy';
import type { CollectiveKind, OpId, Segment } from '../ir/ops';
import type { MeshAxis, TensorType } from '../ir/tensors';

export interface TraceCost {
  time: number;
  // Optional attribution of the trace's work to the ops that caused it, in
  // seconds each op holds each resource, already multiplied by segment
  // repeats. It is a share of the work, not a share of `time`: overlap is a
  // property of the whole trace, so these sum to the fully-sequential total
  // instead. A backend omits it when it cannot answer per op (e.g. a
  // scheduling backend charges an op differently depending on what it ran
  // alongside, so asking what one op costs in isolation has no single answer
  // there).
  busyPerOp?: ReadonlyMap<OpId, Record<HardwareResource, number>>;
}

// A backend bound to one evaluation's context. priceCollective()
// is used by the reshard expander to price candidate collectives.
export interface BoundBackend {
  // priceCollective's identity, computed once at bind: two binds with an
  // equal hash return identical collective prices, so reshard-plan
  // caches are shared across them. Prefix with the backend's name so
  // different backends never collide. A backend whose collective prices
  // are not a pure function of the context (e.g. a stateful simulator)
  // must mint a unique hash per bind - its plans then never share, at
  // the cost of cache entries living until the process ends.
  priceCollectiveHash: string;
  priceTrace(trace: Segment[]): TraceCost;
  priceCollective(
    kind: Exclude<CollectiveKind, 'p2p'>,
    over: MeshAxis[],
    input: TensorType,
    elemBytes: number,
  ): number;
}

// A backend owns all physics: bind it to a deployment to price anything.
export type CostBackend = (deployment: Deployment) => BoundBackend;

// The price a given backend's priceTrace returns: the backend-specific
// cost breakdown of one stage trace.
export type TracePriceOf<TBackend extends CostBackend> = ReturnType<
  ReturnType<TBackend>['priceTrace']
>;
