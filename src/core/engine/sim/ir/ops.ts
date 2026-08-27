import type { AttnConfig } from '../../../model/block/attn';
import { MeshAxis, TensorType } from './tensors';
import type { Dtype } from '../../../model/dtype';

export type CollectiveKind = 'all-reduce' | 'all-gather' | 'reduce-scatter' | 'all-to-all' | 'p2p';

// Opaque DAG node id minted by the lowerer, not a label or other string.
export type OpId = string & { readonly __opId: unique symbol };

interface OpBase {
  id: OpId;
  // human-readable, e.g. "qkv gemm"
  label: string;
  // ids of ops that must finish first
  deps: OpId[];
  // sharded type of this op's output
  out: TensorType;
}

export interface GemmOp extends OpBase {
  kind: 'gemm';
  // input rows type [m, k]
  x: TensorType;
  // weights type [k, n], or [groups, k, n] for a grouped GEMM
  w: TensorType;
  // format this matmul's operands arrive in, which picks the chip's peak for
  // it and sizes the rows it streams
  dtype: Dtype;
  // independent identical matmuls batched in one launch (grouped GEMM:
  // activated experts). Tile padding is per group. FLOPs already count all
  // m rows, so groups does not multiply them.
  groups?: number;
}

export interface AttentionOp extends OpBase {
  kind: 'attention';
  // model-level attention family used by this operation
  variant: AttnConfig['kind'];
  // format the dot-product runs in, picking the chip's peak for it
  dtype: Dtype;
  // per-chip dot-product FLOPs (hand-priced: the sharded types govern
  // the projections around attention, not its internals)
  flops: number;
  // per-chip KV bytes read this step
  kvReadBytes: number;
  // per-chip KV bytes appended this step
  kvWriteBytes: number;
}

export interface WeightLoadOp extends OpBase {
  kind: 'weight-load';
  // format the checkpoint is held in here, which prices the bytes it streams
  dtype: Dtype;
  // fraction streamed this step (activated experts < 1)
  loadFraction: number;
}

// Dependency-only convergence point for branches whose outputs are combined.
export interface JoinOp extends OpBase {
  kind: 'join';
}

// Abstract "get from x's layout to out's layout". Exists only before
// reshard expansion, which replaces it with concrete collectives.
export interface ReshardOp extends OpBase {
  kind: 'reshard';
  // layout the reshard consumes
  x: TensorType;
  // format the moved value is on the wire in
  dtype: Dtype;
}

// One concrete collective, emitted by reshard expansion.
export interface CollectiveOp extends OpBase {
  kind: 'collective';
  variant: Exclude<CollectiveKind, 'p2p'>;
  axes: MeshAxis[];
  // layout the collective consumes (its payload derives from this)
  x: TensorType;
  // format the payload is on the wire in
  dtype: Dtype;
}

export interface P2pOp extends OpBase {
  kind: 'p2p';
  // format the stage boundary's activations are sent in
  dtype: Dtype;
}

// One (subgraph, repeat) unit of a stage trace: the ops run
// sequentially repeat times (a repeated layer is one segment).
export interface Segment<Op = ExpandedOp> {
  // human-readable, e.g. "mla+moe"
  label: string;
  ops: Op[];
  repeat: number;
}

// What lowering emits, reshards still abstract.
export type LoweredOp =
  GemmOp | AttentionOp | WeightLoadOp | JoinOp | ReshardOp | CollectiveOp | P2pOp;

// What reshard expansion emits and cost backends consume.
export type ExpandedOp = Exclude<LoweredOp, ReshardOp>;

// An op minus its DAG identity, before the builder or expander wires it in.
export type Unbuilt<T> = T extends LoweredOp ? Omit<T, 'id' | 'deps'> : never;
