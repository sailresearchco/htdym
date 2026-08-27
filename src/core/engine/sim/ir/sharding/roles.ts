// Parallelism roles: per-phase labels for why a tensor is laid out the
// way it is, consumed by lowering and never present in the IR. The
// attention plane cuts heads x the batch (TP x DPA), the expert plane
// cuts the expert set x each expert's width (EP x ETP), and PP carves
// the machine into pipeline stages.
// TP: attention-head and non-routed-weight cut within a batch group
// (Megatron-style), <= N query heads.
// DPA: batch cut, groups own whole sequences and their KV outright
// (attention DP). MoE only when > 1, for dense models it is plain
// data parallelism.
// EP: whole-expert cut of the routed expert set, <= E.
// ETP: cut of each expert's d_ff (TRT-LLM's moe_tp).
// PP: pipeline-parallel degree (stage count), <= layers.
// Engine flags: vLLM -tp tp -dp dpa [--enable-expert-parallel], SGLang
// --tp world --dp dpa --enable-dp-attention.
export type ShardingRole = 'DPA' | 'TP' | 'EP' | 'ETP' | 'PP';

// Each plane is one whole factorization of the stage mesh: PP plus one
// plane's roles partition the machine, and the boundary between planes
// is where reshard communication is paid.
export const PLANES: readonly (readonly ShardingRole[])[] = [
  ['DPA', 'TP'],
  ['EP', 'ETP'],
];

export const ROLES: readonly ShardingRole[] = ['PP', ...PLANES.flat()];
