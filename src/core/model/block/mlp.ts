// 'gated' (SwiGLU etc., 3DF params) or 'standard' (2DF params).
export type MlpVariant = 'gated' | 'standard';

export type MoeMlpConfig = {
  kind: 'moe';
  experts: number;
  topK: number;
  expertDim: number;
  sharedExperts: number;
  variant: MlpVariant;
  // width the routed experts run at, when narrower than the residual: a
  // per-layer projection pair brackets them and the dispatch ships the
  // narrower tokens. Absent = experts run at modelDim.
  latentDim?: number;
};

export type DenseMlpConfig = { kind: 'dense'; ffDim: number; variant: MlpVariant };

export type MlpConfig = DenseMlpConfig | MoeMlpConfig;

export function denseMlp(ffDim: number, variant: MlpVariant = 'gated'): MlpConfig {
  return { kind: 'dense', ffDim, variant };
}

export function moeMlp(opts: {
  experts: number;
  topK: number;
  expertDim: number;
  sharedExperts?: number;
  variant?: MlpVariant;
  latentDim?: number;
}): MlpConfig {
  return {
    kind: 'moe',
    experts: opts.experts,
    topK: opts.topK,
    expertDim: opts.expertDim,
    sharedExperts: opts.sharedExperts ?? 0,
    variant: opts.variant ?? 'gated',
    latentDim: opts.latentDim,
  };
}
