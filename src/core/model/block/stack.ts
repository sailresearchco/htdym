import type { AttnConfig } from './attn';
import { BlockGroup, BlockRun, sliceRuns } from './index';
import type { MlpConfig } from './mlp';

export function repeat(n: number, attn: AttnConfig, mlp: MlpConfig): BlockGroup[] {
  return n > 0 ? [{ pattern: [{ block: { attn, mlp }, count: n }], repeat: 1 }] : [];
}

// Dense-MLP prefix then MoE for the remaining layers (GLM/Kimi layout).
export function denseThenMoe(
  total: number,
  denseCount: number,
  attn: AttnConfig,
  dense: MlpConfig,
  moe: MlpConfig,
): BlockGroup[] {
  return [...repeat(denseCount, attn, dense), ...repeat(total - denseCount, attn, moe)];
}

// Hybrid attention stacks (Gemma/gpt-oss SWA, Qwen linear:full): repeating
// localPerPattern local layers then globalPerPattern full-attention layers
// (globals last in each repetition).
export function swaStack(opts: {
  layers: number;
  localPerPattern: number;
  globalPerPattern: number;
  local: AttnConfig;
  global: AttnConfig;
  mlp: MlpConfig;
}): BlockGroup[] {
  const pattern: BlockRun[] = [
    { block: { attn: opts.local, mlp: opts.mlp }, count: opts.localPerPattern },
    { block: { attn: opts.global, mlp: opts.mlp }, count: opts.globalPerPattern },
  ].filter((r) => r.count > 0);
  const period = opts.localPerPattern + opts.globalPerPattern;

  const out: BlockGroup[] = [];
  const full = Math.floor(opts.layers / period);
  const rem = opts.layers % period;
  if (full) out.push({ pattern, repeat: full });
  if (rem) out.push({ pattern: sliceRuns(pattern, 0, rem), repeat: 1 }); // stack ends mid-pattern
  return out;
}
