import { BlockGroup, sliceRuns } from '../../../model/block';
import type { ModelSpec } from '../../../model/models';
import { layerCount } from '../../../model/utils';

// One pipeline stage's layer slice, the input to lowerStage.
export interface Stage {
  index: number;
  // the exact layers this stage owns, in order, still grouped
  groups: BlockGroup[];
  // stage 0 additionally embeds, the last stage unembeds
  hasEmbedding: boolean;
  hasUnembedding: boolean;
}

// The layers [start, end) of the grouped stack, regrouped: whole pattern
// repeats stay one group, a cut inside a pattern peels off a partial copy.
function sliceGroups(groups: BlockGroup[], start: number, end: number): BlockGroup[] {
  const out: BlockGroup[] = [];
  let at = 0;
  for (const g of groups) {
    const period = g.pattern.reduce((s, r) => s + r.count, 0);
    const lo = Math.max(start, at);
    const hi = Math.min(end, at + g.repeat * period);
    let [a, b] = [lo - at, hi - at];
    at += g.repeat * period;
    if (a >= b) continue;

    const head = a % period ? Math.min(period - (a % period), b - a) : 0;
    if (head)
      out.push({ pattern: sliceRuns(g.pattern, a % period, (a % period) + head), repeat: 1 });
    a += head;

    const whole = Math.floor((b - a) / period);
    if (whole > 0) out.push({ pattern: g.pattern, repeat: whole });
    a += whole * period;

    if (a < b) out.push({ pattern: sliceRuns(g.pattern, 0, b - a), repeat: 1 });
  }
  return out;
}

const stageMapCache = new WeakMap<ModelSpec, Map<number, Stage[]>>();

// Partition the model into pp pipeline stages (even split, remainder on early stages).
export function partitionIntoStages(m: ModelSpec, pp: number): Stage[] {
  let byPp = stageMapCache.get(m);
  if (!byPp) stageMapCache.set(m, (byPp = new Map()));
  const hit = byPp.get(pp);
  if (hit) return hit;

  const layers = layerCount(m);
  const base = Math.floor(layers / pp);
  const extra = layers % pp;

  const stages: Stage[] = [];
  let at = 0;
  for (let s = 0; s < pp; s++) {
    const take = base + (s < extra ? 1 : 0);
    stages.push({
      index: s,
      groups: sliceGroups(m.blocks, at, at + take),
      hasEmbedding: s === 0,
      hasUnembedding: s === pp - 1,
    });
    at += take;
  }
  byPp.set(pp, stages);
  return stages;
}
