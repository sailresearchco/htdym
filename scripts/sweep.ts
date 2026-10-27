import { searchTuples } from '../src/core/engine/optimizer/search';
import { roleSize, type MoeDispatch } from '../src/core/engine/surface/deploy';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import { evaluateServingStep, type ServingOptions } from '../src/core/engine/sim/run/serving';
import { ROLES } from '../src/core/engine/sim/ir/sharding/roles';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';

// The serving sweep of rank.ts on one machine, every placement of every
// tuple: one line per tuple (its best placement and dispatch), then the
// winner and the wall-clock, time and peak heap.
//   npm exec vite-node scripts/sweep.ts -- <chip> <domain>x<nodes> [P G hit slo skew]
//   npm exec vite-node scripts/sweep.ts -- b300 8x4
const [chipId = 'b300', machineArg = '8x1', ...rest] = process.argv.slice(2);
const [P, G, hit, slo, skew] = (rest.length ? rest : ['88000', '413', '0.91', '50', '1.3']).map(
  Number,
);
const [domain, nodes = 1] = machineArg.split('x').map(Number);
const model = MODEL_PRESETS.find((m) => m.name.startsWith('DeepSeek V4.1 Flash'))!;
const opts = { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0.9, commsOverlap: 0.65 }) };
const serving: ServingOptions = {
  speculative: { draftTokens: 5, acceptance: 0.6 },
  mixedPrefill: true,
  prefixHitRate: hit,
  maxBatchedTokens: 8192,
};
const workload = { prefillLen: P, generateLen: G, routingSkew: skew };
const MAX_SEQS_PER_RANK = 256;

const t0 = performance.now();
const chip = CHIPS_BY_ID[chipId];
const axes = deployedAxes(chip.interconnect, { domain, nodes });
const chips = domain * nodes;
let best: { tps: number; line: string } | undefined;
let [evals, placements] = [0, 0];
for (const sizes of searchTuples(model, chips)) {
  if (sizes.PP > 1) continue; // speculative serving is modeled without PP
  let tupleBest: { tps: number; line: string } | undefined;
  const meshes = enumeratePlacements(axes, sizes);
  placements += meshes.length;
  for (const mesh of meshes)
    for (const moeDispatch of (sizes.EP > 1
      ? ['coalesced-a2a', 'ring-of-experts']
      : ['ring-of-experts']) as MoeDispatch[]) {
      const input = { model, deployment: { chip, mesh, moeDispatch }, workload };
      const dpa = roleSize(mesh, 'DPA');
      const at = (perGroup: number) => {
        evals++;
        return evaluateServingStep(input, perGroup * dpa, serving, opts);
      };
      const meets = (perGroup: number) => {
        const r = at(perGroup);
        return r.ok && 1 / r.tpot >= slo;
      };
      if (!meets(1)) continue;
      let [lo, hi] = [1, MAX_SEQS_PER_RANK * (chips / dpa)];
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        if (meets(mid)) lo = mid;
        else hi = mid - 1;
      }
      const r = at(lo);
      if (!r.ok || (tupleBest && r.outputTokPerSecPerChip <= tupleBest.tps)) continue;
      const layout = ROLES.filter((x) => roleSize(mesh, x) > 1)
        .map((x) => `${x}${roleSize(mesh, x)}[${mesh.roles[x].join('')}]`)
        .join(' ');
      tupleBest = {
        tps: r.outputTokPerSecPerChip,
        line: `${layout.padEnd(40)} ${moeDispatch.padEnd(15)} ${String(lo * dpa).padStart(5)} seqs ${(1 / r.tpot).toFixed(6)} tok/s/user ${r.outputTokPerSecPerChip.toFixed(6)} out tok/s/chip`,
      };
    }
  console.log(tupleBest?.line ?? `${JSON.stringify(sizes)} infeasible`);
  if (tupleBest && (!best || tupleBest.tps > best.tps)) best = tupleBest;
}
console.log(`\nbest ${chipId} x${chips}: ${best?.line ?? 'infeasible'}`);
console.error(
  `${placements} placements, ${evals} evaluations in ${((performance.now() - t0) / 1000).toFixed(1)}s`,
);
