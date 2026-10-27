import { searchTuples } from '../src/core/engine/optimizer/search';
import { roleSize, type MoeDispatch } from '../src/core/engine/surface/deploy';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import { evaluateServingStep, type ServingOptions } from '../src/core/engine/sim/run/serving';
import { ROLES } from '../src/core/engine/sim/ir/sharding/roles';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';

// The UI's chip ranking (requests per dollar against an H100 node) under the
// serving model: DSpark, mixed prefill and prefix-cache hits, for one shape.
//   npm exec vite-node scripts/rank.ts -- <prefillLen> <generateLen> <hitRate> <tok/s/user> <skew>
//   npm exec vite-node scripts/rank.ts -- 88000 413 0.91 50 1.3     (median AgentX request)
const [P, G, hit, slo, skew] = (
  process.argv.slice(2).length ? process.argv.slice(2) : ['88000', '413', '0.91', '50', '1.3']
).map(Number);
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

const rows: { label: string; reqPerUsd: number; line: string }[] = [];
for (const [chipId, machine] of [
  ['h100-sxm', { domain: 8, nodes: 1 }],
  ['h200-sxm', { domain: 8, nodes: 1 }],
  ['b200', { domain: 8, nodes: 1 }],
  ['b300', { domain: 8, nodes: 1 }],
  ['mi355x', { domain: 8, nodes: 1 }],
  ['gb200-nvl72', { domain: 64, nodes: 1 }],
] as const) {
  const chip = CHIPS_BY_ID[chipId];
  const axes = deployedAxes(chip.interconnect, machine);
  const chips = machine.domain * machine.nodes;
  let best: { tps: number; line: string } | undefined;
  for (const sizes of searchTuples(model, chips)) {
    if (sizes.PP > 1) continue; // speculative serving is modeled without PP
    for (const mesh of enumeratePlacements(axes, sizes).slice(0, 4))
      for (const moeDispatch of (sizes.EP > 1
        ? ['coalesced-a2a', 'ring-of-experts']
        : ['ring-of-experts']) as MoeDispatch[]) {
        const input = { model, deployment: { chip, mesh, moeDispatch }, workload };
        const dpa = roleSize(mesh, 'DPA');
        const at = (perGroup: number) => evaluateServingStep(input, perGroup * dpa, serving, opts);
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
        if (!r.ok || (best && r.outputTokPerSecPerChip <= best.tps)) continue;
        const layout = ROLES.filter((x) => roleSize(mesh, x) > 1)
          .map((x) => `${x}${roleSize(mesh, x)}`)
          .join(' ');
        best = {
          tps: r.outputTokPerSecPerChip,
          line: `${layout.padEnd(16)} ${moeDispatch.padEnd(15)} ${String(lo * dpa).padStart(5)} seqs ${(1 / r.tpot).toFixed(0).padStart(4)} tok/s/user ${r.outputTokPerSecPerChip.toFixed(0).padStart(6)} out tok/s/chip`,
        };
      }
  }
  const label = `${chipId} x${chips}`;
  if (!best) {
    rows.push({ label, reqPerUsd: 0, line: 'infeasible' });
    continue;
  }
  const reqPerUsd = ((best.tps / G) * 3600) / chip.costPerHour!;
  rows.push({ label, reqPerUsd, line: `${best.line}  $${chip.costPerHour}/chip-h` });
}
const h100 = rows.find((r) => r.label.startsWith('h100'))!.reqPerUsd;
console.log(
  `${model.name}: ${P} -> ${G}, prefix hit ${hit}, DSpark k=5 a=0.6, skew ${skew}, >= ${slo} tok/s/user, 8192-token step budget per rank`,
);
for (const r of rows.sort((a, b) => b.reqPerUsd - a.reqPerUsd))
  console.log(
    `${r.label.padEnd(16)} ${(r.reqPerUsd / h100).toFixed(2).padStart(5)}x  ${Math.round(r.reqPerUsd).toString().padStart(6)} req/$  ${r.line}`,
  );
