import { searchShardings } from '../src/core/engine/optimizer/search';
import { roleSize } from '../src/core/engine/surface/deploy';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import { evaluateDecodeAtBatch } from '../src/core/engine/sim/run/decode';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { memoryFootprint } from '../src/core/engine/sim/run/memory';
import { partitionIntoStages } from '../src/core/engine/sim/lowering/stages';
import { ROLES } from '../src/core/engine/sim/ir/sharding/roles';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';

// Cross-chip comparison for one model and workload on 8-GPU nodes.
//   npm exec vite-node scripts/compare.ts -- "<model prefix>" [prefillLen] [generateLen] [slo tok/s/user]
// ROUTING_SKEW=<hottest/mean EP-rank load> sets the MoE routing skew (default 1).
const [modelArg = 'DeepSeek V4.1 Flash', pArg = '1024', gArg = '1024', sloArg] =
  process.argv.slice(2);
const model = MODEL_PRESETS.find((m) => m.name.startsWith(modelArg));
if (!model) throw new Error(`unknown model ${modelArg}`);
const workload = {
  prefillLen: Number(pArg),
  generateLen: Number(gArg),
  routingSkew: Number(process.env.ROUTING_SKEW ?? 1),
};
const opts = { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0.9, commsOverlap: 0.65 }) };
const slo = sloArg ? Number(sloArg) : undefined;

const fmtMesh = (mesh: Parameters<typeof roleSize>[0]) =>
  ROLES.filter((r) => roleSize(mesh, r) > 1)
    .map((r) => `${r}${roleSize(mesh, r)}`)
    .join(' ') || 'single';

console.log(
  `${model.name}  ${workload.prefillLen}->${workload.generateLen}  slo=${slo ?? 'none'}  skew=${workload.routingSkew}`,
);
console.log(
  'chip          nodes  best layout                 dispatch        batch  tok/s/chip  $/1M out   tok/s/user',
);
for (const [chipId, nodes] of [
  ['b300', 1],
  ['b200', 1],
  ['h200-sxm', 1],
  ['h200-sxm', 2],
  ['h100-sxm', 2],
  ['mi355x', 1],
  ['rtx-pro-6000', 1],
] as const) {
  const chip = CHIPS_BY_ID[chipId];
  let best: { score: number; line: string } | undefined;
  for (const step of searchShardings(model, chip, { domain: 8, nodes }, workload, {
    ...opts,
    phase: { kind: 'decode', policy: { sloTokPerSecPerUser: slo } },
  })) {
    const c = step.candidate;
    if (!c || (best && c.score <= best.score)) continue;
    const r = c.result as Extract<typeof c.result, { tpot: number }>;
    const usdPerM = (chip.costPerHour! / 3600 / c.score) * 1e6;
    best = {
      score: c.score,
      line:
        `${chipId.padEnd(13)} ${String(nodes).padStart(5)}  ${fmtMesh(c.deployment.mesh).padEnd(27)} ` +
        `${c.deployment.moeDispatch.padEnd(15)} ${String(c.batch).padStart(5)} ` +
        `${c.score.toFixed(0).padStart(11)} ${usdPerM.toFixed(3).padStart(9)} ${(1 / r.tpot).toFixed(1).padStart(12)}`,
    };
  }
  console.log(best?.line ?? `${chipId.padEnd(13)} ${String(nodes).padStart(5)}  infeasible`);
}

// Our production layout: B300 x8 DEP8 (TP1 x DPA8, EP8), at a sweep of batches.
const chip = CHIPS_BY_ID['b300'];
const axes = deployedAxes(chip.interconnect, { domain: 8, nodes: 1 });
const sizes = { PP: 1, TP: 1, DPA: 8, EP: 8, ETP: 1 };
const mesh = enumeratePlacements(axes, sizes)[0];
for (const moeDispatch of ['coalesced-a2a', 'ring-of-experts'] as const) {
  const input = { model, deployment: { chip, mesh, moeDispatch }, workload };
  const mem = memoryFootprint(
    model,
    input.deployment,
    partitionIntoStages(model, 1),
    workload.prefillLen + workload.generateLen,
  );
  console.log(
    `\nB300 DEP8 ${moeDispatch}: weights ${(mem.weightBytesPerChip / 1e9).toFixed(1)} GB/chip, ` +
      `KV ${(mem.kvBytesPerSeqPerChip / 1e6).toFixed(2)} MB/seq, ${mem.maxResidentSeqsPerChip} seqs/chip fit`,
  );
  for (const batch of [8, 64, 256, 1024, 2048]) {
    const r = evaluateDecodeAtBatch(input, batch, 1, opts);
    if (!r.ok) {
      console.log(`  batch ${batch}: ${r.diags.map((d) => d.code).join(',')}`);
      continue;
    }
    const b = (r.cost as any).busy;
    console.log(
      `  batch ${String(batch).padStart(5)}: ${r.tokPerSecPerChip.toFixed(0).padStart(6)} tok/s/chip, ` +
        `${(1 / r.tpot).toFixed(1).padStart(6)} tok/s/user, step ${(r.stepTime * 1e3).toFixed(2)} ms ` +
        `(compute ${(b.compute * 1e3).toFixed(2)}, memory ${(b.memory * 1e3).toFixed(2)}, comms ${(b.comms * 1e3).toFixed(2)})`,
    );
  }
  const p = evaluatePrefill(input, 8, 'throughput', opts);
  if (p.ok)
    console.log(
      `  prefill 8 seqs x ${workload.prefillLen}: ${p.tokPerSecPerChip.toFixed(0)} tok/s/chip`,
    );
}
