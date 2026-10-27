import { cheapestNodeRatio, evaluateDisagg } from '../src/core/engine/sim/run/disagg';
import { operatingBatch } from '../src/core/engine/optimizer/policy';
import { enumeratePlacements } from '../src/core/engine/surface/placements';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';

// Disaggregated prefill/decode on B300 x8 DEP8 pools (TP1 x DPA8, EP8)
// with KV moving over the 100 GB/s/chip ConnectX-8 scale-out NICs.
//   npm exec vite-node scripts/disagg.ts -- "<model prefix>" [prefix hit rate] [slo tok/s/user]
const [modelArg = 'DeepSeek V4 Flash', hArg = '0', sloArg] = process.argv.slice(2);
const model = MODEL_PRESETS.find((m) => m.name.startsWith(modelArg));
if (!model) throw new Error(`unknown model ${modelArg}`);
const opts = { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0.9, commsOverlap: 0.65 }) };
const h = Number(hArg);
const slo = sloArg ? Number(sloArg) : undefined;

const chip = CHIPS_BY_ID['b300'];
const axes = deployedAxes(chip.interconnect, { domain: 8, nodes: 1 });
const mesh = enumeratePlacements(axes, { PP: 1, TP: 1, DPA: 8, EP: 8, ETP: 1 })[0];
const deployment = { chip, mesh, moeDispatch: 'coalesced-a2a' as const };

console.log(`${model.name}  B300x8 DEP8 -> B300x8 DEP8  h=${h}  slo=${slo ?? 'none'}`);
for (const [P, G] of [
  [1024, 1024],
  [8192, 512],
  [32768, 1024],
]) {
  const workload = { prefillLen: P, generateLen: G };
  const decodeBatch = operatingBatch(
    { model, deployment, workload },
    { sloTokPerSecPerUser: slo },
    opts,
  );
  const e = evaluateDisagg(
    { model, prefill: deployment, decode: deployment, workload, decodeBatch, prefixHitRate: h },
    opts,
  );
  if (!e.ok) {
    console.log(`\n${P}->${G}: ${e.diags.map((d) => d.code).join(',')}`);
    continue;
  }
  const n = cheapestNodeRatio(e);
  const f = (x: number, d = 1) => x.toFixed(d);
  console.log(
    `\n${P}->${G}  decode batch ${decodeBatch} (${f(1 / e.decode.result.tpot)} tok/s/user)\n` +
      `  KV/request ${f(e.kvBytesPerRequest / 1e6, 2)} MB, transfer ${f(e.transferTime * 1e3, 3)} ms (TTFT add)\n` +
      `  prefill: ${f(e.prefill.result.tokPerSecPerChip, 0)} tok/s/chip -> ${f(e.prefill.computeReqPerSecPerChip, 2)} req/s/chip compute, ` +
      `${f(e.prefill.linkReqPerSecPerChip, 0)} link, NIC util ${f(100 * e.prefill.linkUtilisation, 2)}%\n` +
      `  decode:  ${f(e.decode.result.tokPerSecPerChip, 0)} tok/s/chip -> ${f(e.decode.computeReqPerSecPerChip, 2)} req/s/chip compute, ` +
      `${f(e.decode.linkReqPerSecPerChip, 0)} link, NIC util ${f(100 * e.decode.linkUtilisation, 2)}%\n` +
      `  balance: ${f(e.decodeChipsPerPrefillChip, 2)} decode chips per prefill chip, ${f(e.reqPerSecPerChip, 2)} req/s/chip, ` +
      `link-bound ${e.linkBound}\n` +
      `  cost: $${f(e.usdPerMOutput, 3)}/1M out, $${f(e.usdPerMInput, 3)}/1M in\n` +
      `  nodes: ${n.prefillNodes}P:${n.decodeNodes}D, ${f(n.reqPerSec)} req/s, $${f(n.usdPerMOutput, 3)}/1M out`,
  );
}
