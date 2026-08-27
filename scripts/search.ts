import { searchShardings } from '../src/core/engine/optimizer/search';
import { Mesh, roleSize } from '../src/core/engine/surface/deploy';
import { ROLES, ShardingRole } from '../src/core/engine/sim/ir/sharding/roles';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { MODEL_PRESETS } from '../src/core/model/models';

// Streaming search CLI: one line per role-size tuple the moment it
// prices (the tuple's best placement and dispatch), new global bests
// marked with *. Usage:
//   npm run search -- <model> <chip> <machine> [prefill|decode]
//   npm run search -- Kimi tpu-v5p 2x4x4
//   npm run search -- "LLaMA 3 70B" h100-sxm 8x2 decode   (switched: domain x nodes)

const [modelArg = 'Kimi', chipArg = 'tpu-v5p', machineArg = '2x4x4', phaseArg = 'prefill'] =
  process.argv.slice(2);

const model = MODEL_PRESETS.find((m) => m.name.startsWith(modelArg));
const chip = CHIPS_BY_ID[chipArg];
if (!model || !chip) throw new Error(`unknown model or chip`);

const machine = chip.interconnect.topologies
  ? chip.interconnect.topologies.find((t) => t.name === machineArg)
  : (([domain, nodes = 1]) => ({ domain, nodes }))(machineArg.split('x').map(Number));
if (!machine) throw new Error(`no ${machineArg} in the ${chipArg} catalog`);

const fmt = (mesh: Mesh, f: (r: ShardingRole, size: number) => string) =>
  ROLES.filter((r) => roleSize(mesh, r) > 1)
    .map((r) => f(r, roleSize(mesh, r)))
    .join(' ');

const chips = Array.isArray(machine)
  ? 0
  : 'count' in machine
    ? machine.count
    : machine.domain * machine.nodes;

const t0 = performance.now();
let best = -Infinity;
let total = 0;
for (const step of searchShardings(
  model,
  chip,
  machine,
  { prefillLen: 4096, generateLen: 1024 },
  {
    costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 }),
    phase:
      phaseArg === 'decode'
        ? { kind: 'decode' }
        : { kind: 'prefill', seqs: chips, mode: 'throughput' },
  },
)) {
  total = step.total;
  const c = step.candidate;
  if (!c) continue;
  const isBest = c.score > best;
  if (isBest) best = c.score;
  const { mesh } = c.deployment;
  console.log(
    `[+${(performance.now() - t0).toFixed(0).padStart(6)}ms] ` +
      `${step.done.toString().padStart(3)}/${step.total} ${isBest ? '*' : ' '} ` +
      `${Math.round(c.score).toString().padStart(6)} tok/s/chip  ` +
      `${(fmt(mesh, (r, s) => `${r}=${s}`) || 'single chip').padEnd(30)}  ` +
      `best: ${c.batch !== undefined ? `batch=${c.batch} ` : ''}` +
      // with EP = 1 there is nothing to route, so the dispatch is noise
      `${roleSize(mesh, 'EP') > 1 ? `${c.deployment.moeDispatch} ` : ''}` +
      `${fmt(mesh, (r) => `${r}[${mesh.roles[r].join('')}]`)}`,
  );
}
console.log(
  `\n${total} tuples in ${((performance.now() - t0) / 1000).toFixed(1)}s, best ${Math.round(best)} tok/s/chip`,
);
