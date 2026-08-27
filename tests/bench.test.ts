import { expect, test } from 'vitest';
import { searchShardings } from '../src/core/engine/optimizer/search';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { MODEL_PRESETS } from '../src/core/model/models';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';

const backend = makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 });
const workload = { prefillLen: 4096, generateLen: 1024 };

// prettier-ignore
const COMBOS: { model: string; chip: string; machine: string | { domain: number; nodes: number } }[] = [
  { model: 'LLaMA 3 8B',        chip: 'h100-sxm', machine: { domain: 8, nodes: 1 } },
  { model: 'LLaMA 3.1 405B',    chip: 'h100-sxm', machine: { domain: 8, nodes: 2 } },
  { model: 'Gemma 4 31B',       chip: 'tpu-v6e',  machine: '4x4' },
  { model: 'Gemma 4 26B A4B',   chip: 'tpu-v6e',  machine: '4x4' },
  { model: 'gpt-oss-20b',       chip: 'tpu-v6e',  machine: '2x4' },
  { model: 'gpt-oss-120b',      chip: 'tpu-v6e',  machine: '4x4' },
  { model: 'gpt-oss-120b',      chip: 'tpu-v6e',  machine: '4x8' },
  { model: 'GLM 5.2',       chip: 'b200',     machine: { domain: 8, nodes: 1 } },
  { model: 'GLM 5.2',       chip: 'tpu-v7x',  machine: '2x2x2' },
  { model: 'GLM 5.2',       chip: 'tpu-v7x',  machine: '2x2x4' },
  // { model: 'GLM 5.2',       chip: 'tpu-v7x',  machine: '2x4x4' },
  { model: 'Kimi K2.6', chip: 'tpu-v5p',  machine: '2x4x4' },
  { model: 'Kimi K3',       chip: 'b300',     machine: { domain: 8, nodes: 1 } },
  { model: 'Kimi K3',       chip: 'b300',     machine: { domain: 8, nodes: 4 } },
  { model: 'Kimi K3',       chip: 'mi355x',   machine: { domain: 8, nodes: 1 } },
  // { model: 'Kimi K2.6', chip: 'tpu-v5p',  machine: '4x4x4' },
];

test('search wall-clock on iconic combos', { timeout: 600_000 }, () => {
  let timeSum = 0;
  for (const c of COMBOS) {
    const [model, chip] = [
      MODEL_PRESETS.find((m) => m.name.startsWith(c.model))!,
      CHIPS_BY_ID[c.chip],
    ];

    const machine =
      typeof c.machine === 'string'
        ? chip.interconnect.topologies!.find((t) => t.name === c.machine)!
        : c.machine;

    const label =
      'dims' in machine
        ? `${c.chip} ${machine.name}`
        : `${machine.domain * machine.nodes}x ${c.chip}`;

    const t0 = performance.now();
    const table = [
      ...searchShardings(model, chip, machine, workload, {
        costBackend: backend,
        phase: { kind: 'decode' },
      }),
    ].flatMap((s) => s.candidate ?? []);
    const dt = performance.now() - t0;
    timeSum += dt;
    console.log(`${c.model} @ ${label}: ${dt.toFixed(0)} ms`);
    expect(table.length).toBeGreaterThan(0);
  }
  console.log(`\nTotal search time: ${timeSum.toFixed(0)} ms`);
});
