import { Candidate, searchShardings, searchTuples } from '../core/engine/optimizer/search';
import { ServingPolicy } from '../core/engine/optimizer/policy';
import { matmulSeconds, roofline } from '../core/engine/roofline';
import { makeNaiveOpCostSumBackend } from '../core/engine/sim/cost/naiveOpCostSum';
import { evaluateDecodeAtBatch } from '../core/engine/sim/run/decode';
import { evaluatePrefill } from '../core/engine/sim/run/prefill';
import { runnableOn } from '../core/engine/sim/run/validate';
import { ROLES } from '../core/engine/sim/ir/sharding/roles';
import { roleSize } from '../core/engine/surface/deploy';
import { flopsPerDecodeToken, flopsPerPrefillToken, hasMoeLayers } from '../core/model/utils';
import { ChipSpec } from '../core/hardware/chips';
import { MODEL_PRESETS, ModelSpec } from '../core/model/models';
import { machineChips, machineKey, machineLabel, machineOf } from './machines';
import { Boundedness, ComponentTimes, UiResult } from './results';
import type { SearchRequest, SearchUpdate } from './searchClient';

// the client pins one worker per chip, so this instance only ever sees one
// chip's requests: each supersedes everything older than it
let latest = 0;

const post = (u: SearchUpdate) => (self as { postMessage(m: unknown): void }).postMessage(u);

self.onmessage = (e: MessageEvent<SearchRequest>) => {
  latest = e.data.id;
  void run(e.data);
};

async function run(req: SearchRequest) {
  const model = MODEL_PRESETS.find((m) => m.name === req.modelName);
  if (!model) return;

  const backend = makeNaiveOpCostSumBackend(req.overlap);
  const policy: ServingPolicy = {
    batching: req.workload.batching,
    sloTokPerSecPerUser: req.workload.sloTokPerSecPerUser || undefined,
  };
  const workload = { prefillLen: req.workload.prefillLen, generateLen: req.workload.generateLen };

  // every group posts upfront with its search size, so the app knows the
  // full totals before any search runs
  const machines = req.chips.map((chip) => {
    const hardware = roofline(model, chip, workload, chip.realizableFlopsFrac);
    const nChips = machineChips(chip);
    const total = hardware ? searchTuples(model, nChips).length : 0;
    const group = {
      id: req.id,
      key: machineKey(chip),
      kind: 'group' as const,
      chipId: chip.id,
      machineLabel: machineLabel(chip),
      nChips,
      hardware,
      total,
      error: hardware ? undefined : `${chip.name} does not support this model's compute dtype`,
    };
    post(group);
    return { chip, hardware, nChips, group };
  });

  for (const { chip, hardware, nChips, group } of machines) {
    if (req.id < latest) return;
    if (!hardware) continue;

    try {
      const gen = searchShardings(model, chip, machineOf(chip), workload, {
        costBackend: backend,
        phase: { kind: 'decode', policy },
      });
      for (const step of gen) {
        if (req.id < latest) return;
        post({
          id: req.id,
          key: group.key,
          kind: 'row',
          done: step.done,
          row:
            step.candidate &&
            toRow(model, chip, group.key, nChips, workload, backend, step.candidate),
        });
        await new Promise((r) => setTimeout(r));
      }
    } catch (err) {
      post({ ...group, error: String(err) });
    }
  }
}

type Backend = ReturnType<typeof makeNaiveOpCostSumBackend>;

// A decode-search winner filled out into the UI's config row: prefill is
// evaluated on the same deployment (throughput at a full-machine batch of
// sequences, plus a single-sequence pass for TTFT).
function toRow(
  model: ModelSpec,
  chip: ChipSpec,
  key: string,
  nChips: number,
  workload: { prefillLen: number; generateLen: number },
  backend: Backend,
  c: Candidate<Backend>,
): UiResult | undefined {
  const { mesh } = c.deployment;
  const input = { model, deployment: c.deployment, workload };
  const dec = c.result;
  if (!('tpot' in dec)) return undefined;

  const runnableModel = runnableOn(model, chip);
  const hw = roofline(model, chip, workload, chip.realizableFlopsFrac)!;
  const dpa = roleSize(mesh, 'DPA');
  const pp = roleSize(mesh, 'PP');
  const ctxAvg = workload.prefillLen + workload.generateLen / 2;
  const prefillBatchSeqs = dpa * Math.max(1, Math.ceil(hw.critTokens / workload.prefillLen));

  const opts = { costBackend: backend };
  const pfThrough = evaluatePrefill(input, prefillBatchSeqs, 'throughput', opts);
  const pfSingle = evaluatePrefill(input, 1, 'ttft', opts);
  // the same config re-priced at an effectively infinite batch (KV gate
  // off) — its own batch-scaling ceiling
  const sat = evaluateDecodeAtBatch(input, 65536 * dpa, pp, { ...opts, ignoreKvCapacity: true });

  const bound = (b: ComponentTimes): Boundedness =>
    b.compute >= b.memory && b.compute >= b.comms
      ? 'compute'
      : b.memory >= b.comms
        ? 'memory'
        : 'comms';

  // the expert plane is structural noise on dense models (never read by
  // lowering), so hide it from the displayed sharding
  const shown = ROLES.filter(
    (r) => roleSize(mesh, r) > 1 && (hasMoeLayers(model) || (r !== 'EP' && r !== 'ETP')),
  );
  const sizes = Object.fromEntries(shown.map((r) => [r, roleSize(mesh, r)]));
  // DCP is not a role: it owns no dims, it re-spends TP's on the sequence.
  // It still belongs in the identity and the label, since two rows can
  // otherwise differ only by it.
  const dcp = c.deployment.decodeContextParallel ?? 1;
  const placement =
    shown.map((r) => `${r}[${mesh.roles[r].join('')}]`).join(' ') +
    (dcp > 1 ? ` DCP=${dcp} of TP` : '');

  return {
    id: `${key}|${JSON.stringify(sizes)}|dcp${dcp}`,
    chipId: chip.id,
    sizes,
    decodeContextParallel: dcp,
    placement,
    dispatch: c.deployment.moeDispatch,
    mesh,
    nChips,
    workload,
    diagnostics: dec.diags,
    memory: {
      weightBytesPerChip: dec.memory.weightBytesPerChip,
      kvSpaceBytesPerChip: Math.max(0, chip.hbmCapacity - dec.memory.weightBytesPerChip),
      kvBytesPerSeqPerChip: dec.memory.kvBytesPerSeqPerChip,
      maxResidentSeqs: dec.memory.maxResidentSeqsPerChip * dpa,
    },
    decode: {
      tokPerSecPerChip: dec.tokPerSecPerChip,
      tokPerSecPerUser: 1 / dec.tpot,
      tpot: dec.tpot,
      stepTime: dec.stepTime,
      batchPerStage: c.batch!,
      residentSeqs: c.batch! * pp,
      mfu:
        dec.tokPerSecPerChip * matmulSeconds(flopsPerDecodeToken(runnableModel, ctxAvg), chip, 1)!,
      fracOfCeiling: dec.tokPerSecPerChip / hw.decodeCeilingOverlapped,
      batchSaturation: sat.ok ? dec.tokPerSecPerChip / sat.tokPerSecPerChip : undefined,
      boundBy: bound(dec.cost.busy),
      components: dec.cost.busy,
    },
    prefill: pfThrough.ok
      ? {
          tokPerSecPerChip: pfThrough.tokPerSecPerChip,
          ttft: pfSingle.ok ? pfSingle.latency : pfThrough.latency,
          batchSeqs: prefillBatchSeqs,
          mfu:
            pfThrough.tokPerSecPerChip *
            matmulSeconds(flopsPerPrefillToken(runnableModel, workload.prefillLen), chip, 1)!,
          fracOfCeiling: pfThrough.tokPerSecPerChip / hw.prefillCeiling,
          boundBy: bound(pfThrough.cost.busy),
          components: pfThrough.cost.busy,
        }
      : undefined,
  };
}
