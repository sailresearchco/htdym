import { expect, test } from 'vitest';
import { searchShardings } from '../src/core/engine/optimizer/search';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { ROLES } from '../src/core/engine/sim/ir/sharding/roles';
import { roleSize, type Deployment } from '../src/core/engine/surface/deploy';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { MODEL_PRESETS } from '../src/core/model/models';

// Pins sweep results to what the planner and pricer produced before their
// caches were restructured, so a speedup that changes a winner, a batch
// or a score fails here. Each row is a tuple's best as
// [deployment, batch, score].

type Row = [string, number, number];

const describe = (d: Deployment) =>
  [
    ...ROLES.filter((r) => roleSize(d.mesh, r) > 1).map(
      (r) => `${r}${roleSize(d.mesh, r)}[${d.mesh.roles[r].join('')}]`,
    ),
    d.moeDispatch,
    `dcp${d.decodeContextParallel ?? 1}`,
  ].join(' ');

const model = (name: string) => MODEL_PRESETS.find((m) => m.name.startsWith(name))!;

function search(
  modelName: string,
  chipId: string,
  machine: string | { domain: number; nodes: number },
  phase: Parameters<typeof searchShardings>[4]['phase'],
): Row[] {
  const chip = CHIPS_BY_ID[chipId];
  const m =
    typeof machine === 'string'
      ? chip.interconnect.topologies!.find((t) => t.name === machine)!
      : machine;
  return [
    ...searchShardings(
      model(modelName),
      chip,
      m,
      { prefillLen: 4096, generateLen: 1024 },
      { costBackend: makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 }), phase },
    ),
  ].flatMap(({ candidate: c }): Row[] =>
    c ? [[describe(c.deployment), c.batch ?? 0, c.score]] : [],
  );
}

function expectRows(got: Row[], want: Row[]) {
  expect(got.map(([d, b]) => [d, b])).toEqual(want.map(([d, b]) => [d, b]));
  got.forEach(([, , score], i) =>
    expect(Math.abs(score - want[i][2])).toBeLessThanOrEqual(1e-9 * Math.abs(want[i][2])),
  );
}

// prettier-ignore
const V4_B300_8_DECODE_SLO: Row[] = [
  ["DPA8[D] EP8[D] coalesced-a2a dcp1", 9216, 57989.4086283125],
  ["DPA4[D0] TP2[D1] EP8[D0D1] coalesced-a2a dcp2", 6820, 42626.49974272729],
  ["DPA2[D1] TP4[D0D2] EP8[D0D1D2] ring-of-experts dcp2", 4834, 30217.497173501208],
  ["TP8[D] EP8[D] ring-of-experts dcp1", 2327, 14545.559912728699],
  ["DPA8[D0D1] EP4[D0] ETP2[D1] coalesced-a2a dcp1", 8192, 51910.29822413309],
  ["PP2[D0] DPA4[D1] EP4[D1] coalesced-a2a dcp1", 3412, 42650.058147813834],
  ["DPA4[D0] TP2[D1] EP4[D0] ETP2[D1] coalesced-a2a dcp2", 7084, 44283.12840082971],
  ["PP2[D0] DPA2[D1] TP2[D2] EP4[D1D2] ring-of-experts dcp2", 2602, 32528.687298230154],
  ["DPA2[D0] TP4[D1D2] EP4[D0D1] ETP2[D2] ring-of-experts dcp2", 4748, 29676.974938301606],
  ["PP2[D1] TP4[D0D2] EP4[D0D2] ring-of-experts dcp2", 1704, 21301.805152883902],
  ["TP8[D0D1D2] EP4[D0D2] ETP2[D1] ring-of-experts dcp4", 3107, 19421.004672194387],
  ["DPA8[D0D1D2] EP2[D1] ETP4[D0D2] coalesced-a2a dcp1", 7384, 46157.42945185529],
  ["PP2[D0] DPA4[D1D2] EP2[D1] ETP2[D2] coalesced-a2a dcp1", 3168, 39600.391368155564],
  ["PP4[D0] DPA2[D1] EP2[D1] coalesced-a2a dcp1", 376, 9407.50467950348],
  ["DPA4[D0] TP2[D1] EP2[D1] ETP4[D0] coalesced-a2a dcp2", 6144, 38940.090451208394],
  ["PP2[D0] DPA2[D1] TP2[D2] EP2[D1] ETP2[D2] coalesced-a2a dcp2", 2778, 34730.63107713642],
  ["PP4[D0] TP2[D1] EP2[D1] coalesced-a2a dcp1", 357, 8929.13387471209],
  ["DPA2[D1] TP4[D0D2] EP2[D1] ETP4[D0D2] coalesced-a2a dcp2", 5314, 33217.42328375195],
  ["PP2[D0] TP4[D1D2] EP2[D1] ETP2[D2] ring-of-experts dcp2", 1672, 20902.94410114038],
  ["TP8[D0D1D2] EP2[D1] ETP4[D0D2] ring-of-experts dcp4", 3070, 19190.568601765815],
  ["DPA8[D] ETP8[D] ring-of-experts dcp1", 6600, 41251.9829811818],
  ["PP2[D1] DPA4[D0D2] ETP4[D0D2] ring-of-experts dcp1", 3072, 39227.560553002244],
  ["PP4[D0] DPA2[D1] ETP2[D1] ring-of-experts dcp1", 372, 9312.806981053225],
  ["PP8[D] ring-of-experts dcp1", 30, 1513.7454430915259],
  ["DPA4[D0D2] TP2[D1] ETP8[D0D1D2] ring-of-experts dcp2", 5836, 36485.96720539082],
  ["PP2[D0] DPA2[D1] TP2[D2] ETP4[D1D2] ring-of-experts dcp2", 2462, 30777.160634767202],
  ["PP4[D0] TP2[D1] ETP2[D1] ring-of-experts dcp1", 365, 9134.624350615131],
  ["DPA2[D1] TP4[D0D2] ETP8[D0D1D2] ring-of-experts dcp2", 4352, 27380.441814429993],
  ["PP2[D1] TP4[D0D2] ETP4[D0D2] ring-of-experts dcp2", 1920, 24020.941185549756],
  ["TP8[D] ETP8[D] ring-of-experts dcp1", 2805, 17537.96207553405],
];
// prettier-ignore
const GLM_V7X_222_PREFILL: Row[] = [
  ["DPA8[XYZ] EP8[XYZ] ring-of-experts dcp1", 0, 15154.389744762591],
  ["DPA4[XY] TP2[Z] EP8[XYZ] ring-of-experts dcp1", 0, 8157.035387438605],
  ["DPA2[X] TP4[YZ] EP8[XYZ] ring-of-experts dcp1", 0, 6594.519357531945],
  ["TP8[XYZ] EP8[XYZ] ring-of-experts dcp1", 0, 5294.901008728084],
  ["DPA8[XYZ] EP4[XY] ETP2[Z] ring-of-experts dcp1", 0, 14888.840320543764],
  ["PP2[X] DPA4[YZ] EP4[YZ] ring-of-experts dcp1", 0, 16580.696221535785],
  ["DPA4[XY] TP2[Z] EP4[XY] ETP2[Z] ring-of-experts dcp1", 0, 8309.869531062392],
  ["PP2[X] DPA2[Y] TP2[Z] EP4[YZ] ring-of-experts dcp1", 0, 8128.204264840756],
  ["DPA2[X] TP4[YZ] EP4[XY] ETP2[Z] ring-of-experts dcp1", 0, 6543.7321074241145],
  ["PP2[X] TP4[YZ] EP4[YZ] ring-of-experts dcp1", 0, 6151.6976136295525],
  ["TP8[XYZ] EP4[XY] ETP2[Z] ring-of-experts dcp1", 0, 5262.109322395017],
  ["DPA8[XYZ] EP2[X] ETP4[YZ] ring-of-experts dcp1", 0, 14384.715383546845],
  ["PP2[X] DPA4[YZ] EP2[Y] ETP2[Z] ring-of-experts dcp1", 0, 16275.792876085057],
  ["PP4[XY] DPA2[Z] EP2[Z] coalesced-a2a dcp1", 0, 17561.640151102787],
  ["DPA4[XY] TP2[Z] EP2[X] ETP4[YZ] ring-of-experts dcp1", 0, 7928.685484576716],
  ["PP2[X] DPA2[Y] TP2[Z] EP2[Y] ETP2[Z] ring-of-experts dcp1", 0, 8386.233337911593],
  ["PP4[XY] TP2[Z] EP2[Z] coalesced-a2a dcp1", 0, 8507.161671840011],
  ["DPA2[X] TP4[YZ] EP2[Y] ETP4[XZ] ring-of-experts dcp1", 0, 6444.4688156487555],
  ["PP2[X] TP4[YZ] EP2[Y] ETP2[Z] ring-of-experts dcp1", 0, 6109.235741706311],
  ["TP8[XYZ] EP2[X] ETP4[YZ] ring-of-experts dcp1", 0, 5197.729529050521],
  ["DPA8[XYZ] ETP8[XYZ] ring-of-experts dcp1", 0, 13472.385687016485],
  ["PP2[X] DPA4[YZ] ETP4[YZ] ring-of-experts dcp1", 0, 15698.43411897841],
  ["PP4[XY] DPA2[Z] ETP2[Z] ring-of-experts dcp1", 0, 17165.953421067654],
  ["PP8[XYZ] ring-of-experts dcp1", 0, 24677.891989358395],
  ["DPA4[XY] TP2[Z] ETP8[XYZ] ring-of-experts dcp1", 0, 7643.391049244933],
  ["PP2[X] DPA2[Y] TP2[Z] ETP4[YZ] ring-of-experts dcp1", 0, 7910.270524515716],
  ["PP4[XY] TP2[Z] ETP2[Z] ring-of-experts dcp1", 0, 10306.152471941037],
  ["DPA2[X] TP4[YZ] ETP8[XYZ] ring-of-experts dcp1", 0, 6254.710689268574],
  ["PP2[X] TP4[YZ] ETP4[YZ] ring-of-experts dcp1", 0, 7328.419140798539],
  ["TP8[XYZ] ETP8[XYZ] ring-of-experts dcp1", 0, 6697.35395832914],
];
// prettier-ignore
const V4_B300_8X2_DECODE_SLO: Row[] = [
  ["DPA16[DN] EP16[DN] ring-of-experts dcp1", 10144, 31725.395903071087],
  ["DPA8[D0D2N] TP2[D1] EP16[D0D1D2N] ring-of-experts dcp2", 9216, 28822.263544411657],
  ["DPA4[D1N] TP4[D0D2] EP16[D0D1D2N] ring-of-experts dcp2", 7364, 23017.45909699308],
  ["DPA2[N] TP8[D] EP16[DN] ring-of-experts dcp1", 3976, 12428.433476156719],
  ["TP16[DN] EP16[DN] ring-of-experts dcp8", 2235, 6984.777299790635],
  ["DPA16[D0D1D2N] EP8[D0D2N] ETP2[D1] ring-of-experts dcp1", 10240, 32659.71954014124],
  ["PP2[N] DPA8[D] EP8[D] coalesced-a2a dcp1", 6264, 39166.78683261321],
  ["DPA8[D0D1N] TP2[D2] EP8[D0D2N] ETP2[D1] ring-of-experts dcp2", 9088, 28403.72188381707],
  ["PP2[D0] DPA4[D1N] TP2[D2] EP8[D1D2N] ring-of-experts dcp2", 4740, 29634.848775967734],
  ["DPA4[D0N] TP4[D1D2] EP8[D0D1N] ETP2[D2] ring-of-experts dcp2", 7304, 22830.194125408383],
  ["PP2[D1] DPA2[N] TP4[D0D2] EP8[D0D2N] ring-of-experts dcp2", 3496, 21856.11051137768],
  ["DPA2[N] TP8[D0D1D2] EP8[D0D2N] ETP2[D1] ring-of-experts dcp4", 5288, 16526.515823280628],
  ["PP2[D1] TP8[D0D2N] EP8[D0D2N] ring-of-experts dcp4", 1734, 10839.038735167438],
  ["TP16[D0D1D2N] EP8[D0D2N] ETP2[D1] ring-of-experts dcp4", 2834, 8856.365595297835],
  ["DPA16[D0D1D2N] EP4[D1N] ETP4[D0D2] ring-of-experts dcp1", 10240, 32009.849060179964],
  ["PP2[N] DPA8[D0D1] EP4[D0] ETP2[D1] coalesced-a2a dcp1", 5888, 36848.20199398278],
  ["PP4[D0N] DPA4[D1] EP4[D1] coalesced-a2a dcp1", 2396, 29960.72828868705],
  ["DPA8[D0D1N] TP2[D2] EP4[D0N] ETP4[D1D2] ring-of-experts dcp2", 8872, 27753.410139565523],
  ["PP2[D0] DPA4[D1N] TP2[D2] EP4[D2N] ETP2[D1] ring-of-experts dcp2", 4656, 29102.230933344137],
  ["PP4[D0] DPA2[N] TP2[D1] EP4[D1N] ring-of-experts dcp2", 1818, 22728.457536757454],
  ["DPA4[D0N] TP4[D1D2] EP4[D1N] ETP4[D0D2] ring-of-experts dcp2", 7168, 22561.89424413057],
  ["PP2[D0] DPA2[N] TP4[D1D2] EP4[D1N] ETP2[D2] ring-of-experts dcp2", 3456, 21624.36213168164],
  ["PP4[D1N] TP4[D0D2] EP4[D0D2] ring-of-experts dcp2", 861, 10766.123621644392],
  ["DPA2[N] TP8[D0D1D2] EP4[D1N] ETP4[D0D2] ring-of-experts dcp4", 5206, 16269.578148686],
  ["PP2[D0] TP8[D1D2N] EP4[D1N] ETP2[D2] ring-of-experts dcp4", 1724, 10776.853397423225],
  ["TP16[D0D1D2N] EP4[D1N] ETP4[D0D2] ring-of-experts dcp4", 2816, 8821.616277111692],
  ["DPA16[D0D1D2N] EP2[D1] ETP8[D0D2N] ring-of-experts dcp1", 9680, 30264.6943631709],
  ["PP2[D1] DPA8[D0D2N] EP2[N] ETP4[D0D2] coalesced-a2a dcp1", 5600, 35020.161464444114],
  ["PP4[D0] DPA4[D1N] EP2[N] ETP2[D1] coalesced-a2a dcp1", 2292, 28657.235694758187],
  ["PP8[D0N] DPA2[D1] EP2[D1] coalesced-a2a dcp1", 138, 3453.0376560891405],
  ["DPA8[D0D2N] TP2[D1] EP2[N] ETP8[D0D1D2] ring-of-experts dcp2", 8464, 26450.9904494527],
  ["PP2[D0] DPA4[D1N] TP2[D2] EP2[N] ETP4[D1D2] ring-of-experts dcp2", 4600, 28755.419981481373],
  ["PP4[D0] DPA2[N] TP2[D1] EP2[D1] ETP2[N] ring-of-experts dcp2", 1792, 22629.59268028601],
  ["PP8[D0N] TP2[D1] EP2[D1] coalesced-a2a dcp1", 127, 3176.867362581647],
  ["DPA4[D1N] TP4[D0D2] EP2[N] ETP8[D0D1D2] ring-of-experts dcp2", 6960, 21756.84516318222],
  ["PP2[D0] DPA2[N] TP4[D1D2] EP2[D1] ETP4[D2N] ring-of-experts dcp2", 3390, 21193.37382594917],
  ["PP4[D0N] TP4[D1D2] EP2[D1] ETP2[D2] ring-of-experts dcp2", 854, 10679.803456049916],
  ["DPA2[N] TP8[D0D1D2] EP2[D1] ETP8[D0D2N] ring-of-experts dcp4", 5078, 15870.732282889227],
  ["PP2[D1] TP8[D0D2N] EP2[N] ETP4[D0D2] ring-of-experts dcp4", 1705, 10656.35710377104],
  ["TP16[D0D1D2N] EP2[D1] ETP8[D0D2N] ring-of-experts dcp4", 2782, 8694.616727377557],
  ["DPA16[DN] ETP16[DN] ring-of-experts dcp1", 8288, 25928.89378668522],
  ["PP2[D1] DPA8[D0D2N] ETP8[D0D2N] ring-of-experts dcp1", 5120, 33003.21431353179],
  ["PP4[D0] DPA4[D1N] ETP4[D1N] ring-of-experts dcp1", 2140, 26761.702341662203],
  ["PP8[D0N] DPA2[D1] ETP2[D1] ring-of-experts dcp1", 136, 3406.5797105837496],
  ["PP16[DN] ring-of-experts dcp1", 29, 1480.3558607723378],
  ["DPA8[D0D2N] TP2[D1] ETP16[D0D1D2N] ring-of-experts dcp2", 7888, 24668.62403717644],
  ["PP2[D0] DPA4[D1N] TP2[D2] ETP8[D1D2N] ring-of-experts dcp2", 4296, 26861.08466915161],
  ["PP4[D0] DPA2[N] TP2[D1] ETP4[D1N] ring-of-experts dcp2", 1770, 22129.851010666214],
  ["PP8[D0N] TP2[D1] ETP2[D1] ring-of-experts dcp1", 128, 3217.3540202568656],
  ["DPA4[D1N] TP4[D0D2] ETP16[D0D1D2N] ring-of-experts dcp2", 6540, 20444.67713611648],
  ["PP2[D1] DPA2[N] TP4[D0D2] ETP8[D0D2N] ring-of-experts dcp2", 3288, 20555.891864697336],
  ["PP4[D0] TP4[D1N] ETP4[D1N] ring-of-experts dcp2", 961, 12016.767702527593],
  ["DPA2[N] TP8[D] ETP16[DN] ring-of-experts dcp1", 3632, 11351.496247131974],
  ["PP2[D1] TP8[D0D2N] ETP8[D0D2N] ring-of-experts dcp4", 2183, 13645.153543954844],
  ["TP16[DN] ETP16[DN] ring-of-experts dcp8", 3029, 9465.633014499617],
];

test('decode search under an SLO pins V4 Flash on a B300 x8', () => {
  expectRows(
    search(
      'DeepSeek V4 Flash',
      'b300',
      { domain: 8, nodes: 1 },
      { kind: 'decode', policy: { sloTokPerSecPerUser: 50 } },
    ),
    V4_B300_8_DECODE_SLO,
  );
});

test('prefill search pins GLM on a TPU v7x 2x2x2 torus', () => {
  expectRows(
    search('GLM 5.3', 'tpu-v7x', '2x2x2', { kind: 'prefill', seqs: 8, mode: 'throughput' }),
    GLM_V7X_222_PREFILL,
  );
});

test('decode search across the scale-out network pins V4 Flash on two B300 x8 nodes', () => {
  expectRows(
    search(
      'DeepSeek V4 Flash',
      'b300',
      { domain: 8, nodes: 2 },
      { kind: 'decode', policy: { sloTokPerSecPerUser: 50 } },
    ),
    V4_B300_8X2_DECODE_SLO,
  );
});
