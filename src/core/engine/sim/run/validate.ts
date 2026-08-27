import { dcpAxes, dcpSize, Diagnostic, roleSize, validateSizes } from '../../surface/deploy';
import { ChipSpec, runsAs } from '../../../hardware/chips';
import type { CategoryDtypes, Dtype } from '../../../model/dtype';
import type { ModelSpec } from '../../../model/models';
import { partitionIntoStages } from '../lowering/stages';
import { memoryFootprint } from './memory';
import type { SimInput } from '../../surface/api';

// The model as this chip will actually run it: an activation format it has
// no unit for becomes one it does, the substitution the dtype-widened
// warning below reports. Weights are untouched, they stay packed in the
// stored format whatever the chip's table says: a widening kernel is
// assumed to exist, and weights-kernel-missing warns where one still has to
// be written. Unchanged models are returned as-is, so the common case
// allocates nothing.
export function runnableOn(m: ModelSpec, chip: ChipSpec): ModelSpec {
  const { activations, indexer } = m.precision;
  const runs = { ...activations };
  let same = true;
  for (const k of Object.keys(runs) as (keyof CategoryDtypes)[]) {
    runs[k] = runsAs(chip, activations[k]);
    same &&= runs[k] === activations[k];
  }
  const runnableIndexer = indexer === undefined ? undefined : runsAs(chip, indexer);
  same &&= runnableIndexer === indexer;
  return same
    ? m
    : { ...m, precision: { ...m.precision, activations: runs, indexer: runnableIndexer } };
}

// Full feasibility of a sim input: the role-size checks, the chip's dtype
// support, and the weights-residency gate. Only batch-dependent KV room
// (kv-no-room) is left to evaluation time.
export function validateInput(input: SimInput): Diagnostic[] {
  const { model, deployment, workload } = input;
  const pp = roleSize(deployment.mesh, 'PP');
  const dcp = dcpSize(deployment);
  const diags = validateSizes(model, {
    TP: roleSize(deployment.mesh, 'TP'),
    PP: pp,
    EP: roleSize(deployment.mesh, 'EP'),
    DCP: dcp,
  });
  // A legal size still needs TP dims that multiply to it on THIS
  // placement. Unlike dcp-not-divisor this is not a property of the size
  // tuple: another placement that splits the axis differently can hold the
  // same DCP, which is why the search prices every placement.
  if (dcp > 1 && dcpAxes(deployment.mesh, dcp) === null)
    diags.push({
      severity: 'error',
      code: 'dcp-unplaceable',
      message: `no subset of this placement's TP dims multiplies to DCP = ${dcp}`,
    });

  // If a chip doesn't support doing matmuls in one of a model's activation
  // type, we'll do it in the next-widest type that the chip does support.
  const chip = deployment.chip;
  const asked = [
    ...new Set<Dtype>([
      ...Object.values(model.precision.activations),
      ...(model.precision.indexer === undefined ? [] : [model.precision.indexer]),
    ]),
  ];
  const widened = asked
    .map((want) => [want, runsAs(chip, want)] as const)
    .filter(([want, got]) => got !== want);
  if (widened.length)
    diags.push({
      severity: 'warning',
      code: 'dtype-widened',
      message: `${chip.name} has no ${widened.map(([w, g]) => `${w} unit: it runs as ${g}`).join(', ')}`,
    });

  // A weight format the chip has no matmul unit for cannot be multiplied as
  // stored, so a special kernel is needed to unpack it into a wider format.
  // Either way the weights are priced packed and the unpack free. If a known
  // kernel implementation exists, an info note is enough, but if no known
  // kernel implementation exists, we warn this will require some work.
  const stored = [...new Set<Dtype>(Object.values(model.precision.weights))];
  const kernelMissing = stored.filter((d) => chip.formats[d] === undefined);
  if (kernelMissing.length)
    diags.push({
      severity: 'warning',
      code: 'weights-kernel-missing',
      message: `no known kernel can unpack ${kernelMissing.join(' or ')} on ${chip.name}: priced as if one existed, but serving this means writing it`,
    });
  const kernelWidened = stored.filter((d) => chip.formats[d] === 'kernel-widened');
  if (kernelWidened.length)
    diags.push({
      severity: 'info',
      code: 'weights-unpacked',
      message: `${chip.name} has no ${kernelWidened.join(' or ')} unit: a kernel unpacks those weights into a wider matmul on the way in`,
    });

  if (diags.some((d) => d.severity === 'error')) return diags;

  const stages = partitionIntoStages(model, pp);
  const memory = memoryFootprint(
    model,
    deployment,
    stages,
    workload.prefillLen + workload.generateLen,
  );
  if (memory.weightBytesPerChip >= deployment.chip.hbmCapacity)
    diags.push({
      severity: 'error',
      code: 'weights-dont-fit',
      message: `weights need ${(memory.weightBytesPerChip / 1e9).toFixed(1)} GB/chip but HBM is ${(deployment.chip.hbmCapacity / 1e9).toFixed(0)} GB`,
    });
  return diags;
}
