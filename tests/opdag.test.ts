import { expect, test } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { evaluateDecodeAtBatch } from '../src/core/engine/sim/run/decode';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { makeMesh } from '../src/core/engine/surface/deploy';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';
import { graphElements, visibleOps } from '../src/ui/components/OpDagView';

// The node styles are computed from times in seconds, which are small enough
// to serialise as 1e-7 - not a valid CSS number, so the browser drops the
// whole declaration and the element silently falls back (a scaleX(1e-7) bar
// renders full width). Asserting on the emitted CSS is the only check that
// sees this; recomputing the ratios here would just re-derive the numbers
// that produced them.
const chip = CHIPS_BY_ID['tpu-v6e'];
const backend = makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 });

function segments(modelName: string, phase: 'prefill' | 'decode') {
  const model = MODEL_PRESETS.find((m) => m.name === modelName)!;
  const slice = chip.interconnect.topologies!.find(
    (t) => t.dims.reduce((a, b) => a * b, 1) === 32,
  )!;
  const axes = deployedAxes(chip.interconnect, slice);
  const names = axes.map((a) => a.name);
  const deployment = {
    chip,
    mesh: makeMesh(axes, { DPA: names, TP: [], EP: names, ETP: [], PP: [] }),
    moeDispatch: 'ring-of-experts' as const,
  };
  const input = { model, deployment, workload: { prefillLen: 8192, generateLen: 10 } };
  const r =
    phase === 'prefill'
      ? evaluatePrefill(input, 32, 'throughput', { costBackend: backend })
      : evaluateDecodeAtBatch(input, 64, 1, { costBackend: backend, ignoreKvCapacity: true });
  if (!r.ok) throw new Error(`eval failed`);

  const bound = backend(deployment);
  return r.perStageTrace[0].map((seg) => {
    const ops = visibleOps(seg.ops);
    return graphElements(ops, bound.priceTrace([{ label: seg.label, ops, repeat: 1 }]).busyPerOp);
  });
}

test('op nodes emit CSS a browser will actually apply', () => {
  for (const name of ['Gemma 4 12B', 'gpt-oss-120b MXFP4/BF16'])
    for (const phase of ['prefill', 'decode'] as const)
      for (const { nodes } of segments(name, phase))
        for (const node of nodes) {
          const html = renderToStaticMarkup((node.data as { label: JSX.Element }).label);

          for (const [, decl] of html.matchAll(/style="([^"]*)"/g)) {
            expect(decl, `${node.id} in ${name}/${phase}`).not.toMatch(/\de[+-]?\d/i);
            expect(decl).toMatch(/^(height:[\d.]+%|transform:scaleX\([\d.]+\))$/);
          }

          // the rail is the op's resource split, so its parts fill it exactly
          const rail = [...html.matchAll(/op-rail-\w+" style="height:([\d.]+)%/g)].map((m) =>
            Number(m[1]),
          );
          if (rail.length)
            expect(
              rail.reduce((a, b) => a + b, 0),
              `${node.id} rail`,
            ).toBeCloseTo(100, 1);

          const [, bar] = html.match(/scaleX\(([\d.]+)\)/) ?? [];
          if (bar !== undefined) expect(Number(bar)).toBeLessThanOrEqual(1);
        }
});

test('the slowest op in a block fills its cost bar, and only it', () => {
  for (const { nodes } of segments('Gemma 4 12B', 'decode')) {
    const bars = nodes
      .map((n) => renderToStaticMarkup((n.data as { label: JSX.Element }).label))
      .flatMap((html) => Number(html.match(/scaleX\(([\d.]+)\)/)?.[1] ?? NaN))
      .filter((v) => !Number.isNaN(v));

    expect(Math.max(...bars)).toBe(1);
    expect(bars.filter((v) => v === 1)).toHaveLength(1);
  }
});
