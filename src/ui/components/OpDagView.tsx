import { useMemo, useState } from 'react';
import dagre from '@dagrejs/dagre';
import {
  Background,
  Controls,
  MarkerType,
  Position,
  ReactFlow,
  useNodesState,
  type Edge,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import type { ModelSpec } from '../../core/model/models';
import { makeNaiveOpCostSumBackend } from '../../core/engine/sim/cost/naiveOpCostSum';
import { evaluateDecodeAtBatch } from '../../core/engine/sim/run/decode';
import { evaluatePrefill } from '../../core/engine/sim/run/prefill';
import type { ExpandedOp, OpId } from '../../core/engine/sim/ir/ops';
import type { HardwareResource } from '../../core/engine/surface/api';
import { typeEq, type TensorType } from '../../core/engine/sim/ir/tensors';
import type { Deployment } from '../../core/engine/surface/deploy';
import type { ChipSpec } from '../../core/hardware/chips';
import { fmtSI, fmtTime } from '../format';
import type { UiOverlap, UiResult } from '../results';

type Phase = 'decode' | 'prefill';

const RESOURCES = ['compute', 'memory', 'comms'] as const;

const NODE_WIDTH = 200;
const NODE_HEIGHT = 50;

const fmtDim = (n: number) => (Number.isInteger(n) ? `${n}` : `${Math.round(n * 100) / 100}`);
const axStr = (axes: string[]) => axes.join(',');

// plain-text layout / label / detail, for the node's hover title
const layoutText = (t: TensorType) =>
  `[${t.shape
    .map((s, i) => (t.sharding[i]?.length ? `${fmtDim(s)}[${axStr(t.sharding[i])}]` : fmtDim(s)))
    .join(', ')}]${t.unreduced.length ? ` {U:${axStr(t.unreduced)}}` : ''}`;
const labelText = (op: ExpandedOp) =>
  op.kind === 'collective' ? `${op.variant} (${axStr(op.axes)})` : op.label;
const detailText = (op: ExpandedOp) =>
  op.kind === 'attention' ? `${layoutText(op.out)} · ${fmtSI(op.flops)} FLOP` : layoutText(op.out);

// which axes shard something + which are partial, ignoring dim placement:
// two layouts differ here only on a genuine relayout (not a shape-only view)
const layoutKey = (t: TensorType) =>
  `${[...new Set(t.sharding.flat())].sort().join(',')}|${[...t.unreduced].sort().join(',')}`;

// each mesh axis in a distinct font/color; separators stay in the base style
const Axes = ({ axes }: { axes: string[] }) => (
  <>
    {axes.map((a, i) => (
      <span key={a}>
        {i > 0 ? ',' : ''}
        <span className="op-flow-axis">{a}</span>
      </span>
    ))}
  </>
);

// scaling-book layout: global shape, each sharded dim's axes subscripted,
// trailing unreduced (partial-sum) axes in braces
function LayoutView({ t }: { t: TensorType }) {
  return (
    <>
      [
      {t.shape.map((s, i) => (
        <span key={i}>
          {i > 0 ? ', ' : ''}
          {fmtDim(s)}
          {t.sharding[i]?.length ? (
            <sub className="op-flow-sub">
              <Axes axes={t.sharding[i]} />
            </sub>
          ) : null}
        </span>
      ))}
      ]
      {t.unreduced.length ? (
        <>
          {' {U:'}
          <Axes axes={t.unreduced} />
          {'}'}
        </>
      ) : null}
    </>
  );
}

// A reshard whose cheapest plan is empty (nothing to reduce, layout already
// right) survives expansion as a join that hands its input straight on. It
// moves no data and costs nothing, so draw it as an edge, not a node.
export function visibleOps(ops: ExpandedOp[]): ExpandedOp[] {
  const byId = new Map(ops.map((o) => [o.id as string, o]));
  const passThrough = new Map<string, OpId>();
  const resolve = (id: OpId) => passThrough.get(id) ?? id;

  // deps precede their consumers, so a chain of them collapses in one pass
  for (const op of ops) {
    const dep = op.kind === 'join' && op.deps.length === 1 ? byId.get(op.deps[0]) : undefined;
    if (dep && typeEq(dep.out, op.out)) passThrough.set(op.id, resolve(op.deps[0]));
  }

  return ops
    .filter((op) => !passThrough.has(op.id))
    .map((op) => ({ ...op, deps: [...new Set(op.deps.map(resolve))] }));
}

// what an op holds each resource for, summed
const busyTotal = (cost: Record<HardwareResource, number>) =>
  cost.compute + cost.memory + cost.comms;

export function graphElements(
  ops: ExpandedOp[],
  // absent when the active backend cannot attribute its cost per op
  busy?: ReadonlyMap<OpId, Record<HardwareResource, number>>,
): { nodes: Node[]; edges: Edge[] } {
  const slowest = Math.max(...[...(busy?.values() ?? [])].map(busyTotal), Number.MIN_VALUE);
  const byId = new Map(ops.map((o) => [o.id as string, o]));

  // MoE dispatch/combine (and similar) are zero-cost views the lowerer folds
  // into a producer's id, so a consumer's input layout can differ from what
  // its producing node shows. Surface each such relayout as its own node.
  const reshapes = ops.flatMap((op) => {
    if (!('x' in op)) return [];
    const producer = byId.get(op.deps[0]);
    if (!producer || producer.kind === 'weight-load') return [];
    if (layoutKey(producer.out) === layoutKey(op.x)) return [];
    return [
      { id: `${op.id}~reshape`, from: op.deps[0] as string, to: op.id as string, type: op.x },
    ];
  });
  const reshapeOf = new Map(reshapes.map((r) => [`${r.from}->${r.to}`, r]));

  // A block reads the residual the block before it wrote, so its producer is
  // in another segment (in the very first, in no op at all: it is the stage
  // input). Stand one node in for it, and route every out-of-segment dep
  // there, so a block's graph starts at the value it consumes.
  const RESIDUAL = 'residual';
  const entry = ops.find(
    (op) => 'x' in op && !op.deps.some((d) => byId.get(d) && byId.get(d)!.kind !== 'weight-load'),
  );
  const inputs = new Map(
    ops.map((op) => [
      op.id as string,
      [
        ...new Set([
          ...(op === entry ? [RESIDUAL] : []),
          ...op.deps.map((d) => (byId.has(d) ? (d as string) : RESIDUAL)),
        ]),
      ],
    ]),
  );

  const graph = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));
  graph.setGraph({
    rankdir: 'TB',
    ranker: 'tight-tree',
    nodesep: 28,
    edgesep: 12,
    ranksep: 58,
    marginx: 20,
    marginy: 20,
  });
  for (const id of [
    ...(entry ? [RESIDUAL] : []),
    ...ops.map((o) => o.id as string),
    ...reshapes.map((r) => r.id),
  ])
    graph.setNode(id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  for (const op of ops)
    for (const dep of inputs.get(op.id)!) {
      const r = reshapeOf.get(`${dep}->${op.id}`);
      if (r) graph.setEdge(dep, r.id).setEdge(r.id, op.id);
      else graph.setEdge(dep, op.id);
    }
  dagre.layout(graph);

  const place = (id: string) => {
    const p = graph.node(id);
    return { x: p.x - NODE_WIDTH / 2, y: p.y - NODE_HEIGHT / 2 };
  };
  const shell = (id: string, className: string, label: JSX.Element): Node => ({
    id,
    position: place(id),
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
    className,
    data: { label },
  });

  const nodes: Node[] = [
    ...(entry && 'x' in entry
      ? [
          shell(
            RESIDUAL,
            'op-flow-node op-flow-node-residual',
            <div title={`residual — ${layoutText(entry.x)}`}>
              <div className="op-flow-label">residual</div>
              <div className="op-flow-detail">
                <LayoutView t={entry.x} />
              </div>
            </div>,
          ),
        ]
      : []),
    ...ops.map((op) => {
      const cost = busy?.get(op.id);
      const took = cost && busyTotal(cost);
      return shell(
        op.id,
        `op-flow-node op-flow-node-${op.kind}`,
        <div
          title={`${labelText(op)} — ${detailText(op)}${took === undefined ? '' : ` — ${fmtTime(took)}`}`}
        >
          <div className="op-flow-label">
            <span>
              {op.kind === 'collective' ? (
                <>
                  {op.variant} (<Axes axes={op.axes} />)
                </>
              ) : (
                op.label
              )}
            </span>
            {took ? <span className="op-flow-time">{fmtTime(took)}</span> : null}
          </div>
          <div className="op-flow-detail">
            <LayoutView t={op.out} />
            {op.kind === 'attention' ? ` · ${fmtSI(op.flops)} FLOP` : null}
          </div>
          {cost && took ? (
            // where this op's time actually goes, over the static rail the
            // node kind paints: attention is compute in prefill and KV reads
            // in decode, and only the priced split can tell them apart
            <i className="op-flow-rail">
              {RESOURCES.map((resource) =>
                cost[resource] > 0 ? (
                  <i
                    key={resource}
                    className={`op-rail-${resource}`}
                    // explicit shares, not flex-grow: factors summing below 1
                    // only claim that fraction of the rail, and a raw ratio
                    // can serialise as 1e-7, which is not a valid CSS number
                    style={{ height: `${((cost[resource] / took) * 100).toFixed(2)}%` }}
                  />
                ) : null,
              )}
            </i>
          ) : null}
          {took === undefined ? null : (
            <i
              className="op-flow-cost"
              style={{ transform: `scaleX(${(took / slowest).toFixed(4)})` }}
            />
          )}
        </div>,
      );
    }),
    ...reshapes.map((r) =>
      shell(
        r.id,
        'op-flow-node op-flow-node-reshape',
        <div title={`reshape — ${layoutText(r.type)}`}>
          <div className="op-flow-label">reshape</div>
          <div className="op-flow-detail">
            <LayoutView t={r.type} />
          </div>
        </div>,
      ),
    ),
  ];

  const edge = (source: string, target: string, stroke?: string): Edge => ({
    id: `${source}->${target}`,
    source,
    target,
    type: 'bezier',
    markerEnd: { type: MarkerType.ArrowClosed, color: stroke },
    style: stroke ? { stroke } : undefined,
  });
  // a matmul's operands: weight-load in red, the activation input in green,
  // everything else (grouped-GEMM routing, etc.) stays the default gray
  const operandColor = (dep: string, op: ExpandedOp, first: string) => {
    if (op.kind !== 'gemm') return undefined;
    if (byId.get(dep)?.kind === 'weight-load') return 'var(--red)';
    return dep === first ? 'var(--green)' : undefined;
  };
  const edges: Edge[] = ops.flatMap((op) => {
    const deps = inputs.get(op.id)!;
    return deps.flatMap((dep) => {
      const r = reshapeOf.get(`${dep}->${op.id}`);
      // a reshape only ever sits on the activation input (deps[0])
      if (r)
        return [
          edge(dep, r.id),
          edge(r.id, op.id, op.kind === 'gemm' ? 'var(--green)' : undefined),
        ];
      return [edge(dep, op.id, operandColor(dep, op, deps[0]))];
    });
  });

  return { nodes, edges };
}

function DagCanvas({
  ops,
  busy,
}: {
  ops: ExpandedOp[];
  busy?: ReadonlyMap<OpId, Record<HardwareResource, number>>;
}) {
  const initial = useMemo(() => graphElements(ops, busy), [ops, busy]);
  const [nodes, , onNodesChange] = useNodesState(initial.nodes);

  return (
    <div className="op-flow-canvas">
      <ReactFlow
        nodes={nodes}
        edges={initial.edges}
        onNodesChange={onNodesChange}
        fitView
        fitViewOptions={{ padding: 0.12, maxZoom: 0.85 }}
        minZoom={0.03}
        maxZoom={2}
        nodesConnectable={false}
        deleteKeyCode={null}
        nodesFocusable
        onlyRenderVisibleElements
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={18} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

export function OpDagView({
  result,
  model,
  chip,
  overlap,
}: {
  result: UiResult;
  model: ModelSpec;
  chip: ChipSpec;
  overlap: UiOverlap;
}) {
  const [phase, setPhase] = useState<Phase>(result.decode ? 'decode' : 'prefill');
  // null = follow the phase's own critical stage / dominant segment
  const [stageSel, setStageSel] = useState<number | null>(null);
  const [segSel, setSegSel] = useState<number | null>(null);

  const { run, backend } = useMemo(() => {
    const deployment: Deployment = { chip, mesh: result.mesh, moeDispatch: result.dispatch };
    const input = { model, deployment, workload: result.workload };
    const costBackend = makeNaiveOpCostSumBackend({
      memoryOverlap: overlap.memoryOverlap,
      commsOverlap: overlap.commsOverlap,
    });
    const pp = result.sizes.PP ?? 1;
    const ev =
      phase === 'decode'
        ? evaluateDecodeAtBatch(input, result.decode!.batchPerStage, pp, {
            costBackend,
            ignoreKvCapacity: true,
          })
        : evaluatePrefill(input, result.prefill!.batchSeqs, 'throughput', { costBackend });
    return {
      run: ev.ok ? ev : null,
      // same context the run bound, so this shares its reshard-plan cache
      backend: costBackend(deployment),
    };
  }, [result, model, chip, overlap.memoryOverlap, overlap.commsOverlap, phase]);

  const stages = run?.perStageTrace ?? [];
  const stageCount = stages.length;
  const stageIndex = Math.min(stageSel ?? run?.criticalStage ?? 0, Math.max(0, stageCount - 1));
  const segments = stages[stageIndex] ?? [];
  // default to the segment that repeats most (the layer that dominates the step)
  const segDefault = segments.reduce((best, s, i, a) => (s.repeat > a[best].repeat ? i : best), 0);
  const segIndex = Math.min(segSel ?? segDefault, Math.max(0, segments.length - 1));
  const ops = useMemo(() => visibleOps(segments[segIndex]?.ops ?? []), [segments, segIndex]);
  // one block's worth of the backend's own cost, attributed per op. Backends
  // that cannot answer per op leave it out and the nodes simply show no time.
  const busy = useMemo(
    () =>
      ops.length ? backend.priceTrace([{ label: 'block', ops, repeat: 1 }]).busyPerOp : undefined,
    [backend, ops],
  );

  const choosePhase = (next: Phase) => {
    setPhase(next);
    setStageSel(null);
    setSegSel(null);
  };
  const chooseStage = (next: number) => {
    setStageSel(next);
    setSegSel(null);
  };

  return (
    <div className="op-dag">
      <div className="op-dag-controls">
        <div className="op-dag-toggle" aria-label="Trace phase">
          {result.decode && (
            <button
              type="button"
              className={phase === 'decode' ? 'active' : ''}
              onClick={() => choosePhase('decode')}
            >
              Decode
            </button>
          )}
          {result.prefill && (
            <button
              type="button"
              className={phase === 'prefill' ? 'active' : ''}
              onClick={() => choosePhase('prefill')}
            >
              Prefill
            </button>
          )}
        </div>
        {segments.length > 1 && (
          <label>
            Block
            <select value={segIndex} onChange={(e) => setSegSel(Number(e.target.value))}>
              {segments.map((s, index) => (
                <option key={index} value={index}>
                  {s.label}
                  {s.repeat > 1 ? ` ×${s.repeat}` : ''}
                </option>
              ))}
            </select>
          </label>
        )}
        {stageCount > 1 && (
          <label className="op-dag-stage">
            Stage
            <select value={stageIndex} onChange={(e) => chooseStage(Number(e.target.value))}>
              {stages.map((_, index) => (
                <option
                  key={index}
                  value={index}
                  title={index === run?.criticalStage ? 'critical stage' : undefined}
                >
                  {index + 1}
                  {index === run?.criticalStage ? ' ★' : ''}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {ops.length ? (
        <DagCanvas key={`${result.id}:${phase}:${stageIndex}:${segIndex}`} ops={ops} busy={busy} />
      ) : (
        <div className="op-flow-canvas op-flow-empty">Trace unavailable for this config.</div>
      )}

      <div className="op-dag-legend">
        <span>
          <i className="op-dag-key op-dag-key-compute" />
          Compute
        </span>
        <span>
          <i className="op-dag-key op-dag-key-memory" />
          HBM
        </span>
        <span>
          <i className="op-dag-key op-dag-key-comms" />
          Collective
        </span>
        <span className="op-dag-count">{ops.length} ops</span>
      </div>
    </div>
  );
}
