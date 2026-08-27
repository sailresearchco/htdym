import { AttentionOp, OpId, LoweredOp, Segment, Unbuilt } from '../ir/ops';
import { MeshAxis, TensorType, dimEq, sameAxes, tt, typeEq } from '../ir/tensors';
import type { Dtype } from '../../../model/dtype';

export interface Val {
  id: OpId | null;
  type: TensorType;
  // format this value is materialized in, carried so a reshard of it ships
  // what it is actually made of rather than a model-wide guess
  dtype: Dtype;
}

export class TraceBuilder {
  private readonly segments: Segment<LoweredOp>[] = [];
  private cur: Segment<LoweredOp> = { label: 'stage', ops: [], repeat: 1 };
  private seq = 0;

  // close the current segment and open a new one for the ops that follow
  startSegment(label: string, repeat = 1): void {
    if (this.cur.ops.length) this.segments.push(this.cur);
    this.cur = { label, ops: [], repeat };
  }

  private push(op: Unbuilt<LoweredOp>, deps: Val[], dtype: Dtype): Val {
    const full = {
      ...op,
      id: `${op.label}#${this.seq++}` as OpId,
      deps: deps.map((v) => v.id).filter((id): id is OpId => id != null),
    } as LoweredOp;
    this.cur.ops.push(full);
    return { id: full.id, type: full.out, dtype };
  }

  tt(shape: number[], sharding?: MeshAxis[][], unreduced?: MeshAxis[]): TensorType {
    return tt(shape, sharding, unreduced);
  }

  // stage-input value produced by no op in this trace
  source(shape: number[], sharding: MeshAxis[][] | undefined, dtype: Dtype): Val {
    return { id: null, type: this.tt(shape, sharding), dtype };
  }

  // change only a value's logical shape without moving data.
  reshape(
    x: Val,
    shape: number[],
    sharding: MeshAxis[][] = x.type.sharding,
    unreduced: MeshAxis[] = x.type.unreduced,
  ): Val {
    if (shape.length !== sharding.length) throw new Error(`rank mismatch`);
    return { id: x.id, type: this.tt(shape, sharding, unreduced), dtype: x.dtype };
  }

  // change only a value's sharding/partial-sum layout without moving data.
  // `dtype` casts the payload on the way (an MoE dispatch narrows tokens to
  // the format its experts consume before putting them on the wire).
  reshard(
    x: Val,
    sharding: MeshAxis[][],
    unreduced: MeshAxis[] = x.type.unreduced,
    opts: { dtype?: Dtype; extraDeps?: Val[] } = {},
  ): Val {
    const dtype = opts.dtype ?? x.dtype;
    return this.push(
      {
        kind: 'reshard',
        label: 'reshard',
        x: x.type,
        dtype,
        out: this.tt(x.type.shape, sharding, unreduced),
      },
      [x, ...(opts.extraDeps ?? [])],
      dtype,
    );
  }

  // `streams` is the hyper-connection expansion rate: the state handed
  // between blocks is that many residual streams wide even though each
  // block reads and writes one stream's worth, so only what crosses a
  // stage boundary carries the multiple.
  pipelineSend(x: Val, streams = 1): Val {
    const last = x.type.shape.length - 1;
    const out =
      streams === 1
        ? x.type
        : this.tt(
            x.type.shape.map((n, i) => (i === last ? n * streams : n)),
            x.type.sharding,
            x.type.unreduced,
          );
    return this.push({ kind: 'p2p', label: 'p2p', dtype: x.dtype, out }, [x], x.dtype);
  }

  weight(
    label: string,
    opts: {
      shape: number[];
      sharding?: MeshAxis[][];
      // format the checkpoint is held in on this chip
      dtype: Dtype;
      loadFraction?: number;
    },
  ): Val {
    const out = this.tt(opts.shape, opts.sharding);
    return this.push(
      { kind: 'weight-load', label, out, dtype: opts.dtype, loadFraction: opts.loadFraction ?? 1 },
      [],
      opts.dtype,
    );
  }

  // x[m, k] * w[k, n], with operands arriving in `dtype`. Contraction dims
  // must agree in size and sharding; a sharded contraction leaves the output
  // unreduced over those axes.
  gemm(label: string, x: Val, w: Val, dtype: Dtype): Val {
    const [xt, wt] = [x.type, w.type];

    if (xt.shape.length !== 2 || wt.shape.length !== 2) throw new Error(`wrong shape`);

    const [m, k1] = xt.shape;
    const [k2, n] = wt.shape;

    const [mSharding, k1Sharding] = xt.sharding;
    const [k2Sharding, nSharding] = wt.sharding;

    if (xt.unreduced.length) throw new Error(`unreduced partials`);
    if (!dimEq(k1, k2)) throw new Error(`contraction size mismatch`);
    if (!sameAxes(k1Sharding, k2Sharding)) throw new Error(`contraction sharding mismatch`);

    return this.push(
      {
        kind: 'gemm',
        label,
        out: this.tt(
          [m, n],
          [mSharding, nSharding],
          k1Sharding, // a sharded contraction leaves partial sums
        ),
        x: xt,
        w: wt,
        dtype,
      },
      [x, w],
      dtype,
    );
  }

  groupedGemm(
    label: string,
    x: Val,
    w: Val,
    opts: { groups: number; dtype: Dtype; extraDeps?: Val[] },
  ): Val {
    const [xt, wt] = [x.type, w.type];

    if (xt.shape.length !== 2 || wt.shape.length !== 3)
      throw new Error(`expected x[m,k] * w[g,k,n]`);

    const [m, k1] = xt.shape;
    const [, k2, outputCols] = wt.shape;

    const [mSharding, k1Sharding] = xt.sharding;
    const [gSharding, k2Sharding, nSharding] = wt.sharding;

    const partialGroups = gSharding.filter((axis) => !mSharding.includes(axis));

    if (!xt.unreduced.every((axis) => partialGroups.includes(axis)))
      throw new Error(`unreduced partials outside group axes`);
    if (!dimEq(k1, k2)) throw new Error(`contraction size mismatch`);
    if (!sameAxes(k1Sharding, k2Sharding)) throw new Error(`contraction sharding mismatch`);

    return this.push(
      {
        kind: 'gemm',
        label,
        out: this.tt([m, outputCols], [mSharding, nSharding], [...partialGroups, ...k1Sharding]),
        x: xt,
        w: wt,
        dtype: opts.dtype,
        groups: opts.groups,
      },
      [x, w, ...(opts.extraDeps ?? [])],
      opts.dtype,
    );
  }

  ffn(
    prefix: string,
    x: Val, // [B_dpa, D]
    opts: {
      ff: number;
      ffSharding: MeshAxis[];
      variant: 'gated' | 'standard';
      // format the weights are held in; `dtype` is what the matmuls run in
      weights: Dtype;
      dtype: Dtype;
    },
  ): Val {
    const { ff, ffSharding } = opts;

    const expandedFf = opts.variant === 'gated' ? 2 * ff : ff;
    const d = x.type.shape[1];

    const w1Load = this.weight(`${prefix}-in-weight`, {
      shape: [d, expandedFf],
      sharding: [[], ffSharding],
      dtype: opts.weights,
    }); // [D, expandedFf_ffSharding]

    const w2Load = this.weight(`${prefix}-out-weight`, {
      shape: [ff, d],
      sharding: [ffSharding, []],
      dtype: opts.weights,
    }); // [ff_ffSharding, D]

    // [B_dpa, D] * [D, expandedFf_ffSharding] = [B_dpa, expandedFf_ffSharding]
    const up = this.gemm(`${prefix}-in`, x, w1Load, opts.dtype);

    // GLU combines the gate/up halves in place [B_dpa, expandedFf_ffSharding] -> [B_dpa, ff_ffSharding]
    const hidden = this.reshape(up, [up.type.shape[0], ff]);

    // [B_dpa, ff_ffSharding] * [ff_ffSharding, D] = [B_dpa, D] {U_ffSharding}
    return this.gemm(`${prefix}-out`, hidden, w2Load, opts.dtype);
  }

  expertFfn(
    prefix: string,
    x: Val, // [(B*topK)_epSharding, D], the routed token-expert pairs
    opts: {
      ff: number;
      ffSharding: MeshAxis[];
      variant: 'gated' | 'standard';
      // format the weights are held in; `dtype` is what the matmuls run in
      weights: Dtype;
      dtype: Dtype;
      loadFraction?: number;
      experts: { count: number; epSharding: MeshAxis[]; activeGroups: number };
      // Extra dependencies of the first gemm (routing decisions).
      extraDeps?: Val[];
    },
  ): Val {
    const { ff, ffSharding, experts } = opts;

    const expandedFf = opts.variant === 'gated' ? 2 * ff : ff;
    const d = x.type.shape[1];

    const w1Load = this.weight(`${prefix}-in-weight`, {
      shape: [experts.count, d, expandedFf],
      sharding: [experts.epSharding, [], ffSharding],
      dtype: opts.weights,
      loadFraction: opts.loadFraction,
    }); // [E_epSharding, D, expandedFf_ffSharding]

    const w2Load = this.weight(`${prefix}-out-weight`, {
      shape: [experts.count, ff, d],
      sharding: [experts.epSharding, ffSharding, []],
      dtype: opts.weights,
      loadFraction: opts.loadFraction,
    }); // [E_epSharding, ff_ffSharding, D]

    const groups = Math.max(1, experts.activeGroups);

    // [(B*topK)_ep, D] * [E_ep, D, expandedFf_ffSharding] = [(B*topK)_ep, expandedFf_ffSharding]
    const up = this.groupedGemm(`${prefix}-in`, x, w1Load, {
      groups,
      dtype: opts.dtype,
      extraDeps: opts.extraDeps,
    });

    // GLU combines [(B*topK)_ep, expandedFf_ffSharding] -> [(B*topK)_ep, ff_ffSharding]
    const hidden = this.reshape(up, [up.type.shape[0], ff]);

    // [(B*topK)_ep, ff_ffSharding] * [E_ep, ff_ffSharding, D] = [(B*topK)_ep, D] {U_ffSharding}
    return this.groupedGemm(`${prefix}-out`, hidden, w2Load, { groups, dtype: opts.dtype });
  }

  attention(
    label: string,
    x: Val,
    opts: Omit<AttentionOp, 'kind' | 'id' | 'label' | 'deps' | 'out'> & { outCols: number },
  ): Val {
    const { outCols, ...rest } = opts;
    const out = this.tt([x.type.shape[0], outCols], [...x.type.sharding]);
    return this.push({ kind: 'attention', label, out, ...rest }, [x], rest.dtype);
  }

  // dependency-only convergence of branches that must agree in type but
  // not in width: `dtype` states the width their sum lands in
  join(label: string, vals: Val[], dtype: Dtype): Val {
    for (const v of vals.slice(1))
      if (!typeEq(v.type, vals[0].type)) throw new Error(`disagreeing types`);
    return this.push({ kind: 'join', label, out: vals[0].type }, vals, dtype);
  }

  expect(x: Val, type: TensorType): Val {
    if (!typeEq(x.type, type)) throw new Error(`expected type mismatch`);
    return x;
  }

  build(): Segment<LoweredOp>[] {
    if (this.cur.ops.length) this.segments.push(this.cur);
    return this.segments;
  }
}
