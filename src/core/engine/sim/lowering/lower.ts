import { expertReadFraction } from '../../../model/utils';
import { ModelSpec } from '../../../model/models';
import { DTYPE_BYTES } from '../../../model/dtype';
import {
  blockAttnFlopsDecodeParts,
  blockAttnFlopsPrefillParts,
  blockAttnParams,
  blockKvBytesParts,
  blockKvHeads,
} from '../../../model/block';
import { dcpAxes, dcpSize, kvFraction, Deployment, roleSize } from '../../surface/deploy';
import { LoweredOp, Segment } from '../ir/ops';
import type { Stage } from './stages';
import { TraceBuilder } from './builder';
import { lowerMoe } from './moe';

export interface LowerArgs {
  model: ModelSpec;
  deployment: Deployment;
  phase: 'decode' | 'prefill';
  // tokens the whole stage processes this step (all batch groups)
  tokensTotal: number;
  // tokens on one batch group (TTFT lowers a single active group)
  tokensPerGroup: number;
  // sequences per batch group (prefill attention is per-seq quadratic)
  seqsPerGroup: number;
  // decode: mean context, prefill: prompt length T
  ctx: number;
  // emit the stage-boundary send (PP > 1 and not the last stage)
  includeBoundary: boolean;
}

export function lowerStage(args: LowerArgs, stage: Stage): Segment<LoweredOp>[] {
  const { model, tokensPerGroup, tokensTotal, seqsPerGroup } = args;
  const { vocab, modelDim: d_model, precision } = model;
  const acts = precision.activations;
  const { roles } = args.deployment.mesh;
  const tp = roleSize(args.deployment.mesh, 'TP');
  const dcp = dcpSize(args.deployment);
  // null means no subset of the TP dims multiplies to dcp on this
  // placement; validateInput rejects that before lowering ever runs
  const dcpGroup = dcpAxes(args.deployment.mesh, dcp) ?? [];
  const tpMinusDcp = roles['TP'].filter((a) => !dcpGroup.includes(a));

  const trace = new TraceBuilder();

  let residual = trace.source([tokensTotal, d_model], [roles['DPA'], []], precision.residual); // [B_dpa, D]
  const residualT = residual.type;

  for (const group of stage.groups)
    for (const { block, count } of group.pattern) {
      const { attn, mlp } = block;
      // sliding-window and full-attention layers are both "gqa"; mark the
      // windowed ones so they read as distinct segments (Gemma's 5:1 stack)
      const attnLabel = attn.kind === 'gqa' && attn.window != null ? 'gqa-swa' : attn.kind;
      trace.startSegment(`${attnLabel}+${mlp.kind}`, group.repeat * count);

      residual = trace.expect(residual, residualT); // [B_dpa, D]

      // --------------------------- Attention ------------------------

      const { qkv: qkvParams, o: oParams } = blockAttnParams(model, block);

      const wQkv = trace.weight('attn-qkv', {
        shape: [d_model, qkvParams / d_model],
        sharding: [[], roles['TP']],
        dtype: precision.weights.attention,
      }); // [D, QKV_tp]

      const qkv = trace.gemm('qkv', residual, wQkv, acts.attention); // [B_dpa, D] * [D, QKV_tp] = [B_dpa, QKV_tp]

      // TODO: comms in sharded attention? should we model attn manually?
      const kvHeadFrac = kvFraction(blockKvHeads(block), tp, dcp);
      const flops =
        args.phase === 'decode'
          ? blockAttnFlopsDecodeParts(block, args.ctx)
          : blockAttnFlopsPrefillParts(block, args.ctx);
      const kvRead =
        args.phase === 'decode'
          ? blockKvBytesParts(block, DTYPE_BYTES[precision.kv], args.ctx, 'read')
          : { core: 0, indexer: 0 };
      const kvWrite = blockKvBytesParts(
        block,
        DTYPE_BYTES[precision.kv],
        args.phase === 'decode' ? 1 : args.ctx,
        'store',
      );
      const scaledFlops = (work: number) =>
        args.phase === 'decode' ? (tokensPerGroup * work) / tp : (seqsPerGroup * work) / tp;
      const readBytes = (bytes: number) =>
        args.phase === 'decode' ? tokensPerGroup * kvHeadFrac * bytes : 0;
      const writeBytes = (bytes: number) =>
        (args.phase === 'decode' ? tokensPerGroup : seqsPerGroup) * kvHeadFrac * bytes;

      // CSA/DSA index selection is a separate dot product and can run at a
      // narrower width than core attention. It preserves the projected
      // value here; the edge simply makes core attention wait for its picks.
      const indexed =
        flops.indexer > 0
          ? trace.attention('indexer', qkv, {
              variant: block.attn.kind,
              dtype: precision.indexer ?? acts.attention,
              outCols: qkvParams / d_model,
              flops: scaledFlops(flops.indexer),
              kvReadBytes: readBytes(kvRead.indexer),
              kvWriteBytes: writeBytes(kvWrite.indexer),
            })
          : qkv;

      // Under DCP this rank holds only its slice of the sequence, so it
      // needs every query head to attend over it: the queries all-gather
      // across the group on the way in, and the partial outputs (with
      // their softmax normalizers, which ride along in the same message)
      // reduce-scatter back on the way out. Both are per-layer, per-step
      // and small, which is why dcpAxes takes the innermost dims.
      const gathered = dcpGroup.length
        ? trace.reshard(indexed, [roles['DPA'], tpMinusDcp])
        : indexed;

      const attnOp = trace.attention('attn', gathered, {
        variant: block.attn.kind,
        dtype: acts.attention,
        outCols: oParams / d_model,
        flops: scaledFlops(flops.core),
        kvReadBytes: readBytes(kvRead.core),
        kvWriteBytes: writeBytes(kvWrite.core),
      }); // [B_dpa, QKV_tp] -> [B_dpa, O_tp]

      const combined = dcpGroup.length
        ? trace.reshard(
            // the slices are partial sums over the group until combined
            trace.reshape(attnOp, attnOp.type.shape, attnOp.type.sharding, dcpGroup),
            [roles['DPA'], roles['TP']],
            [],
          )
        : attnOp;

      const wO = trace.weight('attn-o', {
        shape: [oParams / d_model, d_model],
        sharding: [roles['TP'], []],
        dtype: precision.weights.attention,
      }); // [O_tp, D]

      const attnOut = trace.reshard(
        trace.gemm('o-proj', combined, wO, acts.attention), // [B_dpa, O_tp] * [O_tp, D] = [B_dpa, D] {U_tp}
        [roles['DPA'], []],
        [],
        { dtype: precision.residual },
      ); // [B_dpa, D] {U_tp} -> [B_dpa, D]

      // ------------------------------ MLP ---------------------------

      if (mlp.kind === 'dense') {
        const mlpOut = trace.ffn('mlp', attnOut, {
          ff: mlp.ffDim,
          ffSharding: roles['TP'],
          variant: mlp.variant,
          weights: precision.weights.denseMlp,
          dtype: acts.denseMlp,
        }); // [B_dpa, D] -> [B_dpa, D] {U_tp}

        // [B_dpa, D] {U_tp} -> [B_dpa, D]
        residual = trace.reshard(mlpOut, [roles['DPA'], []], [], { dtype: precision.residual });

        continue;
      }

      // ------------------------------ MoE ---------------------------

      residual = lowerMoe(trace, attnOut, {
        deployment: args.deployment,
        mlp,
        tokensTotal,
        weights: precision.weights,
        acts,
        residual: precision.residual,
        activatedFrac: expertReadFraction(model, tokensTotal),
      });

      residual = trace.expect(residual, residualT); // [B_dpa, D]
    }

  if (stage.hasUnembedding) {
    trace.startSegment('lm_head');

    const logits = trace.gemm(
      'unembed',
      residual, // [B_dpa, D]
      trace.weight('unembed', {
        shape: [d_model, vocab],
        sharding: [[], roles['TP']],
        dtype: precision.weights.embeddings,
      }), // [D, V_tp]
      acts.embeddings,
    ); // [B_dpa, D] * [D, V_tp] = [B_dpa, V_tp]

    trace.reshard(logits, [roles['DPA'], []]); // [B_dpa, V_tp] -> [B_dpa, V]
  }

  if (args.includeBoundary) trace.pipelineSend(residual, model.residualStreams ?? 1);

  return trace.build();
}
