import { ModelSpec } from '../../../model/models';
import { DTYPE_BYTES } from '../../../model/dtype';
import {
  blockAttnFlopsDecodeParts,
  blockAttnFlopsPrefillParts,
  blockAttnParams,
  blockKvBytesParts,
  blockKvHeads,
  type AttnFlopParts,
  type KvByteParts,
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
  // A decode step can carry chunked prefill work for other sequences
  // (continuous batching): `seqs` prompt-equivalents per batch group, each
  // a prompt of length `ctx` whose first `cached` tokens are prefix-cache
  // hits and cost nothing. Its tokens are already counted in
  // tokensTotal / tokensPerGroup; this adds only their attention.
  // `chunkLen` is how many prompt tokens one step takes from a prompt; a
  // longer uncached part spans steps, each re-reading what came before.
  chunk?: { seqs: number; ctx: number; cached: number; chunkLen?: number };
  // prefill: the first `cached` tokens of each prompt are prefix-cache hits
  cached?: number;
  // emit the stage-boundary send (PP > 1 and not the last stage)
  includeBoundary: boolean;
  // MoE hottest/mean EP-rank load at large batch (SimInput's workload)
  routingSkew?: number;
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
  let residualT = residual.type;

  for (const group of stage.groups)
    for (const { block, count } of group.pattern) {
      const { attn, mlp } = block;
      // sliding-window and full-attention layers are both "gqa"; mark the
      // windowed ones so they read as distinct segments (Gemma's 5:1 stack)
      const attnLabel = attn.kind === 'gqa' && attn.window != null ? 'gqa-swa' : attn.kind;
      trace.startSegment(`${attnLabel}+${mlp.kind}`, group.repeat * count);

      residual = trace.expect(residual, residualT); // [B_dpa, D]

      // ---------------------------- Rows ----------------------------

      // The attention work of this block for one batch group, and the rows
      // its matmuls carry. Prompt work covers positions [lo, P) of each
      // prompt: lo skips prefix-cache hits, and a block with a prefill
      // tail skips all but the prompt's last few tokens. A decode step's
      // tokens are its sequences' query tokens: one each, or the k+1 a
      // speculative verify scores. Every query does its own dot products
      // and stores its own entry, but a sequence's cache is read once.
      const kvB = DTYPE_BYTES[precision.kv];
      const tail = block.prefillTail;
      const prompt = (seqs: number, P: number, cached: number, chunkLen = P - cached) => {
        const lo = tail === undefined ? cached : Math.max(cached, P - tail);
        const none = { core: 0, indexer: 0 };
        const full = blockAttnFlopsPrefillParts(block, P);
        const hit = lo > 0 ? blockAttnFlopsPrefillParts(block, lo) : none;
        // the cache gains exactly what a (P - lo)-token sequence stores:
        // its compressed entries and as much window as it fills
        const written = blockKvBytesParts(block, kvB, P - lo, 'store');
        // Each chunk's queries attend over everything before it, which
        // comes from HBM: the hit prefix, then what earlier chunks wrote.
        // A chunk reads it once, as a decode step reads a sequence's cache
        // once for all its queries (sparse picks taken as one set). A tail
        // block runs only in the prompt's last chunk.
        const starts: number[] = [];
        for (let at = cached; at < P; at += Math.max(1, chunkLen)) starts.push(at);
        const prefix = (tail === undefined ? starts : starts.slice(-1))
          .filter((at) => at > 0)
          .map((at) => blockKvBytesParts(block, kvB, at, 'read'))
          .reduce((a, r) => ({ core: a.core + r.core, indexer: a.indexer + r.indexer }), none);
        return {
          rows: seqs * (P - lo),
          flops: {
            core: seqs * (full.core - hit.core),
            indexer: seqs * (full.indexer - hit.indexer),
          },
          write: { core: seqs * written.core, indexer: seqs * written.indexer },
          read: { core: seqs * prefix.core, indexer: seqs * prefix.indexer },
        };
      };
      let rowsPerGroup: number;
      let flops: AttnFlopParts;
      let kvRead: KvByteParts = { core: 0, indexer: 0 };
      let kvWrite: KvByteParts;
      if (args.phase === 'prefill') {
        // TTFT runs one prompt on one group, so rows are prompt tokens
        // times however many prompts the group holds
        const w = prompt(seqsPerGroup, args.ctx, args.cached ?? 0);
        rowsPerGroup = (tokensPerGroup * w.rows) / (seqsPerGroup * args.ctx);
        [flops, kvWrite, kvRead] = [w.flops, w.write, w.read];
      } else {
        const chunk = args.chunk
          ? prompt(args.chunk.seqs, args.chunk.ctx, args.chunk.cached, args.chunk.chunkLen)
          : null;
        const chunkTokens = args.chunk ? args.chunk.seqs * (args.chunk.ctx - args.chunk.cached) : 0;
        const decodeTokens = tokensPerGroup - chunkTokens;
        const per = blockAttnFlopsDecodeParts(block, args.ctx);
        const read = blockKvBytesParts(block, kvB, args.ctx, 'read');
        const write = blockKvBytesParts(block, kvB, 1, 'store');
        rowsPerGroup = decodeTokens + (chunk?.rows ?? 0);
        flops = {
          core: decodeTokens * per.core + (chunk?.flops.core ?? 0),
          indexer: decodeTokens * per.indexer + (chunk?.flops.indexer ?? 0),
        };
        kvRead = {
          core: seqsPerGroup * read.core + (chunk?.read.core ?? 0),
          indexer: seqsPerGroup * read.indexer + (chunk?.read.indexer ?? 0),
        };
        kvWrite = {
          core: decodeTokens * write.core + (chunk?.write.core ?? 0),
          indexer: decodeTokens * write.indexer + (chunk?.write.indexer ?? 0),
        };
      }
      const rowsTotal = (tokensTotal * rowsPerGroup) / tokensPerGroup;
      if (rowsTotal !== residual.type.shape[0]) {
        // dropping (or restoring) rows moves no data
        residual = trace.reshape(residual, [rowsTotal, d_model]);
        residualT = residual.type;
      }

      // ---------------------------- Engram --------------------------

      if (block.engram) {
        const e = block.engram;
        // Each token's hashed rows come over the host link, its heads split
        // over the TP ranks (gathering them back under TP is not priced).
        const looked = trace.hostRead(
          'engram-rows',
          [rowsTotal, e.projIn],
          [roles['DPA'], []],
          (rowsPerGroup * e.lookups * e.rowBytes) / tp,
          acts.attention,
        ); // [B_dpa, In]
        const wEngram = trace.weight('engram-proj-weight', {
          shape: [e.projIn, e.projOut],
          sharding: [[], roles['TP']],
          dtype: precision.weights.attention,
        }); // [In, Out_tp]
        // the gate against the residual is elementwise and not priced
        trace.gemm('engram-proj', looked, wEngram, acts.attention); // [B_dpa, In] * [In, Out_tp]
      }

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
      const scaledFlops = (work: number) => work / tp;
      const readBytes = (bytes: number) => kvHeadFrac * bytes;
      const writeBytes = (bytes: number) => kvHeadFrac * bytes;

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
        tokensTotal: rowsTotal,
        weights: precision.weights,
        acts,
        residual: precision.residual,
        routingSkew: args.routingSkew,
      });

      residual = trace.expect(residual, residualT); // [B_dpa, D]
    }

  // the stage hands on (and unembeds) every row it computes: all of them,
  // less a prefill's prefix-cache hits
  const outRows =
    args.phase === 'prefill' && args.cached
      ? (tokensTotal * (args.ctx - args.cached)) / args.ctx
      : tokensTotal;
  if (residual.type.shape[0] !== outRows) residual = trace.reshape(residual, [outRows, d_model]);

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
