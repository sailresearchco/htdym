import type { ModelSpec } from '../../../model/models';
import { DTYPE_BYTES } from '../../../model/dtype';
import { kvBytesPerSeq } from '../../../model/utils';
import { roleSize, totalChips } from '../../surface/deploy';
import type { Deployment } from '../../surface/deploy';
import type { CostBackend } from '../cost/types';
import { partitionIntoStages } from '../lowering/stages';
import type {
  DecodeEvaluation,
  EvalFailure,
  EvalOptions,
  PrefillEvaluation,
  SimInput,
} from '../../surface/api';
import { evaluateDecodeAtBatch } from './decode';
import { evaluateServingStep, type ServingEvaluation, type ServingOptions } from './serving';
import { evaluatePrefill } from './prefill';
import { memoryFootprint } from './memory';

// Prefill/decode disaggregated serving (Dynamo + NIXL style): a prefill
// pool computes each prompt's KV cache, ships it over a link to a decode
// pool, and the decode pool generates. Both pools are priced by the
// ordinary phase evaluators as black boxes; this module only adds the KV
// hand-off between them and balances the two rates.

// The path one KV transfer takes, per chip: each source chip pushes its
// shard to its destination chips in parallel (RDMA, so it overlaps
// compute on both sides and only bounds the rate, never the step).
export interface KvLink {
  // one-way bytes/s each chip can push or pull
  bandwidthPerChip: number;
  // fixed setup cost of one transfer (handshake + NIC + switch)
  latency: number;
}

export interface DisaggInput {
  model: ModelSpec;
  prefill: Deployment;
  decode: Deployment;
  workload: SimInput['workload'];
  // decode operating batch (the serving policy's operatingBatch picks one)
  decodeBatch: number;
  // sequences per prefill pass. Default: ~8192 tokens per DPA group per
  // pipeline stage, vLLM's default max_num_batched_tokens.
  prefillSeqs?: number;
  // Default: the slower of the two chips' scale-out NICs, i.e. transfers
  // cross the inter-node network. Override for NVLink-local pools.
  link?: KvLink;
  // Fraction of each prompt's tokens found in the prefix cache, h. Those
  // need no prefill compute, and we assume the decode side already holds
  // their KV too (KV-aware routing sends a request to a decode worker that
  // shares its prefix), so they need no transfer either. Default 0.
  prefixHitRate?: number;
  // Speculative decoding on the decode pool (the prefill pool only fills
  // the cache). Absent = plain decode.
  speculative?: ServingOptions['speculative'];
}

// One pool's side of the balance, all rates per chip of that pool.
export interface DisaggPool<R> {
  result: R;
  chips: number;
  // requests/s/chip compute alone sustains
  computeReqPerSecPerChip: number;
  // requests/s/chip the pool's NICs sustain moving the KV
  linkReqPerSecPerChip: number;
  // the pool's rate: the lower of the two
  reqPerSecPerChip: number;
  usdPerHourPerChip: number;
  // KV bytes the busiest chip of a request's group sends or receives
  transferBytesPerChip: number;
  // link busy fraction at the balanced rate (1 = the link is the bound)
  linkUtilisation: number;
}

type Ok<E> = Extract<E, { ok: true }>;

export type DisaggEvaluation<TBackend extends CostBackend> =
  | EvalFailure
  | {
      ok: true;
      prefill: DisaggPool<Ok<PrefillEvaluation<TBackend>>>;
      decode: DisaggPool<Ok<DecodeEvaluation<TBackend>> | Ok<ServingEvaluation<TBackend>>>;
      link: KvLink;
      // KV one request moves, unsharded (the uncached prompt tail at P)
      kvBytesPerRequest: number;
      // what the transfer adds to TTFT: setup + the busiest chip's shard
      transferTime: number;
      // decode chips per prefill chip that balance the two rates
      decodeChipsPerPrefillChip: number;
      // requests/s per chip of the balanced fleet (both pools)
      reqPerSecPerChip: number;
      // true when either pool's NIC, not its compute, sets its rate
      linkBound: boolean;
      // balanced-fleet chip cost of one request, then the same cost per
      // 1M output and per 1M input tokens (two views of one bill, not a split)
      usdPerRequest: number;
      usdPerMOutput: number;
      usdPerMInput: number;
    };

export function defaultKvLink(prefill: Deployment, decode: Deployment): KvLink {
  const outs = [prefill.chip, decode.chip].map((c) => {
    const so = c.interconnect.scaleOut;
    if (!so) throw new Error(`${c.id} has no scale-out network, pass a KV link explicitly`);
    return so;
  });
  return {
    bandwidthPerChip: Math.min(...outs.map((o) => o.bandwidthPerChip)),
    latency: Math.max(...outs.map((o) => o.latency)),
  };
}

// KV bytes the busiest chip of one request's group holds for len tokens,
// under this deployment's cache sharding (heads over TP, sequence over
// DCP, layers over PP; one request lives in exactly one DPA group).
function kvBytesPerChip(m: ModelSpec, d: Deployment, len: number): number {
  const stages = partitionIntoStages(m, roleSize(d.mesh, 'PP'));
  return memoryFootprint(m, d, stages, len).kvBytesPerSeqPerChip;
}

// Requests/s/chip a pool's NICs sustain: each of a group's chips moves its
// shard of every request that group serves, so a group clears bw / bytes
// requests/s, shared by its totalChips / DPA chips. Latency pipelines
// across requests and does not cap the rate.
function linkRate(d: Deployment, bytesPerChip: number, link: KvLink): number {
  const groupChips = totalChips(d) / roleSize(d.mesh, 'DPA');
  return bytesPerChip > 0 ? link.bandwidthPerChip / bytesPerChip / groupChips : Infinity;
}

export function evaluateDisagg<TBackend extends CostBackend>(
  input: DisaggInput,
  opts: EvalOptions<TBackend>,
): DisaggEvaluation<TBackend> {
  const { model, workload } = input;
  const P = workload.prefillLen;
  const G = workload.generateLen;
  const h = input.prefixHitRate ?? 0;
  if (!(h >= 0 && h < 1)) throw new Error(`prefix hit rate must be in [0, 1)`);
  const link = input.link ?? defaultKvLink(input.prefill, input.decode);

  // Prefill computes only each prompt's uncached tail, which attends over
  // the cached prefix (read from HBM). The token budget fills with tails.
  // at least one token is always computed, however close h is to 1
  const cached = Math.min(Math.round(P * h), P - 1);
  const pDpa = roleSize(input.prefill.mesh, 'DPA');
  const pPp = roleSize(input.prefill.mesh, 'PP');
  const seqs = input.prefillSeqs ?? pDpa * pPp * Math.max(1, Math.floor(8192 / (P - cached)));
  const pre = evaluatePrefill(
    { model, deployment: input.prefill, workload },
    seqs,
    'throughput',
    opts,
    cached,
  );
  if (!pre.ok) return pre;

  const decodeInput = { model, deployment: input.decode, workload };
  const dec = input.speculative
    ? evaluateServingStep(decodeInput, input.decodeBatch, { speculative: input.speculative }, opts)
    : evaluateDecodeAtBatch(
        decodeInput,
        input.decodeBatch,
        roleSize(input.decode.mesh, 'PP'),
        opts,
      );
  if (!dec.ok) return dec;
  const decodeTokPerSecPerChip =
    'outputTokPerSecPerChip' in dec ? dec.outputTokPerSecPerChip : dec.tokPerSecPerChip;

  // Only the uncached tail moves. Measuring it as the KV of a tail-length
  // sequence keeps windowed caches honest: a window full of new tokens
  // moves whole, however long the cached prefix behind it.
  const tail = P - cached;
  const kvBytesPerRequest = kvBytesPerSeq(model, DTYPE_BYTES[model.precision.kv], tail);
  // The destination group needs what its chips hold, replicated heads
  // included, and the source group's chips split sending it: copies the
  // source holds redundantly go out once, copies the destination needs
  // twice go out twice.
  const groupChips = (d: Deployment) => totalChips(d) / roleSize(d.mesh, 'DPA');
  const dstBytes = kvBytesPerChip(model, input.decode, tail);
  const srcBytes = (dstBytes * groupChips(input.decode)) / groupChips(input.prefill);
  const transferTime = link.latency + Math.max(srcBytes, dstBytes) / link.bandwidthPerChip;

  const pCompute = pre.tokPerSecPerChip / P;
  const dCompute = decodeTokPerSecPerChip / G;
  const pLink = linkRate(input.prefill, srcBytes, link);
  const dLink = linkRate(input.decode, dstBytes, link);
  const pRate = Math.min(pCompute, pLink);
  const dRate = Math.min(dCompute, dLink);

  const pPrice = input.prefill.chip.costPerHour;
  const dPrice = input.decode.chip.costPerHour;
  if (!pPrice || !dPrice) throw new Error(`missing price`);
  const usdPerRequest = pPrice / 3600 / pRate + dPrice / 3600 / dRate;

  const ratio = pRate / dRate;
  return {
    ok: true,
    prefill: {
      result: pre,
      chips: totalChips(input.prefill),
      computeReqPerSecPerChip: pCompute,
      linkReqPerSecPerChip: pLink,
      reqPerSecPerChip: pRate,
      usdPerHourPerChip: pPrice,
      transferBytesPerChip: srcBytes,
      linkUtilisation: pRate / pLink,
    },
    decode: {
      result: dec,
      chips: totalChips(input.decode),
      computeReqPerSecPerChip: dCompute,
      linkReqPerSecPerChip: dLink,
      reqPerSecPerChip: dRate,
      usdPerHourPerChip: dPrice,
      transferBytesPerChip: dstBytes,
      linkUtilisation: dRate / dLink,
    },
    link,
    kvBytesPerRequest,
    transferTime,
    decodeChipsPerPrefillChip: ratio,
    reqPerSecPerChip: pRate / (1 + ratio),
    linkBound: pLink < pCompute || dLink < dCompute,
    usdPerRequest,
    usdPerMOutput: (usdPerRequest / G) * 1e6,
    usdPerMInput: (usdPerRequest / P) * 1e6,
  };
}

export interface NodeRatio {
  prefillNodes: number;
  decodeNodes: number;
  // fleet rate: the slower pool's, the other idles the difference
  reqPerSec: number;
  usdPerRequest: number;
  usdPerMOutput: number;
}

// The cheapest whole-replica fleet for an evaluated pair: x prefill and y
// decode replicas (one replica = one deployment, e.g. an 8-GPU node), at
// most maxNodes of each. Real fleets come in replicas, so the exact
// balance is rounded to the pair that wastes the least; ties go to the
// smaller fleet.
export function cheapestNodeRatio<TBackend extends CostBackend>(
  e: Extract<DisaggEvaluation<TBackend>, { ok: true }>,
  maxNodes = 16,
): NodeRatio {
  const perNode = (p: DisaggPool<unknown>) => p.chips * p.reqPerSecPerChip;
  const usdPerHour = (p: DisaggPool<unknown>) => p.chips * p.usdPerHourPerChip;

  let best: NodeRatio | undefined;
  for (let x = 1; x <= maxNodes; x++)
    for (let y = 1; y <= maxNodes; y++) {
      const reqPerSec = Math.min(x * perNode(e.prefill), y * perNode(e.decode));
      const usdPerRequest =
        (x * usdPerHour(e.prefill) + y * usdPerHour(e.decode)) / 3600 / reqPerSec;
      const size = (r: NodeRatio) => r.prefillNodes + r.decodeNodes;
      const tie = best && Math.abs(usdPerRequest - best.usdPerRequest) <= 1e-9 * best.usdPerRequest;
      if (!best || (tie ? x + y < size(best) : usdPerRequest < best.usdPerRequest))
        best = {
          prefillNodes: x,
          decodeNodes: y,
          reqPerSec,
          usdPerRequest,
          // same 1/G scale the evaluation used
          usdPerMOutput: usdPerRequest * (e.usdPerMOutput / e.usdPerRequest),
        };
    }
  return best!;
}
