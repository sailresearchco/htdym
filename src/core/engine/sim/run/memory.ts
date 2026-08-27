import { ModelSpec } from '../../../model/models';
import {
  blockAttnParams,
  blockDenseMlpParams,
  blockKvBytes,
  blockKvHeads,
  blockLatentProjParams,
  blockRouterParams,
  blockRoutedExpertParams,
  blockSharedExpertParams,
} from '../../../model/block';
import { chipsPerStage, dcpSize, kvFraction, Deployment, roleSize } from '../../surface/deploy';
import type { Stage } from '../lowering/stages';
import type { MemoryFootprint } from '../../surface/api';
import { DTYPE_BYTES } from '../../../model/dtype';

// Exact per-chip memory footprint from the ordered stage map. Weights are
// counted at the stored format on every chip: they stay packed, with a
// widening kernel assumed where none ships (validate warns there).
export function memoryFootprint(
  m: ModelSpec,
  d: Deployment,
  stages: Stage[],
  fullLen: number,
): MemoryFootprint {
  const tp = roleSize(d.mesh, 'TP');
  const B = (c: keyof typeof m.precision.weights) => DTYPE_BYTES[m.precision.weights[c]];

  let worstWeights = 0;
  let worstKv = 0;
  let residents = Infinity;
  for (const s of stages) {
    let weights = 0;
    let kvPerSeq = 0;
    for (const g of s.groups)
      for (const { block: b, count } of g.pattern) {
        const n = g.repeat * count;
        weights +=
          (n *
            (blockAttnParams(m, b).total * B('attention') +
              (blockDenseMlpParams(m, b) + blockLatentProjParams(m, b)) * B('denseMlp') +
              blockSharedExpertParams(m, b) * B('sharedExperts'))) /
          tp;

        // router is replicated, not sharded
        weights += n * blockRouterParams(m, b) * B('router');

        weights +=
          (n * blockRoutedExpertParams(m, b).total * B('routedExperts')) / chipsPerStage(d);

        kvPerSeq +=
          n *
          blockKvBytes(b, DTYPE_BYTES[m.precision.kv], fullLen, 'store') *
          kvFraction(blockKvHeads(b), tp, dcpSize(d));
      }

    const emb = (m.vocab * m.modelDim * B('embeddings')) / tp;
    if (s.hasEmbedding) weights += emb;
    if (s.hasUnembedding && !(m.tiedEmbeddings && s.hasEmbedding)) weights += emb;

    worstWeights = Math.max(worstWeights, weights);
    worstKv = Math.max(worstKv, kvPerSeq);

    const free = d.chip.hbmCapacity - weights;
    residents = Math.min(
      residents,
      kvPerSeq > 0 ? Math.max(0, Math.floor(free / kvPerSeq)) : Infinity,
    );
  }
  return {
    weightBytesPerChip: worstWeights,
    kvBytesPerSeqPerChip: worstKv,
    maxResidentSeqsPerChip: residents,
  };
}
