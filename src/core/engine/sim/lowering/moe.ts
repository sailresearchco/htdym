import type { CategoryDtypes, Dtype } from '../../../model/dtype';
import type { MoeMlpConfig } from '../../../model/block/mlp';
import { roleSize, type Deployment } from '../../surface/deploy';
import { sameAxes } from '../ir/tensors';
import type { TraceBuilder, Val } from './builder';

export interface MoeArgs {
  deployment: Deployment;
  mlp: MoeMlpConfig;
  tokensTotal: number;
  // per-category held weight formats (latent projections count as denseMlp)
  weights: CategoryDtypes;
  // per-category activation formats: the routed experts consume a narrower
  // one than the rest of the block on releases that quantize only them
  acts: CategoryDtypes;
  // width of the residual stream the block's output reductions land in
  residual: Dtype;
  // steady-state hottest/mean EP-rank load at large batch (1 = balanced),
  // see rankImbalance. Absent = 1.
  routingSkew?: number;
}

export function lowerMoe(trace: TraceBuilder, x: Val, args: MoeArgs): Val {
  const { mlp } = args;
  const { mesh, moeDispatch: dispatch } = args.deployment;
  const { roles } = mesh;
  const ep = roleSize(mesh, 'EP');
  const d = x.type.shape[1];
  const inner = mlp.latentDim || d;
  const skew = args.routingSkew ?? 1;

  // The step waits for the most loaded EP rank, so everything below is
  // sized for that rank rather than the mean one: its token-expert pairs
  // (what its GEMMs run) and the tokens it receives and returns over the
  // a2a. Every rank's slice is padded to the hottest one, which prices
  // the a2a at the hottest port - the one the collective waits for.
  const imb = rankImbalance(args.tokensTotal, pairLoad(mlp.experts, mlp.topK, ep), ep, skew);

  // Under uniform routing, an expert is untouched only if every token
  // misses it. The hottest rank's experts are hit imb times as often,
  // so it touches (and streams) more of its experts than the mean rank.
  const hitRate = Math.min(1, (imb * mlp.topK) / mlp.experts);
  const activatedFrac = 1 - Math.pow(1 - hitRate, args.tokensTotal);

  // the small [D, E] router replicates on every rank
  // so each can pick global top-k without gathering logits
  const routerWeights = trace.weight('router', {
    shape: [d, mlp.experts],
    sharding: [[], []],
    dtype: args.weights.router,
  }); // [D, E]

  const router = trace.gemm('router', x, routerWeights, args.acts.router); // [B_dpa, D] * [D, E] = [B_dpa, E]

  const [sharedOut, mergeSharedPartials] =
    mlp.sharedExperts > 0
      ? [
          trace.ffn('shared', x, {
            ff: mlp.sharedExperts * mlp.expertDim,
            ffSharding: roles['TP'],
            variant: mlp.variant,
            weights: args.weights.sharedExperts,
            dtype: args.acts.sharedExperts,
          }), // [B_dpa, D] -> [B_dpa, D] {U_tp}
          sameAxes(roles['TP'], roles['ETP']),
        ]
      : [null, false];

  const outPartials = mergeSharedPartials ? roles['ETP'] : [];

  // Only the tokens going out are narrowed to what the experts consume:
  // the trip back carries their outputs, which are unreduced partial sums
  // and are summed in the stream's own width.
  //
  // A latent MoE runs its routed experts narrower than the residual, so
  // only the narrow tokens cross the dispatch. The pair of projections
  // brackets the whole routed path TP-sharded on the latent dim, which
  // lets the trip back reduce the experts' ETP partials straight into the
  // up-projection's contraction.
  const routedIn = mlp.latentDim
    ? trace.gemm(
        'moe-down',
        x,
        trace.weight('moe-down-weight', {
          shape: [d, inner],
          sharding: [[], roles['TP']],
          dtype: args.weights.denseMlp,
        }), // [D, L_tp]
        args.acts.denseMlp,
      ) // [B_dpa, D] * [D, L_tp] = [B_dpa, L_tp]
    : x;

  const backSharding = mlp.latentDim ? roles['TP'] : [];
  const backPartials = mlp.latentDim ? [] : outPartials;

  let routedOut: Val; // [B_dpa, L_tp] or [B_dpa, D] {U_etp if mergeSharedPartials}

  if (dispatch === 'ring-of-experts') {
    const replicated = trace.reshard(routedIn, [[], []], undefined, {
      dtype: args.acts.routedExperts,
    }); // [B_dpa, L_tp] -> [B, L]

    // routed view: the B*topK token-expert pairs, padded to the hottest
    // expert shard's share (no data moves)
    const routed = trace.reshape(
      replicated,
      [args.tokensTotal * mlp.topK * imb, inner],
      [roles['EP'], []],
    ); // [B, L] -> [(B*topK*imb)_ep, L]

    const expertOut = trace.expertFfn('experts', routed, {
      ff: mlp.expertDim,
      ffSharding: roles['ETP'],
      variant: mlp.variant,
      weights: args.weights.routedExperts,
      dtype: args.acts.routedExperts,
      loadFraction: activatedFrac,
      experts: {
        count: mlp.experts,
        epSharding: roles['EP'],
        activeGroups: (mlp.experts * activatedFrac) / ep,
      },
      extraDeps: [router],
    }); // [(B*topK)_ep, D] -> [(B*topK)_ep, D] {U_etp}

    // collapse each token's pairs, leaving partials over EP and ETP
    const combined = trace.reshape(
      expertOut,
      routedIn.type.shape,
      [[], []],
      [...expertOut.type.unreduced, ...roles['EP']],
    ); // [(B*topK)_ep, L] {U_etp} -> [B, L] {U_etp,ep}

    routedOut = trace.reshard(combined, [roles['DPA'], backSharding], backPartials, {
      dtype: routedIn.dtype,
    }); // [B, L] {U_etp,ep} -> [B_dpa, L_tp]
  } else if (dispatch === 'expanded-a2a' || dispatch === 'coalesced-a2a') {
    // Each token creates topK copies, each copy chooses each expert shard
    // (chip with E/EP experts) with a 1/EP probability, assuming balanced
    // routing. Expanded ships every copy; coalesced ships each token at
    // most once per destination shard and merges each shard's expert
    // outputs before the trip back, so only the expected distinct-shard
    // count crosses the wire while a received token fans out to its
    // local experts for free. Either is then padded to what the hottest
    // shard receives: its pairs, or (coalesced) its distinct tokens.
    const copies =
      dispatch === 'expanded-a2a'
        ? mlp.topK * imb
        : shardsHit(mlp.experts, mlp.topK, ep) *
          rankImbalance(args.tokensTotal, tokenLoad(mlp.experts, mlp.topK, ep), ep, skew);

    // [B_dpa, L_tp] -> [B_dpa, EP, copies/EP, L_tp]
    const send = trace.reshape(
      routedIn,
      [args.tokensTotal, ep, copies / ep, inner],
      [roles['DPA'], [], [], backSharding],
    );

    // [B_dpa, EP, copies/EP, L_tp] -> [B, EP_ep, copies/EP, L]
    const dispatched = trace.reshard(send, [[], roles['EP'], [], []], [], {
      dtype: args.acts.routedExperts,
      extraDeps: [router],
    });

    // [B, EP_ep, copies/EP, L] -> [(B * topK * imb)_ep, L]
    const grouped = trace.reshape(
      dispatched,
      [args.tokensTotal * mlp.topK * imb, inner],
      [roles['EP'], []],
    );

    const expertOut = trace.expertFfn('experts', grouped, {
      ff: mlp.expertDim,
      ffSharding: roles['ETP'],
      variant: mlp.variant,
      weights: args.weights.routedExperts,
      dtype: args.acts.routedExperts,
      loadFraction: activatedFrac,
      experts: {
        count: mlp.experts,
        epSharding: roles['EP'],
        activeGroups: (mlp.experts * activatedFrac) / ep,
      },
    }); // [(B * topK * imb)_ep, L] -> [(B * topK * imb)_ep, L] {U_etp}

    // [(B * topK * imb)_ep, L] {U_etp} -> [B, EP_ep, copies/EP, L] {U_etp}
    const ungrouped = trace.reshape(
      expertOut,
      [args.tokensTotal, ep, copies / ep, inner],
      [[], roles['EP'], [], []],
    );

    // [B, EP_ep, copies/EP, L] {U_etp} -> [B_dpa, EP, copies/EP, L_tp]
    const receivedBack = trace.reshard(
      ungrouped,
      [roles['DPA'], [], [], backSharding],
      backPartials,
      { dtype: routedIn.dtype },
    );

    // [B_dpa, EP, copies/EP, L_tp] -> [B_dpa, L_tp]
    routedOut = trace.reshape(receivedBack, routedIn.type.shape, routedIn.type.sharding);
  } else throw new Error(`unsupported dispatch: ${dispatch}`);

  if (mlp.latentDim)
    routedOut = trace.reshard(
      trace.gemm(
        'moe-up',
        routedOut,
        trace.weight('moe-up-weight', {
          shape: [inner, d],
          sharding: [roles['TP'], []],
          dtype: args.weights.denseMlp,
        }), // [L_tp, D]
        args.acts.denseMlp,
      ), // [B_dpa, L_tp] * [L_tp, D] = [B_dpa, D] {U_tp}
      [roles['DPA'], []],
      outPartials,
      { dtype: args.residual },
    ); // [B_dpa, D] {U_tp} -> [B_dpa, D] {U_etp if mergeSharedPartials}

  if (!sharedOut) {
    return trace.expect(routedOut, x.type); // [B_dpa, D]
  } else if (mergeSharedPartials) {
    // [B_dpa, D] {U_etp=tp}
    const merged = trace.join('moe-output', [routedOut, sharedOut], args.residual);
    // [B_dpa, D] {U_etp=tp} -> [B_dpa, D]
    const reducedMerged = trace.reshard(merged, [roles['DPA'], []], [], { dtype: args.residual });
    return trace.expect(reducedMerged, x.type); // [B_dpa, D]
  } else {
    // [B_dpa, D] {U_tp} -> [B_dpa, D]
    const reducedShared = trace.reshard(sharedOut, [roles['DPA'], []], [], {
      dtype: args.residual,
    });
    const merged = trace.join('moe-output', [routedOut, reducedShared], args.residual); // [B_dpa, D]
    return trace.expect(merged, x.type); // [B_dpa, D]
  }
}

// Expected number of distinct expert shards a token's topK distinct
// experts land on under balanced routing: a shard of E/EP experts
// receives none of the topK draws with probability C(E-E/EP, topK) /
// C(E, topK), taken as a running product to keep the binomials tame.
function shardsHit(experts: number, topK: number, ep: number): number {
  let miss = 1;
  for (let i = 0; i < topK; i++) miss *= (experts - experts / ep - i) / (experts - i);
  return ep * (1 - miss);
}

// One token's load on one EP rank, as moments of a per-token draw.
interface TokenLoad {
  mean: number;
  variance: number;
  // the most one token can put on one rank
  max: number;
}

// Pairs: how many of a token's topK distinct experts land on one shard of
// E/EP experts. Hypergeometric, topK draws from E with E/EP successes.
function pairLoad(experts: number, topK: number, ep: number): TokenLoad {
  const p = 1 / ep;
  const fpc = experts > 1 ? (experts - topK) / (experts - 1) : 0;
  return { mean: topK * p, variance: topK * p * (1 - p) * fpc, max: Math.min(topK, experts / ep) };
}

// Tokens: whether a token reaches one shard at all, what a coalesced
// dispatch ships. Bernoulli with the shardsHit/EP probability.
function tokenLoad(experts: number, topK: number, ep: number): TokenLoad {
  const p = shardsHit(experts, topK, ep) / ep;
  return { mean: p, variance: p * (1 - p), max: 1 };
}

// Expected hottest/mean EP-rank load for B tokens of per-token `load`.
// Two sources stack:
//
// - routing skew: real routers have hot experts. `skew` is the hottest
//   rank's steady-state load over the mean at large batch, set by the
//   caller after any EPLB rebalancing (which is not modelled: the caller
//   states the post-EPLB skew).
// - finite batch: a rank's load is a sum of B independent per-token
//   draws, so it is ~normal with mean B*mean and sd sqrt(B*variance),
//   and the max of EP such ranks sits about sqrt(2 ln EP) sds above the
//   mean (the Gaussian extreme-value bound). Relative to the mean that
//   falls as 1/sqrt(B), which is why small batches suffer more.
//
// So hottest ~ skew*B*mean + sqrt(2 ln EP * B*variance), over B*mean,
// capped where every token puts its most on the hottest rank. Exactly 1
// at EP = 1, where there is no other rank to be hotter than.
export function rankImbalance(tokens: number, load: TokenLoad, ep: number, skew: number): number {
  if (ep <= 1 || load.mean <= 0 || tokens <= 0) return 1;
  const spread = Math.sqrt(2 * Math.log(ep) * load.variance * tokens) / (tokens * load.mean);
  return Math.min(skew + spread, load.max / load.mean);
}
