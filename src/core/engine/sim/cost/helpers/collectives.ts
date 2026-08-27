import { CollectiveKind } from '../../ir/ops';
import { localElems, MeshAxis, TensorType } from '../../ir/tensors';
import { MeshDim } from '../../../../hardware/topology';

// One realized communication ring/domain a collective can run over.
interface CommChannel {
  n: number;
  // ring: usable one-way bandwidth per link, switch: per-chip egress
  bw: number;
  // per-step latency: one logical hop (its physical links in series) or
  // one switch traversal
  lat: number;
  // closed ring (wraparound or a multi-axis grid), open path otherwise
  closed: boolean;
  switched: boolean;
}

// The channel a single mesh dim provides on its own.
function channelOf(d: MeshDim): CommChannel {
  if (d.axis.kind === 'switch')
    return { n: d.size, bw: d.axis.bandwidth, lat: d.axis.latency, closed: false, switched: true };

  return {
    n: d.size,
    // a hop between stride-s logical neighbors crosses s physical links,
    // and the s sibling groups (the other offsets within the stride
    // window) run the same collective concurrently over those links, so
    // each group effectively owns 1/s of every link it touches
    bw: d.axis.bandwidth / d.stride,
    // the same s links cross in series, so the hop pays each one's latency
    lat: d.axis.latency * d.stride,
    // wraparound only helps a group spanning its whole parent axis
    closed: d.axis.wrap && d.size * d.stride === d.axis.size,
    switched: false,
  };
}

// All axes of the spanned grid working concurrently as one channel: the
// payload splits across the dims and each part pipelines over a different
// axis's links at the same time, so a chip's usable ingress is the sum of
// every dim's usable directions (2 per closed ring, 1 per open path,
// stride-derated as usual). On a wrapped torus this recovers the k-axis
// bandwidth multiplier of the standard multi-axis algorithms, and on an
// even open mesh it lands exactly on the snaked single ring (a 2x4 mesh
// is an 8-ring using only mesh edges).
function gridChannel(group: MeshDim[]): CommChannel | null {
  // one dim per parent: factors of the same axis contend for the same
  // links (composeFactors merges the chainable case instead)
  const parents = new Set(group.map((d) => d.axis));
  if (group.length < 2 || parents.size < group.length) return null;

  // switches have no grid to spread over
  if (group.some((d) => d.axis.kind !== 'ring')) return null;

  const rings = group.map(channelOf);
  return {
    n: group.reduce((p, d) => p * d.size, 1),
    // agTime divides by 2*bw when closed, so half the summed one-way
    // ingress makes the formulas line up
    bw: rings.reduce((s, ch) => s + (ch.closed ? 2 : 1) * ch.bw, 0) / 2,
    // the grid's latency critical path is per-dim, and the per-dim cut
    // floors below (latency-inclusive) already carry it
    lat: 0,
    closed: true,
    switched: false,
  };
}

// All-gather of one shard per chip: after n-1 pipelined steps every chip
// has received the other n-1 shards. A closed ring sends half the shards
// each way, giving 2 usable links per chip. An open path effectively gets
// one (the standard 2x mesh penalty), and switch egress is the same
// single-pipe shape. Reduce-scatter is the time-reverse with identical
// traffic and all-reduce is one of each, so all three price through here.
function agTime(shard: number, ch: CommChannel): number {
  // latency counts the pipeline's sequential steps: a closed ring works
  // both directions at once, so its longest chain is half the ring
  const steps = ch.closed ? Math.ceil((ch.n - 1) / 2) : ch.n - 1;
  return (shard * (ch.n - 1)) / (ch.closed ? 2 * ch.bw : ch.bw) + steps * ch.lat;
}

// All-to-all where each chip holds local bytes
// and sends an equal 1/n slice of it to every peer.
function a2aTime(local: number, ch: CommChannel): number {
  // switch: every pair is one hop, so each chip just pushes its (n-1)/n
  // outbound share through its own egress
  if (ch.switched) return (local * (ch.n - 1)) / (ch.n * ch.bw) + ch.lat;

  // ring: bisection-bound. Slices travel n/4 hops on average (closed,
  // taking the shorter way around) or n/2 open, and spreading that total
  // link traffic over the ring's links leaves local*n/8 bytes per link
  // closed, twice that open. Latency is the farthest slice's hop chain.
  return (
    (local * ch.n) / (ch.closed ? 8 * ch.bw : 4 * ch.bw) +
    (ch.closed ? Math.floor(ch.n / 2) : ch.n - 1) * ch.lat
  );
}

// Dimension-wise staging: run the collective one channel at a time in the
// given order, tracking how each stage changes the next stage's payload.
function stagedTime(kind: CollectiveKind, order: CommChannel[], bytes: number): number {
  let t = 0;
  if (kind === 'all-gather') {
    // each stage gathers the current shard across its channel, so the
    // shard every chip holds grows n-fold per stage
    let shard = bytes;
    for (const ch of order) {
      t += agTime(shard, ch);
      shard *= ch.n;
    }
  } else if (kind === 'reduce-scatter' || kind === 'all-reduce') {
    // each reduce-scatter stage leaves a chip 1/n of what it held, so
    // later stages move less. This shrinkage is why all-reduce scatters
    // first and why stage order can matter.
    let full = bytes;
    for (const ch of order) {
      // same wire traffic as all-gathering the shard this stage leaves
      t += agTime(full / ch.n, ch);
      full /= ch.n;
    }
    if (kind === 'all-reduce') {
      // then gather the reduced shards back out, retracing the stages in
      // reverse so payload sizes mirror the way down
      let shard = full;
      for (const ch of [...order].reverse()) {
        t += agTime(shard, ch);
        shard *= ch.n;
      }
    }
  } else {
    // all-to-all routes dimension by dimension (row exchange, then column
    // exchange), and every stage carries the full local payload
    for (const ch of order) t += a2aTime(bytes, ch);
  }
  return t;
}

// Merge factors of one parent whose strides chain contiguously: X0 (size
// 2, stride 1) and X1 (size 2, stride 2) address exactly the chips of one
// size-4 stride-1 factor, and merging all factors of an axis recovers the
// whole physical ring, wraparound included. A switch has no stride
// structure (any subset of its chips is one uniform domain), so its
// factors always merge.
function composeFactors(group: MeshDim[]): MeshDim[] {
  const axes = [...new Set(group.map((d) => d.axis))];
  return axes.flatMap((axis) => {
    // innermost factor first, so chainable neighbors become adjacent
    const factors = group.filter((d) => d.axis === axis).sort((a, b) => a.stride - b.stride);
    const out: MeshDim[] = [];
    for (const f of factors) {
      const prev = out[out.length - 1];
      // f starts exactly where prev's window ends, so together they tile
      // one contiguous stride window and act as a single factor
      if (prev && (axis.kind === 'switch' || f.stride === prev.stride * prev.size)) {
        out[out.length - 1] = { ...prev, name: `${prev.name}+${f.name}`, size: prev.size * f.size };
      } else out.push(f);
    }
    return out;
  });
}

// stagedTime is a pure function of the channel sequence, so only DISTINCT
// sequences are worth pricing: identical channels are common (the split
// factors of one switch domain are indistinguishable) and k of them would
// otherwise inflate the orderings k!-fold
function channelOrders(chans: CommChannel[]): CommChannel[][] {
  const key = (c: CommChannel) => `${c.n} ${c.bw} ${c.lat} ${c.closed} ${c.switched}`;
  const sorted = [...chans].sort((a, b) => key(a).localeCompare(key(b)));
  const keys = sorted.map(key);
  const out: CommChannel[][] = [];
  const used: boolean[] = new Array(sorted.length).fill(false);
  const cur: CommChannel[] = [];
  const rec = () => {
    if (cur.length === sorted.length) return void out.push([...cur]);
    for (let i = 0; i < sorted.length; i++) {
      // equal channels take slots in index order, so each distinct
      // sequence is built exactly once
      if (used[i] || (i > 0 && !used[i - 1] && keys[i] === keys[i - 1])) continue;
      used[i] = true;
      cur.push(sorted[i]);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

// cache of collective costs, keyed by the collective kind, the number of
// bytes in the payload, and the group structure (concatenated dim descriptions)
const costCache = new Map<string, number>();

// Time for one collective over the named mesh dims. input is the op's
// INPUT tensor type (sharded over the dims for all-gather/all-to-all,
// carrying them unreduced for all-reduce/reduce-scatter). Enumerates
// the legal realizations and takes the cheapest.
export function collectiveCost(
  kind: CollectiveKind,
  over: MeshAxis[],
  input: TensorType,
  bytesPerElem: number,
  dims: MeshDim[],
): number {
  const group = over
    .map((name) => {
      const dim = dims.find((d) => d.name === name);
      if (!dim) throw new Error(`collective over unknown mesh dim`);
      return dim;
    })
    // size-1 dims carry no traffic and would inflate the grid's ingress
    .filter((d) => d.size > 1);

  // gathered/routed dims must shard the input, reduced dims must be
  // unreduced partials on it (p2p sends the whole tensor and is exempt)
  if (kind !== 'p2p') {
    const carried =
      kind === 'all-reduce' || kind === 'reduce-scatter' ? input.unreduced : input.sharding.flat();
    for (const d of group)
      if (!carried.includes(d.name)) throw new Error(`collective dims absent from its input`);
  }

  // what one chip actually holds, all sharding divided out
  const bytes = localElems(input, dims) * bytesPerElem;

  // pure in (kind, payload, group structure) and re-priced thousands of
  // times across placements, so memoize on exactly those inputs. The
  // parent axis name is part of the structure: co-parent factors compose,
  // same-shaped factors of different parents do not
  const ckey = `${kind} ${bytes}|` + group.map(descOf).sort().join('|');
  const memo = costCache.get(ckey);
  if (memo !== undefined) return memo;
  const price = (t: number) => (costCache.set(ckey, t), t);

  const composed = composeFactors(group);

  // p2p is one neighbor hop, bounded by the slowest fabric it crosses.
  // It reads the composed group: consecutive stages along chained
  // factors of one parent are plain neighbors on that ring, so the
  // stride derate (which models sibling groups sharing links) does not
  // apply to the hop, and one message pays the slowest tier's latency
  // once rather than every dim's
  if (kind === 'p2p')
    return price(
      composed.length
        ? bytes / Math.min(...composed.map((d) => channelOf(d).bw)) +
            Math.max(...composed.map((d) => d.axis.latency))
        : 0,
    );

  // price the group both as written and with chainable factors merged
  const groups = composed.length < group.length ? [group, composed] : [group];

  const candidates = groups.flatMap((grp) => {
    // every distinct stage ordering (payload growth makes order matter once
    // the channels' bandwidths differ); a2a stages all carry the full
    // payload, so its sum is order-blind and one ordering suffices
    const chans = grp.map(channelOf);
    const orders = kind === 'all-to-all' ? [chans] : channelOrders(chans);
    const cands = orders.map((order) => stagedTime(kind, order, bytes));

    // plus all axes working at once, which staging cannot express
    const grid = gridChannel(grp);
    if (grid) {
      if (kind === 'all-to-all') {
        // restricted to any one dim, the traffic is a full-payload a2a
        // over that ring no matter how routing is staged, so each dim's
        // link load is fixed and pipelining chunks through the dims
        // finishes at the slowest dim rather than the staged sum
        cands.push(Math.max(...grp.map((d) => a2aTime(bytes, channelOf(d)))));
      } else {
        // the summed-ingress claim cannot beat any single dim's cut:
        // data from a different slice of a dim must cross that dim's
        // links somewhere (once per neighboring slice, replicating
        // after), so each dim floors the grid at its own ring time for
        // the final shard
        const shard = kind === 'all-gather' ? bytes : bytes / grid.n;
        const both = kind === 'all-reduce' ? 2 : 1;
        const floor = Math.max(...grp.map((d) => both * agTime(shard, channelOf(d))));
        cands.push(Math.max(stagedTime(kind, [grid], bytes), floor));
      }
    }

    return cands;
  });

  return price(Math.min(...candidates));
}

// key fragment per dim, built once: dim objects are shared across every price of their mesh
const dimDesc = new WeakMap<MeshDim, string>();

function descOf(d: MeshDim): string {
  let s = dimDesc.get(d);
  if (s === undefined)
    dimDesc.set(
      d,
      (s = `${d.axis.name} ${d.axis.kind} ${d.axis.bandwidth} ${d.axis.latency} ${d.axis.wrap} ${d.axis.size} ${d.stride} ${d.size}`),
    );
  return s;
}
