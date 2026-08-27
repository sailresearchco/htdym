interface FabricBase {
  // Aggregate one-way egress bandwidth per chip in bytes/s, summed over all
  // links (e.g. NVLink4 "900 GB/s bidirectional" => 450e9 here, a v5p with
  // 6 ICI links of 9e10 B/s each => 5.4e11). deployedAxes derives per-axis
  // link bandwidths from this.
  bandwidthPerChip: number;
  // Effective latency of one collective step over the fast fabric in
  // seconds: one hop between ring neighbors, or one pipelined step
  // through the switch. 0 = model bandwidth only.
  latency: number;
  // Chips reachable over the fast fabric (NVLink domain / ICI slice).
  domainSize: number;
}

// A ring/torus fabric sold as discrete slices (TPU pods, Trainium boxes).
export interface SlicedFabric extends FabricBase {
  // The discrete list of acquirable slice/instance topologies: the single
  // source of truth for what can be rented.
  topologies: SliceTopology[];
  // Chips housed by one host machine (a v5e host holds a 4x2 block);
  // translates a global node count into a slice size.
  chipsPerHost: number;
  scaleOut?: never;
}

// A switched fabric with per-chip granularity (GPU boards).
export interface SwitchedFabric extends FabricBase {
  topologies?: never;
  chipsPerHost?: never;
  // Optional second fabric tier: nodes of domainSize chips joined by a
  // switched scale-out network (InfiniBand/RoCE). Extends the reachable
  // world to domainSize*maxNodes.
  scaleOut?: {
    // one-way egress per chip over the inter-node network (one 400 Gb
    // NDR NIC per GPU = 50e9 in HGX/DGX reference designs)
    bandwidthPerChip: number;
    // per-step latency over this tier (NIC + fabric switch traversal)
    latency: number;
    // nodes reachable over the scale-out network
    maxNodes: number;
  };
}

export type InterconnectSpec = SlicedFabric | SwitchedFabric;

// One acquirable slice/instance topology of a fabric.
export interface SliceTopology {
  // the shape's name as sold, e.g. "4x4x12"
  name: string;
  // chips per axis
  dims: number[];
  // per-axis wraparound links
  wrapped: boolean[];
  // per-axis twisted-torus flag: a twist relands the axis's wraparound
  // cables on a shifted row, ~doubling bisection bandwidth across it.
  twisted?: boolean[];
  // chips in the slice: product of dims
  count: number;
}

// A real interconnect dimension of a deployed machine.
export interface PhysAxis {
  name: string; // X/Y/Z for torus dims, D for a switch domain, N for scale-out
  size: number;
  kind: 'ring' | 'switch';
  // ring: one-way bandwidth per link
  // switch: one-way per-chip egress into this tier
  bandwidth: number;
  // effective latency of one collective step over this axis (one
  // neighbor hop / one switch traversal), 0 = pure-bandwidth pricing
  latency: number;
  // ring only: wraparound links close the ring
  wrap: boolean;
}

// One factor of a physical axis.
export interface MeshDim {
  // parent name when unsplit, parent name + factor index when split ("Y0")
  name: string;
  size: number;
  // the parent axis, shared by all its factors
  axis: PhysAxis;
  // chip stride along the parent: 1 = contiguous neighbors
  stride: number;
}

export function reachableChips(ic: InterconnectSpec): number {
  return ic.domainSize * (ic.scaleOut?.maxNodes ?? 1);
}

// Resolve the machine a deployment occupies into named physical axes.
export function deployedAxes(
  ic: InterconnectSpec,
  use: SliceTopology | { domain: number; nodes: number },
): PhysAxis[] {
  if ('dims' in use) {
    // ring fabric
    if (!ic.topologies) throw new Error(`fabric has no slices`);

    // per-axis link bandwidth: egress splits over 2 links per torus axis
    const link = ic.bandwidthPerChip / (2 * use.dims.length);
    return use.dims.map((size, i): PhysAxis => ({
      name: 'XYZ'[i],
      size,
      kind: 'ring',
      bandwidth: link,
      latency: ic.latency,
      wrap: use.wrapped[i],
    }));
  } else {
    // switched fabric
    if (ic.topologies) throw new Error(`slice fabric needs an acquired slice`);
    if (use.domain > ic.domainSize) throw new Error(`chips exceed the domain`);

    const axes: PhysAxis[] = [
      {
        name: 'D',
        size: use.domain,
        kind: 'switch',
        bandwidth: ic.bandwidthPerChip,
        latency: ic.latency,
        wrap: false,
      },
    ];

    if (use.nodes > 1) {
      if (use.nodes > (ic.scaleOut?.maxNodes ?? 1)) throw new Error(`nodes exceed the scale-out`);

      axes.push({
        name: 'N',
        size: use.nodes,
        kind: 'switch',
        bandwidth: ic.scaleOut!.bandwidthPerChip,
        latency: ic.scaleOut!.latency,
        wrap: false,
      });
    }

    return axes;
  }
}

// Factor a physical axis into multiple logical mesh dimensions.
export function factorAxes(axes: PhysAxis[], splits: Record<string, number[]> = {}): MeshDim[] {
  for (const name of Object.keys(splits))
    if (!axes.some((ax) => ax.name === name)) throw new Error(`split of unknown axis`);

  return axes.flatMap((ax) => {
    const factors = splits[ax.name];

    if (!factors) return { name: ax.name, size: ax.size, axis: ax, stride: 1 };

    if (factors.reduce((p, v) => p * v, 1) !== ax.size)
      throw new Error(`splits do not multiply to the axis size`);

    let stride = 1;
    return factors.map((size, i) => {
      const dim = { name: `${ax.name}${i}`, size, axis: ax, stride };
      stride *= size;
      return dim;
    });
  });
}
