# How to Deploy Your Model

A static cost estimator for LLM inference. Given a model, a hardware topology, a
sharding configuration, and a workload, it answers one question:

> **How fast does this run, and what does it cost?**

The motivating use case is capacity and hardware planning. When a new model lands,
you want to compare deployment options quantitatively — TPU v7 pod vs. GB200 NVL72,
under which sharding on each — without standing up a real deployment for every
candidate.

## What it does

You give it four things:

- **Model** — layers with explicit attention (MHA / GQA / MLA) and FFN (dense / MoE)
  dimensions. No hardcoded architectures.
- **Hardware** — per-chip FLOP/s, HBM bandwidth and capacity, and the topology's
  physical axes, each with its geometry (ring, switch, network hop) and link speeds.
- **Sharding** — sizes for the parallelism roles (DPA, TP, EP, ETP, PP), with an
  optional role→axis placement. Omit the placement and it searches for the best one.
- **Workload** — prefill or decode, token count, batch size.

It lowers that into a graph of costed operations (GEMMs, weight and KV loads,
collectives), and a cost backend prices the graph to produce a time estimate.

## Design principles

- **Structure is shared; physics is pluggable.** The model description, topology, and
  lowering pipeline are fixed shared machinery — the product. How a costed graph turns
  into a time estimate is delegated to an interchangeable cost backend. Today that's a
  simple roofline model; a more detailed scheduling simulator can be added later without
  touching anything upstream. The pipeline is agnostic to which backend is active.

- **It never makes a decision it can't justify with a price.** Wherever there's a
  choice — which collectives to emit at a sharding boundary, which physical axes a
  role should occupy — it enumerates the legal options and lets the active backend
  pick the cheapest. There are no hand-coded "on TPUs, prefer X" policies, so the tool
  stays correct on hardware nobody had in mind when it was written.

## Quick start

```
npm install
npm run dev      # interactive explorer
npm test         # run the suite
```

The explorer ranks sharding configurations per chip, and drills into any one with a
fabric view (the chips on their real interconnect) and an execution trace (the op
graph, colored by what each op is bound on).

## Layout

```
src/core   engine (lowering, placement search, cost backends), model & hardware specs
src/ui     the explorer — leaderboard, fabric view, execution trace
tests      property / fuzz tests
```
