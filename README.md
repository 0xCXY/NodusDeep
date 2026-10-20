<p align="center">
  <img src="logo.svg" width="110" alt="NodusDeep">
</p>

<h1 align="center">NodusDeep</h1>

<p align="center">
  <strong>Distributed AI Inference Network</strong><br>
  <em>One Model. Many Nodes.</em>
</p>

<p align="center">
  <img alt="License: PolyForm Noncommercial 1.0.0" src="https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-0D9488">
  <img alt="Status: pre-alpha" src="https://img.shields.io/badge/status-pre--alpha-orange">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue">
</p>

---

## What is NodusDeep

NodusDeep is a distributed inference network for large Mixture-of-Experts (MoE) models. It turns a
loose collection of ordinary computers — gaming rigs, office desktops, laptops, home servers — into
a single serving cluster for models that no one of them could hold alone.

A model is split into independent pieces and spread across many machines. When a request arrives,
each machine computes only its own piece, and the pieces cooperate to produce one coherent answer.
No single node ever holds the whole model. No datacenter is required.

The people who lend their hardware are paid in the network's own token for every unit of real work
they perform. The people who use the network pay in the same token, and calling the API costs no
network fees.

**In short: share your compute, run models you could never run alone, and get paid for it.**

---

## Why this works

Modern MoE models are architecturally built for distribution. A model may carry tens of billions of
parameters, but only a small fraction of them — often under 3% — are active for any given token.
The rest sit idle.

That is the opening. If only a slice of the model is active at a time, then only a slice of the
machine needs to be big. NodusDeep exploits this directly:

- **Experts are independent.** Each expert is a self-contained feed-forward block. Experts can live
  on different machines and never need to talk to each other.
- **Pieces are small.** A single expert of a modern MoE model can be a few hundred megabytes. It fits
  comfortably on hardware that would choke on the full checkpoint.
- **Work is verifiable.** Every node signs a receipt for the activations it processed. Contribution
  is measured, not trusted.

This is why NodusDeep can run a model that is orders of magnitude larger than any individual
participant's hardware.

---

## How it works

```
                         ┌──────────────────┐
      client  ──────────▶│   API Gateway    │  OpenAI-compatible
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │    Scheduler     │  routes tokens to the
                         │                  │  experts that own them
                         └────────┬─────────┘
                                  │
        ┌───────────────┬─────────┴────────┬───────────────┐
        │               │                  │               │
   ┌────▼────┐     ┌────▼────┐        ┌────▼────┐     ┌────▼────┐
   │  Node   │     │  Node   │        │  Node   │     │  Node   │
   │ experts │     │ experts │  ....  │ experts │     │ experts │
   │  0..11  │     │ 12..23  │        │ 24..35  │     │ 36..47  │
   └─────────┘     └─────────┘        └─────────┘     └─────────┘
        │               │                  │               │
        └───────────────┴──────────────────┴───────────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Signed receipts │  contribution proof
                         │  → settlement    │  → token payout
                         └──────────────────┘
```

1. **Join.** A contributor runs one command. The node probes its hardware, announces the capacity it
   can offer, and receives a slice of the model — only the bytes it needs, nothing more.
2. **Place.** The scheduler measures latency between nodes and lays out the model so that the
   pipeline is as short and as balanced as possible. A weaker machine simply holds fewer pieces.
3. **Serve.** A request streams through the assigned nodes. Each node loads only its own expert
   shard, keeps its own KV cache, and returns its output.
4. **Prove.** Every node signs a receipt covering the input and output it handled. The receipt set
   must tile the model with no gaps and no overlaps, and it carries a per-job nonce so it cannot be
   replayed.
5. **Settle.** Verified receipts are converted into token payouts, weighted by the work each node
   actually performed.

---

## Features

### For people who need models

- **OpenAI-compatible API** — point any existing SDK or tool at the endpoint and it just works.
- **Models far larger than your hardware** — the network pools capacity, so the model size is no
  longer bounded by your machine.
- **Streaming responses** with low time-to-first-token through pipelined execution.
- **No network fees** — API calls are paid in the network token, with settlement costs borne by the
  protocol rather than charged per request.
- **Deterministic, verifiable output** — receipts let any result be traced back to the nodes that
  produced it.

### For people who contribute compute

- **Any hardware counts.** A single consumer GPU, a multi-GPU workstation, a CPU-only box, an Apple
  Silicon machine. Capacity is assigned by measurement, not by brand.
- **Download only your slice.** A node fetches the expert shards it was assigned, not the whole
  checkpoint. Joining a 200B+ model does not require 200B of disk.
- **One-command join.** No cluster configuration, no manual peer lists.
- **Paid per unit of work.** Rewards are proportional to the layers served and the tokens produced,
  verified by signed receipts.
- **Fault tolerant.** If a node drops mid-request, its work is replayed on a spare and the request
  completes without failing.

### For the network

- **Contribution is measurable.** Receipts + coverage checks + tolerance-based challenge-response
  make it expensive to claim work that was not done.
- **Stake and slash.** Nodes bond tokens to participate; provably bad output is punished from that
  bond.
- **Heterogeneity is first-class.** Mixed hardware is the normal case, not an edge case.
- **Zero-fee ledger.** The settlement chain is operated by the network itself, with transaction fees
  set to zero and validators compensated through issuance.

---

## Token economics

NodusDeep has a native token with three jobs: paying contributors, paying for inference, and
governing the network.

| Mechanism | Design |
|---|---|
| **Wallet** | Every participant — contributor or user — has a self-custodial wallet created in one step. |
| **Mining-style rewards** | A fixed issuance schedule pays nodes in proportion to verified work: layers served × tokens produced × uptime. Comparable to a mining pool's share-based payout. |
| **Stake to serve** | A node bonds tokens before it can take work. The bond is what makes cheating expensive and Sybil identities uneconomical. |
| **Slashing** | A node that returns provably incorrect output, fabricates receipts, or repeatedly goes offline loses part of its bond. |
| **Pay-per-use** | Users spend tokens to call the API. Spent tokens are consumed by the protocol, tying token demand to real usage rather than speculation. |
| **Free transactions** | The settlement chain charges no transaction fees. Validators are paid from issuance, so moving tokens and settling jobs costs the user nothing. |
| **Open transfer** | The token is freely transferable. Contributors can hold, spend, or trade what they earn. |

The intent is straightforward and deliberately unglamorous: **contributors are the miners, inference
is the work, and the token is the payroll.**

---

## Architecture

NodusDeep is built as four cooperating layers, each of which can be developed and tested on its own.

| Layer | Responsibility |
|---|---|
| **Inference engine** | Loads expert shards from GGUF checkpoints by byte offset, runs quantized MoE forward passes, manages per-node KV cache. Native backends for CPU and GPU. |
| **Network layer** | Peer discovery, authenticated and encrypted transport, NAT traversal, pipeline assembly, ring ordering, mid-request recovery. |
| **Verification layer** | Per-stage signed receipts, coverage validation, challenge-response sampling with numerical tolerance for heterogeneous kernels, reputation scoring. |
| **Settlement layer** | Wallet, staking, slashing, issuance, per-token payout, zero-fee ledger. |

The inference engine is the part we own end to end, because serving MoE models efficiently on mixed
consumer hardware is the core of the product. Model architectures are adopted from the open
ecosystem — there is no value in re-deriving every attention variant by hand.

---

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| **0 — Engine** | Single-node MoE inference from GGUF with correct quantization and KV cache. Verifiable against a reference run. | Planned |
| **1 — Local cluster** | Several machines on one network serve a model together. Expert-level sharding. Output matches single-node execution. | Planned |
| **2 — Join flow** | One-command onboarding, hardware probing, capacity announcement, automatic placement. | Planned |
| **3 — Verification** | Signed receipts live in the serving path. Challenge-response auditing with stake-backed penalties. | Planned |
| **4 — Public network** | NAT traversal, relay fallback, geographic placement, mid-request healing over the open internet. | Planned |
| **5 — Settlement** | Wallet, token issuance, per-node payouts, zero-fee ledger, slashing wired to verification. | Planned |
| **6 — Scale** | Larger models, more architectures, batch throughput, long-context serving. | Planned |

Phase 0 and Phase 1 are where the near-term work lives. Everything later depends on the engine
being correct and fast.

---

## Status

**Pre-alpha.** This repository is at the design and bootstrap stage. The architecture is specified
in [`DESIGN.md`](DESIGN.md); implementation has not started.

What exists today:

- [x] Product definition and architecture
- [x] Token and incentive design
- [x] License
- [ ] Inference engine
- [ ] Network layer
- [ ] Verification layer
- [ ] Settlement layer

We would rather state this plainly than ship a roadmap dressed up as a product.

---

## Getting involved

NodusDeep is noncommercial open source. If you want to work on distributed inference, MoE serving,
verifiable computation, or incentive design, this is the kind of problem where one more pair of
hands changes the timeline.

- **Read the design.** [`DESIGN.md`](DESIGN.md) is the current source of truth.
- **Open an issue.** Design critique, hardware reports, and benchmark methodology are all welcome.
- **Bring hardware.** Reports from real machines — especially odd ones — are worth more than
  synthetic benchmarks.

---

## License

**PolyForm Noncommercial License 1.0.0.** See [`LICENSE`](LICENSE).

The full source is available and may be read, studied, modified, and redistributed for any
noncommercial purpose — including personal research, hobby projects, education, public research
institutions, charities, and government use.

**Commercial use is not permitted.** If you want to use NodusDeep commercially, you need a separate
written license.

> A note on terminology: because it restricts commercial use, this license is not OSI-approved
> "open source" in the strict sense. The accurate description is **source-available,
> noncommercial**. We state this so nobody discovers it later and feels misled.

---

<p align="center">
  <sub>NodusDeep — One Model. Many Nodes.</sub>
</p>
