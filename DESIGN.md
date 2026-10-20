# NodusDeep — Technical Design

> Status: pre-alpha design document. This describes what we intend to build and why each choice was
> made. It is not a description of working code.

---

## 1. The problem

Large MoE models are too big for ordinary hardware, and the usual answer — buy more GPU — excludes
almost everyone. Meanwhile a huge amount of consumer compute sits idle: gaming GPUs between
sessions, workstations overnight, laptops in drawers, machines that are individually unremarkable.

Two facts make the idle capacity usable:

1. **MoE models are sparse.** A model may hold tens of billions of parameters while activating only a
   small fraction per token. Most weights are cold at any instant.
2. **Experts are independent.** Each expert is a standalone feed-forward block. Two experts never
   need to exchange data. They can live on different machines, in different countries, on different
   hardware.

NodusDeep exists to convert those two facts into a product: a network that runs models larger than
any single participant could hold, using hardware participants already own.

---

## 2. Design principles

**Sparse by default.** Every decision assumes only a fraction of the model is active at a time. If a
design requires the full model to be resident anywhere, it is wrong.

**Heterogeneity is the normal case.** A network of identical datacenter GPUs is a special case we do
not optimize for. The common case is mixed: different vendors, different memory sizes, different
speeds, different operating systems. Capacity is admitted by measurement, never by model name.

**Verification before payment.** Nothing is paid for unless it is proven. Proof must be cheap enough
to run on every job and robust enough that faking work costs more than doing it.

**Small downloads.** A participant should never download a checkpoint they cannot use. A node
receives only the bytes for the shards it was assigned.

**Own the engine, adopt the models.** Efficient MoE serving on mixed consumer hardware is the
product, so the engine is ours. Model architectures are inherited from the open ecosystem, because
re-deriving every attention variant is a treadmill with no payoff.

**Fail visibly.** A broken edge must raise a specific error naming the peer, the layer, and whether
it dropped or timed out. Silent hangs and opaque "broken pipe" failures are treated as bugs.

---

## 3. Architecture

Four layers, each independently testable.

```
┌──────────────────────────────────────────────────────────────┐
│  Client  (OpenAI-compatible API, Python SDK, streaming)       │
├──────────────────────────────────────────────────────────────┤
│  Scheduler                                                    │
│    · routes each token to the experts that own it             │
│    · places shards against measured latency and memory        │
│    · maintains pipeline order and redundancy                  │
├──────────────────────────────────────────────────────────────┤
│  Network                                                      │
│    · peer discovery and capacity announcement                 │
│    · authenticated, encrypted transport                       │
│    · NAT traversal and relay fallback                         │
│    · mid-request recovery                                     │
├──────────────────────────────────────────────────────────────┤
│  Inference engine                                             │
│    · GGUF shard loading by byte offset                        │
│    · quantized MoE forward pass (CPU + GPU)                   │
│    · per-node KV cache, batching, speculative decoding        │
├──────────────────────────────────────────────────────────────┤
│  Verification + Settlement                                    │
│    · per-stage signed receipts and coverage proof             │
│    · challenge-response auditing                              │
│    · staking, slashing, issuance, zero-fee ledger             │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Sharding strategy

Sharding happens at two different granularities, chosen per parameter class.

### 4.1 Shared parameters — sharded by layer

Attention blocks, embeddings, normalization layers and the router are present in every forward pass.
They are sharded by **contiguous layer blocks**, one block per node.

A node holding layers `[12, 24)` runs attention for those layers and forwards activations onward. It
keeps a KV cache covering its own layers only.

### 4.2 Expert parameters — sharded by expert

Inside a MoE layer, experts are dispatched individually. A node may hold experts `{3, 7, 11, 19}`
rather than a contiguous range, and different nodes may hold overlapping sets for redundancy.

This is the important difference from a pure layer split. Because a single expert of a modern MoE
model is often a few hundred megabytes, a machine with 8 GB of memory can participate meaningfully in
a model with hundreds of billions of parameters by holding a modest number of experts.

### 4.3 Why not one granularity

Pure layer sharding makes the smallest participating unit equal to a full layer, which locks out weak
hardware and forces every participant to hold parameters they will rarely use. Pure expert sharding
leaves attention unsharded, which reintroduces a single machine holding the whole attention stack —
a de facto central node.

The hybrid gets both properties: weak machines participate through experts, and no machine becomes
structurally central.

---

## 5. Weight distribution

### 5.1 Format

GGUF. It is memory-mappable, self-describing, quantization-native, and produces single files with
metadata embedded. A node can map a file and read only the ranges it needs.

### 5.2 Byte-offset shard loading

Within a MoE layer, all experts of a projection live in one tensor:

```
blk.{layer}.ffn_gate_exps.weight   →  [n_experts, d_model, d_ff]
```

Expert `e` therefore occupies a contiguous byte range:

```
offset(e) = tensor_base + e × bytes_per_expert
```

So loading one expert is a seek plus a bounded read — not a full tensor load.

```
  tensor base
  │
  ├── expert 0  ──┐
  ├── expert 1    │  each is bytes_per_expert, contiguous,
  ├── expert 2    │  independently addressable
  ├── ...         │
  └── expert N-1 ─┘
```

Two consequences matter:

- **Distribution.** Shards can be served over HTTP range requests or peer-to-peer transfers, so a
  node fetches only its own experts.
- **Demand paging.** Weights stay quantized on disk; the engine maps and decompresses on use. A node
  can hold more experts than fit in memory and page the cold ones in.

### 5.3 Memory policy

Weights are **kept in their quantized form** and decompressed into a reusable buffer at compute time.
We do not expand quantized weights into a float32 copy at load — that multiplies memory by 2–4× and
defeats the point of sharding small.

---

## 6. Inference engine

### 6.1 Compute backends

| Backend | Target | Notes |
|---|---|---|
| CPU (SIMD) | x86-64, ARM64 | Must work out of the box. Slowly is acceptable; not at all is not. |
| CUDA | NVIDIA | Primary GPU path. |
| Metal | Apple Silicon | Second-class citizen but supported. |
| Vulkan | Broad GPU coverage | Fallback for hardware the two above miss. |

Quantization formats in scope for the first release: `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, and `F16`.
Unsupported formats must fail loudly with the format name, never silently miscompute.

### 6.2 MoE forward pass

Per layer, the sequence is:

```
h  = attention(rms_norm(x)) + x
g  = softmax(router(rms_norm(h)))
top_k = argmax_k(g)
y  = Σ_{e ∈ top_k}  g[e] · expert_e(rms_norm(h))
x' = y + h
```

The engine must handle:

- **Grouped-query attention** with configurable KV head count
- **RoPE** with configurable base and scaling
- **Router variants** including bias-corrected and sigmoid-gated routers
- **Shared experts** alongside routed experts

### 6.3 KV cache

A real KV cache is mandatory, not optional. Without it, generation cost grows quadratically with
context length and the product is unusable.

```
keys[layer][kv_head][position][head_dim]
values[layer][kv_head][position][head_dim]
```

Requirements:

- Append-only growth, allocated in blocks to avoid repeated reallocation
- Per-node residency: each node caches only its own layers
- Cache eviction policy for memory pressure
- Truncation support for speculative decoding rollback

### 6.4 Speculative decoding

Over a wide-area network, round-trip latency — not compute — is the scarce resource. Speculative
decoding amortizes one traversal over several committed tokens.

A small draft model on the entry node proposes K tokens. The pipeline verifies all K in a single
traversal. The longest accepted prefix is committed. Pipelining multiple verify chunks keeps the ring
at throughput rather than latency.

Acceptance must be exact: the committed distribution has to match what the full model would have
produced. Anything else is a correctness regression dressed up as a speedup.

---

## 7. Network layer

### 7.1 Discovery

Peers are discovered through a distributed hash table. Each node publishes:

- its identity key
- the shards it holds
- measured throughput and latency
- current load and available capacity

Entries expire quickly so that a dead node stops being routed to within seconds.

### 7.2 Transport

Authenticated and encrypted, with no language-level serialization in the hot path. Frames are
length-prefixed and seals are per-edge. A forged frame is a parse error, never code execution.

Per-edge supervision is mandatory:

- explicit timeouts on every edge
- errors carry peer identity, layer index, and failure mode
- health logging per edge: round-trip latency and bytes moved

### 7.3 NAT traversal

Direct hole-punching first, relay fallback for symmetric NAT. A node behind a home router with no
port forwarding must be able to join with one command.

### 7.4 Placement and ordering

Given a set of nodes with measured capacities:

1. Fit contiguous layer blocks to each node's memory, largest first, to minimize hop count.
2. Order the ring to minimize total round-trip latency.
3. Place the coordinating node in-region — a coordinator far from the ring pays two long round-trips
   per token for no benefit.
4. Assign, then link.

Rebalance on join or leave, but never reshuffle a running pipeline for a single arrival. New capacity
lands in a pool and is used at the next natural boundary.

### 7.5 Mid-request recovery

When a node dies mid-generation:

1. Detect it within seconds via the edge supervisor.
2. Preserve already-committed tokens.
3. Splice a spare into the failed slot.
4. Re-link only the failed node's predecessor and successor.
5. Re-prefill and resume.

The generated continuation must be byte-identical to what an uninterrupted run would have produced.

---

## 8. Verification

Three mechanisms, layered by cost.

### 8.1 Signed receipts

Every node signs a receipt for the work it did:

```json
{
  "job_id": "...",
  "node_pubkey": "...",
  "layer_start": 24,
  "layer_end": 36,
  "n_chunks": 128,
  "in_root": "sha256 of chained inputs",
  "out_root": "sha256 of chained outputs",
  "nonce": "coordinator-issued, per job",
  "sig": "ed25519 over canonical form"
}
```

Validation requires:

- every signature valid against the claimed node key
- the union of all layer ranges tiles `[0, layer_count)` with no gap and no overlap
- the nonce matches the job's issued nonce (no replay)
- on the lossless path, `out_root[i] == in_root[i+1]` across adjacent stages

Property that falls out: a node cannot be paid without a receipt signed by its own key, and a
coordinator cannot fabricate a node's receipt because it does not hold the key. This is what prevents
whoever coordinates from capturing all the pay.

### 8.2 Challenge-response

Receipts prove that a node claims to have run its block. They do not prove the output was correct.
For that, the network occasionally samples a node:

1. The verifier derives a deterministic input from a fresh random seed.
2. It publishes a commitment to the seed before the challenge goes out.
3. The challenged node computes its block's output on that input.
4. The verifier compares against a trusted recomputation.

Comparison uses **cosine similarity with a tolerance**, not exact equality. Outputs are not
bit-reproducible across different GPU vendors, kernels, and quantization paths — demanding exact
matches would flag every honest heterogeneous node.

| Difference source | Typical cosine |
|---|---|
| Same kernel, same hardware | ≥ 0.9999 |
| Different kernel, same quantized weights | ≥ 0.999 |
| Different quantization format | ≥ 0.99 |
| Fabricated or wrong block | far below 0.9 |

The threshold sits above the largest honest drift and below any plausible fake.

The seed must be unpredictable and committed before the challenge. A fixed projection seed lets a
cheater learn which coordinates are sampled once and forge forever.

### 8.3 Economic enforcement

Verification produces evidence; staking produces consequences. A node bonds tokens to participate.
Failed challenges, fabricated receipts, and repeated unavailability draw penalties against that bond,
escalating from warnings to freezing to partial slashing to removal.

This is the current mechanism, and we are explicit about its limit: it catches cheating with high
probability over time, not with certainty per job. A succinct cryptographic proof of computation
would close that gap. The receipt's `in_root`/`out_root` slots are where such a proof would drop in,
without changing the surrounding protocol.

---

## 9. Settlement

### 9.1 Ledger

Settlement runs on a purpose-built chain operated by the network itself. Validators are compensated
from token issuance rather than from transaction fees, which allows **transaction fees to be set to
zero**. Moving tokens and settling jobs costs the participant nothing.

Commercial and technical rationale:

- Users should not need a second asset to use the product. Requiring a separate gas token is friction
  that kills adoption.
- Zero fees make per-token micro-settlement viable. If settlement cost money, small jobs could not be
  settled economically.
- Validator income from issuance keeps the security budget tied to network growth rather than to
  transaction volume, which would otherwise collapse when usage is bursty.

### 9.2 Token roles

| Function | Mechanism |
|---|---|
| Pay contributors | Fixed issuance, distributed in proportion to verified work |
| Pay for inference | Users spend tokens per API call |
| Secure the network | Nodes bond tokens; bad behavior is slashed from the bond |
| Govern | Token-weighted proposals on parameters and upgrades |

### 9.3 Work accounting

Payout for a job is proportional to verified contribution:

```
share(node) = layers_served(node) × tokens_processed(node) × quality(node)
payout(node) = reward_pool(job) × share(node) / Σ share(all nodes)
```

`quality` starts at 1.0 and decays with failed challenges and outages. A node that serves more layers
per token earns more per token, which correctly prices the fact that holding more of the model costs
more.

### 9.4 Consumption, not speculation

Tokens spent on API calls are consumed by the protocol rather than transferred to validators. Demand
for the token is therefore tied to actual inference usage. A contributor who wants to use the network
must either earn tokens by serving or buy them from someone who did.

---

## 10. Failure modes we expect

| Failure | Consequence | Mitigation |
|---|---|---|
| Node dies mid-request | Request stalls | Mid-request recovery with committed-token preservation (§7.5) |
| Node returns plausible garbage | Corrupted output | Challenge-response with tolerance (§8.2) |
| Node fabricates work claims | Theft from reward pool | Signed receipts + coverage proof (§8.1) |
| Coordinator under-reports work | Node underpaid | Per-node signed receipts make attribution independent of the coordinator (§8.1) |
| Too many hops | Latency dominates throughput | Placement minimizes hop count; speculative decoding amortizes traversals |
| Weak node drags the ring | Whole pipeline slows | Placement sizes blocks to measured capacity; weak nodes hold fewer layers |
| Network partition | Split accounting | Settlement finality requires quorum; partitions halt rather than double-spend |

---

## 11. Open problems

Stated plainly, because pretending they are solved would be a disservice to anyone joining.

**Trustless verification at low cost.** Current verification is probabilistic and economic. A
succinct proof that `out = block(in)` without re-execution remains an open research direction.

**Privacy of activations.** A node must decrypt the activations it processes in order to compute on
them, so a malicious node can reconstruct a fraction of a user's content. Mitigations under
consideration: pinning the most informative boundary layers to staked nodes, per-request
trusted-only routing, and evaluation of split-inference privacy bounds.

**Cross-hardware numerical drift.** Honest nodes on different backends produce slightly different
outputs. Tolerances must be set per format pair, and a format mismatch between nodes in one pipeline
must be either prevented at placement or accounted for in verification.

**Cold start.** A permissionless network has the usual chicken-and-egg problem. The intended path is
to make private swarms genuinely useful first — a team or community running its own set of nodes —
so the software earns its place before the public network needs to.

---

## 12. Milestones

| Milestone | Definition of done |
|---|---|
| **M0 — Engine correctness** | One machine loads a MoE model from GGUF and produces output matching a reference implementation within tolerance, with a working KV cache. |
| **M1 — Expert sharding** | The same model, with experts split across two processes, produces output identical to M0. |
| **M2 — Multi-node** | Three machines on a LAN serve the model together. Killing one mid-request does not fail the request. |
| **M3 — Join flow** | An unmodified machine joins with one command, receives only its shards, and serves. |
| **M4 — Verification live** | Receipts are emitted in the serving path and validated before payout. A deliberately broken node is caught by challenge-response. |
| **M5 — WAN** | Nodes on different networks behind NAT serve together, with honest latency numbers published. |
| **M6 — Settlement** | Tokens are issued, distributed by verified work, and spendable on API calls on a zero-fee ledger. |

Each milestone has a pass/fail criterion and a published measurement. Numbers without a reproducible
measurement are not reported.
