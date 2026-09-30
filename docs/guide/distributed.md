---
layout: docs
title: Distributed Inference (Research)
parent: Guide
nav_order: 10
description: "Research evaluation of a Mac orchestrating USB-tethered iPhones for speculative decoding and embeddings (measured results, transport and iOS findings, what AuraLocal would need); no mesh API ships, and the multilingual-e5-small provider it evaluated is documented in Document RAG."
---

# Distributed Inference over USB (research)
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Status and verdict

{: .warning }
> **Research only. Not shipped, no API.** AuraLocal has no mesh, node, USB transport or
> speculative-decoding API. This page records an evaluation (2026-09-29) so its results and pitfalls
> are not rediscovered. The one piece that shipped, the multilingual-e5-small embedding provider, is
> documented in [Document RAG]({{ '/guide/rag' | relative_url }}#multilingual-e5-small-opt-in).

Evidence labels, applied to every number (no iPhone was connected, so no phone figure is [Verified]):

- **[Verified]** measured on a MacBook Pro M1 Pro (32 GB, 16-core GPU, macOS 26), or read in source
  code, SDK headers or Apple documentation during the evaluation. Phone specifications count as
  [Third-party] even when Apple publishes them.
- **[Third-party]** published by someone else and not reproduced here.
- **[Inferred]** derived from verified or third-party inputs.
- **[Needs device]** only a connected iPhone can answer it.

**Verdict.** A proof of concept is worth building, but not for speed. On this hardware it would show
discovery, topology and failover for cable-attached iPhones (no reference project does this); a remote
speculative-decoding loop that matches greedy up to kernel near-ties but runs slower; and a light node
that could pay off only for bulk indexing or near saturation of the Mac's Neural Engine (ANE) [Inferred].

Four findings change the naive design:

1. **Speculative decoding does not speed up an M1 Pro, with a local draft or a loopback stand-in for a
   remote one; a real phone draft is modeled no better [Inferred].** Verifying k draft tokens costs
   1.64×, 2.6× and 3.95× a decode step for k = 1, 2, 4 [Verified]. The phone should not be the draft.
2. **Offloading a query embedding is not expected to lower its typical latency; only the Mac ANE's noisy
   tail (p90 up to ~65 ms) leaves room for a phone [Inferred].** The Mac ANE runs e5-small (128 tokens)
   in 1.6–1.9 ms [Verified]; the phone adds its own compute plus a USB round trip [Inferred].
3. **`iproxy` 2.1.1 listens on all interfaces** by default, despite its help text [Verified], so the
   tunnel is reachable from the LAN unless a firewall blocks it [Inferred].
4. **usbmux connects only from the Mac to the phone** [Verified]. That rules out MLX distributed over
   the cable and means a phone cannot call the Mac without a reverse channel.

## What was evaluated

| Node | Hardware | Proposed role | Role after evaluation |
|---|---|---|---|
| Mac | M1 Pro, 32 GB | Orchestrator, vector store, 14B–32B reasoning LLM | Same, with a 14B target (32B is marginal: ~19.7 GiB at 8k context against a 20.0 GiB limit [Inferred]; see [Model choices](#model-choices)) |
| iPhone 17 Pro | A19 Pro, 12 GB, USB-C | Draft model (1B–3B) for speculative decoding, or perception (Whisper, Vision) | Perception, embeddings, bulk indexing; not draft |
| iPhone 12 | A14, 4 GB, Lightning | Embeddings, classification, VAD | Small Core ML models on the ANE only; bulk indexing capacity |

Phone figures in the table are [Third-party]. The proposed flow: discover phones with libimobiledevice,
open one `iproxy` tunnel per device, query `GET /device/capabilities` from an `NWListener` in an iOS 17+
app, build a topology table, assign roles, and run a RAG pipeline that reports per-node latency.

The revised design always treats the Mac as a candidate. A phone gets a role only if it is ready, in the
foreground, `nominal` or `fair` thermally, runs the same artifact (embedding cosine ≥ 0.999) or tokenizer
(draft), has `1.25 × model + 150 MB` of headroom and an RTT p95 ≤ 20 ms, and beats the Mac by ≥ 10%. With
the measured M1 Pro costs, that keeps everything on the Mac with no draft [Inferred].

## Measured results

LLM rows used MLX 0.32.3 and mlx-lm 0.31.3 (the llama.cpp row, build 9660); e5 rows time the Core ML
prediction alone from a Swift benchmark, without tokenization. Latencies are p50, batch 1. End-to-end
figures for the shipped provider are in [Document RAG]({{ '/guide/rag' | relative_url }}#multilingual-e5-small-opt-in).

| Measurement | Result | Evidence |
|---|---|---|
| Qwen3-14B-4bit greedy decode | 19.0–19.8 tok/s on a clean run; 51.5 ms/token inside the decode loop | [Verified] |
| 14B forward cost by tokens per forward, T = 1…6+ | 57.5 · 94.4 · 150.1 · 181.8 · 226.7 · ~370 ms | [Verified] |
| Speculative, local draft (mlx-lm built-in) | 0.6B: k=1 11.4–16.5, k=2 12.2–15.8, k=4 9.7–12.4 tok/s. 1.7B: k=1 16.1–18.0, k=2 13.5–17.4 tok/s | [Verified] |
| Speculative, remote draft (separate process over loopback TCP) | 0.6B k=1 13.4–17.7 tok/s; 1.7B k=1 15.4–18.1 tok/s | [Verified] |
| Acceptance per draft token, k=1 | ≈ 0.75 with 0.6B (0.64–0.86), ≈ 0.80 with 1.7B (0.70–0.95) | [Verified] |
| Remote loop output vs greedy | 48 of 48 runs identical token for token; remote acceptance equal to mlx-lm's in all 20 remote/local pairs | [Verified] |
| Loopback transport per draft round (JSON over TCP) | 0.4–0.8 ms median per run (two runs 1.7–2.1 ms) | [Verified] |
| 0.6B draft on the Mac GPU | Alone 218 tok/s (4.6 ms/token). Concurrently with the 14B: 119 tok/s, and the 14B forward +30% (T=1) / +34% (T=2) | [Verified] |
| llama.cpp (build 9660), same 14B as Q4_K_M | 13.9 tok/s; with a 0.6B Q8_0 draft 10.3–14.15 tok/s | [Verified] |
| e5-small, fixed 128 tokens, by compute unit | `.cpuAndNeuralEngine` 1.93 ms (1.55 ms with the ANE layout); `.all` 2.48; `.cpuAndGPU` 4.50; `.cpuOnly` 4.98 ms | [Verified] |
| e5-small, enumerated 64/128/256/512 tokens, ANE | ANE layout 1.60 · 1.63 · 3.40 · 9.57 ms; stock graph 1.48 · 2.02 · 5.50 · 18.01 ms | [Verified] |
| First ANE load (on-device compile) | ~3.2 s for one fixed shape; 15 s for four shapes (stock); 29–31.5 s for four shapes (ANE layout). Next process from the compiled cache: 82–122 ms. First predict at an unwarmed shape: up to 6.2 s | [Verified] |
| e5-small fp16 memory | `phys_footprint` 11–22 MB after load (weights are file-backed). An int8 copy showed 388–575 MB in some runs and crashed on `.cpuAndGPU` | [Verified] |
| Contention: e5 on the ANE during a 14B-sized decode | Decode −10% at 457 embeddings/s (saturated, 128 tokens); −2.5% at 93/s (512 tokens). Under 0.3% at 1–10 queries/s [Inferred] | [Verified] |
| Contention: e5 on the GPU / CPU | GPU: decode −6.6%, embedding 4.5 → 28.5 ms. CPU at 150/s: decode −2.4% | [Verified] |
| ANE noise from other system clients (GPU idle) | Clean: p50 1.63, p90 1.75 ms. Noisy stretch: p50 4–6 ms, p90 ~65 ms, minimum 1.9 ms | [Verified] |
| Host-side transport (no USB involved) | Apple's usbmuxd control round trip 0.11 ms. Against a fake usbmuxd and fake device: new connection through `iproxy` 2.26 ms (p99 8.5); keep-alive 0.16 ms through `iproxy`, 0.05 ms through a direct `Connect`; direct `Connect` bulk 422–486 MB/s | [Verified] |
| Tokenizing e5 input on the Mac | Rust `tokenizers` 0.025 ms per sentence; swift-transformers 0.23–0.25 ms, plus ~0.57 s to load the 17 MB `tokenizer.json` | [Verified] |
| Mac memory | 14B-4bit: 7.74 GiB weights, 8.22 GiB peak at 512 context; GPU wired limit 20.0 GiB (`iogpu.wired_limit_mb` set to 20480 on this Mac; the stock default was not measured) | [Verified] |

The contention rows used a synthetic memory-bound 4-bit decode sized like the 14B model (8.46 GB,
18.5–19.2 tok/s). Phone-side figures, none measured here:

| Measurement | Figure | Evidence |
|---|---|---|
| iPhone 17 Pro, MLX 4-bit decode | Qwen3-0.6B 164–179 tok/s in bursts, ~99 sustained; Qwen3-1.7B ~65 tok/s | [Third-party] |
| iPhone 17 Pro, drift and thermals | 20–30% between sessions; reports `fair` when plugged in and warm, even idle | [Third-party] |
| iPhone 17 Pro, per-app memory ceiling | ~6.44 GB with both `increased-memory-limit` and `extended-virtual-addressing` | [Third-party] |
| 4 GB iPhones, per-app memory | ~2.0–2.3 GB with or without `increased-memory-limit` (iPhone 12 without it: 2,098 MB; iPhone 13 with it: 2.2–2.3 GB) | [Inferred] from [Third-party] |
| A14 Neural Engine | 16 cores, 11 TOPS, the same rating as M1 | [Third-party] |
| e5-small on the A14 ANE | ~2–4 ms at 128 tokens, ~10–25 ms at 512 | [Inferred] |
| iPhone 12 LLM, embedding and Whisper speed | No published figures | [Needs device] |
| usbmux RTT and throughput | No published figures; the models below assume 1–3 ms | [Needs device] |
| USB link rate | iPhone 17 Pro: up to 10 Gb/s, only with a 10 Gb/s USB 3 cable (Apple spec). iPhone 12 (Lightning): USB 2, 480 Mb/s | [Third-party] / [Inferred] |

## Why speculative decoding does not pay on M1-class GPUs

Speculative decoding assumes that verifying k draft tokens in one forward costs about one decode step.
On the M1 Pro it costs **1.64× (k=1), 2.6× (k=2), 3.95× (k=4) and ~6.4× (k ≥ 5)** [Verified]. In MLX
v0.32.3 (`quantized.cpp`), the affine `qmv_wide` kernel, which reuses weights across rows, is enabled
only on GPU generation 15 or later (M3 and newer); the M1 Pro reports `applegpu_g13s`, so each extra row
re-runs plain `qmv`, and at six or more rows MLX switches to `qmm_splitk` [Verified]. Never use k ≥ 5 on
this class of Mac; the mlx-lm CLI and server default of k = 3 already costs 3.2× a decode step [Verified].

No speculative configuration beat the clean greedy baseline [Verified]. Two remote k=1 runs exceeded
the greedy of a session disturbed by a concurrent download (17.7 vs 17.4, 13.4 vs 12.1 tok/s) [Verified].

A real phone draft would not help either. Modeled with Leviathan et al. (Theorem 3.8) from the Mac's
measured costs and third-party phone draft speeds [Inferred]:

- Sequential remote draft, k=1, 0.6B: 16.6–17.2 tok/s, no better than a local draft.
- Perfectly pipelined remote draft: 18.7 tok/s (0.6B) and 19.4 tok/s (1.7B), at best equal to greedy.
  Real pipelining gains less: pre-drafted tokens count only when the whole round is accepted and the
  phone guessed the bonus token (≈ 0.56 at k=1).
- The link is not the bottleneck: an assumed 2–3 ms RTT is about 2–3% of the 93 ms in-loop k=1
  verification [Verified]. Nor is drafting: the Mac drafts at 4.6 ms/token [Verified], the phone at
  ~6.1 ms in bursts and ~10 ms sustained [Third-party].
- A cost model must use greedy's in-loop 51.5 ms/token, not the isolated 57.5 ms forward, which inflates
  every predicted speedup by ~12%, more than a 10% decision margin. With in-loop inputs, a remote k=1
  draft scores 0.89× greedy.
- Qwen3-32B would gain about +3% (local) and +6–10% (pipelined remote) over 7.7 tok/s: within noise.

It could pay on a GPU-generation-15+ Mac or with an nvfp4/mxfp4 target [Inferred]: on Qwen3-4B, a
five-token verification cost 2.29× (nvfp4) and 2.68× (mxfp4) a decode step, against 3.79× for affine
4-bit [Verified]. For 14B that extrapolates to about +20% (local, k=2) and at most +35% (pipelined
remote) [Inferred]; the quality of those quantizations was not measured.

## Transport findings

- **Use usbmuxd `Connect`, not `iproxy`.** A standard-library Python client (16-byte little-endian
  header plus XML plist) answered every protocol probe from Apple's daemon, including `Connect`'s error
  path [Verified]; its data path was tested only against a fake daemon. The port goes in network byte
  order (8080 is sent as 36895); after `Connect` the socket is a byte stream to the phone, no Mac port.
- **`iproxy` 2.1.1 listens on `*:PORT` (IPv4 and IPv6)**, not 127.0.0.1 as its help says, since libusbmuxd
  commit `303ece5f` (2024-03-26; issue #140 open). The Mac's LAN address reached it from the Mac itself;
  the firewall is on but permits `iproxy`. Pass `-s 127.0.0.1`; `-s localhost` binds only `::1` [Verified].
- **An open `iproxy` port does not mean a device is there**: it binds before any lookup [Verified]. Take
  attach and detach from a usbmuxd `Listen` socket opened before `ListDevices`, keyed by UDID and filtered
  to `ConnectionType == "USB"`; one UDID can also appear over Wi-Fi with another device ID [Verified] in
  source, [Inferred] for Apple's daemon.
- **`iproxy` can lose data silently.** A send blocked for 10 s drops that chunk and keeps the connection
  open; issue #151 (open) reports the same loss when `send` returns `EAGAIN` after `select` said writable.
  Measured: 1 MiB sent, 8 KiB back after 12 s, no error; direct `Connect` moved 256 MiB intact [Verified].
- **`iproxy` children outlive their parent** after a `SIGKILL`, still listening; a simple `sh` watchdog
  then hid both port clashes and the child's death [Verified].
- **Host to device only.** libusbmuxd can open a connection to a device port (`usbmuxd_connect`) but
  cannot accept one from the device; its `Listen` only streams attach/detach events [Verified]. The MLX
  ring backend needs every rank to dial its neighbor, so neither `mlx.distributed` nor mlx-swift's
  `MLXDistributed` runs over USB alone; an IP link needs Local Network permission for the phone's outgoing
  TCP (TN3179) [Verified]. JACCL/RDMA needs Thunderbolt 5, which iPhones lack.
- **Some libimobiledevice 1.4.0 commands trigger pairing**: `idevicepair validate` can show "Trust This
  Computer"; `idevicedevmodectl list` without `-u` handshakes with every device; plain `ideviceinfo` pairs
  (`-s` does not); `idevicepair` exits 1 for every error and `idevicepair list` is empty on macOS [Verified].
  Safe order: `ReadPairRecord` (passive), `idevicepair -u UDID validate`, `idevicedevmodectl -u UDID list`.
- **USB speed.** `system_profiler SPUSBDataType` is empty on macOS 26; use `SPUSBHostDataType` or the
  faster `ioreg -p IOUSB -l -a -r -c IOUSBHostDevice`, matching serial to UDID minus its dash [Verified].
- **The usbmuxd socket is world-writable**, so any local process can reach the phone's port: the node API
  needs a token [Verified]. A replugged, locked phone exposes nothing until unlocked (Apple docs) [Verified].

## iOS node findings

- **Listening needs no Local Network permission** (TN3179), so an `NWListener` on all interfaces is
  silently reachable over Wi-Fi. Bind with `requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback),
  port: 8080)`: in the simulator, `requiredInterfaceType = .loopback` still accepted a connection to the
  Mac's LAN IP, and `acceptLocalOnly` counts Wi-Fi as local [Verified]. That usbmux delivers on the
  device's loopback is [Inferred] from PeerTalk and the usbmuxd README. Add a token; skip Bonjour.
- **Background suspension.** A suspended app's socket accepts but never answers (TN2277), and
  `isIdleTimerDisabled` does not survive a lock or an app switch [Verified]. Cancel the listener and its
  connections on background, rebuild on foreground, and have the Mac send heartbeats.
- **Memory.** `os_proc_available_memory()` is headroom to the process's current limit, not free device
  memory; it returns 0 over the limit, outside an app and in the simulator, where `physicalMemory` is
  the host's RAM [Verified]. The limit is `phys_footprint + limit_bytes_remaining` from `TASK_VM_INFO`
  [Inferred]. Size models against measured headroom; per-app budgets are far below device RAM (see above).
- **Core ML placement.** `MLComputeUnits` has no Neural-Engine-only case, and `.cpuAndNeuralEngine`
  allows CPU fallback [Verified]. `MLComputePlan` (iOS 17.4+) gives the anticipated device per operation:
  for e5 at 128 tokens, the 12-layer encoder (290 ops) runs on the ANE and 28 ops on the CPU, including
  the 250k-row embedding lookup the ANE cannot run; `.all` moves the lookup to the slower GPU path
  [Verified]. The ANE core count is `MLNeuralEngineComputeDevice.totalCoreCount` (iOS 17).
- **Device identity.** `utsname.machine` is `arm64` in the simulator [Verified]; read `ProductType` from
  the Mac and key by model, not chip: A19 Pro has 5 GPU cores in iPhone Air, 6 in 17 Pro [Third-party].
- **MLX** does not run in the iOS Simulator (mlx-swift docs). Open upstream on 2026-09-29 [Verified]:
  mlx-swift #491 (0.32.2 crashes at launch on iOS and macOS before 26.4; fix PRs #493 and #494 unmerged)
  and #485 (an app linking 0.31.6 hangs before `main()` on iOS 27.0). Pin the 0.31.x release AuraLocal resolves (0.31.3) in a
  separate app target and test it on iOS 27 first (#485 was reported against 0.31.6), and cap MLX's buffer cache below the headroom before loading a draft, or jetsam may kill it [Inferred].

## Model choices

**Draft and target: the dense Qwen3 family.** Qwen3 0.6B through 32B share one tokenizer: SHA-256
prefix `486b1a1eacc7` over vocabulary and merges, `61eec40e4844` over added tokens; `vocab_size` is a
padded 151,936, with 151,643 BPE entries plus 26 added tokens, highest real id 151,668 [Verified].
mlx-lm checks only `vocab_size`, which is too weak for a network protocol: compare a tokenizer hash and
reject draft ids ≥ 151,669. Qwen3.5 uses a different vocabulary (248,320), and both Qwen3.5 and Qwen3-Next
(hybrid linear attention) keep caches that cannot be trimmed, so neither can take either role [Verified].

**Target size: 14B, not 32B.** Qwen3-32B-4bit has 17.16 GiB of weights and 256 KiB of KV cache per
token [Verified]: about 18.7 GiB at 4k and 19.7 GiB at 8k context against the 20.0 GiB wired limit, so
only short contexts fit [Inferred]. Qwen3-14B-4bit (7.74 GiB, 160 KiB/token, 8.22 GiB peak at 512
context) [Verified] reaches about 13.5 GiB at 32k; a phone draft frees only 0.3–0.9 GiB [Inferred].

**Embeddings: `intfloat/multilingual-e5-small`** (MIT, 118M parameters, 384 dimensions): 65.09 on the
mean of the seven Spanish-only MMTEB v2 subsets (our aggregation of public MTEB results), within one point
of models 2.6× its size; English-only small models lose 10–18 points [Verified]. Rejected: EmbeddingGemma
(no fp16 activations, which the ANE uses), Qwen3-Embedding-0.6B (1.19 GB), jina-embeddings-v5-text-nano
(non-commercial). No published Core ML conversion was usable as is (wrong pooler, `RangeDim`, `-inf`
masks, missing licenses) [Verified]. Three conversion pitfalls, all measured [Verified]:

1. **The Hugging Face BERT mask becomes `-inf` in fp16** and returns NaN on CPU and GPU; the ANE hides it
   (and `.all` gave cosine 0.79). Use a finite −1e4 mask and validate every conversion on `CPU_ONLY`.
2. **Enumerated shapes with pooling inside the graph fall back to fp32 on the CPU**: 28 ms instead of
   1.9 ms at 128 tokens, 160 ms instead of 18 ms at 512. `RangeDim` keeps only its default shape on the
   ANE. Output per-token states and pool in the caller.
3. **Below an iOS 18 target, coremltools rejects enumerated shapes on more than one input.** Derive the
   attention mask in the graph from the pad id, leaving `input_ids` as the only input.

Apple's ANE layout (`(B, C, 1, S)`, 1×1 convolutions, LayerNorm epsilon 1e-5) was up to 1.9× faster at
512 tokens [Verified]. This recipe **shipped** as AuraLocal's opt-in e5 provider
(`CoreMLTextEmbeddingTool`, `CoreMLEmbeddingProvider`, `scripts/embeddings/convert_e5_coreml.py`; see
[Document RAG]({{ '/guide/rag' | relative_url }}#multilingual-e5-small-opt-in)). A mesh would tokenize
on the Mac and send ids so phones ship no tokenizer; that needs a new ids-in entry point, because the
shipped tool embeds only from text and its bundle check requires `tokenizer.json` and `tokenizer_config.json`.

## How it would integrate into AuraLocal

No mesh code exists. Everything below is unimplemented except module 1 and these shipped AuraLocal
features, which a mesh would reuse (claims about current code were checked on the source tree):

- **Capabilities.** `HardwareProfile.availableMemoryBytes()` reads `os_proc_available_memory()` on iOS
  and reports *unknown* instead of 0, as a node needs; `HardwareAnalyzer` rates fit and context window.
- **Node tasks.** The MLX and GGUF backends, and the [on-device ML tools]({{ '/guide/ml-tools' | relative_url }})
  (`SystemTool` conformers, see `SystemToolRegistry`): OCR, image classification, barcodes, faces,
  language, entities, sentiment, sound classification, NaturalLanguage sentence embeddings
  (`NLEmbeddingTool`), generic Core ML, Create ML text classifiers and the e5 embedding tool.
- **Egress governance.** `ConsentGate`, `PIIRedactor` and `CostLedger` from [Hybrid Inference]({{ '/guide/hybrid' | relative_url }})
  would apply unchanged to a cable-attached Mac as the escalation target [Inferred].

**What is missing:**

1. **A token-level backend API with position-based sync.** `InferenceBackend` takes text
   (`prompt: String` or `messages: [[String: String]]`) and streams `String` tokens; a draft needs
   "generate from these ids, return ids, keep the longest common prefix". The resolved mlx-swift-lm
   3.31.3 exposes `trimPromptCache` and `SpeculativeTokenIterator`, unused by AuraLocal.
2. **A reverse channel for phone-to-Mac escalation.** `OpenAICompatibleProvider` opens connections to a
   `baseURL`, which a phone cannot do over usbmux; the Mac would hold a connection open for its requests.
3. **Minimum iOS version.** AuraLocal requires iOS 18 and C++ interoperability; the evaluated node
   design (not built) targeted iOS 17, so it did not depend on AuraLocal.

**If this work continues, the modules would come in this order. Each one depends on device measurements
that have not been taken, and there is no commitment or timeline:**

| # | Module | Scope |
|---|---|---|
| 1 | e5 embedding provider for AuraDocs | **Shipped**, independent of any mesh ([Document RAG]({{ '/guide/rag' | relative_url }}#multilingual-e5-small-opt-in)) |
| 2 | iOS node | Loopback server with a token; memory from `HardwareProfile.availableMemoryBytes()` (not `HardwareProfile.current()`, which substitutes 60% of physical RAM, the host's RAM in the simulator); perception and embedding roles over `SystemTool` conformers |
| 3 | USB transport for remote escalation | The reverse channel above, with consent and cost receipts; no Wi-Fi and no Local Network prompt [Inferred] |
| 4 | macOS orchestrator in the `aura` CLI | A Swift usbmuxd client; the protocol is small and needs no dependencies |
| 5 | Draft role | Last, and only on a GPU-generation-15+ Mac or with an nvfp4/mxfp4 target |

## Phases, exit criteria and risks

| Phase | Content | Exit criterion |
|---|---|---|
| 0: Proof of concept, no device | Python unit tests (HTTP framing, usbmuxd protocol against a fake daemon, role scenarios); speculative equivalence with tiny random Qwen3 models and injected faults; a fake node on the Mac; an iOS Simulator node | Tests green; the simulator node reports honest capabilities (memory `null`, simulated flag); the pipeline runs with an all-Mac baseline; the speculative loop matches greedy |
| 1: Devices | Day-one checks: the heavy phone's iOS version, a minimal MLX app launched before any draft code, loopback delivery, `Connect` before Trust, real memory limits; then the pre-registered suites below | The rules below. Expected: the draft fails on this Mac, query embeddings fail, bulk indexing and ANE isolation might pass |
| 2: Integration | Only if phase 1 passes: the modules above, in order | Each step with tests and an external consumer that resolves the package |

Pre-registered rules (same-session Mac baseline, ABAB interleaving; runs starting at thermal `serious`
or with > 256 MB swap growth are discarded; simulator latencies are never reported):

- **Query embedding** (500 real queries, half Spanish): pays only if phone p95 ≤ 0.9 × Mac p95.
- **Bulk indexing:** pays if aggregate throughput ≥ 1.2× the Mac alone, or decode improves ≥ 5% at equal
  throughput. Use ≥ 100k passages and include ANE compile time: at 5k passages of 256 tokens the Mac's
  ANE compute alone (3.40 ms each) takes about 17 s, before a phone completes its first compile [Inferred].
- **Speculative decoding** (k ∈ {1, 2, 4}): pays only if the remote draft reaches ≥ 1.05× the best of
  greedy and local draft with non-overlapping 95% confidence intervals. Every output must equal greedy up
  to kernel near-ties, defined as a top-2 logit gap within 2 bf16 ulps (0.25 for logits of magnitude 16–32).
- **Embedding parity:** cosine ≥ 0.999 against the Mac on 64 Spanish and English sentences.

Risks:

- **An open mlx-swift issue on iOS 27.** #485 may stop an MLX app from launching on iOS 27;
  keep a no-MLX node app installed on the same phone, and confirm Xcode can deploy to that iOS version.
- **Manual first run.** Unlock, "Trust This Computer" and Developer Mode (restart plus passcode) cannot
  be automated; the orchestrator must report "needs user" rather than trigger pairing.
- **Memory entitlements** may be unavailable with free provisioning [Needs device]; a 0.6B draft probably fits without them [Inferred].
- **Custom HTTP framing** on both ends: strict RFC 9112 subset, tests against curl, fuzzing.
- **Numeric near-ties.** Verification uses other kernels than a one-row decode: equal to greedy up to near-ties.
- **Noise.** The shared Mac ANE (p90 up to ~65 ms) and phone thermal drift (20–30% [Third-party]) can
  flip conclusions between runs: report p90/p99 and interleave.
- **Scope.** VAD, text classification and Whisper/Vision stay out until the core numbers exist.

## Reference projects

Checked on 2026-09-29. None offers per-node role assignment plus a capability endpoint over USB.

| Project | Relevant facts | Taken | Not taken |
|---|---|---|---|
| [exo](https://github.com/exo-explore/exo) (Apache-2.0) | Rewritten in Python and Rust on zenoh; no iOS support on `main`; places models by memory only (FLOPS is a TODO); no speculative decoding on `main` | Node identity check (launch the app with a node id and verify the echo); largest-remainder split for sharding bulk indexing | Multicast discovery (does not cross usbmux); its README's latency-aware placement, which the code does not implement |
| [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT) | React Native with llama.rn; Metal only from iOS 18; reads `os_proc_available_memory`, `phys_footprint` and `MTLDevice` | Fit estimator: ceiling = max(largest successful load, observed headroom), cold start min(0.6 × RAM, RAM − 1.2 GB); report `phys_footprint`, not RSS; `devicectl`-driven device benchmarks | Its device-rules table (no license) and leaderboard (aggregate ratings only) |
| [apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench) (MIT) | Decode, prefill, TTFT and `phys_footprint`; admits only runs that start thermal-nominal; iPhone 17 Pro figures, none for A14 or M1 Pro | Methodology and the iPhone 17 Pro figures used above | Core AI rows (iOS 27 only) |
| llama.cpp RPC | Embeddable `rpc-server`; the Mac-to-phone direction fits usbmux; the handshake checks only major/minor while op-list changes bump only the patch (a `static_assert` on `GGML_OP_COUNT`), so build client and server from the same commit; automatic layer split uses Metal's working set, not the jetsam limit | A stock alternative to A/B against a custom draft protocol (not run) | Anything relying on its automatic memory split |
| MLX distributed (ring), `MLXDistributed` | Every rank dials its neighbor | Nothing | Impossible over USB alone |
| [cake](https://github.com/evilsocket/cake) | Rust/Candle with an iOS worker app; shards layers by VRAM; manual `host:port` topology | The manual-topology idea | Its "FAIR" license restricts commercial use; layer sharding, not role offload |
