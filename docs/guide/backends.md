---
layout: docs
title: Backends
parent: Guide
nav_order: 2
description: "How AuraLocal's dual-backend architecture selects between MLX and llama.cpp/GGUF for each model and device."
---

# Backends
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AuraLocal uses two inference engines internally, selected automatically by `BackendRouter`:

| Backend | Engine | Formats | Platforms | Peak RAM (estimate) |
|---------|--------|---------|-----------|---------|
| `MLXBackend` | mlx-swift | `.mlx` | iOS devices + macOS (not the Simulator) | ~1 – 20 GB |
| `LlamaCppBackend` | llama.cpp | `.gguf` | iOS + macOS, when the model fits in memory | ~1 – 40 GB |
| `LayerStreamingBackend` | llama.cpp + mmap | `.gguf` | iOS + macOS, when it does not fit (in practice iPhone and iPad) | ~1 GB for a 7B–8B |

All three implement the same `InferenceBackend` protocol — your code looks identical regardless of which backend is running.

{: .warning }
> The API is identical, the behaviour is not. **Tools** (`ModelManager.shared.load(_:tools:onProgress:)`, `AuraProfile.tools`) reach only the GGUF backends; an MLX model silently ignores them. The llama.cpp backends are text-only (an `image:` argument is ignored) and flatten a multi-turn history into one prompt (roles dropped, only the last system turn kept); MLX passes the messages with their roles. A `constraint:` on `chat`/`stream` is enforced only by `LlamaCppBackend`; MLX, layer-streaming and remote throw `GenerationConstraintError.unsupportedBackend` (see [Constrained generation](#constrained-generation-gguf-only)). `AuraSession` enforces `AuraProfile.outputSchema` the same way on every backend, by validating each reply and re-prompting, not by constrained decoding; other APIs ignore it (see [Structured output]({{ '/api/core' | relative_url }}#structured-output)). Of `AuraProfile.sampling` only `maxTokens` reaches generation.

`BackendKind` also declares `.remote` and `.hybrid`, but no AuraCore API returns them; remote inference goes through `HybridEscalator` (see [Hybrid inference]({{ '/guide/hybrid' | relative_url }})).

---

## Automatic Selection

`BackendRouter` selects a backend when `AuraEngine` is initialized:

```swift
// Internal to AuraCore, shown for illustration: AuraEngine calls it on init.
// From app code, call BackendRouter.recommendedBackend(for:) instead.
let backend = BackendRouter.selectBackend(for: model, temperature: temperature, tools: tools)
```

The routing logic:

```
model.format == .mlx
    → MLXBackend  (device or Mac; `tools` are ignored)
    → UnavailableBackend on the iOS Simulator
      (load throws "MLX models can't run on the iOS Simulator")

model.format == .gguf
    fitLevel == .excellent / .good / .marginal
        → LlamaCppBackend  (full model loaded; Metal on device, CPU in the Simulator)
    fitLevel == .streamingRequired
        → LayerStreamingBackend  (mmap; Metal on device, CPU in the Simulator)
    fitLevel == .tooLarge
        → LlamaCppBackend  (ModelManager.load refuses it first with AuraError.modelTooLarge)
```

To check which backend a model will use on the current device:

```swift
import AuraCore

let kind = BackendRouter.recommendedBackend(for: .llama3_1_8b_gguf)
// → .llamaCpp        on Mac with 16 GB RAM
// → .layerStreaming  on a 6 GB iPhone (~3 GB app budget with the entitlement)
```

`recommendedBackend` answers only *which engine*, not *whether it fits*: a `.tooLarge` GGUF still reports `.llamaCpp`. Check `HardwareAnalyzer.assess(model).fitLevel.isRunnable` first.

---

## MLX Backend

Used for all `.mlx` format models, from Granite Docling 258M up to Ornith 1.5 35B A3B in the catalog. The 12B-and-larger entries exceed every iPhone budget preset and fit only on a Mac; check `HardwareAnalyzer.assess(_:)`.

**Characteristics:**
- Runs entirely on the Apple Silicon GPU via Metal
- Model weights loaded into unified memory as MLX arrays
- Supports text generation and vision (multimodal) models
- GPU cache scaled by model weight size (~1/12 of the weights), clamped to 128 MB – 512 MB
- 20–45 tokens/second for ≤4B models on iPhone 15 Pro / M-series Mac

**When it's used:** Any model case without `_gguf` suffix — Qwen3, Llama 3.2, Gemma 3, Phi-3.5, SmolVLM, FastVLM, Granite Docling.

---

## LlamaCpp Backend

Used for `.gguf` models when the full file fits in available RAM.

**Characteristics:**
- Uses `LocalLLMClient` → llama.cpp under the hood
- Full Metal GPU offload (all layers, the llama.cpp default) on a Mac or iOS device; CPU-only in the Simulator
- Context window: `HardwareAnalyzer.recommendedContextWindow(for:)` derives it from the model's GQA KV cost and the memory left after the weights (0.4 GB reserve), rounded down to a multiple of 512, never below 1024 and never above 32768 on macOS / 8192 on iOS, and capped at the entry's `maxContextLength`. Entries without KV metadata (`kvHeads == 0`) fall back to 8192 (Mac with ≥32 GB), 4096 (smaller Macs), 2048 (iOS with ≥8 GB) or 1024.
- Up to 8 threads on macOS, up to 4 on iOS
- 8–20 tokens/second on M-series Mac depending on model size

**When it's used:** GGUF models that fit the app's memory budget: 7B–70B on a Mac depending on RAM, and 1.2B–3B models on a 6 GB iPhone with the entitlement.

```swift
// On a Mac with 32 GB RAM, Llama 3.1 8B uses this backend
let llm = try await ModelManager.shared.load(.llama3_1_8b_gguf)
```

---

## Constrained Generation (GGUF only)

`chat` and `stream` take an optional `constraint:` that llama.cpp's sampler enforces token by token:

```swift
let llm = try await ModelManager.shared.load(.llama3_1_8b_gguf)

let json = try await llm.chat(
    "Return a JSON object with name and age for: Ana, 31.",
    maxTokens: 128,
    constraint: .json                                    // a JSON object at the root
)
let verdict = try await llm.chat(
    "Is 4 even? Answer yes or no.",
    maxTokens: 8,
    constraint: .grammar(gbnf: #"root ::= "yes" | "no""#)  // root defaults to "root"
)
```

| Backend | `constraint:` |
|---------|---------------|
| `LlamaCppBackend` | Enforced |
| `MLXBackend` | Throws `GenerationConstraintError.unsupportedBackend(.mlx)` |
| `LayerStreamingBackend` | Throws `.unsupportedBackend(.layerStreaming)` |
| `RemoteBackend` | Throws `.unsupportedBackend(.remote)` |

`.json` uses the `json.gbnf` grammar bundled with LocalLLMClient, whose root is an object. A `.grammar` that is
blank, or that llama.cpp rejects (syntax error, undefined rule, no rule named `root`, left recursion), throws
`GenerationConstraintError.invalidGrammar(root:)`.

### Why each call gets its own client

The pinned LocalLLMClient (0.5.0) builds one sampler chain per client and never resets it. After the first
constrained answer the grammar sits in its end state, so every later generation on that client can only emit
end-of-generation or the grammar's trailing whitespace: the replies come back empty (or whitespace-only for
`.json`). Penalty history leaks across calls the same way. Until a release
carries the reset (prepared as `contrib/localllmclient/0003`), `LlamaCppBackend` runs each constrained call on a
dedicated, short-lived client built on the same GGUF file:

- **Context:** UTF-8 bytes of the prompt + `maxTokens` + 256 for the chat template, rounded up to a multiple of
  256 (llama.cpp pads to that anyway) and capped at `HardwareAnalyzer.recommendedContextWindow(for:)`. A token
  never covers less than one byte, so below the cap the prompt always fits; the price is over-reserving, since
  English text averages several bytes per token. When the cap applies, the window can fill before the answer
  ends, and generation then stops without an error, cutting the output as `maxTokens` does.
- **Sampling:** the main session's temperature, top-p and thread count. No tools.
- **Scheduling:** calls on one backend, constrained or not, take turns in arrival order; the dedicated client is
  freed before the call returns, and the main session is never touched. A call cancelled before its turn throws
  `CancellationError` without decoding. The turns do **not** cover decodes LocalLLMClient keeps running in its own
  tasks after a main-session call has returned: a call cancelled during its prefill, or a call with tools cut
  short by `maxTokens`, keeps decoding, and the next call overlaps it (a constrained one on its own context, an
  unconstrained one on the same context, where it has crashed or hung).

Measured on an M1 Pro (32 GB, macOS 26.7.1) with Qwen2.5-0.5B-Instruct Q4_K_M (491 MB), a ~150-byte prompt,
`maxTokens: 128` and ~26 generated tokens (768-token dedicated context; the main session ran at 8192). Other builds
were loading the machine (1-minute load average 146 at the start and 182 at the end of the final run; not logged
for the others), so read the latencies as upper bounds.

| | Measured |
|---|---|
| Build the dedicated client, page cache warm, main session loaded | median 238 ms (220–281, n=7); 450 ms under heavier load |
| First build in a fresh process, GGUF not in the page cache | 549–1133 ms (n=3) |
| First build in a fresh process, GGUF in the page cache | 260–499 ms (n=3) |
| `chat(constraint: .json)` end to end | median 1.56 s (2.78 s under heavier load), against 0.25 s (0.31 s) for the same unconstrained `chat` |
| Grammar sampling alone | ~66–74 ms per generated token on top of unconstrained decoding |
| Dedicated client after build (no generation) | +10 MB at the 768-token window; +94 to +97 MB had it used the 8192-token main window |
| Peak per constrained call, over the loaded session | first call +145 to +157 MB; later calls +82 to +152 MB |
| Process footprint after 20 constrained calls | +94 to +141 MB over the loaded session; stops growing after ~13 calls (both runs) |

Most of a constrained call is grammar sampling, not the client build: the pinned client runs the grammar over the
whole vocabulary (152k tokens here) twice per token, and a reused client pays the same on its one working call.
The footprint that stays after the calls is freed memory the allocator keeps for reuse (`vmmap` reports it as empty
`MALLOC_LARGE`/`MALLOC_SMALL` regions; `malloc_zone_pressure_relief` returned none of it). It stops growing, but on
iOS it counts toward the jetsam limit, and so does the per-call peak.

### Limits

- `maxTokens` still cuts the output, and a cut JSON object does not parse. Budget for the whole object. Reaching
  the context cap cuts it the same way, also without an error.
- A cancelled constrained call throws `CancellationError`; it never returns the partial output.
- The prompt is flattened the same way as an unconstrained call (`System: …` / `User: …` in one user turn).
- Only `chat(_:systemPrompt:maxTokens:constraint:)` and `stream(_:systemPrompt:maxTokens:constraint:)` take a
  constraint; the history, vision and `AuraSession` paths do not.

---

## Layer-Streaming Backend

Used for `.gguf` models that are too large for monolithic loading but pass the streaming gate (`.streamingRequired`), which in practice means iPhone and iPad.

**How it works:**

Traditional LLM loading reads the entire model into RAM before inference. For a 4.7 GB Llama 3.1 8B file on a 6 GB iPhone (~3 GB for the app with the increased-memory-limit entitlement, ~1.5 GB without, where 8B is refused), full loading is impossible without crashing.

Layer-streaming uses **mmap** — the OS maps the file into virtual address space but only loads pages that are actually accessed. Inference processes one transformer layer at a time:

```
Disk (GGUF file)
  → OS page cache (mmap, demand-paged)
    → Current layer weights (~130 MB)
    → Next layer prefetch (~130 MB)
    → Embedding table (~130 MB)
    → KV cache (~128 MB at 1024 ctx, ~256 MB at the 2048 default, GQA)
    → Activations + overhead (~80 MB)
─────────────────────────────────────
Total peak footprint: small and roughly constant, whatever the model's size
```

**Characteristics:**
- Metal GPU offload of all layers on a device (the llama.cpp default); CPU-only in the Simulator. Weights are mmapped (`use_mmap = true`), so resident memory stays low.
- Context window: 2048 tokens on iOS (4096 on macOS), chosen at load and cut to 1024 when 512 MB or less is free above a 250 MB margin, and to 512 when 256 MB or less is free or the OS reports unknown. Under pressure (less than 250 MB free, or unknown) a reply is also capped at 256 tokens.
- 2–6 tokens/second on iPhone 15 Pro
- `MemoryBudgetManager` checks `os_proc_available_memory()` every 32 tokens and stops early if memory becomes critical

**When it's used:** GGUF models on iPhones and iPads where `HardwareAnalyzer` returns `.streamingRequired`.

### Memory Budget (7B Q4_K_M on 6 GB iPhone)

| Component | RAM Usage |
|-----------|-----------|
| iOS system | ~2.5 GB |
| App budget available | ~1.5 GB |
| Current layer (Q4) | ~130 MB |
| Next layer prefetch | ~130 MB |
| Embedding table | ~130 MB |
| KV cache (1024–2048 ctx, GQA; 8 KV heads) | ~128–256 MB |
| Activations + overhead | ~80 MB |
| Safety margin | ~250 MB |
| **Total app peak** | **~0.85 GB at 1024 ctx, ~1 GB at 2048 (incl. 250 MB margin)** |

### GQA-Aware KV Cache

Modern models use Grouped-Query Attention (GQA), which dramatically reduces KV cache size compared to naive full-attention estimates:

| Model | Q heads | KV heads | KV cache (2048 ctx, FP16) |
|-------|---------|---------|--------------------------|
| Llama 3.1 8B | 32 | 8 | ~256 MB |
| Qwen 2.5 7B | 28 | 4 | ~112 MB |
| Mistral 7B | 32 | 8 | ~256 MB |
| Gemma 2 9B | 16 | 8 | ~672 MB |

`model.estimatedKVCacheGB(contextLength:)` applies this formula (`4 × layers × KV heads × head dim × context` bytes) to the catalog's metadata, not a naive full-attention estimate. The catalog's Gemma 2 9B entry currently declares 4 KV heads, so it returns about half (~336 MB) for that model.

---

## macOS vs iOS Differences

| | macOS | iOS |
|---|---|---|
| GGUF backend | By fit: `LlamaCppBackend` if it fits, else `LayerStreamingBackend` | Same rule. At ~3 GB (6 GB iPhone, entitled) 1.2B–3B GGUFs load fully and larger ones stream |
| GPU layers | All (llama.cpp default) | All on device; 0 in the Simulator |
| Context window | Full load: memory-derived, 1024–32768; streaming: 4096, cut to 1024/512 under pressure | Full load: memory-derived, 1024–8192; streaming: 2048, cut to 1024/512 |
| Threads | ≤8 (full) / ≤6 (streaming) | ≤4 (full) / ≤3 (streaming) |
| Max viable model | 70B (full load at a 64 GB Mac's ~48 GB budget estimate) | 32B (streaming at the ~6.4 GB `iphone-17-pro` estimate); 14B at ~3 GB |
| Background inference | Continues | GGUF: paused by LocalLLMClient while the app is inactive; MLX: continues |

---

## Background Lifecycle (iOS)

{: .warning }
> `BackgroundLifecycle` does not pause inference. `BackgroundLifecycle.shared` observes `didEnterBackground` / `willEnterForeground` and sets `isPaused`, but nothing in AuraCore reads `isPaused`, and its observers exist only after your code first touches `BackgroundLifecycle.shared`.

What actually happens when the app leaves the foreground:

- **GGUF:** LocalLLMClient itself pauses llama.cpp between tokens from `willResignActive` until `didBecomeActive`.
- **MLX:** generation keeps running. Cancel the task that consumes the stream when the app backgrounds.

To free memory on backgrounding:

```swift
// Registers the observers; on backgrounding, evicts every model
// except the most recently used (ModelManager.shared.evictAllButMostRecent()).
BackgroundLifecycle.shared.aggressiveMemorySaving = true
```

This is a no-op on macOS.
