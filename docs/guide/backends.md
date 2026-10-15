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
After each generation, `AuraLocal.lastGenerationStats` reports token counts and timings: MLX passes on its runtime
counters, the llama.cpp backends time the stream and estimate the prompt from its rendered text (see
[Generation Stats]({{ '/api/core' | relative_url }}#generation-stats)).

{: .warning }
> The API is identical, the behaviour is not. **Tools** (`ModelManager.shared.load(_:tools:onProgress:)`, `AuraProfile.tools`) reach only the GGUF backends; an MLX model silently ignores them. The llama.cpp backends are text-only (an `image:` argument is ignored). Both families pass a multi-turn history with its roles; the llama.cpp backends render it with the GGUF's own chat template (see [Multi-turn conversations](#multi-turn-conversations-gguf)). A `constraint:` on `chat`/`stream` is enforced only by `LlamaCppBackend`; MLX, layer-streaming and remote throw `GenerationConstraintError.unsupportedBackend` (see [Constrained generation](#constrained-generation-gguf-only)). `AuraSession` enforces `AuraProfile.outputSchema` the same way on every backend, by validating each reply and re-prompting, not by constrained decoding; other APIs ignore it (see [Structured output]({{ '/api/core' | relative_url }}#structured-output)). Of `AuraProfile.sampling` only `maxTokens` reaches generation.

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
- KV cache bounded by `HardwareAnalyzer.recommendedContextWindow(for:)` (`maxKVSize`). Past it MLX overwrites the oldest tokens without an error, which `AuraLocal.lastContextOverflow` reports. Qwen 3.5 text models, Gemma 3 and 4, and LFM2 build their own caches and ignore the bound (see [Memory]({{ '/guide/memory' | relative_url }}#mlx-kv-window))
- 20–45 tokens/second for ≤4B models on iPhone 15 Pro / M-series Mac

**When it's used:** Any model case without `_gguf` suffix — Qwen3, Llama 3.2, Gemma 3, Phi-3.5, SmolVLM, FastVLM, Granite Docling.

---

## LlamaCpp Backend

Used for `.gguf` models when the full file fits in available RAM.

**Characteristics:**
- Uses `LocalLLMClient` → llama.cpp under the hood
- Full Metal GPU offload (all layers, the llama.cpp default) on a Mac or iOS device; CPU-only in the Simulator
- Context window: `HardwareAnalyzer.recommendedContextWindow(for:)` derives it from the model's GQA KV cost (per layer type when the entry has `kvLayers`) and the memory left after the weights (0.4 GB reserve): the largest multiple of 512 that fits, never below 1024 and never above 32768 on macOS / 8192 on iOS, and capped at the entry's `maxContextLength`. Entries without KV metadata (`kvHeads == 0`) fall back to 8192 (Mac with ≥32 GB), 4096 (smaller Macs), 2048 (iOS with ≥8 GB) or 1024.
- Up to 8 threads on macOS, up to 4 on iOS
- 8–20 tokens/second on M-series Mac depending on model size

**When it's used:** GGUF models that fit the app's memory budget: 7B–70B on a Mac depending on RAM, and 1.2B–3B models on a 6 GB iPhone with the entitlement.

```swift
// On a Mac with 32 GB RAM, Llama 3.1 8B uses this backend
let llm = try await ModelManager.shared.load(.llama3_1_8b_gguf)
```

### Multi-turn conversations (GGUF)

Both llama.cpp backends treat every call (`chat`, `stream`, `chat(_:in:)`, `stream(_:in:)`, `generate(messages:)`)
as one self-contained turn: it starts from an empty KV cache and an empty decode batch, and the model stays loaded.
The pinned LocalLLMClient (0.5.0) never cleared the cache when a new prompt did not extend the cached one, so the
cache grew every turn until the context overflowed (around turn 8 at 1024 tokens) and later turns failed or
aborted; the fix it needs upstream is prepared as `contrib/localllmclient/0004`.

- **Roles:** the history is rendered with the GGUF's `tokenizer.chat_template` and special tokens (`bos_token`,
  `eos_token`, …), read from the file the way llama.cpp resolves them, with the same Jinja engine and message shapes
  LocalLLMClient's `.chat` input uses. A single `chat(_:systemPrompt:)` call sends the system prompt as a system
  turn, not as text inside the user turn. A GGUF without a template gets a `System:` / `User:` / `Assistant:`
  transcript.
- **What every template accepts:** all system turns (a `summarizeAndPrune` summary included) merge, in order, into
  one leading system turn; assistant turns before the first user turn join the system text; consecutive user or
  assistant turns merge; empty assistant turns are dropped. A template that rejects a system role (Gemma 2) or
  renders part of the conversation empty gets plain-string content, then the system text at the start of the first
  user turn.
- **Overflow:** when the rendered history does not fit the context, its oldest exchanges are dropped until it does;
  the system turn and the newest user turn always stay. LocalLLMClient exposes no tokenizer, so the first overflow
  costs a refused prefill; from it the backend learns a tokens-per-byte ratio and from then on trims before
  prefilling, keeping room for the reply (`maxTokens`, at most a quarter of the context). Trimming is not reported
  in `lastContextOverflow`. A system turn plus newest user turn that alone exceed the context throw
  `LLMError.failedToDecode` (`context size exceeded`).
- **Cost:** the whole history is prefilled on every turn, so prefill time grows linearly with it: 1.28 s for a
  3038-token history with Qwen2.5-0.5B Q4_K_M on an M1 Pro (about 0.42 ms per token, measured with other builds
  running); larger models cost proportionally more. The previous path paid the same, since its prompt cache never
  matched.
- **Cut replies:** a reply ended by `maxTokens`, cancellation or backgrounding leaves the next turn unaffected. The
  decode runs in a task of its own, so the call returns at once; the next turn waits until that decode has stopped
  (at most one token, or the end of a prefill) and flushes the token it left queued.
- **With tools** a turn runs on a LocalLLMClient `LlamaClient` (its tool-call parser), with the same rendering, the
  call → execute → resume loop `LLMSession` runs, and the same clean start. That client decodes in a task nobody can
  cancel, so a reply cut short returns at once but the **next** turn waits until that decode reaches the end of
  generation or fills the context (seconds on a 0.5B model). A tool that throws fails the turn with its own error;
  arguments that do not decode fail it with LocalLLMClient's `ToolError.argumentDecodingFailed`.
- **Not reset:** the sampler's repetition-penalty window and random state carry over between turns (no public reset
  in LocalLLMClient 0.5.0; see `contrib/localllmclient/0003`). At temperature 0 they have no effect.

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

- **Context:** UTF-8 bytes of the rendered prompt + `maxTokens` + 256, rounded up to a multiple of
  256 (llama.cpp pads to that anyway) and capped at `HardwareAnalyzer.recommendedContextWindow(for:)`. A token
  never covers less than one byte, so below the cap the prompt always fits; the price is over-reserving, since
  English text averages several bytes per token. When the cap applies, the window can fill before the answer
  ends, and generation then stops without an error, cutting the output as `maxTokens` does.
- **Sampling:** the loaded model's temperature, top-p and thread count. No tools.
- **Scheduling:** calls on one backend, constrained or not, take turns in arrival order; the dedicated client is
  freed before the call returns, and the loaded model's context is never touched. A call cancelled before its turn
  throws `CancellationError` without decoding. A chat turn cut short can still be finishing its last token (with
  tools, decoding to the end of generation) when a constrained call starts; the constrained call overlaps it on
  its own context.

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
- The prompt is rendered with the model's chat template like an unconstrained call, the system prompt in its own
  turn, but without tool instructions.
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
| Background inference | Continues | Cancelled on entering the background; the call throws `GenerationInterrupted` with the text so far |

---

## Background Lifecycle (iOS)

`BackgroundLifecycle.shared` starts observing when a model loads. On `didEnterBackground` it cancels every
running generation and, until `willEnterForeground`, refuses new ones; both throw `GenerationInterrupted`
(`reason == .appBackgrounded`, `partialText` = the text so far).

- **MLX:** stops at the next token.
- **GGUF** (full load and streaming): the call returns at once. LocalLLMClient pauses llama.cpp between
  tokens from `willResignActive`; the paused decode runs one more token after `didBecomeActive` and then
  stops. The next turn on that model waits for it; a constrained call, on its own context, can overlap it.

To free memory on backgrounding as well:

```swift
// Evicts every model except the most recently used (ModelManager.shared.evictAllButMostRecent()).
BackgroundLifecycle.shared.aggressiveMemorySaving = true
```

Nothing is cancelled on macOS. Details and the opt-in `DeviceConditions` guard (heat, Low Power Mode,
battery): [Memory]({{ '/guide/memory' | relative_url }}#background-lifecycle-ios).
