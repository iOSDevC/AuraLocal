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
counters, the llama.cpp backends time the stream and estimate the prompt from its rendered text, and with tools the
output as well (see
[Generation Stats]({{ '/api/core' | relative_url }}#generation-stats)).

{: .warning }
> The API is identical, the behaviour is not. **Tools** (`ModelManager.shared.load(_:tools:temperature:topP:onProgress:)`, `AuraProfile.tools`) run on the GGUF backends and on MLX text models; with an MLX vision or document model `ModelManager.load` and `AuraSession` throw `ModelConfigurationError.toolsUnsupported(.mlx)` before loading anything. The llama.cpp backends are text-only (an `image:` argument is ignored). Both families pass a multi-turn history with its roles; the llama.cpp backends render it with the GGUF's own chat template (see [Multi-turn conversations](#multi-turn-conversations-gguf)). A `constraint:` on `chat`/`stream` is enforced only by `LlamaCppBackend`; MLX, layer-streaming and remote throw `GenerationConstraintError.unsupportedBackend` (see [Constrained generation](#constrained-generation-gguf-only)). `AuraSession` enforces `AuraProfile.outputSchema` on every backend by validating each reply and re-prompting; on `LlamaCppBackend`, for a profile without tools, it also decodes each reply under a grammar built from the schema (see [Schema grammars](#schema-grammars)); other APIs ignore it (see [Structured output]({{ '/api/core' | relative_url }}#structured-output)). `AuraSession` loads the model with `AuraProfile.sampling`'s `temperature` and `topP` and uses its `maxTokens` per turn; `topK` and `contextOverride` are not applied (see [Sampling](#sampling-temperature-and-top-p)).

`BackendKind` also declares `.remote` and `.hybrid`, but no AuraCore API returns them; remote inference goes through `HybridEscalator` (see [Hybrid inference]({{ '/guide/hybrid' | relative_url }})).

---

## Automatic Selection

`BackendRouter` selects a backend when `AuraEngine` is initialized:

```swift
// Internal to AuraCore, shown for illustration: AuraEngine calls it on init.
// From app code, call BackendRouter.recommendedBackend(for:) instead.
let backend = BackendRouter.selectBackend(for: model, sampling: sampling, tools: tools)
```

The routing logic:

```
model.format == .mlx
    → MLXBackend  (device or Mac; with tools, text models only)
    → UnavailableBackend for a vision or document model with tools
      (load throws ModelConfigurationError.toolsUnsupported(.mlx); the public paths throw it first)
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

## Sampling (temperature and top-p)

`ModelManager.shared.load(_:tools:temperature:topP:onProgress:)` and the factories
(`AuraLocal.text(_:temperature:topP:onProgress:)`, `.vision`, `.specialized`) take a temperature and a top-p. They
hold for the instance's lifetime; there is no per-call setting.

```swift
// Repeatable extraction: temperature 0 is greedy on both backends.
let llm = try await ModelManager.shared.load(.llama3_1_8b_gguf, temperature: 0)
let total = try await llm.chat("Coffee 3.50, bagel 2.25. Reply with the total only.", maxTokens: 16)
```

| | `nil` (the default) | Temperature 0 |
|---|---|---|
| GGUF (`LlamaCppBackend`, `LayerStreamingBackend`) | temperature 0.7, top-p 0.95 | llama.cpp's temperature sampler keeps only the most likely token |
| MLX | temperature 0.7 (text), 0.1 (vision), 0 (OCR / document); top-p off (1) | `GenerateParameters.sampler()` returns `ArgMaxSampler` |

- **Why per instance:** llama.cpp builds its sampler chain when the context is created, and the pinned
  LocalLLMClient (0.5.0) keeps the chain private, so a loaded GGUF cannot change its temperature without a new
  context, which reloads the model. MLX takes its parameters per call, but an instance keeps one setting on both
  backends so a shared instance never changes under another caller.
- **Cache:** `ModelManager` caches one instance per model and setting, after filling in the defaults: `load(model)`
  and `load(model, temperature: 0.7, topP: 0.95)` share a GGUF instance. Each other setting is a separate instance
  that loads the model again, counts against `memoryBudget` and is evicted like any other.
  `evict(_:)`, `isLoaded(_:)` and `cancelLoad(_:)` act on every setting of the model.
- **Validation:** a temperature below 0, infinite or NaN throws `ModelConfigurationError.invalidTemperature`; a
  top-p outside `0 < topP <= 1` throws `.invalidTopP`. Both are thrown before anything is downloaded or loaded.
- **Constrained calls** run on a dedicated client with the instance's temperature and top-p.
- **At temperature 0** on GGUF, the repetition penalty (1.1 over the last 64 tokens) has no effect, since
  LocalLLMClient applies it after the temperature sampler. Greedy decoding can repeat itself; budget `maxTokens`.
- `AuraSession` loads its model with `AuraProfile.sampling.temperature` and `.topP` (`SamplingParams.default` is
  0.7 / 0.95, `.precise` 0.2 / 0.95). On MLX that turns top-p on, unlike a plain `load`.

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
- Text models run tool calls with the same loop as the GGUF backends; vision and document models with tools throw
  `ModelConfigurationError.toolsUnsupported(.mlx)` from `ModelManager.load` and `AuraSession` (see
  [Prompt cache and tool rounds](#prompt-cache-and-tool-rounds-mlx))
- A text model keeps the KV cache of its last generation and prefills only the part of the next prompt it does
  not hold (see [Prompt cache and tool rounds](#prompt-cache-and-tool-rounds-mlx))

**When it's used:** Any model case without `_gguf` suffix — Qwen3, Llama 3.2, Gemma 3, Phi-3.5, SmolVLM, FastVLM, Granite Docling.

### Prompt cache and tool rounds (MLX)

- **Prompt cache:** a text model's generation keeps its KV cache and the exact tokens it holds: the prompt and every
  sampled token, the stop token included. The next prompt is compared with them token by token; the cache is
  trimmed to the longest prefix the two share, leaving at least one prompt token, and only the rest is prefilled.
  A prompt that shares nothing, a cache a layer cannot trim (the recurrent layers of Qwen 3.5, a sliding-window
  layer of Gemma 3 once the conversation outgrows its window) and a prompt at or past `maxKVSize` get a new cache
  and a full prefill. Vision and document models, and a call with an image, always prefill in full. There is no
  public switch.
- **What it saves:** over an 8-turn greedy conversation with Qwen3 0.6B 4-bit on an M1 Pro (two `swift test`
  runs), 7 prompts reused the cache, 86% of prompt tokens were not prefilled again, and prefill took 0.46–0.61 s in
  all against 1.44–1.61 s without the cache (the 8th turn, 494 tokens: 53 ms against 321 ms in the first run). A
  template that rewrites earlier assistant turns stops the reuse where it rewrites them: Qwen3 drops the `<think>`
  block of earlier replies, so each turn prefills the previous reply and the new user turn again.
- **Same tokens, not always the same text:** a hit leaves the cache as a full prefill split at the same token
  would, and the reply matches that bit for bit. fp16 results still depend on how a prompt is split into batches,
  so a greedy reply can drift from an unsplit full prefill after a few dozen tokens: in that conversation 4 of 8
  replies differed, as many as changing only the prefill chunk size, with no reuse, changed.
- **Memory and stats:** the cache stays allocated between generations, as large as the conversation's KV;
  `unload()`, and so an eviction, frees it. On a hit `lastGenerationStats.promptTime` covers only the tokens
  prefilled while `promptTokens` counts the whole prompt, so `prefillTokensPerSecond` is not a prefill speed. To
  time a full prefill, evict the model (`ModelManager.shared.evict(_:)`) and load it again.
- **With tools:** a text model loaded with tools runs the loop described for GGUF below and in
  [Tool calling]({{ '/api/core' | relative_url }}#tool-calling-gguf-and-mlx): `toolLoop` rounds (default 1),
  `ToolEvent`, `lastToolTurn`, `{"error": …}` results, sorted-keys JSON output, and one `maxTokens` for every reply
  of the turn, in which every generated token counts, call text included.
  - **Parsing:** mlx-swift-lm's parser for the format it infers from the model's `config.json` (`model_type`), or
    JSON in `<tool_call>` tags when it infers none. A model that writes another format gets no calls and its call
    text streams as text; a call whose text does not parse is dropped, neither run nor shown. Every call in a
    reply runs, the same tool twice included.
  - **What the model sees:** the tools go to the chat template as `tools`, with no instructions of AuraLocal's, so
    a model whose template ignores `tools` never learns about them. A round goes back as an assistant turn with
    `tool_calls` and a `tool` turn per result (with `tool_call_id` and `name`); when the template throws on that,
    every round of the turn goes as one assistant turn of `<tool_call>` and `<tool_response>` text. Each reply
    renders the whole conversation, and the prompt cache prefills only what it does not hold: in the live test the
    reply after a round reused 149 of its 200 prompt tokens. Checked on Qwen3 0.6B 4-bit only.
  - **Cancelling** while a tool runs cancels its task and throws `CancellationError`; cancelling during a reply
    cuts it, and the turn returns the text so far. Nothing keeps decoding afterwards.
- **Shared weights:** with a cached instance of the same model, `ModelManager.load(_:tools:)` runs the tool instance
  on its weights instead of loading a second copy. The two generate one at a time, and the weights stay in memory
  until both are released.

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
  prefilling, keeping room for the reply (`maxTokens`, at most a quarter of the context). Every load of the model
  starts without that ratio, an eviction and reload on iOS included, so until its first overflow a history that
  fits but leaves less room than `maxTokens` gets a reply cut at the end of the context, with no error. Resuming a
  saved conversation is the usual case: `chat(_:in:)` sends up to `maxContextTokens` (3072 by default), more than
  the 1024 or 2048 tokens iOS falls back to. On a small context pass a `maxContextTokens` that leaves room for
  `maxTokens`, with a margin, since it is an estimate (about 4 characters per token). Trimming is not reported in
  `lastContextOverflow`. A system turn plus newest user turn that alone exceed the context throw
  `LLMError.failedToDecode` (`context size exceeded`).
- **Cost:** the whole history is prefilled on every turn, so prefill time grows linearly with it: 1.28 s for a
  3038-token history with Qwen2.5-0.5B Q4_K_M on an M1 Pro (about 0.42 ms per token, measured with other builds
  running); larger models cost proportionally more. The previous path paid the same, since its prompt cache never
  matched.
- **Cut replies:** a reply ended by `maxTokens`, cancellation or backgrounding leaves the next turn unaffected. The
  decode runs in a task of its own, so the call returns at once; the next turn waits until that decode has stopped
  (at most one token, or the end of a prefill) and flushes the token it left queued.
- **With tools** a turn runs on a LocalLLMClient `LlamaClient` (its tool-call parser), with the same rendering and
  the same clean start, in **rounds**. A round is every call one reply asks for: the calls run in the order the
  model wrote them, then the model replies again with their results. The turn ends at a reply that asks for no
  tools, at a cut reply, or at a reply that asks for tools after `toolLoop.maxToolRounds` rounds (default 1, as many
  rounds as before); those last calls do not run and are reported as `ToolEvent.roundLimitReached` and in
  `lastToolTurn.skipped`. Every round shares the turn's `maxTokens` and prefills the whole conversation again, so
  each round costs a full prefill (see
  [Tool calling]({{ '/api/core' | relative_url }}#tool-calling-gguf-and-mlx)).
  - **Errors:** a tool that throws, arguments that do not decode and an unknown tool name reach the model as
    `{"error": "…"}` and the turn goes on. `ToolLoopOptions(toolErrors: .failTurn)` makes the turn throw instead:
    the tool's own error, LocalLLMClient's `ToolError.argumentDecodingFailed`, or `LLMError.invalidParameter`.
    Cancelling the turn while a tool runs cancels the tool's task and throws `CancellationError`.
  - **What the model sees:** a tool's output as JSON with sorted keys (`key: value` pairs sorted by key when the
    output holds a value JSON cannot, such as a `Date`). With a chat template that mentions `tools` (Qwen2.5's, for
    example) a round is sent in the template's own format: the reply's calls as an assistant turn with `tool_calls`,
    each result as a `tool` turn. Other templates, and any whose native rendering throws, get each round as one
    assistant turn: the reply's calls as `<tool_call>` text, then each result wrapped in `<tool_response>` (the
    wrapping LocalLLMClient gives a result), so templates that require alternating user and assistant turns, such
    as Gemma 3's, accept it. The call turn is AuraLocal's: LocalLLMClient's own resume sends only the results. On a
    template that does not mention `tools`, only the first reply's prompt lists the tools and the call format; later
    rounds do not repeat them, so a chain that moves on to another tool in a later round needs a template with
    native tool support.
  - **Limits:** every round's prompt is trimmed like the first, but the newest user turn and the rounds after it
    always stay, so tool output too large for the context fails the turn with `LLMError.failedToDecode`.
    LocalLLMClient keeps only the first call per tool name in one reply (`LlamaClient.swift:119-124` in 0.5.0).
    The client decodes in a task nobody can cancel, so a reply cut short returns at once but the **next** turn
    waits until that decode reaches the end of generation or fills the context (seconds on a 0.5B model).
    After `unload()` (an eviction too) the model stays in memory until that decode ends, so a reload in the
    meantime holds two copies, and a process that exits before it ends aborts in `ggml_metal_device_free`.
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
  constraint; the history and vision paths do not. `AuraSession` uses one internally for a profile's
  `outputSchema` (below).

### Schema grammars

For a profile with an `outputSchema` and no tools, `AuraSession` runs every generation, repairs included, through
the constrained path above with a GBNF grammar that AuraLocal builds from the compiled schema. Validation and
repair still run on every reply. MLX and layer-streaming keep validation only, and so does
`AuraSession.usesSchemaGrammar = false`. What the grammar enforces is listed in
[the API reference]({{ '/api/core' | relative_url }}#schema-grammars).

Measured with `GGUFSchemaGrammarLiveTests` on the same M1 Pro (32 GB, macOS 26.7.1): temperature 0, the person
schema (`name`, `age` with `minimum: 0`), four one-sentence prompts, `maxRepairAttempts = 0`, two runs per model.
Times are wall clock per `stream` call. Other builds were loading the machine (1-minute load average 5.5 after the
first Qwen run and 27–29 after the second runs, whose times are in parentheses), so read them as upper bounds:

| Model | With the grammar | Validation only |
|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M (152k-token vocabulary) | 1.00–1.35 s (1.11–2.93 s); 4 of 4 replies conform; 12–18 tokens | 0.41–0.60 s (0.45–0.69 s); 4 of 4 conform; the same replies |
| SmolLM2-360M-Instruct Q4_K_M (49k-token vocabulary) | 0.54–0.60 s (1.08–1.66 s); 4 of 4 conform; 18–20 tokens | 0.82–0.92 s (1.54–2.32 s); 0 of 4 conform: each reply, 58–61 tokens, restated the schema |

Memory was not measured for schema grammars; the per-call peak above was measured for `.json` only.

- The grammar fixes the member order (required properties in `required` order), so a reply can start differently
  from what the model would write unprompted. With `required: ["age", "name"]` both models wrote `age` first.
- `maxTokens` still cuts a reply, and a string, a number or a free-form object can run until it does: on an
  open-ended prompt under the schema `{}`, Qwen2.5-0.5B repeated members until `maxTokens` (200). A cut reply fails
  validation and is repaired.
- Bounds the grammar cannot express (`maximum`, lengths above 64) still cost repairs.

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

`model.estimatedKVCacheGB(contextLength:)` applies this formula (`4 × layers × KV heads × head dim × context` bytes) to the catalog's metadata and returns the figures above; an entry with `kvLayers` is summed per layer type (see [Models › KV cache per layer type]({{ '/guide/models' | relative_url }}#kv-cache-per-layer-type-and-mixtures-of-experts)).

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

Nothing is cancelled on macOS or visionOS. Details and the opt-in `DeviceConditions` guard (heat, Low Power Mode,
battery): [Memory]({{ '/guide/memory' | relative_url }}#background-lifecycle-ios).
