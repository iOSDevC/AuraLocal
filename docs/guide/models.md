---
layout: docs
title: Model Catalog
parent: Guide
nav_order: 3
description: "All models supported by AuraLocal — MLX and GGUF models for iOS and macOS."
---

# Model Catalog
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Format Overview

AuraLocal supports two weight formats:

- <span class="badge badge-mlx">MLX</span> — Quantized Apple MLX arrays. GPU-accelerated on Apple Silicon. Best for ≤4B models on any Apple device.
- <span class="badge badge-gguf">GGUF</span> — Universal quantized format used by llama.cpp. Best for 7B–70B models on Mac. Uses layer-streaming on iOS. <span class="badge badge-stream">STREAM</span>

The complete catalog is `Sources/AuraCore/Resources/models.json`; `Model.allModels` returns it at runtime. Entries
without a static member (e.g. `ornith_1_5_35b_a3b_mlx`, `minicpm5_1b`, `swift15_qwen38_27b_mlx`) are loaded by id; see
[Catalog entries without a static member](#catalog-entries-without-a-static-member). Sizes in these tables are the
catalog's `approximateSizeMB` (MiB) in binary MB / GB; [Verifying the catalog](#verifying-the-catalog) keeps them,
and the layer and KV fields, in line with Hugging Face.

---

## Text Models — MLX

Fast, GPU-accelerated text generation. Downloads from `mlx-community` on HuggingFace.

| Model case | Display name | Size | HuggingFace repo |
|-----------|-------------|------|-----------------|
| `.qwen3_0_6b` | Qwen3 0.6B | ~320 MB | `mlx-community/Qwen3-0.6B-4bit` |
| `.qwen3_1_7b` ⭐ | Qwen3 1.7B | ~923 MB | `mlx-community/Qwen3-1.7B-4bit` |
| `.qwen3_4b` | Qwen3 4B | ~2.1 GB | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| `.gemma3_1b` | Gemma 3 1B | ~700 MB | `mlx-community/gemma-3-1b-it-4bit` |
| `.phi3_5_mini` | Phi-3.5 Mini | ~2.0 GB | `mlx-community/Phi-3.5-mini-instruct-4bit` |
| `.llama3_2_1b` | Llama 3.2 1B | ~663 MB | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| `.llama3_2_3b` | Llama 3.2 3B | ~1.7 GB | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| `.gemma4_12b` | Gemma 4 12B | ~6.3 GB | `mlx-community/gemma-4-12B-it-4bit` |
| `.granite4_tiny` | Granite 4.0 Tiny | ~3.5 GB | `mlx-community/granite-4.0-tiny-preview-4bit` |
| `.cogito_v1_3b` | Cogito v1 Preview 3B | ~1.7 GB | `mlx-community/deepcogito-cogito-v1-preview-llama-3B-4bit` |
| `.lfm2_1_2b` | LFM 2 1.2B | ~628 MB | `mlx-community/LFM2-1.2B-4bit` |
| `.qwen35_2b_text` | Qwen 3.5 2B (Text) | ~1.6 GB | `mlx-community/Qwen3.5-2B-MLX-4bit` |
| `.smollm3_3b` | SmolLM 3 3B | ~1.6 GB | `mlx-community/SmolLM3-3B-4bit` |

⭐ Recommended default. `.gemma4_12b` exceeds every iPhone [device preset](#device-presets), and mlx-swift-lm 3.31.3
cannot load it yet: its repo declares `model_type` `gemma4_unified`, which neither factory registers.

```swift
let llm = try await AuraLocal.text(.qwen3_1_7b)
```

---

## Vision Models — MLX

Multimodal image + text models. Pass a `UIImage` or `NSImage` for analysis.

| Model case | Display name | Size | Best for |
|-----------|-------------|------|---------|
| `.qwen35_0_8b` ⭐ | Qwen3.5 0.8B (default) | ~596 MB | Default, iPhone |
| `.qwen35_2b` | Qwen3.5 2B | ~1.6 GB | Higher accuracy, iPad |
| `.smolvlm_500m` | SmolVLM 500M | ~968 MB | Minimum memory footprint |
| `.smolvlm_2b` | SmolVLM2 2B | ~1.4 GB | SmolVLM, balanced |
| `.lfm25_vl_1_6b` | LFM 2.5 VL 1.6B | ~1.4 GB | — |

```swift
let vlm = try await AuraLocal.vision(.qwen35_0_8b)
let description = try await vlm.analyze("What's in this photo?", image: photo)
```

---

## Specialized Vision Models — MLX

Optimized for structured document and receipt extraction.

| Model case | Display name | Size | Output format |
|-----------|-------------|------|--------------|
| `.fastVLM_0_5b_fp16` ⭐ | FastVLM 0.5B FP16 | ~1.2 GB | JSON |
| `.fastVLM_1_5b_int8` | FastVLM 1.5B Int8 | ~1.8 GB | JSON |
| `.graniteDocling_258m` | Granite Docling 258M (IBM) | ~601 MB | DocTags → Markdown |

```swift
// FastVLM — structured JSON output
let ocr = try await AuraLocal.specialized(.fastVLM_0_5b_fp16)
let json = try await ocr.extractDocument(receiptImage)

// Granite Docling — DocTags markup converted to Markdown
let docOCR = try await AuraLocal.specialized(.graniteDocling_258m)
let raw = try await docOCR.extractDocument(documentScan)
let markdown = AuraLocal.parseDocTags(raw)
```

---

## Large Models — GGUF

7B–70B models using the llama.cpp backend. Downloaded from HuggingFace as single `.gguf` files (Q4_K_M quantization).

Use `ModelManager.shared.load(_:)`, which downloads the file, refuses `.tooLarge` models and then loads.
`AuraLocal.text(_:)` does **not** download GGUF files — with an uncached GGUF it throws *GGUF file not found*.

| Model case | Display name | File size | Min RAM (iOS) | Min RAM (macOS) |
|-----------|-------------|-----------|--------------|----------------|
| `.llama3_1_8b_gguf` ⭐ | Llama 3.1 8B (GGUF) | ~4.6 GB | 4 GB (streaming) | 16 GB |
| `.qwen2_5_7b_gguf` | Qwen 2.5 7B (GGUF) | ~4.3 GB | 4 GB (streaming) | 16 GB |
| `.mistral_7b_gguf` | Mistral 7B v0.2 (GGUF) | ~4.0 GB | 4 GB (streaming) | 16 GB |
| `.phi3_medium_gguf` | Phi-3 Medium 14B (GGUF) | ~8.0 GB | 6 GB (streaming) | 16 GB |
| `.gemma2_9b_gguf` | Gemma 2 9B (GGUF) | ~5.4 GB | 4 GB (streaming) | 16 GB |
| `.qwen2_5_32b_gguf` | Qwen 2.5 32B (GGUF) | ~18.5 GB | 12 GB, iPhone 17 Pro (streaming) | 32 GB (streams on 16 GB) |
| `.llama3_1_70b_gguf` | Llama 3.1 70B (GGUF) | ~39.1 GB | Not viable | 64 GB (streams on 32 GB) |

{: .note }
> Each cell is the smallest [device preset](#device-presets) at which `HardwareAnalyzer.assess` rates the model
> runnable. Every budget except `mac-32gb` is an estimate; the 6 GB and 8 GB iPhone classes assume the
> increased-memory-limit entitlement and the iPhone 17 Pro both memory entitlements. "Not viable" means `.tooLarge`
> at every iPhone preset: streaming needs RAM for at least a third of the weights. The smallest Mac preset is 16 GB.
> The 4 GB class (≈2.05 GB) is a jetsam measurement without entitlements; streaming a multi-GB GGUF on iOS still
> needs both memory entitlements (see [Memory Management]({{ '/guide/memory' | relative_url }}#entitlement)), so a
> "4 GB (streaming)" cell is the fit rule's answer, not a tested setup.

`.qwen2_5_7b_gguf` and `.qwen2_5_32b_gguf` download bartowski's single-file Q4_K_M: Qwen's own repos ship that quant
only in parts, which the GGUF downloader skips.

```swift
// Backend selected automatically by HardwareAnalyzer
let llm = try await ModelManager.shared.load(.llama3_1_8b_gguf)
let reply = try await llm.chat("Explain quantum computing")
```

### Quantization

All GGUF models use **Q4_K_M** quantization by default — the best size/quality trade-off for on-device inference:

| Quantization | Size vs FP16 | Quality loss | Notes |
|-------------|-------------|-------------|-------|
| Q4_K_M | ~25% of FP16 | Minimal | **AuraLocal default** |
| Q5_K_M | ~31% of FP16 | Very minimal | Higher quality, more RAM |
| Q8_0 | ~50% of FP16 | Near-zero | macOS only for 7B |

---

## Domain Models

Models tagged with a `Model.Domain` (`code`, `security`, `finance`, `medicine`).

| Model case | Display name | Format | Size | Domain | Notes |
|-----------|-------------|--------|------|--------|-------|
| `.medgemma_4b` | MedGemma 4B | MLX (vision) | ~2.8 GB | medicine | load with `AuraLocal.vision(_:)`; HAI-DEF license terms apply |
| `.medgemma_1_5_4b` | MedGemma 1.5 4B | MLX (vision) | ~3.2 GB | medicine | load with `AuraLocal.vision(_:)`; HAI-DEF license terms apply |
| `.mediphi_3_8b_mlx` | MediPhi Instruct 3.8B | MLX | ~2.0 GB | medicine | English |
| `.apollo2_1_5b_gguf` | Apollo2 1.5B (GGUF) | GGUF | ~1.0 GB | medicine | |
| `.medpsy_1_7b_gguf` | MedPsy 1.7B (GGUF) | GGUF | ~1.2 GB | medicine | validated in English by its author |
| `.fingpt_mt_llama3_8b_gguf` | FinGPT MT 8B (Llama 3) | GGUF | ~4.6 GB | finance | |
| `.fingpt_sentiment_lfm2_1_2b_gguf` | FinGPT Sentiment 1.2B | GGUF | ~769 MB | finance | sentiment only, tuned for single-token outputs; not for chat |
| `.security_gemma_e2b_gguf` | Security SLM Gemma 4 E2B (Mobile Sec) | GGUF | ~3.2 GB | security | vulnerability triage, not the sole authority |
| `gemma4_12b_coder_gguf` | Gemma 4 12B Coder (Fable5·Composer2.5) | GGUF | ~6.9 GB | code | no static member |
| `gemma4_12b_agentic_gguf` | Gemma 4 12B Agentic (Fable5·Composer2.5) | GGUF | ~6.9 GB | code | no static member |

```swift
let financeModels = Model.models(in: .finance)
```

---

## Catalog entries without a static member

These entries are in `models.json` but have no `Model.<id>` constant; look them up by id with
`ModelRegistry.shared.model(id:)`. The last column is the smallest [device preset](#device-presets) at which
`HardwareAnalyzer.assess` rates the entry runnable.

| Model id | Display name | Format | Size | Smallest preset |
|----------|-------------|--------|------|-----------------|
| `minicpm5_1b` | MiniCPM5 1B | MLX | ~580 MB | `iphone-4gb` |
| `minicpm5_2b` | MiniCPM5 2B | MLX | ~1.3 GB | `iphone-4gb` |
| `minicpm4_1_8b` | MiniCPM4.1 8B | MLX | ~4.3 GB | `iphone-17-pro` |
| `swift15_qwen38_27b_mlx` | Swift 1.5 Qwen3.8 27B (MLX 3-bit) | MLX | ~11.0 GB | `mac-32gb` |
| `qwen38_27b_uncensored_mlx` | Qwen3.8 27B Uncensored (MLX 4-bit) | MLX (vision) | ~15.0 GB | `mac-32gb` |
| `ornith_1_5_35b_a3b_mlx` | Ornith 1.5 35B A3B (MLX 4-bit) | MLX | ~18.2 GB | `mac-32gb` |
| `gemma4_12b_coder_gguf` | Gemma 4 12B Coder (Fable5·Composer2.5) | GGUF | ~6.9 GB | `iphone-6gb` (streaming) |
| `gemma4_12b_agentic_gguf` | Gemma 4 12B Agentic (Fable5·Composer2.5) | GGUF | ~6.9 GB | `iphone-6gb` (streaming) |

```swift
if let model = ModelRegistry.shared.model(id: "minicpm5_1b") {
    let llm = try await AuraLocal.text(model)
    print(try await llm.chat("Hello"))
}
```

---

## Verifying the catalog

`scripts/verify_catalog.py` checks every `models.json` entry against live Hugging Face data, prints the differences
and exits 1 when any remain. It uses only the Python standard library and reads layers, KV heads, head width,
`kvLayers`, `experts`, the trained context and the loaded weight bytes from `aura models check <repo> --json`, the
code behind `aura models check --entry`.

```sh
swift build --product aura                       # the script runs .build/debug/aura (or --aura <path>)
uv run scripts/verify_catalog.py                 # report only
uv run scripts/verify_catalog.py --fix           # also write the fixable corrections to both catalog copies
uv run scripts/verify_catalog.py --only gemma4_12b,granite4_tiny --report /tmp/catalog.json
uv run scripts/verify_catalog.py --no-aura           # repos, files and sizes only, no aura build needed
```

| Field | Live source | `--fix` |
|-------|-------------|---------|
| `repoID` | `/api/models/{repo}?blobs=true` without following redirects: a 3xx is a rename, `gated` is reported as a note, 401/403/404 is gone or private (anonymous requests cannot tell which) | renames only |
| `ggufFilename` | the repo listing: the exact name at the repo root; a split (`-0000N-of-0000M`) or subfolder file is a difference | case-only mismatches |
| MLX files | a top-level `config.json` and `*.safetensors` | — |
| `approximateSizeMB` | GGUF: that file's size; MLX: the safetensors `aura` loads (`weightsBytes`); drift above `--size-tolerance` (2 %) | yes, in MiB |
| `numLayers`, `kvHeads`, `headDim` | `aura`'s `architecture` object: `config.json`, or the header of the entry's own GGUF (`--gguf`) | yes |
| `kvLayers`, `experts` | the same; `kvLayers` stays out when every layer keeps the whole context at one width, and `experts` is MLX only | yes |
| `maxContextLength` | the trained context: a cap above it, or no cap while it is below 32768 | yes |

Requests stay under `--max-requests` (450) per 300 s, below Hugging Face's 500; each `aura` call is charged at an
upper bound (5 plus 2 per top-level safetensors file). A 429 waits for `Retry-After`, else the `RateLimit` header's
`t=`, else 15 s doubling up to 300 s; an `aura` report that met a 429 is retried after 60 s, doubling. `HF_TOKEN`, when set, is sent to huggingface.co with the script's own requests;
`aura` uses the Keychain token. `--device` (default `mac-64gb`) only affects the "not runnable" notes, which report
the pinned runtimes and never fail the run. Differences that must stay go in `scripts/catalog-exceptions.json`
(`{"entries": {"<id>": {"fields": ["approximateSizeMB"], "reason": "…"}}}`), which the run lists without failing.

**Last run, 2026-10-14:** 59 entries before, 53 after, 0 differences.

- Same model, live repo: `cogito_v1_3b` → `mlx-community/deepcogito-cogito-v1-preview-llama-3B-4bit`, `smollm3_3b` →
  `mlx-community/SmolLM3-3B-4bit`, `lfm2_1_2b` → `mlx-community/LFM2-1.2B-4bit`, `lfm25_vl_1_6b` →
  `mlx-community/LFM2.5-VL-1.6B-4bit`, `qwen35_2b_text` → `mlx-community/Qwen3.5-2B-MLX-4bit` (Qwen3.5-2B has no
  separate Instruct release); `qwen2_5_7b_gguf` / `qwen2_5_32b_gguf` → bartowski's single-file Q4_K_M.
- Renamed on Hugging Face (the old names still redirect): `dphn/Dolphin3.0-Llama3.1-8B-GGUF`,
  `mlx-community/gemma-4-12B-it-4bit`. Downloads live in a folder named after `repoID`, so a Dolphin 3.0 8B
  downloaded under the old id downloads again.
- Removed, no loadable repo: `gemma4_1b`, `gemma4_4b` (Gemma 4 ships E2B, E4B, 12B, 26B-A4B, 31B),
  `granite4_compact` (no such Granite 4.0 model), `lfm25_3_2b` (no LFM2.5 3.2B), `ministral_3b` (the 2412 3B weights
  were never released), `graniteVision_3_3` (its repo holds no weights, and mlx-swift-lm 3.31.3 has no
  `granite_vision`). Their `Model` constants are `@available(*, unavailable)` with the reason.
- Corrected from the live data: every MLX entry's `kvHeads` / `headDim` (all were 0), 12 `kvLayers`, 2 `experts`
  (`granite4_tiny`, `ornith_1_5_35b_a3b_mlx`), `numLayers` of 10 entries (`qwen35_2b` 36 → 24, `granite4_tiny`
  18 → 40), the `Dolphin3.0-Qwen2.5-3b` file name case, sizes, and `maxContextLength` caps below 32768.
- Not a catalog difference: `gemma4_12b` declares `model_type` `gemma4_unified`, which mlx-swift-lm 3.31.3 does not
  register, so it does not load until the pin moves.
- Re-run the same day with the per-layer GGUF sizing: `gemma4_12b_coder_gguf`, `gemma4_12b_agentic_gguf` and
  `security_gemma_e2b_gguf` now carry their sliding layers' `headDim` 256 (`key_length_swa`) and a `kvLayers` with the
  global layers' 512-wide heads; before, every layer was sized at 512. 53 entries, 0 differences.

---

## Hardware Compatibility Check

```swift
import AuraCore

// Check a specific model on the current device
let result = HardwareAnalyzer.assess(.llama3_1_8b_gguf)
print(result.fitLevel.label)       // "Streaming", "Good", "Too Large", etc.
print(result.fitLevel.isRunnable)  // true/false
print(result.model.estimatedRuntimeMemoryGB)   // ~5 GB full-load (monolithic)
print(result.model.estimatedStreamingMemoryGB) // ~0.65 GB in layer-streaming mode

// Every catalog model: .tooLarge last, then downloaded first, then best fit
let compatible = HardwareAnalyzer.compatibleModels().filter { $0.fitLevel.isRunnable }
for c in compatible {
    print("\(c.model.displayName): \(c.fitLevel.label)")
}

// Check for a custom device profile
let profile = HardwareProfile(
    totalMemoryGB: 16.0,
    availableMemoryGB: 8.0,
    deviceName: "M3 MacBook Pro"
)
let results = HardwareAnalyzer.compatibleModels(profile: profile)
```

### Fit Levels

| Level | Meaning | `isRunnable` |
|-------|---------|-------------|
| `.excellent` | >40% RAM headroom | ✅ |
| `.good` | 20–40% RAM headroom | ✅ |
| `.marginal` | <20% RAM headroom | ✅ |
| `.streamingRequired` | Full load impossible, mmap streaming viable | ✅ |
| `.tooLarge` | Exceeds even streaming budget | ❌ |

### KV cache per layer type and mixtures of experts

Two optional `models.json` objects refine the estimate. Entries without them keep the old math: every one of
`numLayers` caches the whole context (`4 × kvHeads × headDim` bytes per token per layer), and an entry with
`kvHeads` or `headDim` 0 gets a flat 0.15 GB.

```json
"kvLayers": {"fullAttentionLayers": 6, "slidingLayers": 0, "slidingWindow": 0, "recurrentLayers": 18, "recurrentStateBytes": 10100736},
"experts": {"routedExperts": 256, "expertsPerToken": 8, "activeSizeMB": 1865}
```

- **`kvLayers`** (`KVCacheLayout`) mirrors the caches mlx-swift-lm 3.31.3 allocates in `newCache(parameters:)`:
  full-attention layers grow with the context, sliding-window layers stop at `slidingWindow` tokens once decoding
  starts (`RotatingKVCache`), and linear-attention / state-space layers hold a fixed `recurrentStateBytes`
  (`MambaCache`). Gemma 4 adds `fullAttentionKVHeads` / `fullAttentionHeadDim` for its wider global layers.
  `estimatedKVCacheGB(contextLength:)` sums them at their peak, which for a sliding layer comes during prefill:
  `slidingPrefillStep` (512, `GenerateParameters.prefillStepSize`) means the prompt is fed in steps, so the layer
  briefly holds `slidingWindow + 511` tokens; no `slidingPrefillStep` means the runtime prefills the whole prompt in
  one pass and the layer holds all of it until the first decoded token. That is the case for `gemma3`,
  `gemma3_text` and `gemma3n` (their `prepare` does not chunk) and for the sliding-window VLMs (Gemma 3, Gemma 4,
  Mistral 3), so their peak equals every layer keeping the whole context. `MLXBackend` re-prefills the whole
  conversation every turn, so every turn reaches it.
- **`experts`** (`ExpertRouting`) holds the MiB a decoded token reads: the non-expert tensors plus
  `expertsPerToken / routedExperts` of the expert tensors (`.experts.`, `switch_mlp`, Granite's
  `input_linear` / `output_linear`). Shared experts count as non-expert; vision towers and `mtp` heads are left out.
  Memory fit still uses the whole download, because every expert stays resident.

The checker derives `kvLayers` from `config.json` (the top level, else `text_config`):

| Family (`model_type`) | Rule |
|-----------------------|------|
| `qwen3_next`, `qwen3_5`, `qwen3_5_moe`, `qwen3_5_text` | layer `i` is full attention when `(i + 1) % full_attention_interval == 0` (default 4), else Gated DeltaNet; `layer_types` is ignored, as the runtime ignores it |
| `gemma3`, `gemma3_text` | global when `i % sliding_window_pattern == pattern - 1` (default 6), else sliding; missing heads take the runtime defaults (4 / 256 for `gemma3`, 1 / 256 for `gemma3_text`). `gemma4` / `gemma4_text` without `layer_types` use the same rule with default 5 |
| `gemma3n`, `gemma4`, `gemma4_text`, `gpt_oss`, `olmo3`, `mistral3`, `afmoe` | `layer_types`; the last `num_kv_shared_layers` layers hold no cache (`gemma3n`, `gemma4*`) |
| `exaone4` | `sliding_window_pattern` string: `L` is sliding |
| `lfm2`, `lfm2_moe`, `lfm2_vl` | `full_attn_idxs` (else `layer_types`), the rest short convolutions |
| `nemotron_h` | `hybrid_override_pattern`: `M` Mamba, `*` attention, others no cache |
| anything else | `layer_types`, where `sliding_attention` counts as full (the runtime gives it a full cache) and `linear` / `mamba` / `conv` names are recurrent; else `full_attention_interval`; else every layer full |

Recurrent states are sized at 2 bytes per element from `linear_*` keys (Gated DeltaNet), `mamba_*` keys (Mamba 2) or
`conv_L_cache` × `hidden_size` (LFM2). A sliding layer with no plausible `sliding_window` counts as full. Families the
table does not list (Jamba, Falcon H1, Baichuan M1, MiMo V2 Flash) are sized as every layer full attention,
without the recurrent state some of them also keep.
A GGUF header that lists its sliding-window layers (`attention.sliding_window_pattern`, one flag per layer, as in
Gemma 4) gets a `kvLayers` too, sized as llama.cpp b8851 allocates it: `llama_context_default_params()` sets
`swa_full`, so sliding-window layers are sized for the whole context like the global ones (`slidingWindow` 0), at
`attention.key_length_swa` × their own `head_count_kv`; the entry's `kvHeads` / `headDim` are the sliding layers',
and the global layers' go in `fullAttentionKVHeads` / `fullAttentionHeadDim`. The last
`attention.shared_kv_layers` layers hold no cache. Other GGUF headers stay uniform at the largest `head_count_kv`
and `key_length`, which llama.cpp's own sizing matches when every layer has the same width. The `experts` object
comes from MLX repos only. The 256-token allocation step of `KVCacheSimple` is ignored.

`HardwareAnalyzer.recommendedContextWindow(for:)` searches the largest multiple of 512 whose cache fits, so a
sliding or hybrid model earns a longer window than its layer count suggests.

**Decode range for a mixture of experts.** `ModelCompatibility.estimatedDecodeTokensPerSecondRange` (and
`FitEstimate.tokensPerSecondRange`) is the decode estimate ([Benchmarks › Speed estimate]({{ '/guide/benchmarks' | relative_url }}#speed-estimate))
on `activeSizeMB` at its top and half of it at its bottom
(`uncalibratedMoEFloor` in `HardwareAnalyzer.swift`): MLX MoE decode also pays for routing and expert gathers, and no MoE has been
measured on this hardware yet. `estimatedDecodeTokensPerSecond` and `speedLevel` use the low end; `verdict` says
"uncalibrated MoE". Dense models keep a single figure and a `nil` range.

---

## Finding compatible models

A download that *fits* is not a download that *runs*. `ModelCompatibilityChecker` answers the second question
for any Hugging Face repo: will AuraLocal, with the runtimes it pins — **mlx-swift-lm 3.31.3** (MLX) and
**llama.cpp b8851** (GGUF, via LocalLLMClient 0.5.0) — load and run it on a given device, and if not, exactly why.

It reads the repo listing (`/api/models/{id}?blobs=true`), `config.json`, `model.safetensors.index.json` and, with
HTTP Range requests, the first 4 MiB of one GGUF file (`ggufFile` when given, else the smallest whole quant at the
repo root: subfolders hold MTP draft heads and other files the downloader never fetches) or the headers of the shipped
safetensors files (when there is
no index or it is stale; a `qwen3_5` vision repo's first shard when its `__metadata__` decides the verdict; every
loaded shard, up to 64, when `config.json` declares more than one routed expert, to size each tensor from its
`data_offsets`). Weights are never downloaded. Network errors, 401/403 and 404 come back as findings — the verdict is `unknown` when they hide
the repo listing, an MLX `config.json` or a GGUF architecture — and the checker never throws. Gated repos use the
Hugging Face token saved in the Keychain (`download.huggingface`).

```swift
import AuraCore

let checker = ModelCompatibilityChecker()
let report = await checker.check("mlx-community/Qwen3.5-27B-4bit", on: .mac32GB)
print(report.status.label, "—", report.headline)   // Runs — Vision · Good · 16.0 of 20.0 GB
for finding in report.blockers {
    print(finding.title, "—", finding.detail)
}
if let entry = report.suggestedEntry {
    print(entry.jsonText())                        // the models.json entry, catalog key order
}

// Load a repo the checker cleared, without editing models.json (registered for this launch only).
if report.status.isRunnable, let model = report.suggestedEntry?.decodedModel() {
    ModelRegistry.shared.register(model)
    let llm = try await ModelManager.shared.load(model)
    print(try await llm.chat("Hello"))
}

// Fetch once, judge every device preset without refetching.
let snapshot = await checker.snapshot(of: "mlx-community/MiniCPM5-1B-4bit")
let verdicts = DevicePreset.all().map { device in
    (device.displayName, CompatibilityEvaluator.evaluate(snapshot, on: device).status.label)
}
```

### What it checks

Each rule yields a typed finding (`blocker`, `caveat` or `info`) that names the file, key or pinned source that
decides it. Any blocker means **Won't run**.

| Rule | Blocks when | Why (pinned source) |
|------|-------------|--------------------|
| `format` | no MLX weights (MLX quantization in `config.json` or the `mlx` tag) and no GGUF | the catalog loads MLX conversions and GGUF only |
| `generative` | pipeline `fill-mask`, `feature-extraction`, `zero-shot-image-classification`…, a `*ForMaskedLM`-style head, an encoder `model_type`, the `sentence-transformers` / `sentence-similarity` / `feature-extraction` tags, or "embedding" in the repo or `base_model` name | the runtimes chat with text generators only; MLX embedding conversions often keep `pipeline_tag: text-generation` |
| `mlx.config` | an MLX repo has no `config.json`, or it lacks `model_type` (unreadable — 401/404 — is a caveat and the verdict `unknown`) | `BaseConfiguration` decodes `model_type` before anything else |
| `mlx.model-type` | top-level `model_type` is in neither `LLMTypeRegistry` (54 types) nor `VLMTypeRegistry` (17) | the factories dispatch on the top-level key; `text_config.model_type` does not count |
| `mlx.weight-prefixes` | a `qwen3_5` / `qwen3_5_moe` tensor outside `language_model`, `model`, `lm_head`, `vision_tower`, `mtp` — or, for vision weights whose safetensors `__metadata__.format` is `mlx`, outside `language_model` and `vision_tower` | the pinned `sanitize` maps only those (the VLM one returns MLX-format weights unchanged); anything else fails `update(parameters:verify: [.all])` |
| `mlx.towers` | a type registered only as an LLM (e.g. `gemma3n`) whose weights hold vision or audio towers (`vision_tower`, `audio_tower`, `embed_vision`, `embed_audio`, `multi_modal_projector`…) | its pinned text model has no such modules and no LLM-only sanitize drops them |
| `mlx.extra-safetensors` | a top-level `*.safetensors` missing from the weight map, or a map that points at some missing files; a map that references **none** of the shipped files is only a caveat (stale index) | AuraLocal's downloader fetches every top-level file and `loadWeights` merges every `*.safetensors` without reading the index, so a stale index is replaced by the shipped files' headers |
| `mlx.rope` | a rope type `RoPEUtils.initializeRope` does not implement, or a `longrope` without its three fields | `initializeRope` calls `fatalError` — the app crashes |
| `mlx.category` | a VLM-only type without vision tensors | vision needs a registered VLM type **and** vision weights; otherwise the model loads as text |
| `gguf.architecture` | `general.architecture` not in b8851's `LLM_ARCH_NAMES`, added later (`qwen4exp` → b10660), or an encoder / diffusion architecture | llama.cpp rejects unknown architectures |
| `gguf.qwen35-nextn` | `qwen35` / `qwen35moe` with `nextn_predict_layers` > 0 | b8851 marks recurrent layers arithmetically and demands `ssm_*` tensors the MTP block lacks; fixed upstream in b9180 (PR #22673) |
| `gguf.shards` | every quant is split (`-0000N-of-0000M`); a caveat when only some are | the GGUF path loads one file |
| `fit` | the weights (MLX) or every single-file quant (GGUF) are too large for the device | `HardwareAnalyzer.assess` against the device budget |
| `imagegen` | a text-to-image pipeline on an iPhone preset (a caveat on a Mac: mflux support is not verified) | AuraImageGen runs only its built-in Core ML preset on iOS (Stable Diffusion 2.1 base); other diffusion repos are not supported |
| `license` | never blocks: flags gated repos, a missing license, non-commercial (`cc-by-nc*`) and custom (`other`) licenses by name | |

`mlx.tokenizer`, `gguf.header`, `gguf.projector` and `context` add caveats and context (missing tokenizer files,
unreadable headers, an ignored mmproj, trained contexts below 32768 tokens) but never block.

Fit uses the existing `HardwareAnalyzer` math: weights + runtime overhead + a GQA-aware KV cache at 2048 tokens,
with `kvHeads` / `headDim` from `config.json` or the GGUF header and the cache per layer type described in
[KV cache per layer type and mixtures of experts](#kv-cache-per-layer-type-and-mixtures-of-experts). A `kv` finding
reports the layer split when not every layer keeps the whole context, and a `moe` finding the bytes a token reads
and the decode range. Suggested entries carry `kvHeads`, `headDim` and `kvLayers` (omitted when every layer keeps the
whole context at one width); MLX entries also carry `experts`. Every GGUF quant gets its own fit. Sizes are
binary: catalog `approximateSizeMB` is MiB, because `HardwareAnalyzer` divides it by 1024 to get GiB. Layer, KV-head
and head-width values past 4096, 1024 and 8192 come from a corrupt or hostile file and count as unknown.

The pinned tables live in `Sources/AuraCore/ModelCompatibility/PinnedRuntimes.swift`; `scripts/regen_pinned_runtimes.sh`
prints the new sets when a pin moves, and `ModelCompatibilityTests` fails when the MLX sets drift from the
`mlx-swift-lm` checkout.

### Known incompatibilities

Verified on 2026-09-29 with `aura models check <repo> --device mac-32gb` (M1 Pro, 20.0 GB measured budget):

| Repository | Verdict | Deciding finding |
|------------|---------|------------------|
| `ukisai/Swift-1.5-4bit-MLX` | Won't run | `visual.*` (501 tensors) is not mapped by the qwen3_5 sanitize |
| `ukisai/Swift-1.5-3bit-MLX-TextOnly` | Runs, with caveats | text: only `language_model.*`; 10.96 GB of weights, Excellent · 12.0 of 20.0 GB; custom license |
| `mlx-community/Qwen3.5-27B-4bit` | Runs | vision: `language_model` + `vision_tower`; Good · 16.0 of 20.0 GB |
| `ukisai/Swift-1.5-Qwen3.8-27B-GGUF` | Won't run | `qwen35`, `block_count` 65 with 1 NextN layer: needs llama.cpp b9180 or later (17 of 22 quants would otherwise load fully) |
| `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF` | Won't run | `qwen4exp` needs b10660; every quant is split; the smallest (IQ1_S) is 65.3 GB |
| `Edge0/Edge0-35B-A3B-preview` | Won't run | `lora_edge0_35b.safetensors` (620 tensors) and `prerouter_edge0_35b.safetensors` (99) are outside the weight map |
| `medicalai/ClinicalBERT` | Won't run | `fill-mask`, `DistilBertForMaskedLM`; only `pytorch_model.bin` |
| `google/medsiglip-448` | Won't run | SigLIP, `zero-shot-image-classification`; not an MLX conversion; gated, custom license |
| `stanford-crfm/BioMedLM` | Won't run | `gpt2` is not registered; only `.bin` weights; trained context 1024 |
| `mlx-community/MiniCPM5-1B-4bit` | Runs | `model_type` `llama`; 0.57 GB — fits the iPhone 4 GB class |
| `mlx-community/MiniCPM4.1-8B-4bit` | Runs | `minicpm`; its `longrope` carries the fields RoPEUtils needs |
| `mlx-community/MiniCPM3-4B-4bit` | Won't run | `minicpm3` is not registered ("v3 uses a different architecture", MLXLLM README) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | Runs, with caveats | `qwen3_5_moe`, Marginal · 18.8 of 20.0 GB; no license declared (2026-10-14: 18.7 GB with the per-layer KV; decode ~29–58 tok/s, uncalibrated MoE range) |
| `Qwen/Qwen2.5-7B-Instruct-GGUF` | Runs, with caveats | Q4_K_M and 6 other quants are split; Q2_K and Q3_K_M are single files |
| `Qwen/Qwen3.8-Flash-Next` | Won't run | unconverted safetensors (335 GB); `qwen4_exp` is not registered |
| `mlx-community/gemma-3-4b-it-qat-4bit` | Runs, with caveats | vision; its index names two shards the repo no longer ships, so the checker reads `model.safetensors` itself |
| `mlx-community/gemma-3n-E4B-it-bf16` | Won't run | `gemma3n` is LLM-only here; `model.vision_tower` / `model.audio_tower` would be unhandled keys |
| `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | Won't run | an embedding model (`sentence-transformers` tags) despite `pipeline_tag: text-generation` |

Re-checked on 2026-10-14 with the per-layer KV and the MoE estimate (`--device mac-32gb`). One figure means the peak
and the decoding size are the same; in parentheses, every layer keeping the whole context:

| Repository | KV layers | KV at 8192 tokens: peak / once decoding | Decode |
|------------|-----------|-----------------------------------------|--------|
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 10 of 40 full, 30 Gated DeltaNet (31.4 MB state) | 191 MB (640 MB if every layer kept it) | ~29–58 tok/s, uncalibrated MoE range: a token reads ≈1.82 of 18.17 GB |
| `mlx-community/Qwen3.5-2B-4bit` | 6 of 24 full, 18 Gated DeltaNet (9.6 MB state) | 106 MB (384 MB) | ~63 tok/s |
| `mlx-community/gemma-3-12b-it-qat-4bit` | 8 of 48 full, 40 sliding (1024 tokens); the prompt is prefilled in one pass | 3072 MB / 832 MB (3072 MB) | ~20 tok/s |
| `mlx-community/gemma-3-4b-it-4bit` | 5 of 34 full, 29 sliding (1024 tokens), prefilled in one pass; heads from the runtime defaults | 1088 MB / 276 MB (1088 MB) | ~40 tok/s |
| `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF` (`gemma4-coding-Q4_K_M.gguf`) | llama.cpp: 8 global at 1 × 512 and 40 sliding-window at 8 × 256, all sized for the whole context | 2688 MB (6144 MB at the header's 8 × 512 on every layer) | — |

### Device presets

Only this device and the 32 GB Mac are measured; every other budget is an estimate, and `DevicePreset.source`
says where it comes from (the CLI and the app show it). The 32 GB figure was measured on one M1 Pro whose
`iogpu.wired_limit_mb` is set to 20480 by a local LaunchDaemon; a stock 32 GB Mac's default was not measured.

| Preset (`--device`) | Budget | Source |
|---------------------|--------|--------|
| `this-device` | measured now | Mac: Metal `recommendedMaxWorkingSetSize`; iPhone: the app's memory limit, `os_proc_available_memory()` + `phys_footprint`, capped at 74% of RAM |
| `iphone-4gb` | ≈2.0 GB | estimate: third-party jetsam measurement, ActiveHard 2098 MB on an iPhone 12 |
| `iphone-6gb` | ≈3.0 GB | estimate: this guide's [memory budgets]({{ '/guide/memory' | relative_url }}), increased-memory-limit entitlement |
| `iphone-8gb` | ≈4.5 GB | estimate: same source |
| `iphone-17-pro` | ≈6.4 GB | estimate: third-party report, only with both memory entitlements |
| `mac-16gb` | ≈10.7 GB | estimate: default Metal working set ≈ 2/3 of RAM up to 36 GB |
| `mac-32gb` | 20.0 GB | measured on one M1 Pro with `iogpu.wired_limit_mb` set to 20480; `recommendedMaxWorkingSetSize` follows it |
| `mac-64gb` | ≈48 GB | estimate: default Metal working set ≈ 3/4 of RAM above 36 GB |

### From the command line

```
aura models search "<query>" [--format mlx|gguf] [--device <preset>] [--limit N]
aura models check <owner/repo | URL> [--device <preset>] [--gguf <file>] [--json] [--entry]
aura models devices
```

`search` checks every hit (four at a time) and prints a verdict table; `check` prints every finding, the
per-quant fit table and the `models.json` entry; `--entry` prints only the entry, `--json` the whole report (with an
`architecture` object that survives blockers, and HF's `gated` mode). `--gguf <file>` picks the GGUF whose header is read.
`--device` defaults to `this-device`; `--limit` defaults to 15 (1–100). See also the
[CLI reference]({{ '/guide/cli' | relative_url }}#model-compatibility-aura-models).

### The example apps

Both example apps use AuraUI's Hugging Face browser (`HFModelSearchList`, `HFRepoDetailView`,
`DownloadedModelsList`, `HuggingFaceTokenSection`; see the [UI API]({{ '/api/ui' | relative_url }}#hugging-face-browser)),
built on the same checker and on [`ModelDownloads`](#downloading-without-loading): search with format and sort
filters, a verdict badge per result (checked lazily as rows appear), a **Compatible only** toggle, and a detail
view with the findings, the per-quant fit table, **Copy catalog entry**, **Open on Hugging Face** and, for repos
rated runnable, a **Download** section (quant picker, size and free space, progress, Cancel/Resume, **Try it** once
finished). A gated repo shows **Add a token**, which opens the token form in place.

- **AuraExample** (iOS 18.6+ and macOS 26.2+): **Models → Search Hugging Face** and **Models → Downloads**, plus
  **Find on Hugging Face…** at the top of the Text tab's model menu (on iPhone, Models sits under More). Search
  judges repos against this device only and starts with **Compatible only** on. A finished download is listed
  under **Downloaded** in the Text menu (and in the Vision picker for an MLX vision model) without a relaunch,
  and again after one. In the Text menu's **Downloaded** section a model that can't load now is disabled with the
  reason; the Vision picker leaves out downloaded models rated *Too Large*. Nothing loads until you send a
  message.
- **[ModelFinder](https://github.com/iOSDevC/AuraLocal/blob/main/Examples/ModelFinder/README.md)** (iOS 18 and
  macOS 15): the same views in a split view with a device picker (this device, iPhone classes, Macs), a Downloads
  tab and a Settings tab. Its README explains how to generate and open the project.

On the Simulator the format filter starts on GGUF, MLX models are disabled in the Text menu's **Downloaded**
section and in Try it (MLX can't run there; picked anywhere else, they fail to load with an error), and "this
device" reports the Mac's memory, not an iPhone's.

---

## Bring your own GGUF

`Model.fromURL(_:displayName:)` builds a text GGUF model from a Hugging Face file URL (`resolve` or `blob` form,
any revision or subfolder), any other `https` URL ending in `.gguf`, or a `file://` path that is loaded in place
and never copied. `Model.from(_:)` takes the same as a `ModelSource` (`.huggingFace(url:)`, `.directURL(_:)`,
`.localFile(_:)`).

```swift
if let model = Model.fromURL("https://huggingface.co/<owner>/<repo>/resolve/main/<file>.gguf") {
    let llm = try await ModelManager.shared.load(model)   // downloads to <Caches>/models/<owner>/<repo>/
    print(try await llm.chat("Hello"))
}
```

Such a model has size 0 and no KV metadata. The load-time memory guard always rates it as fitting, and its
context window is the platform fallback (8192 or 4096 on a Mac, 2048 or 1024 on iOS). For a sized entry, use
the checker's `suggestedEntry` (below).

---

## Downloading without loading

`ModelDownloads` fetches a model to the exact path its backend loads from, without loading it, so a model
downloaded ahead of time is one `ModelManager.shared.load(_:)` finds on disk:

```swift
// A sized Model for a search result: the checker's suggested entry, or another single-file quant of it.
let report = await ModelCompatibilityChecker().check("Qwen/Qwen2.5-0.5B-Instruct-GGUF", on: .thisDevice())
guard let entry = report.suggestedEntry, let model = entry.decodedModel() else { return }
// let other = entry.replacingQuant(report.quantFits[0].option)?.decodedModel()

let plan = try await ModelDownloads.plan(for: model)          // files and sizes, nothing written
print(plan.totalBytes, plan.bytesOnDisk, ModelDownloads.availableDiskBytes() ?? -1)

let location = try await ModelDownloads.download(model) { done, total in
    print("\(done) / \(total)")                                 // @Sendable, off the main actor
}
// GGUF: location == model.resolvedFileURL; MLX: the snapshots/main folder, and model.isDownloaded is true.

ModelManager.shared.evict(model)                                // unload before deleting
try ModelDownloads.delete(model)                                // file + .partial, or the whole MLX repo folder
```

- **Where files land**: GGUF at `<Caches>/models/<org>/<repo>/<file>.gguf`; MLX at
  `<Caches>/huggingface/hub/models--<org>--<repo>/snapshots/main/` with a `.complete` marker written after the
  last file. The MLX load path (`MLXBackend`) has its own downloader that uses the same folder and marker, so a
  snapshot downloaded here loads without a second download.
- **One folder, pinned**: `ModelDownloads.downloadFolder(repoID:revision:folder:include:auth:progress:)` fetches
  the files under one folder of a repo at a fixed revision into
  `<Caches>/huggingface/hub/models--<org>--<repo>/snapshots/main/<folder>`. It resumes `.partial` files, deletes
  files from any other revision first, and sends no token unless you pass `auth`. `planFolder`, `localFolder`,
  `isFolderComplete` and `deleteRepo` go with it. AuraImageGen downloads its Core ML Stable Diffusion this way
  (see [Image generation]({{ '/guide/imagegen' | relative_url }})).
- **What MLX fetches**: top-level `*.safetensors`, `*.json` and `*.jinja` files of `main` only (mlx-swift-lm's
  patterns). Weights in a subfolder, or no `config.json`, throw `ModelDownloadError.incompleteRepo`.
- **What GGUF fetches**: one single-file quant at the top level of the repo. Split (`-00001-of-0000N`) and
  subfolder quants throw `ModelDownloadError.unsupported`.
- **Resume**: each file is fetched in 32 MB `Range` chunks into `<file>.partial`, which survives a cancel, a
  network cut and a relaunch; the next `download` continues from it. The partial records the ETag of its bytes
  (an extended attribute); when a chunk comes back with another ETag — the file was re-uploaded under the same
  name — or the partial is longer than the file, it restarts from zero instead of splicing two revisions. A
  server that sends no ETag cannot be checked. The file is moved into place only when its size matches.
  `ModelDownloads.bytesOnDisk(for:)` reports partial bytes; `DownloadedModels.partialFiles()` lists the `.partial`
  files; `DownloadedModels.scanSnapshots()` lists MLX snapshots, complete or not.
- **Free space**: `download` throws `ModelDownloadError.insufficientSpace` before writing when the remaining bytes
  plus 20 % do not fit.
- **Errors** (`ModelDownloadError`): `unauthorized` (`X-Error-Code: GatedRepo`, or 401/403 with a token sent:
  invalid token, no access, license not accepted), `notFound` (404, or `RepoNotFound` / `EntryNotFound` /
  `RevisionNotFound`), `notFoundOrPrivate` (an anonymous 401 with no error code: Hugging Face answers a missing,
  renamed or private repo the same way), `insufficientSpace`, `http`, `incompleteRepo`, `sizeMismatch`,
  `unsupported`.
  An error response never becomes a model file.
- **Token**: gated repos need a read token in the Keychain item that `ModelDownloads`, `GGUFModelDownloader`,
  `ChunkedDownloader` and `ModelCompatibilityChecker` read, as does AuraUI's `ModelSection` through `ModelDownloads` (the MLX load path does not; service
  `dev.auralocal.remote`, account `download.huggingface`): `KeychainDownloadAuth.saveHuggingFaceToken(_:)`,
  `deleteHuggingFaceToken()`, `hasHuggingFaceToken`. It is sent to `huggingface.co` / `hf.co` only, never to the
  CDN a file redirects to. On iOS the item belongs to the app that saved it.
- **Limits**: one download at a time per model (do not run it and `GGUFModelDownloader` on the same model at
  once); no background `URLSession`, so a transfer stops when the app is suspended and resumes from the partial
  file later; files live in Caches, which iOS may purge when storage is low.

### Keeping downloads across launches

A model built from a search result (`CatalogEntry.decodedModel()`) is not in the catalog, so the pickers
(`Model.textModels`, `Model.visionModels`, `Model.ggufModels`) list it only after `ModelRegistry.shared.register(_:)`.
`DownloadRecords` keeps the sized `Model` in `UserDefaults` (key `dev.auralocal.downloadRecords`) so it can be
registered again at launch:

```swift
DownloadRecords.standard.remember(model)          // when the download starts, so a partial resumes sized
_ = try await ModelDownloads.download(model) { _, _ in }
DownloadRecords.standard.registerIfDownloaded(model)

// At launch, before the first view reads the pickers:
DownloadRecords.standard.registerDownloaded()

// Delete: unload, remove the files, then forget the record and unregister.
ModelManager.shared.evict(model)
try ModelDownloads.delete(model)
DownloadRecords.standard.remove(model)
```

- Records are keyed by file (`gguf:<repo>/<file>`, `mlx:<repo>`), not by id: recording the same file under
  another id replaces the old record.
- `registerDownloaded()` registers only what is fully on disk. It drops a record when neither the file nor a
  `.partial` is left (Caches purged) and keeps it while a partial exists, so Resume stays sized.
- A record for a file a registry entry already points at (same repo, format and GGUF filename) is not registered:
  that entry loads the same file, and a second id would list it twice.
- `ModelRegistry.unregister(id:)` removes only models added with `register(_:)`; bundled and remote catalog
  entries stay. `DownloadRecords.remove(_:)` calls it.
- Record only sized models. A `Model.fromURL(_:displayName:)` model has size 0, which the load-time memory guard
  always rates as fitting.
- AuraUI's `ModelDownloadCenter` does all of this for the browser views: it records downloads started from a
  search result and recorded files it resumes, never a catalog or unsized model. The Model Finder's old
  `downloadRecords` key is copied over once.

Limits: weights live in Caches, which iOS may purge (the record is then dropped at the next launch); no background
`URLSession`; MLX can't run on the Simulator; on the Simulator "this device" is the Mac's memory; the extra EOS
tokens some MLX catalog entries get (`MLXBackend.configuration(for:)`) are keyed by catalog id, so an MLX repo
added from search does not get them; a GGUF repo's `mmproj` projector is ignored, so GGUF repos always load as
text.

---

## Model Collections

```swift
// All text-purpose models (MLX + GGUF)
let allText = Model.textModels

// Only MLX models
let mlxOnly = Model.mlxModels

// Only GGUF models
let ggufOnly = Model.ggufModels

// All models that can run on this device
let runnable = Model.runnableModels  // filtered by HardwareAnalyzer

// Models recommended for macOS (15 GB+)
let macRecommended = Model.allCases.filter { $0.isMacOSRecommended }
```

---

## Uncensored / Abliterated Models

AuraLocal ships a curated set of uncensored and abliterated models for use cases that require unrestricted output — creative writing, security research, adult content platforms, or any context where the developer controls the system prompt entirely.

{: .warning }
> These models have alignment data removed. You are responsible for appropriate use within your app. AuraLocal does not endorse harmful or illegal use of model output.

### What the labels mean

| Label | Technique | Notes |
|-------|-----------|-------|
| **Abliterated** | Refusal direction subtracted from weights post-training (mlabonne method). No fine-tuning needed. | Fast to produce, widely available for any base model. |
| **Josiefied** | Abliteration + additional DPO fine-tune by Goekdeniz-Guelmez. | Stronger uncensoring than abliteration alone. |
| **Dolphin** | Training dataset filtered to remove alignment/bias data (Eric Hartford; Hugging Face org `dphn`, formerly `cognitivecomputations`). | Model follows the system prompt without imposing ethics. User controls the tone. |
| **Uncensored fine-tune** | Fine-tuned on a no-refusal dataset. | Behavior depends on the training data quality. |

---

### Uncensored Text Models — MLX

| Model case | Display name | Size | Min RAM | Method |
|-----------|-------------|------|---------|--------|
| `.dolphin_qwen2_1_5b` | Dolphin 2.9 Qwen2 1.5B | ~828 MB | 4 GB | Dolphin dataset |
| `.josiefied_qwen3_1_7b` ⭐ | Josiefied Qwen3 1.7B | ~923 MB | 4 GB | Abliterated + Josiefied |
| `.josiefied_qwen3_4b` | Josiefied Qwen3 4B | ~2.1 GB | 6 GB | Abliterated + Josiefied |
| `.josiefied_qwen3_8b` | Josiefied Qwen3 8B | ~4.3 GB | 12 GB (iPhone 17 Pro) | Abliterated + Josiefied |

⭐ Recommended starting point — best quality-to-size ratio for iPhone

`qwen38_27b_uncensored_mlx` ("Qwen3.8 27B Uncensored (MLX 4-bit)", ~15.0 GB, a vision model, Mac only) is also
flagged uncensored, so `Model.uncensoredModels` returns it. It has no static member; see
[Catalog entries without a static member](#catalog-entries-without-a-static-member).

```swift
let llm = try await AuraLocal.text(.josiefied_qwen3_1_7b)
for try await token in llm.stream("Write a story with no restrictions.") {
    print(token, terminator: "")
}
```

---

### Uncensored Text Models — GGUF

| Model case | Display name | File size | Min RAM (iOS) | Method |
|-----------|-------------|-----------|--------------|--------|
| `.dolphin3_qwen25_1_5b_gguf` | Dolphin 3.0 Qwen2.5 1.5B (GGUF) | ~940 MB | 4 GB | Dolphin dataset |
| `.llama32_3b_uncensored_gguf` | Llama 3.2 3B Uncensored (GGUF) | ~2.1 GB | 4 GB (streaming), 6 GB full load | Uncensored fine-tune |
| `.dolphin3_qwen25_3b_gguf` | Dolphin 3.0 Qwen2.5 3B (GGUF) | ~1.8 GB | 4 GB (streaming), 6 GB full load | Dolphin dataset |
| `.dolphin3_llama31_8b_gguf` ⭐ | Dolphin 3.0 Llama 3.1 8B (GGUF) | ~4.6 GB | 4 GB (streaming) | Dolphin dataset |
| `.llama31_8b_abliterated_gguf` | Llama 3.1 8B Abliterated (GGUF) | ~4.6 GB | 4 GB (streaming) | Abliteration |

⭐ Most downloaded uncensored GGUF — 37k+ downloads/month on HuggingFace

```swift
// GGUF uncensored — backend auto-selected based on device RAM
let llm = try await ModelManager.shared.load(.dolphin3_llama31_8b_gguf)
let reply = try await llm.chat("No restrictions. Answer anything.")
```

---

### Filtering uncensored models in code

```swift
// All uncensored / abliterated models
let uncensored = Model.uncensoredModels

// Check if a specific model is uncensored
if Model.josiefied_qwen3_1_7b.isUncensored {
    print("No refusal training.")
}

// Runnable uncensored models on this device
let runnableUncensored = Model.uncensoredModels.filter { model in
    HardwareAnalyzer.assess(model).fitLevel.isRunnable
}
```

---

## Model Cache Location

Models are cached after download and reused across app launches:

| Format | Cache path (`<Caches>` is the app's Caches directory) |
|--------|-----------|
| MLX | `<Caches>/huggingface/hub/models--<sanitized>/snapshots/main/` (loadable once `.complete` exists) |
| GGUF | `<Caches>/models/<repoID>/<filename>.gguf` |

An unfinished file sits next to its destination as `<file>.partial` until its size matches; see
[Downloading without loading](#downloading-without-loading).

```swift
// Check if a model is already downloaded
if Model.qwen3_1_7b.isDownloaded {
    print("Qwen3 1.7B is cached")   // MLX: Caches/huggingface/hub/models--…/snapshots/main
}
let gguf = Model.llama3_1_8b_gguf
if gguf.isDownloaded, let file = gguf.resolvedFileURL {
    print("Cached at: \(file.path)")   // GGUF: Caches/models/<repoID>/<file>.gguf
}
```
