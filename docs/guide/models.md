---
layout: docs
title: Model Catalog
parent: Guide
nav_order: 3
description: "All models supported by AuraLocal — MLX small models and GGUF large models for iOS and macOS."
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

---

## Text Models — MLX

Fast, GPU-accelerated text generation. Downloads from `mlx-community` on HuggingFace.

| Model case | Display name | Size | HuggingFace repo |
|-----------|-------------|------|-----------------|
| `.qwen3_0_6b` | Qwen3 0.6B | ~400 MB | `mlx-community/Qwen3-0.6B-4bit` |
| `.qwen3_1_7b` ⭐ | Qwen3 1.7B | ~1.0 GB | `mlx-community/Qwen3-1.7B-4bit` |
| `.qwen3_4b` | Qwen3 4B | ~2.5 GB | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| `.gemma3_1b` | Gemma 3 1B | ~700 MB | `mlx-community/gemma-3-1b-it-4bit` |
| `.phi3_5_mini` | Phi-3.5 Mini | ~2.2 GB | `mlx-community/Phi-3.5-mini-instruct-4bit` |
| `.llama3_2_1b` | Llama 3.2 1B | ~700 MB | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| `.llama3_2_3b` | Llama 3.2 3B | ~1.8 GB | `mlx-community/Llama-3.2-3B-Instruct-4bit` |

⭐ Recommended default

```swift
let llm = try await AuraLocal.text(.qwen3_1_7b)
```

---

## Vision Models — MLX

Multimodal image + text models. Pass a `UIImage` or `NSImage` for analysis.

| Model case | Display name | Size | Best for |
|-----------|-------------|------|---------|
| `.qwen35_0_8b` ⭐ | Qwen3.5 VL 0.8B | ~625 MB | Default, iPhone |
| `.qwen35_2b` | Qwen3.5 VL 2B | ~1.7 GB | Higher accuracy, iPad |
| `.smolvlm_500m` | SmolVLM 500M | ~1.0 GB | Minimum memory footprint |
| `.smolvlm_2b` | SmolVLM 2B | ~1.5 GB | SmolVLM, balanced |

```swift
let vlm = try await AuraLocal.vision(.qwen35_0_8b)
let description = try await vlm.analyze("What's in this photo?", image: photo)
```

---

## Specialized Vision Models — MLX

Optimized for structured document and receipt extraction.

| Model case | Display name | Size | Output format |
|-----------|-------------|------|--------------|
| `.fastVLM_0_5b_fp16` ⭐ | FastVLM 0.5B FP16 | ~1.25 GB | JSON |
| `.fastVLM_1_5b_int8` | FastVLM 1.5B INT8 | ~800 MB | JSON |
| `.graniteDocling_258m` | Granite Docling 258M | ~631 MB | DocTags → Markdown |
| `.graniteVision_3_3` | Granite Vision 3.3 | ~1.2 GB | Plain text |

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

Use `ModelManager.shared.load(_:)` which handles the download automatically.

| Model case | Display name | File size | Min RAM (iOS) | Min RAM (macOS) |
|-----------|-------------|-----------|--------------|----------------|
| `.llama3_1_8b_gguf` ⭐ | Llama 3.1 8B GGUF | ~4.7 GB | 6 GB (streaming) | 8 GB |
| `.qwen2_5_7b_gguf` | Qwen 2.5 7B GGUF | ~4.4 GB | 6 GB (streaming) | 8 GB |
| `.mistral_7b_gguf` | Mistral 7B GGUF | ~4.1 GB | 6 GB (streaming) | 8 GB |
| `.phi3_medium_gguf` | Phi-3 Medium GGUF | ~8.0 GB | 8 GB (streaming) | 16 GB |
| `.gemma2_9b_gguf` | Gemma 2 9B GGUF | ~5.4 GB | 8 GB (streaming) | 12 GB |
| `.qwen2_5_32b_gguf` | Qwen 2.5 32B GGUF | ~20 GB | Not viable | 48 GB |
| `.llama3_1_70b_gguf` | Llama 3.1 70B GGUF | ~40 GB | Not viable | 80 GB |

{: .note }
> Models marked "Not viable" on iOS exceed the layer-streaming budget even at minimum context. They run on high-end Macs (Mac Studio, Mac Pro).

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

## Hardware Compatibility Check

```swift
import AuraCore

// Check a specific model on the current device
let result = HardwareAnalyzer.assess(.llama3_1_8b_gguf)
print(result.fitLevel.label)       // "Streaming", "Good", "Too Large", etc.
print(result.fitLevel.isRunnable)  // true/false
print(result.model.estimatedRuntimeMemoryGB)   // ~5 GB full-load (monolithic)
print(result.model.estimatedStreamingMemoryGB) // ~0.65 GB in layer-streaming mode

// Get all runnable models sorted by fit level
let compatible = HardwareAnalyzer.compatibleModels()
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

---

## Finding compatible models

A download that *fits* is not a download that *runs*. `ModelCompatibilityChecker` answers the second question
for any Hugging Face repo: will AuraLocal, with the runtimes it pins — **mlx-swift-lm 3.31.3** (MLX) and
**llama.cpp b8851** (GGUF, via LocalLLMClient 0.5.0) — load and run it on a given device, and if not, exactly why.

It reads the repo listing (`/api/models/{id}?blobs=true`), `config.json`, `model.safetensors.index.json` and, with
HTTP Range requests, the first 4 MiB of one GGUF file or the header of a single safetensors file. Weights are
never downloaded. Network errors, 401/403 and 404 come back as findings — the verdict is `unknown` when they hide
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
| `generative` | pipeline `fill-mask`, `feature-extraction`, `zero-shot-image-classification`…, a `*ForMaskedLM`-style head, or an encoder `model_type` | the runtimes chat with text generators only |
| `mlx.model-type` | top-level `model_type` is in neither `LLMTypeRegistry` (54 types) nor `VLMTypeRegistry` (17) | the factories dispatch on the top-level key; `text_config.model_type` does not count |
| `mlx.weight-prefixes` | a `qwen3_5` / `qwen3_5_moe` tensor outside `language_model`, `model`, `lm_head`, `vision_tower`, `mtp` | the pinned `sanitize` maps only those; anything else fails `update(parameters:verify: [.all])` |
| `mlx.extra-safetensors` | a top-level `*.safetensors` missing from the weight map | AuraLocal's downloader fetches every top-level file and `loadWeights` merges every `*.safetensors` |
| `mlx.rope` | a rope type `RoPEUtils.initializeRope` does not implement, or a `longrope` without its three fields | `initializeRope` calls `fatalError` — the app crashes |
| `mlx.category` | a VLM-only type without vision tensors | vision needs a registered VLM type **and** vision weights; otherwise the model loads as text |
| `gguf.architecture` | `general.architecture` not in b8851's `LLM_ARCH_NAMES`, added later (`qwen4exp` → b10660), or an encoder / diffusion architecture | llama.cpp rejects unknown architectures |
| `gguf.qwen35-nextn` | `qwen35` / `qwen35moe` with `nextn_predict_layers` > 0 | b8851 marks recurrent layers arithmetically and demands `ssm_*` tensors the MTP block lacks; fixed in b9495 |
| `gguf.shards` | every quant is split (`-0000N-of-0000M`); a caveat when only some are | the GGUF path loads one file |
| `fit` | the weights (MLX) or every single-file quant (GGUF) are too large for the device | `HardwareAnalyzer.assess` against the device budget |
| `imagegen` | a text-to-image pipeline on an iPhone preset (a caveat on a Mac: mflux support is not verified) | AuraImageGen drives mflux on macOS only |
| `license` | never blocks: flags gated repos, a missing license, non-commercial (`cc-by-nc*`) and custom (`other`) licenses by name | |

`mlx.config`, `mlx.tokenizer`, `gguf.header`, `gguf.projector` and `context` add caveats and context (quantization,
unreadable files, an ignored mmproj, trained contexts below 32768 tokens) but never block.

Fit uses the existing `HardwareAnalyzer` math: weights + runtime overhead + a GQA-aware KV cache at 2048 tokens,
with `kvHeads` / `headDim` from `config.json` or the GGUF header. Every GGUF quant gets its own fit. Sizes are
binary: catalog `approximateSizeMB` is MiB, because `HardwareAnalyzer` divides it by 1024 to get GiB.

The pinned tables live in `Sources/AuraCore/ModelCompatibility/PinnedRuntimes.swift`, with the commands that
regenerate them; `ModelCompatibilityTests` fails when the MLX sets drift from the `mlx-swift-lm` checkout.

### Known incompatibilities

Verified on 2026-09-29 with `aura models check <repo> --device mac-32gb` (M1 Pro, 20.0 GB measured budget):

| Repository | Verdict | Deciding finding |
|------------|---------|------------------|
| `ukisai/Swift-1.5-4bit-MLX` | Won't run | `visual.*` (501 tensors) is not mapped by the qwen3_5 sanitize |
| `ukisai/Swift-1.5-3bit-MLX-TextOnly` | Runs, with caveats | text: only `language_model.*`; 10.96 GB of weights, Excellent · 12.0 of 20.0 GB; custom license |
| `mlx-community/Qwen3.5-27B-4bit` | Runs | vision: `language_model` + `vision_tower`; Good · 16.0 of 20.0 GB |
| `ukisai/Swift-1.5-Qwen3.8-27B-GGUF` | Won't run | `qwen35`, `block_count` 65 with 1 NextN layer: needs llama.cpp b9495 (17 of 22 quants would otherwise load fully) |
| `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF` | Won't run | `qwen4exp` needs b10660; every quant is split; the smallest (IQ1_S) is 65.3 GB |
| `Edge0/Edge0-35B-A3B-preview` | Won't run | `lora_edge0_35b.safetensors` (620 tensors) and `prerouter_edge0_35b.safetensors` (99) are outside the weight map |
| `medicalai/ClinicalBERT` | Won't run | `fill-mask`, `DistilBertForMaskedLM`; only `pytorch_model.bin` |
| `google/medsiglip-448` | Won't run | SigLIP, `zero-shot-image-classification`; not an MLX conversion; gated, custom license |
| `stanford-crfm/BioMedLM` | Won't run | `gpt2` is not registered; only `.bin` weights; trained context 1024 |
| `mlx-community/MiniCPM5-1B-4bit` | Runs | `model_type` `llama`; 0.57 GB — fits the iPhone 4 GB class |
| `mlx-community/MiniCPM4.1-8B-4bit` | Runs | `minicpm`; its `longrope` carries the fields RoPEUtils needs |
| `mlx-community/MiniCPM3-4B-4bit` | Won't run | `minicpm3` is not registered ("v3 uses a different architecture", MLXLLM README) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | Runs, with caveats | `qwen3_5_moe`, Marginal · 18.8 of 20.0 GB; no license declared |
| `Qwen/Qwen2.5-7B-Instruct-GGUF` | Runs, with caveats | Q4_K_M and 6 other quants are split; Q2_K and Q3_K_M are single files |
| `Qwen/Qwen3.8-Flash-Next` | Won't run | unconverted safetensors (335 GB); `qwen4_exp` is not registered |

### Device presets

Only this device and the 32 GB Mac are measured; every other budget is an estimate, and `DevicePreset.source`
says where it comes from (the CLI and the app show it).

| Preset (`--device`) | Budget | Source |
|---------------------|--------|--------|
| `this-device` | measured now | `HardwareProfile.current()` — memory the process can still allocate |
| `iphone-4gb` | ≈2.0 GB | estimate: third-party jetsam measurement, ActiveHard 2098 MB on an iPhone 12 |
| `iphone-6gb` | ≈3.0 GB | estimate: this guide's [memory budgets]({{ '/guide/memory' | relative_url }}), increased-memory-limit entitlement |
| `iphone-8gb` | ≈4.5 GB | estimate: same source |
| `iphone-17-pro` | ≈6.4 GB | estimate: third-party report, only with both memory entitlements |
| `mac-16gb` | ≈10.7 GB | estimate: default Metal working set ≈ 2/3 of RAM up to 36 GB |
| `mac-32gb` | 20.0 GB | measured on an M1 Pro: `iogpu.wired_limit_mb` 20480, `recommendedMaxWorkingSetSize` 20480 MiB |
| `mac-64gb` | ≈48 GB | estimate: default Metal working set ≈ 3/4 of RAM above 36 GB |

### From the command line

```
aura models search "<query>" [--format mlx|gguf] [--device <preset>] [--limit N]
aura models check <owner/repo | URL> [--device <preset>] [--json] [--entry]
aura models devices
```

`search` checks every hit (four at a time) and prints a verdict table; `check` prints every finding, the
per-quant fit table and the `models.json` entry; `--entry` prints only the entry, `--json` the whole report.

### The example app

`Examples/ModelFinder` is a SwiftUI app for iOS 18 and macOS 15 built on the same checker: search with format,
sort and device filters, a verdict badge per result (checked lazily as rows appear), and a detail view with the
findings, the per-quant fit table, **Copy catalog entry** and **Open on Hugging Face**. Its README explains how to
generate and open the project.

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
| **Dolphin** | Training dataset filtered to remove alignment/bias data (Eric Hartford / cognitivecomputations). | Model follows the system prompt without imposing ethics. User controls the tone. |
| **Uncensored fine-tune** | Fine-tuned on a no-refusal dataset. | Behavior depends on the training data quality. |

---

### Uncensored Text Models — MLX

| Model case | Display name | Size | Min RAM | Method |
|-----------|-------------|------|---------|--------|
| `.dolphin_qwen2_1_5b` | Dolphin 2.9 Qwen2 1.5B | ~870 MB | 4 GB | Dolphin dataset |
| `.josiefied_qwen3_1_7b` ⭐ | Josiefied Qwen3 1.7B | ~950 MB | 4 GB | Abliterated + Josiefied |
| `.josiefied_qwen3_4b` | Josiefied Qwen3 4B | ~2.3 GB | 6 GB | Abliterated + Josiefied |
| `.josiefied_qwen3_8b` | Josiefied Qwen3 8B | ~4.6 GB | 8 GB | Abliterated + Josiefied |

⭐ Recommended starting point — best quality-to-size ratio for iPhone

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
| `.dolphin3_qwen25_1_5b_gguf` | Dolphin 3.0 Qwen2.5 1.5B | ~990 MB | 4 GB | Dolphin dataset |
| `.llama32_3b_uncensored_gguf` | Llama 3.2 3B Uncensored | ~2.2 GB | 6 GB | Uncensored fine-tune |
| `.dolphin3_qwen25_3b_gguf` | Dolphin 3.0 Qwen2.5 3B | ~1.9 GB | 6 GB | Dolphin dataset |
| `.dolphin3_llama31_8b_gguf` ⭐ | Dolphin 3.0 Llama 3.1 8B | ~4.9 GB | 6 GB (streaming) | Dolphin dataset |
| `.llama31_8b_abliterated_gguf` | Llama 3.1 8B Abliterated | ~4.9 GB | 6 GB (streaming) | Abliteration |

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

| Format | Cache path |
|--------|-----------|
| MLX | `~/Library/Caches/huggingface/hub/models--<sanitized>/snapshots/main/` |
| GGUF | `~/Library/Caches/models/<repoID>/<filename>.gguf` |

```swift
// Check if a model is already downloaded
if Model.qwen3_1_7b.isDownloaded {
    print("Cached at: \(Model.qwen3_1_7b.cacheDirectory.path)")
}
```
