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
print(result.estimatedRuntimeMemoryGB) // ~0.65 GB in streaming mode

// Get all runnable models sorted by fit level
let compatible = HardwareAnalyzer.compatibleModels()
for (model, result) in compatible {
    print("\(model.displayName): \(result.fitLevel.label)")
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
| MLX | `~/Library/Caches/models/<org>/<repo>/` |
| GGUF | `~/Library/Caches/gguf/<filename>.gguf` |

```swift
// Check if a model is already downloaded
if Model.qwen3_1_7b.isDownloaded {
    print("Cached at: \(Model.qwen3_1_7b.cacheDirectory.path)")
}
```
