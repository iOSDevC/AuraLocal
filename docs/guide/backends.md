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

| Backend | Engine | Formats | Platforms | Peak RAM |
|---------|--------|---------|-----------|---------|
| `MLXBackend` | mlx-swift | `.mlx` | iOS + macOS | 500 MB – 3 GB |
| `LlamaCppBackend` | llama.cpp | `.gguf` | macOS (primary) | 4–40 GB |
| `LayerStreamingBackend` | llama.cpp + mmap | `.gguf` | iOS (memory constrained) | ≤750 MB |

All three implement the same `InferenceBackend` protocol — your code looks identical regardless of which backend is running.

---

## Automatic Selection

`BackendRouter` selects a backend when `AuraEngine` is initialized:

```swift
// This happens automatically inside AuraLocal / ModelManager
let backend = BackendRouter.selectBackend(
    for: model,
    temperature: temperature,
    profile: HardwareProfile.current()
)
```

The routing logic:

```
model.format == .mlx
    → MLXBackend (always)

model.format == .gguf
    fitLevel == .excellent / .good / .marginal
        → LlamaCppBackend  (full model loaded, Metal GPU offload)
    fitLevel == .streamingRequired
        → LayerStreamingBackend  (mmap, CPU-only on iOS)
    fitLevel == .tooLarge
        → LlamaCppBackend  (will fail at load with a clear error)
```

To check which backend a model will use on the current device:

```swift
import AuraCore

let kind = BackendRouter.recommendedBackend(for: .llama3_1_8b_gguf)
// → .llamaCpp   on Mac with 16 GB RAM
// → .layerStreaming  on iPhone 15 with 6 GB RAM
```

---

## MLX Backend

Used for all `.mlx` format models (0.5B – 4B parameters).

**Characteristics:**
- Runs entirely on the Apple Silicon GPU via Metal
- Model weights loaded into unified memory as MLX arrays
- Supports text generation and vision (multimodal) models
- GPU cache scaled by available RAM: 128 MB (low) → 512 MB (high)
- 20–45 tokens/second on iPhone 15 Pro / M-series Mac

**When it's used:** Any model case without `_gguf` suffix — Qwen3, Llama 3.2, Gemma 3, Phi-3.5, SmolVLM, FastVLM, Granite Docling.

---

## LlamaCpp Backend

Used for `.gguf` models when the full file fits in available RAM.

**Characteristics:**
- Uses `LocalLLMClient` → llama.cpp under the hood
- Full Metal GPU offload on macOS (all layers on GPU)
- Context window: 8192 tokens (macOS), 2048 tokens (iOS with enough RAM)
- 8 threads on macOS, 4 threads on iOS
- 8–20 tokens/second on M-series Mac depending on model size

**When it's used:** GGUF models on macOS where the model file fits in RAM (typically 8–40 GB Macs for 7B–70B models).

```swift
// On a Mac with 32 GB RAM, Llama 3.1 8B uses this backend
let llm = try await ModelManager.shared.load(.llama3_1_8b_gguf)
```

---

## Layer-Streaming Backend

Used for `.gguf` models on iOS when the file is too large for monolithic loading.

**How it works:**

Traditional LLM loading reads the entire model into RAM before inference. For a 4.7 GB Llama 3.1 8B file on a 6 GB iPhone (only ~1.5 GB free for apps), this is impossible without crashing.

Layer-streaming uses **mmap** — the OS maps the file into virtual address space but only loads pages that are actually accessed. Inference processes one transformer layer at a time:

```
Disk (GGUF file)
  → OS page cache (mmap, demand-paged)
    → Current layer weights (~130 MB)
    → Next layer prefetch (~130 MB)
    → Embedding table (~130 MB)
    → KV cache (~32 MB at 1024 ctx, GQA)
    → Activations + overhead (~80 MB)
─────────────────────────────────────
Total peak: ≤750 MB  (safe under iOS jetsam ~1.5 GB limit)
```

**Characteristics:**
- CPU-only on iOS (no Metal GPU offload — avoids locking large GPU buffers)
- Context window: adaptive, 512–1024 tokens based on available memory
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
| KV cache (1024 ctx, GQA) | ~32 MB |
| Activations + overhead | ~80 MB |
| Safety margin | ~250 MB |
| **Total app peak** | **≤750 MB** |

### GQA-Aware KV Cache

Modern models use Grouped-Query Attention (GQA), which dramatically reduces KV cache size compared to naive full-attention estimates:

| Model | Q heads | KV heads | KV cache (2048 ctx, FP16) |
|-------|---------|---------|--------------------------|
| Llama 3.1 8B | 32 | 8 | ~256 MB |
| Qwen 2.5 7B | 28 | 4 | ~115 MB |
| Mistral 7B | 32 | 8 | ~230 MB |
| Gemma 2 9B | 16 | 8 | ~180 MB |

`HardwareAnalyzer.estimatedKVCacheGB(model:contextLength:)` uses these GQA-corrected formulas, not naive estimates.

---

## macOS vs iOS Differences

| | macOS | iOS |
|---|---|---|
| GGUF backend | `LlamaCppBackend` | `LayerStreamingBackend` |
| GPU layers | Up to all layers | 0 (CPU only) |
| Context window | 8192 tokens | 512–1024 tokens |
| Threads | 8 | 4 |
| Max viable model | 70B (80 GB Mac) | 13B (streaming) |
| Background inference | Continues | Paused by `BackgroundLifecycle` |

---

## Background Lifecycle (iOS)

On iOS, active Metal GPU buffers can cause jetsam (OS memory termination) when the app is backgrounded. `BackgroundLifecycle` automatically pauses generation:

```swift
// Automatically managed — no action needed in most cases
BackgroundLifecycle.shared.isPaused  // true when app is in background

// Optional: evict models aggressively on background to free RAM
BackgroundLifecycle.shared.aggressiveMemorySaving = true
```

This is a no-op on macOS.
