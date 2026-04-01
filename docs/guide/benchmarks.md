---
layout: docs
title: Benchmarks
parent: Guide
nav_order: 8
description: "AuraLocal performance benchmarks — tokens per second, RAM usage, and first-token latency on iPhone, iPad, and Mac."
---

# Benchmarks
{: .no_toc }

Performance measurements for on-device LLM inference across Apple devices.

{: .note }
> Benchmarks measured with AuraLocal v3.0, default Q4_K_M quantization for GGUF and 4-bit for MLX. Results vary by prompt complexity and thermal state.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Text Generation — Tokens/Second

### MLX Backend (Apple Silicon GPU)

| Model | iPhone 15 | iPhone 15 Pro | M1 iPad Pro | M3 MacBook Air | M3 Pro Mac |
|-------|-----------|---------------|-------------|----------------|------------|
| Qwen3 0.6B | ~40 tok/s | ~50 tok/s | ~55 tok/s | ~70 tok/s | ~90 tok/s |
| Qwen3 1.7B | ~22 tok/s | ~30 tok/s | ~35 tok/s | ~45 tok/s | ~60 tok/s |
| Qwen3 4B | ~10 tok/s | ~14 tok/s | ~18 tok/s | ~25 tok/s | ~38 tok/s |
| Llama 3.2 3B | ~18 tok/s | ~24 tok/s | ~28 tok/s | ~40 tok/s | ~55 tok/s |

### llama.cpp Backend — Full Load (macOS)

| Model | M1 Pro 16 GB | M2 Max 32 GB | M3 Pro 36 GB | M2 Ultra 96 GB |
|-------|-------------|-------------|-------------|----------------|
| Llama 3.1 8B | ~12 tok/s | ~18 tok/s | ~22 tok/s | ~35 tok/s |
| Mistral 7B | ~14 tok/s | ~20 tok/s | ~25 tok/s | ~38 tok/s |
| Qwen 2.5 7B | ~13 tok/s | ~19 tok/s | ~23 tok/s | ~36 tok/s |
| Llama 3.1 70B | — | — | ~4 tok/s | ~12 tok/s |

### Layer-Streaming Backend (iOS)

| Model | iPhone 15 (6 GB) | iPhone 15 Pro (8 GB) | iPad Pro M2 (8 GB) |
|-------|-----------------|---------------------|-------------------|
| Llama 3.1 8B | ~2–3 tok/s | ~3–4 tok/s | ~4–5 tok/s |
| Mistral 7B | ~3–4 tok/s | ~4–5 tok/s | ~5–6 tok/s |
| Qwen 2.5 7B | ~2–3 tok/s | ~3–4 tok/s | ~4–5 tok/s |

{: .note }
> Layer-streaming runs CPU-only on iOS. Speed is lower than macOS GPU inference but enables models that would otherwise be impossible to run on the device at all.

---

## Peak RAM Usage

### MLX Models

| Model | Peak RAM |
|-------|---------|
| Qwen3 0.6B | ~350 MB |
| Qwen3 1.7B | ~900 MB |
| Qwen3 4B | ~2.4 GB |
| Llama 3.2 3B | ~1.7 GB |

### GGUF — Full Load (macOS)

| Model | Peak RAM |
|-------|---------|
| Llama 3.1 8B | ~4.9 GB |
| Mistral 7B | ~4.3 GB |
| Qwen 2.5 7B | ~4.6 GB |

### GGUF — Layer-Streaming (iOS)

| Model | Peak RAM (any device) |
|-------|----------------------|
| Llama 3.1 8B | ~650–750 MB |
| Mistral 7B | ~600–700 MB |
| Qwen 2.5 7B | ~620–720 MB |

The streaming RAM is nearly constant regardless of model size — the OS pages weights from disk on demand.

---

## First-Token Latency

Time from sending a prompt to receiving the first token:

| Backend | Device | First-token latency |
|---------|--------|-------------------|
| MLX | iPhone 15 Pro | 80–150 ms |
| MLX | M3 Mac | 40–80 ms |
| llama.cpp (full load) | M3 Mac | 200–400 ms |
| Layer-streaming (cold) | iPhone 15 | 800–1500 ms |
| Layer-streaming (warm) | iPhone 15 | 300–600 ms |

"Warm" = model weights partially in OS page cache from a recent session.

---

## Model Load Time

Time from `ModelManager.shared.load()` to first inference (after download):

| Model | Cold load | Warm (cached in RAM) |
|-------|-----------|---------------------|
| Qwen3 1.7B (MLX) | ~2 s | ~0 ms |
| Qwen3 4B (MLX) | ~4 s | ~0 ms |
| Llama 3.1 8B (GGUF, full) | ~8–12 s | ~0 ms |
| Llama 3.1 8B (GGUF, streaming) | ~1–2 s | ~0 ms |

GGUF layer-streaming loads faster because only the first few layers are paged in at startup.

---

## How to Reproduce

```swift
import AuraCore

let start = Date()
let llm = try await ModelManager.shared.load(.qwen3_1_7b)
let loadTime = Date().timeIntervalSince(start)
print("Load: \(loadTime)s")

var firstToken = true
var firstTokenTime: TimeInterval = 0
let genStart = Date()

for try await token in llm.stream("Write a haiku about Swift programming.") {
    if firstToken {
        firstTokenTime = Date().timeIntervalSince(genStart)
        firstToken = false
    }
    print(token, terminator: "")
}

let totalTime = Date().timeIntervalSince(genStart)
print("\nFirst token: \(firstTokenTime)s | Total: \(totalTime)s")
```
