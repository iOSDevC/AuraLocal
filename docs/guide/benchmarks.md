---
layout: docs
title: Benchmarks
parent: Guide
nav_order: 8
description: "Tonamox performance benchmarks — tokens per second, RAM usage, and first-token latency on iPhone, iPad, and Mac."
---

# Benchmarks
{: .no_toc }

Illustrative performance figures for on-device LLM inference across Apple devices.

{: .note }
> The per-device tables below are carried over from the first version of these docs (2026-03-31) and have not been re-measured on the current release, so treat them as illustrative. They assume Q4_K_M quantization for GGUF and 4-bit for MLX; results vary by prompt complexity and thermal state. Measured figures are in [Measured on an M1 Pro](#measured-on-an-m1-pro).

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Measured on an M1 Pro

One MacBook Pro M1 Pro, 32 GB, 200 GB/s spec bandwidth. Weights are the file sizes on Hugging Face.

| Model | Runtime | Weights | Decode | Source |
|---|---|---|---|---|
| Gemma 4 12B (11.91 B params) Q8_0 | llama.cpp build 9660, `llama-bench` tg64, one repetition | 11.78 GiB | 12.12 tok/s (prefill pp512: 175.69) | own run, 2026-07 |
| Qwen3-14B Q4_K_M | llama.cpp build 9660, `llama-bench` tg64 | 8.38 GiB | 13.9 tok/s | [Distributed inference]({{ '/guide/distributed' | relative_url }}#measured-results) |
| Qwen3-14B 4-bit (`mlx-community`) | mlx-lm 0.31.3, greedy | 7.74 GiB | 19.0–19.8 tok/s on a clean run | same |
| Qwen3-0.6B 4-bit (`mlx-community`) | mlx-lm 0.31.3, alone on the GPU | 0.31 GiB | 218 tok/s | same |

The 14B MLX baseline moves between sessions: a run with a download going in parallel measured 16.6–18.5 tok/s, with one outlier at 12.1.

## Speed estimate

`HardwareAnalyzer` predicts decode speed before a download as
`1 / (weights ÷ (bandwidth × 0.865) + 0.0066 s)`: each token streams every weight once, plus a fixed per-token cost
that does not shrink with bandwidth. A bandwidth-only formula reads that fixed cost as poor efficiency on the
fastest chips and overestimates them (M2 Ultra, LLaMA 2 7B Q4_0: 169 predicted, 94 measured).

The two constants are a least-squares fit to the M1 and M2 rows of
[llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) (third-party, LLaMA 2 7B,
build 8e672ef). `EstimatorAccuracyTests` fails when a relative error exceeds 25 %:

| Gate | Points | Error |
|---|---|---|
| Our measurements (table above), each point | 4, two of them known misses | 10 % and 0.4 %; the known misses 31 % and 45 % |
| #4167 M3 and M4 rows, not used in the fit, median per chip tier | 5–12 per tier | 3–10 % |
| #4167 M1–M4 rows, median per chip tier | 14–24 per tier | 3–7 % |

The known misses on our own Mac are expected failures: a third miss fails the test, and so does either of them
coming within 25 %. Qwen3-0.6B is predicted at 119 tok/s (measured 218): the fixed 6.6 ms per token caps every
estimate at 1 / 0.0066 ≈ 151 tok/s, while MLX took 4.6 ms per token there. Qwen3-14B Q4_K_M is predicted at 18.2
(measured 13.9). The #4167 rows for M5 and newer are excluded: `HardwareProfile` has no bandwidth for
those chips yet, so the estimate is `nil` there. Those rows, measured with a newer build (c1d0e7a), show a smaller
per-token cost: the formula underestimates M5 Max and Ultra by up to 44 % (median 31 %).

The gate sizes every point without `experts`, so it covers the dense path only. A mixture of experts with an
`experts` entry is charged only the bytes a token reads, and its estimate is an uncalibrated range (see
[Models › KV cache per layer type]({{ '/guide/models' | relative_url }}#kv-cache-per-layer-type-and-mixtures-of-experts)).

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
> Layer-streaming pages weights from disk on demand (mmap) and still offloads layers to Metal on a device. Speed is lower than full-load inference but enables models that would otherwise not fit on the device at all.

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
import Foundation
import TonamoxCore

let start = ContinuousClock.now
let llm = try await ModelManager.shared.load(.qwen3_1_7b)
print("Load: \(ContinuousClock.now - start)")

for try await token in llm.stream("Write a haiku about Swift programming.") {
    print(token, terminator: "")
}

if let stats = llm.lastGenerationStats {
    print("\n\(stats.source.label): TTFT \(stats.timeToFirstToken)s, "
        + "\(stats.generatedTokens) tokens at \(stats.decodeTokensPerSecond ?? 0) tok/s")
}
```

`lastGenerationStats` reports MLX's exact token counts, and times prefill and decode (from the first token to the
last) on every backend; on llama.cpp, prompt tokens are estimated, and with tools registered the generated count is
estimated from text length too (`source == .estimated`). A second prompt on the same MLX text model reuses the KV
cache of the first and prefills only what it does not share, so time a full prefill on a newly loaded model. See
[Generation Stats]({{ '/api/core' | relative_url }}#generation-stats).
