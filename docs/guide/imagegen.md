---
layout: docs
title: Image Generation
parent: Guide
nav_order: 8
description: "FLUX text-to-image + lora.safetensors on macOS via mflux — install, ungated models, the Swift/CLI API, and measured performance."
---

# Image Generation + LoRA (macOS)

`AuraImageGen` adds **FLUX** text-to-image with easy `lora.safetensors` loading. It runs
[mflux](https://github.com/filipstrand/mflux) (FLUX on Apple MLX) as a **subprocess**.

## Why macOS only

FLUX is a ~12B-parameter diffusion model. A Swift package cannot embed the Python runtime mflux
needs, so `AuraImageGen` locates and execs an already-installed `mflux-generate`. Consequences,
stated plainly:

- **macOS only.** Off macOS the API exists but `MFluxEngine.isAvailable == false` and `generate`
  throws `.unsupportedPlatform`. FLUX is far too large for iPhone, and the iOS Simulator has no
  Metal GPU.
- **Non-sandboxed only.** The macOS App Sandbox blocks exec-ing an external CLI, so this serves the
  `aura` CLI and non-sandboxed apps (the Example has `ENABLE_APP_SANDBOX = NO`) — not App-Store
  sandboxed apps. An in-process, sandbox-safe Swift engine (`flux.swift`) is **not** available: it
  pins `swift-transformers 0.1.x` while AuraLocal is on `1.3.x` — an unresolvable dependency conflict.

## Prerequisites

**1. Install mflux** (one-time). The Example app has an **Install mflux** button that runs this:

```bash
uv tool install mflux
```

**2. Pick a model that downloads.** The bare `schnell` / `dev` aliases point at black-forest-labs'
**gated** HuggingFace repos — they return `401` without a login and an accepted license. The easy
path is an **ungated pre-quantized** repo, which downloads with no auth:

- `dhairyashil/FLUX.1-schnell-mflux-4bit` (4-bit, ~9 GB)
- `dhairyashil/FLUX.1-schnell-mflux-8bit` (8-bit)

> Using an alias with `--quantize` downloads the **full** model (tens of GB) and quantizes on load;
> a pre-quantized repo downloads only the quantized weights.

## Swift

```swift
import AuraImageGen

let engine = MFluxEngine()
guard engine.isAvailable else { /* macOS + mflux required */ return }

let result = try await engine.generate(ImageGenRequest(
    prompt: "a red fox in snow, cinematic lighting",
    model: "dhairyashil/FLUX.1-schnell-mflux-4bit",
    baseModel: "schnell",          // required for custom repos; omit for the bare aliases
    steps: 4,
    seed: 42,
    lowRAM: false,                 // true trades speed for a smaller peak
    loras: [LoRA(url: loraURL, scale: 0.9)]
))
result.image     // PlatformImage
result.fileURL   // PNG on disk
```

Errors are typed (`ImageGenError`): `.engineNotFound(installHint:)`, `.loraNotFound`,
`.unsupportedPlatform`, `.generationFailed`, `.outputMissing`.

## CLI

```bash
aura imagegen "a red fox in snow" \
  --model dhairyashil/FLUX.1-schnell-mflux-4bit --base-model schnell \
  --lora /path/to/style.safetensors --lora-scale 0.9 \
  --steps 4 --seed 42
```

Flags map 1:1 onto mflux (`--model`, `--base-model`, `--steps`, `--seed`, `--quantize`, `--low-ram`,
`--lora-paths`, `--lora-scales`).

## Performance (measured)

M1 Pro (32 GB), schnell 4-bit, 1024×1024, 4 steps:

| Metric | Value |
|--------|-------|
| Peak memory footprint | **≈ 20 GB** (runtime peak, not the ~9 GB on-disk size) |
| Speed | **≈ 25 s/step → ~100 s** diffusion; ~2–3 min/image with model load |
| First-run download | ~9 GB (pre-quantized), network-bound |

It fits a 32 GB Mac but runs best **exclusively** — the ~20 GB peak presses the ~21.5 GB Metal wired
limit, so don't co-load an LLM. Lower resolution (512×512) is substantially faster.

## Finding a model that fits (compatibility filter)

`HuggingFaceSearch` searches HF, and `HardwareAnalyzer.fitLevel(forWeightsBytes:kind:)` says whether a
result fits **this** machine. The Example's Image tab has **Search HuggingFace…**, with a
*"Only what fits · N GB free"* toggle, gated/kind badges, and a **Use** button that fills in the model.

```swift
let hits = try await HuggingFaceSearch.search("flux schnell mflux", sort: .downloads)
for hit in hits {
    // sizes live in the repo tree, not the search payload
    guard let bytes = try await HuggingFaceRepo.weightBytes(repoURL: "https://huggingface.co/\(hit.id)",
                                                           kind: hit.kind) else { continue }
    let fit = HardwareAnalyzer.fitLevel(forWeightsBytes: bytes, kind: hit.kind)
    print(hit.id, hit.gated ? "(gated)" : "", fit.label)
}
```

**Why `kind` matters.** Judging a download by its on-disk size is misleading for image models:

| Kind | Peak RAM vs weights | Can layer-stream |
|------|--------------------|------------------|
| `.llm` | ~1.15× (weights + overhead + KV) | yes |
| `.diffusion` | **~2.2×** (text encoder + VAE + activations co-resident) | no |

The 2.2× is measured, not assumed — FLUX schnell 4-bit is 9.2 GB on disk and peaks ~20 GB. Validated
against the live repo: 8.9 GB of `.safetensors` → estimated 19.7 GB peak vs ~20 GB measured. So a 6.8 GB
FLUX correctly reads **too large** on a Mac with 13 GB free, and **good** with ~27 GB free — where naive
size-based math would have called it "excellent" both times.

`hit.gated` flags repos that need a HuggingFace login (black-forest-labs' FLUX repos are gated) so you
can pick an ungated mirror instead.

## LoRA notes

`--lora-paths` accepts **local files or HuggingFace repos**; multiple LoRAs each take a scale. mflux
also ships curated styles (`--lora-style`) and a `mflux-lora-library` / `mflux-train` toolchain for
building your own — outside `AuraImageGen`'s scope, which is loading, not training.
