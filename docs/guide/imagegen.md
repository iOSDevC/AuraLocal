---
layout: docs
title: Image Generation
parent: Guide
nav_order: 8.5
description: "Text-to-image on iPhone and Mac with Core ML Stable Diffusion, Apple Image Playground behind consent, and FLUX + lora.safetensors on macOS via mflux."
---

# Image Generation + LoRA

`AuraImageGen` has two engines and one policy:

| Path | Platforms | Where it runs | Notes |
|------|-----------|---------------|-------|
| `CoreMLDiffusionEngine` | iOS, macOS | This device | Stable Diffusion 2.1 base, 512×512, ~1.07 GB one-time download |
| Apple Image Playground (`ImagePlaygroundPolicy`) | iOS 18.4+ | On device before iOS 27; Apple's servers from iOS 27 | Apple's sheet; AuraLocal only sets its styles and options |
| `MFluxEngine` | macOS | This Mac | FLUX + `lora.safetensors` through the mflux CLI |

## On device (iPhone and Mac): Core ML Stable Diffusion

`CoreMLDiffusionEngine` runs Apple's [Core ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion)
pipeline in process. The only network traffic is the one-time model download; prompts and images stay on
the device.

```swift
import AuraImageGen

let model = CoreMLDiffusionModel.stableDiffusion21Base
if !model.isDownloaded {
    try await model.download { done, total in print(done, "of", total) }   // ~1.07 GB, no account
}

let engine = CoreMLDiffusionEngine()            // .neuralEngine, .automatic memory mode
let output = try await engine.generate(DiffusionRequest(prompt: "a lighthouse at dusk", steps: 20)) { event in
    print(event)                                // .preparingModel, .step(n, of: total), .decoding
}
output.images.first?.image                       // PlatformImage
output.images.first?.fileURL                     // PNG on disk
output.seed                                      // the seed used (random when the request had none)
await engine.unload()                            // free the pipeline; the next run reloads it
```

### The model

`CoreMLDiffusionModel.stableDiffusion21Base` is Apple's 6-bit palettized conversion of Stable Diffusion 2.1
base (`apple/coreml-stable-diffusion-2-1-base-palettized`, `split_einsum_v2/compiled`), pinned to revision
`dca5f2fda9c5143c71aded72bbb4ce39f3c2cc9c` so an upstream re-upload cannot change what runs.

- **Download.** 17 files, 1,072,887,285 bytes: the text encoder, Unet and VAE decoder (`.mlmodelc`), plus
  `vocab.json` and `merges.txt`. The VAE encoder, which only image-to-image needs, is skipped. The repo is
  ungated and no Hugging Face token is sent.
- **Storage.** `ModelDownloads.downloadFolder(repoID:revision:folder:include:auth:progress:)` writes to
  `<Caches>/huggingface/hub/models--apple--coreml-stable-diffusion-2-1-base-palettized/snapshots/main/split_einsum_v2/compiled`
  and resumes `.partial` files. iOS may purge Caches, so `generate` re-checks `isDownloaded` on every call.
  The folder records the revision its files come from, and `isDownloaded` requires the pinned one. Files from
  any other revision, finished or partial, are deleted before the next download.
  AuraUI's Downloads list shows the folder as a paused MLX snapshot of that repo; Resume is off there, Delete works.
- **Output.** 512×512 only: the Unet's input shape is fixed. No LoRA.
- **No safety checker.** The repo ships no `SafetyChecker.mlmodelc`, so nothing filters the images. An app
  that ships this needs its own moderation decision.
- **License.** CreativeML Open RAIL++-M, which restricts some uses and requires passing those restrictions on
  to your users. Read it from the [model card](https://huggingface.co/apple/coreml-stable-diffusion-2-1-base-palettized).
- **Quality.** Clearly below FLUX, SDXL or Image Playground; hands and text in images come out poorly.

### Requests, memory and cancellation

- **Validation.** `DiffusionRequest` takes `steps` 1–50, `guidanceScale` 1–20 and `imageCount` 1–4. An empty
  prompt or an out-of-range value throws `OnDeviceImageGenError.invalidRequest(field:reason:)`; nothing is
  clamped. A `nil` seed picks a random one, returned in `DiffusionOutput.seed`. The images of one run share
  that seed.
- **Memory gate.** Before loading, the engine estimates the peak as the download size × 2.2
  (`HardwareAnalyzer.peakMemoryGB(forWeightsBytes:kind: .diffusion)`, about 2.36 GB) and throws
  `.insufficientMemory` when `HardwareProfile.availableMemoryBytes()` reports less. On iPhone and iPad,
  unknown memory blocks too (`.insufficientMemory` with 0 available): iOS reports unknown when the app is
  already at its jetsam limit. On a Mac, unknown only means the kernel query failed, and it does not block.
- **Memory mode.** `.automatic` keeps the models loaded between runs (`.resident`) when at least twice the
  estimated peak is free, and otherwise loads each stage on demand (`.lowMemory`, slower). Pass `.resident` or
  `.lowMemory` to choose. In `.lowMemory`, a cancelled or failed run unloads the stage it left loaded.
- **Compute units.** `.neuralEngine` (default) or `.gpu`. There is no CPU-only option: on the CPU this model's
  latents are NaN from the first step, so every image comes out black (measured on macOS). The engine checks
  the latents after every step; NaN or infinity stops the run with `.generationFailed`.
- **Not in the Simulator.** It has no Neural Engine, and the CPU path only gives black images, so `generate`
  throws `.simulatorUnsupported` before loading anything. `engine.unavailability` reports it up front, and
  `.unsupportedPlatform` on visionOS, so a UI can say so before offering the download. Test on a device.
- **One at a time.** A second concurrent `generate` on the same engine throws `.busy`.
- **Cancellation.** Cancelling the calling task stops at the next denoising step and throws
  `CancellationError`. Loading the model and the final decode cannot be interrupted.
- **Background and device conditions.** Moving the app to the background stops the run the same way and
  throws `GenerationInterrupted` (`.appBackgrounded`); while the app is in the background, `generate` throws it
  before loading. `CoreMLDiffusionEngine(deviceGuard: DeviceConditions())` opts in to the heat, Low Power Mode
  and battery guard, checked before the run and every `checkInterval` steps; at block it throws
  `GenerationInterrupted` (`.deviceConditions`). See [Memory]({{ '/guide/memory' | relative_url }}#device-conditions).

Errors are `OnDeviceImageGenError`: `.unsupportedPlatform` (visionOS), `.simulatorUnsupported`,
`.modelNotDownloaded`, `.insufficientMemory`, `.invalidRequest`, `.busy`, `.modelLoadFailed`,
`.generationFailed`, `.filteredBySafetyChecker`, `.imagePlaygroundUnavailable` and `.consentDeclined`. Each has an
`errorDescription` and a `recoverySuggestion`.

### Performance (estimated, not measured on A19)

| | Estimate | Basis |
|--|----------|-------|
| Warm run, 512×512, 20 steps, iPhone 17 Pro / Pro Max | ~5–10 s | Apple measured 7.9 s on iPhone 14 Pro Max (6-bit weights, fp16 activations, `SPLIT_EINSUM_V2`, Neural Engine); Apple says low-memory mode added up to 2 s |
| First run on a device | up to a few minutes | Core ML prepares the model for the Neural Engine once and caches the result |
| Peak memory | ~1.5–2.5 GB | the 2.2× heuristic; a third-party figure puts an app's limit on a 12 GB iPhone at about 6.4 GB with both increased-memory entitlements |
| iOS Simulator | no images | no Neural Engine, and on the CPU the latents are NaN; the engine throws `.simulatorUnsupported` |

### Dependency

`AuraImageGen` links `StableDiffusion` from `apple/ml-stable-diffusion`, pinned `exact: "1.1.1"` on iOS, macOS
and Mac Catalyst (1.1.1 declares no visionOS, where the engine throws `.unsupportedPlatform`). It is exact
because upstream `main` pins `swift-transformers` 0.1.8, which conflicts with AuraLocal's 1.3 floor. It adds
one pin to a downstream resolve, `ml-stable-diffusion`. The `swift-argument-parser` its CLI target uses is
already in the graph through mlx-swift, LocalLLMClient and swift-syntax. AuraLocal's own lockfile gained both.
Apple presents Core AI as Core ML's successor, so expect a migration later.

## Apple Image Playground

`ImagePlaygroundPolicy` holds the settings for presenting Apple's `imagePlaygroundSheet`. AuraLocal does not
generate through Image Playground; the sheet is Apple's UI and pipeline.

```swift
import ImagePlayground
import AuraImageGen

content
    .imagePlaygroundSheet(isPresented: $presenting, concept: prompt) { url in /* move the file */ }
    // Always pass the list: the default `in:` is `.all`, which includes the ChatGPT style.
    .imagePlaygroundGenerationStyle(ImagePlaygroundPolicy.defaultStyle, in: ImagePlaygroundPolicy.allowedStyles)
```

- **Styles.** `allowedStyles` is animation, illustration and sketch. `.externalProvider` (ChatGPT) is never
  offered.
- **Where it runs.** `ImagePlaygroundPolicy.processing` follows Apple's statements: on device before iOS 27
  (for every style except the external provider), Private Cloud Compute from iOS 27, with daily limits. No API
  reports it per request. When `processing.leavesDevice` is true, ask the user before every presentation; the
  example does, and declining sends nothing and shows `.consentDeclined`.
- **Options.** On iOS 26.4+, `.imagePlaygroundOptions(ImagePlaygroundPolicy.options)` turns personalization
  off.
- **Availability.** `supportsImagePlayground` is a single Bool, so the reason it is false (Apple Intelligence
  off, region, language, device, Simulator) cannot be told apart.
- **Not used: `ImageCreator`.** It is deprecated in iOS 27, its initializer throws `notSupported` there, and
  Apple says it will not compile at the public release, which would break every integrator's build.

## FLUX via mflux (macOS)

`MFluxEngine` adds **FLUX** text-to-image with easy `lora.safetensors` loading. It runs
[mflux](https://github.com/filipstrand/mflux) (FLUX on Apple MLX) as a **subprocess**.

### Why FLUX stays on macOS

FLUX is a ~12B-parameter diffusion model. A Swift package cannot embed the Python runtime mflux
needs, so `AuraImageGen` locates and execs an already-installed `mflux-generate`. Consequences,
stated plainly:

- **macOS only.** Off macOS the API exists but `MFluxEngine.isAvailable == false` and `generate`
  throws `.unsupportedPlatform`. The quantized weights alone are 6.8–9.2 GB and the peak is about twice
  that, far beyond an iPhone app's memory limit; use `CoreMLDiffusionEngine` there.
- **Non-sandboxed only.** The macOS App Sandbox blocks exec-ing an external CLI, so this serves the
  `aura` CLI and non-sandboxed apps (the Example has `ENABLE_APP_SANDBOX = NO`) — not App-Store
  sandboxed apps. An in-process, sandbox-safe Swift engine (`flux.swift`) is **not** available: it
  pins `swift-transformers 0.1.x` while AuraLocal is on `1.3.x` — an unresolvable dependency conflict.

### Prerequisites

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
> a pre-quantized repo downloads only the quantized weights. This is what the defaults do:
> `ImageGenRequest(prompt:)` and `aura imagegen "…"` use `model: "schnell"` with `quantize: 4`, so
> always pass an ungated repo in `model` (`--model`) plus `baseModel` (`--base-model`).

### Swift

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

### CLI

```bash
aura imagegen "a red fox in snow" \
  --model dhairyashil/FLUX.1-schnell-mflux-4bit --base-model schnell \
  --lora /path/to/style.safetensors --lora-scale 0.9 \
  --steps 4 --seed 42
```

`aura imagegen` flags: `--model`, `--base-model`, `--steps`, `--seed`, `--quantize` and `--low-ram`
pass straight to mflux. `--lora <file>` is repeatable, and `--lora-scale S` sets the scale of the
`--lora` before it; they become mflux's `--lora-paths`/`--lora-scales`. `--out <dir>` picks the folder
for `image.png` (default: a temporary folder), and the CLI prints the PNG's path. Width and height are
fixed at 1024×1024 on the CLI; set `ImageGenRequest(width:height:)` in Swift.

### Performance (measured)

M1 Pro (32 GB), schnell 4-bit, 1024×1024, 4 steps:

| Metric | Value |
|--------|-------|
| Peak memory footprint | **≈ 20 GB** (runtime peak, not the ~9 GB on-disk size) |
| Speed | **≈ 25 s/step → ~100 s** diffusion; ~2–3 min/image with model load |
| First-run download | ~9 GB (pre-quantized), network-bound |

It fits a 32 GB Mac but runs best **exclusively** — the ~20 GB peak presses the ~21.5 GB Metal wired
limit, so don't co-load an LLM. Lower resolution (512×512) is substantially faster.

## Finding a model that fits (compatibility filter)

`HuggingFaceSearch` searches HF, and `ModelCompatibilityChecker` says whether a result runs on **this**
machine — for a diffusion pipeline it sizes every `.safetensors` and applies the peak multiplier below (see
[Finding compatible models]({{ '/guide/models' | relative_url }})). The Example's Image tab (FLUX, macOS) has
**Search HuggingFace…**, with an *"Only what runs"* toggle, gated/kind badges, and a **Use** button that fills
in the model.

```swift
let checker = ModelCompatibilityChecker()
let hits = try await HuggingFaceSearch.search("flux schnell mflux", sort: .downloads)
for hit in hits {
    let report = await checker.check(hit.id, on: .thisDevice())
    print(hit.id, hit.gated ? "(gated)" : "", report.status.label, report.bestFit?.summary ?? "")
}
```

**Why `kind` matters.** Judging a download by its on-disk size is misleading for image models:

| Kind | Peak RAM vs weights | Can layer-stream |
|------|--------------------|------------------|
| `.llm` | ~1.15× (weights + overhead + KV) | yes |
| `.diffusion` | **~2.2×** (text encoder + VAE + activations co-resident) | no |

The 2.2× is measured, not assumed — FLUX schnell 4-bit is 9.2 GB on disk and peaks ~20 GB. Validated
against the live repo: 8.9 GB of `.safetensors` → estimated 19.7 GB peak vs ~20 GB measured. So a 6.8 GB
FLUX correctly reads **too large** on a Mac with 13 GB free, **good** with ~20 GB free and **excellent** only
from ~25 GB free — where naive size-based math would have called it "excellent" every time.

`hit.gated` flags repos that need a HuggingFace login (black-forest-labs' FLUX repos are gated) so you
can pick an ungated mirror instead.

## LoRA notes

`AuraImageGen` accepts **local `.safetensors` files only** (a missing path throws
`ImageGenError.loraNotFound`); download a Hugging Face LoRA first. Multiple LoRAs each take a scale. mflux
also ships curated styles (`--lora-style`) and a `mflux-lora-library` / `mflux-train` toolchain for
building your own — outside `AuraImageGen`'s scope, which is loading, not training.
