---
layout: docs
title: Image Generation
parent: Guide
nav_order: 8.5
description: "Text-to-image on iPhone and Mac with Core ML Stable Diffusion, Apple Image Playground behind consent, and FLUX + lora.safetensors on macOS via mflux."
---

# Image Generation + LoRA

`TonamoxImageGen` has two engines and one policy:

| Path | Platforms | Where it runs | Notes |
|------|-----------|---------------|-------|
| `CoreMLDiffusionEngine` | iOS, macOS | This device | Stable Diffusion 2.1 base built in (512×512, ~1.07 GB one-time download), or a Core ML SD 1.x/2.x folder from Hugging Face that `CoreMLDiffusionInspector` accepts |
| Apple Image Playground (`ImagePlaygroundPolicy`) | iOS 18.4+, macOS 15.4+, visionOS 2.4+ (the Example uses it on iOS only) | On device before OS 27; Apple's servers from OS 27 | Apple's sheet; Tonamox only sets its styles and options |
| `MFluxEngine` | macOS | This Mac | FLUX + `lora.safetensors` through the mflux CLI |

## On device (iPhone and Mac): Core ML Stable Diffusion

`CoreMLDiffusionEngine` runs Apple's [Core ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion)
pipeline in process. The only network traffic is the one-time model download; prompts and images stay on
the device.

```swift
import TonamoxImageGen

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

The Example app's **Image** tab runs this engine. On iOS its segments are **On device** and **Apple**
(Image Playground); on macOS they are **FLUX (mflux)** and **Stable Diffusion (Core ML)**. A **Model** picker
lists the built-in model and every model adopted from Hugging Face, with Ready or Not downloaded per row;
**Find on Hugging Face…** opens a search with a verdict per repo and per folder (see
[Other models from Hugging Face](#other-models-from-hugging-face)). Picking a model applies its steps, guidance
and compute unit. The tab asks before downloading and shows the size and license, offers the compute units and
memory mode under **Advanced** (steps 1–50, so few-step models fit: 4 for turbo-style names, 8 for LCM), and
shares each image with **Share**. The share sheet's **Save Image** adds the image to Photos only when the app declares
`NSPhotoLibraryAddUsageDescription`, which the Example does. **Delete** (or **Remove**, when a Hugging Face
model is not fully downloaded) drops an adopted model with its files and selects the built-in one. On a
device where the engine can run, **Use** (after an alert with the size and license) adds and selects the model and
starts its download; in the Simulator it only adds and selects it, nothing is downloaded, and the fit verdicts
there use the Mac's memory.

### The model

`CoreMLDiffusionModel.stableDiffusion21Base` is Apple's 6-bit palettized conversion of Stable Diffusion 2.1
base (`apple/coreml-stable-diffusion-2-1-base-palettized`, `split_einsum_v2/compiled`), pinned to revision
`dca5f2fda9c5143c71aded72bbb4ce39f3c2cc9c` so an upstream re-upload cannot change what runs.

- **Download.** 17 files, 1,072,887,285 bytes: the text encoder, Unet and VAE decoder (`.mlmodelc`), plus
  `vocab.json` and `merges.txt`. The VAE encoder, which only image-to-image needs, is skipped. The repo is
  ungated and no Hugging Face token is sent.
- **Storage.** `ModelDownloads.downloadFolder(repoID:revision:folder:include:auth:progress:)` writes to
  `<root>/huggingface/hub/models--apple--coreml-stable-diffusion-2-1-base-palettized/snapshots/main/split_einsum_v2/compiled`
  under `ModelStorage.rootDirectory` (Application Support, excluded from backup; see
  [Where models are stored]({{ '/guide/models' | relative_url }}#where-models-are-stored)), resumes `.partial` files
  and checks each LFS file's SHA-256. A folder an earlier version downloaded into Caches is used from there until
  it is migrated (`resourcesDirectory`). The files can be deleted between runs, so `generate` re-checks
  `isDownloaded` on every call.
  The folder records the revision its files come from, and `isDownloaded` requires the pinned one. Files from
  any other revision, finished or partial, are deleted before the next download.
  TonamoxUI's Downloads list shows every image model as a paused MLX snapshot of its repo. Resume is off there;
  its Delete removes every folder of that repo and does not unload a running image engine.
  `CoreMLDiffusionModel.delete()` removes only the model's own folder (`ModelDownloads.deleteFolder(repoID:folder:)`).
- **Output.** 512×512 only: the Unet's input shape is fixed. No LoRA.
- **No safety checker.** The repo ships no `SafetyChecker.mlmodelc`, so nothing filters the images. An app
  that ships this needs its own moderation decision.
- **License.** CreativeML Open RAIL++-M, which restricts some uses and requires passing those restrictions on
  to your users. Read it from the [model card](https://huggingface.co/apple/coreml-stable-diffusion-2-1-base-palettized).
- **Quality.** Clearly below FLUX, SDXL or Image Playground; hands and text in images come out poorly.

### Other models from Hugging Face

`CoreMLDiffusionInspector` finds Core ML Stable Diffusion repos on Hugging Face and judges each folder for a
device without downloading weights. It reads the repo listing at its current commit, then each candidate
folder's Unet `metadata.json` (about 4 KB), pinned to that commit. `CoreMLDiffusionModel(variant:)` turns a
folder rated **Runs** or **Runs, with caveats** into a model pinned to the same commit.

```swift
import TonamoxCore
import TonamoxImageGen

let inspector = CoreMLDiffusionInspector()
let hits = try await inspector.search("dreamshaper")       // sends the text to huggingface.co
let report = try await inspector.inspect(hits[0].id)        // listing + up to 4 metadata.json probes
guard let best = report.recommended, best.verdict.isRunnable else { return }
let model = try CoreMLDiffusionModel(variant: best)        // throws unless the folder runs
CoreMLDiffusionModelStore.standard.remember(model)         // listed again after a relaunch
try await model.download { done, total in print(done, "of", total) }
let engine = CoreMLDiffusionEngine(model: model, computeUnits: model.preferredComputeUnits)
let output = try await engine.generate(DiffusionRequest(prompt: "a lighthouse at dusk",
                                                        steps: model.recommendedSteps,
                                                        guidanceScale: model.recommendedGuidance))
```

- **Search.** `search(_:limit:)` runs three queries: by the `coreml` tag, by the `core-ml` tag, and by name
  with "coreml" added, so repos with either tag or none show up. A plain word alone would not work: its top
  hits are all diffusers repos. A pasted repository URL searches its id. It keeps only repos whose file names
  show a loadable folder, or a Core ML specific reason they are not supported. An empty query lists the most
  downloaded ones. `inspect(_:on:)` takes any `owner/repo` or repository URL, listed by the search or not; the
  Example's finder offers **Inspect** for a pasted one.
- **Layout (file names only).** A folder needs `TextEncoder.mlmodelc`, `Unet.mlmodelc` or both
  `UnetChunk1/2.mlmodelc`, `VAEDecoder.mlmodelc`, `vocab.json` and `merges.txt`, each model directory holding
  `coremldata.bin`, `model.mil` and `weights/weight.bin`. When a folder has both the whole Unet and its chunks,
  only the chunks are downloaded, because the pipeline uses them. `VAEEncoder`, `SafetyChecker` and ControlNet
  models are never downloaded. `CoreMLDiffusionLayout` does this.
- **Rules (from `metadata.json`).** `CoreMLDiffusionRules` blocks a Unet converted without classifier-free
  guidance (batch 1), inpainting and instruct-pix2pix Unets, SDXL, stateful Unets, extra inputs such as
  ControlNet residuals, a newer OS than the device runs, non-square output, an fp16 unchunked Unet of 1 GiB or
  more on the iPhone's Neural Engine, v-prediction models, folders at the repo root and gated repos. ORIGINAL
  attention is a caveat: it runs on the GPU. LCM names are a caveat (the runtime has no LCM scheduler);
  turbo, lightning, Hyper-SD, DMD and TCD names set 4 steps and guidance 1.0, LCM 8 and 1.0 (estimates). A
  missing or `other` license, or one marked non-commercial (`nc`) or `community`, is a caveat. When the probe
  was skipped or failed the verdict is **Unknown**, never **Runs**; gated repos are not probed and are
  **Won't run**. Each finding is a `CompatibilityFinding` whose `rule` is its id (`R1`…`R19`, `FIT`).
- **Memory.** The verdict uses the engine's estimate, 2.2× the folder's exact listed bytes, but compares it
  with the device budget (`HardwareAnalyzer.fitLevel(forWeightsBytes:kind: .diffusion, profile:)`): on
  iPhone the app's whole memory limit, on a Mac the Metal working-set limit. The engine compares it with the
  memory free when it loads (`HardwareProfile.availableMemoryBytes()`, which leaves out what the app already
  holds), so a folder rated **Fits** or **Tight** can still be refused with `.insufficientMemory`. The 2.2
  factor was measured on FLUX, not Core ML, so it is an estimate here. fp16 SD 2.x folders (about 2.5 GB)
  come out **Tight** against the `DevicePreset.iPhone17Pro` budget (6.4 GB, a third-party figure); only a run
  on the device shows the real peak.
- **Persistence.** `CoreMLDiffusionModelStore` keeps adopted models in `UserDefaults`; `all` lists the built-in
  model first, then the records by display name, and the built-in model is never stored. Call `remember(_:)` when
  a download starts, so a partial download is listed after a relaunch; `forget(_:)` drops a record and leaves the
  files (`delete()` removes them). The model id leaves the revision out, so adopting a newer commit of the
  same folder replaces the record. Adopting the built-in model's own folder gives `stableDiffusion21Base` at
  its pinned revision, since both would share one folder on disk. Files on disk stay the source of truth
  (`isDownloaded`).
- **Deleting.** `CoreMLDiffusionModel.delete()` removes the folder's downloaded and partial files, in the storage
  root and in Caches. Core ML keeps
  its compiled copy of a model in a system-managed cache that this does not touch. In one Mac test a GPU
  model with a 0.89 GB download left about 1.9 GB there.
- **What leaves the device.** The search text and repo ids, sent to huggingface.co, and then the download.
  The inspector's requests carry no Hugging Face token, cookie or `Authorization` header, and the download sends
  no token, so gated repos are reported, not opened. Prompts and images stay on the device.
- **Not supported yet.** SDXL (a separate runner and a memory estimate measured on the phone are needed),
  folders at the repo root, repos that ship only `.zip` archives or uncompiled `.mlpackage`s, and Core AI
  `.aimodel`s. The repo's `SafetyChecker` is not downloaded, so images are not filtered. LCM and turbo quality
  under the runtime's DPM-Solver++ is not guaranteed.
- **Unverified on iPhone.** Only Apple's SD 1.5 6-bit folder was run end to end with this code, on a Mac. The
  first Neural Engine compile, latency, peak memory and jetsam for each other model are unmeasured on iPhone.

Errors are `CoreMLDiffusionModelError`: `.invalidRepoID`, `.searchFailed`, `.repositoryUnavailable` (a non-2xx
listing, such as 401, 403 or 404, or a listing that is unreadable or has no full commit sha), `.network` and
`.notRunnable`. Anonymous Hugging Face answers 401 for a repo that does not exist, so a mistyped id reads "not
found, or private"; a gated repo lists fine and gets the R17 finding instead.

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

`TonamoxImageGen` links `StableDiffusion` from `apple/ml-stable-diffusion`, pinned `exact: "1.1.1"` on iOS, macOS
and Mac Catalyst (1.1.1 declares no visionOS, where the engine throws `.unsupportedPlatform`). It is exact
because upstream `main` pins `swift-transformers` 0.1.8, which conflicts with Tonamox's 1.3 floor. It adds
one pin to a downstream resolve, `ml-stable-diffusion`. The `swift-argument-parser` its CLI target uses is
already in the graph through mlx-swift, LocalLLMClient and swift-syntax. Tonamox's own lockfile gained both.
Apple presents Core AI as Core ML's successor, so expect a migration later.

## Apple Image Playground

`ImagePlaygroundPolicy` holds the settings for presenting Apple's `imagePlaygroundSheet`. Tonamox does not
generate through Image Playground; the sheet is Apple's UI and pipeline.

```swift
import ImagePlayground
import TonamoxImageGen

content
    .imagePlaygroundSheet(isPresented: $presenting, concept: prompt) { url in /* move the file */ }
    // Always pass the list: the default `in:` is `.all`, which includes the ChatGPT style.
    .imagePlaygroundGenerationStyle(ImagePlaygroundPolicy.defaultStyle, in: ImagePlaygroundPolicy.allowedStyles)
```

- **Styles.** `allowedStyles` is animation, illustration and sketch. `.externalProvider` (ChatGPT) is never
  offered.
- **Where it runs.** `ImagePlaygroundPolicy.processing` follows Apple's statements: on device before OS 27
  (for every style except the external provider), Private Cloud Compute from OS 27, with daily limits. It
  reads the OS major version, so the same rule applies on iOS, macOS and visionOS. No API reports it per
  request. When `processing.leavesDevice` is true, ask the user before every presentation; the example
  does, and declining sends nothing and shows `.consentDeclined`.
- **Options.** On iOS, macOS and visionOS 26.4+, `.imagePlaygroundOptions(ImagePlaygroundPolicy.options)`
  turns personalization off.
- **Availability.** `supportsImagePlayground` is a single Bool, so the reason it is false (Apple Intelligence
  off, region, language, device, Simulator) cannot be told apart.
- **Not used: `ImageCreator`.** It is deprecated in iOS 27, its initializer throws `notSupported` there, and
  Apple says it will not compile at the public release, which would break every integrator's build.

## FLUX via mflux (macOS)

`MFluxEngine` adds **FLUX** text-to-image with easy `lora.safetensors` loading. It runs
[mflux](https://github.com/filipstrand/mflux) (FLUX on Apple MLX) as a **subprocess**.

### Why FLUX stays on macOS

FLUX is a ~12B-parameter diffusion model. A Swift package cannot embed the Python runtime mflux
needs, so `TonamoxImageGen` locates and execs an already-installed `mflux-generate`. Consequences,
stated plainly:

- **macOS only.** Off macOS the API exists but `MFluxEngine.isAvailable == false` and `generate`
  throws `.unsupportedPlatform`. The quantized weights alone are 6.8–9.2 GB and the peak is about twice
  that, far beyond an iPhone app's memory limit; use `CoreMLDiffusionEngine` there.
- **Non-sandboxed only.** The macOS App Sandbox blocks exec-ing an external CLI, so this serves the
  `tonamox` CLI and non-sandboxed apps (the Example has `ENABLE_APP_SANDBOX = NO`) — not App-Store
  sandboxed apps. An in-process, sandbox-safe Swift engine (`flux.swift`) is **not** available: it
  pins `swift-transformers 0.1.x` while Tonamox is on `1.3.x` — an unresolvable dependency conflict.

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
> `ImageGenRequest(prompt:)` and `tonamox imagegen "…"` use `model: "schnell"` with `quantize: 4`, so
> always pass an ungated repo in `model` (`--model`) plus `baseModel` (`--base-model`).

### Swift

```swift
import TonamoxImageGen

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
tonamox imagegen "a red fox in snow" \
  --model dhairyashil/FLUX.1-schnell-mflux-4bit --base-model schnell \
  --lora /path/to/style.safetensors --lora-scale 0.9 \
  --steps 4 --seed 42
```

`tonamox imagegen` flags: `--model`, `--base-model`, `--steps`, `--seed`, `--quantize` and `--low-ram`
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

`TonamoxImageGen` accepts **local `.safetensors` files only** (a missing path throws
`ImageGenError.loraNotFound`); download a Hugging Face LoRA first. Multiple LoRAs each take a scale. mflux
also ships curated styles (`--lora-style`) and a `mflux-lora-library` / `mflux-train` toolchain for
building your own — outside `TonamoxImageGen`'s scope, which is loading, not training.
