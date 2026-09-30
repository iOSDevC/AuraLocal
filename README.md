<p align="center">
  <img src="https://raw.githubusercontent.com/iOSDevC/AuraLocal/main/docs/media/logo.png" alt="AuraLocal" width="400">
</p>

# AuraLocal

Lightweight on-device LLM & VLM Swift package for iOS/macOS/visionOS. Run Qwen3, Llama 3, Mistral, Gemma, SmolVLM and more — locally and privately by default (no API keys unless you opt into cloud escalation). Supports both MLX (Apple Silicon GPU) and llama.cpp/GGUF (up to 70B, including layer-streaming for memory-constrained devices).

**Documentation:** [iosdevc.github.io/AuraLocal](https://iosdevc.github.io/AuraLocal/) — installation, hybrid inference, RAG, the CLI and the other guides.
Research notes, not a shipped feature: [Distributed inference](docs/guide/distributed.md).

---

## Highlights

- **Dual-backend inference** — MLX for GPU inference (≤4B on most iPhones, 8B at the iPhone 17 Pro budget, up to 35B on a Mac); llama.cpp/GGUF for 1B–70B, with full Metal acceleration or layer-streaming when a model doesn't fit.
- **Layer-streaming mode** — Run 7B–12B models on entitled 6–8 GB iPhones at ~2–5 tok/s. The OS pages weights from disk on demand, so the app's own footprint stays small — but mapped pages that are faulted in still count toward the resident size jetsam measures, so they are evictable rather than free. Without both memory entitlements an 8B is refused outright.
- **`InferenceBackend` protocol** — Clean abstraction over all backends; `BackendRouter` selects the optimal engine automatically based on model format and device RAM.
- **GGUF model catalog** — 7 large models (Llama 3.1, Qwen 2.5, Mistral, Phi-3, Gemma 2; 5 download today, see [the GGUF table](#text-models--gguf-llamacpp)) with automatic download from HuggingFace, resume support, and `@Published` progress.
- **GQA-aware memory estimates** — KV cache calculations account for Grouped-Query Attention (Llama 3.1 8B: 256 MB at FP16/2048 ctx vs ~1 GB with naive full-attention assumption).
- **Unified model management** — `ModelManager.shared.load()` provides LRU caching, in-flight deduplication, automatic memory-pressure eviction, and backend-aware GGUF downloading.
- **Swift 6 concurrency** — All public APIs are `@MainActor`-isolated or `Sendable`, with `actor`-based stores for data-race safety.
- **OOM prevention that fails safe** — One source of truth (`HardwareProfile.availableMemoryBytes()`) asks the OS and returns `nil` rather than inventing a number. iOS reports "unknown" exactly when the process is at its jetsam limit, so unknown counts as *pressure*: layer-streaming shrinks its context to 512 tokens and stops a reply early (checked every 32 tokens), and the model cache keeps one model. Paired with `DispatchSource` listeners and per-generation checks.
- **Hybrid RAG pipeline** — FTS5 keyword pre-filter + Accelerate cosine re-ranking, stored in SQLite. Zero external dependencies.
- **Local provider detection** — `AuraLocal.detectLocalProviders()` discovers a running Ollama (`:11434`) or llama.cpp `llama-server` (`:8080/v1`) and the models each exposes, via a dependency-free URLSession probe (never throws — a down server is a normal result).
- **Token-optimized hybrid inference** — Stay local by default: `AuraLocal.stream()` runs on-device models, and `HybridEscalator` sends a request to your own llama-server/Ollama box or a cloud model only when needed, streaming its answer through an `onToken` callback. It **cuts the tokens sent to the remote** via selective-context compression (input-dependent), an in-memory response cache (repeat calls in a session cost **$0**), and opt-in PII redaction. Every escalation returns its token counts, and `CostLedger` records its cost, for a receipt. The policy path is off by default and consent-gated. See [Hybrid Inference](#hybrid-inference-local--remote).
- **Agent orchestration (Architect-step escalation)** — `AgentCrew` (Extractor → Reviewer → Architect → Reporter) escalates the Architect's local draft to a bigger model when it looks weak, reusing the same compression + consent + cost machinery.
- **On-device ML tools, not just LLMs** — Typed, availability-checked wrappers over Apple's ML frameworks, using models that ship with the OS and no extra dependencies: Vision (OCR lines, image classification, barcodes & QR, face detection), NaturalLanguage (language ID, named entities, sentiment, embeddings), SoundAnalysis (303 everyday sounds), a runner for **your own Core ML models**, and **on-device Create ML training** of text classifiers. `SystemToolRegistry` reports which built-in tools run on each device. See [On-device ML tools](#on-device-ml-tools).
- **`aura` CLI & binaries** — A headless integration harness (`aura providers | tools | ocr | ml | models | imagegen`) plus build scripts for a release CLI and a drag-to-Applications `.dmg`. See [CLI & Binaries](#cli--binaries).
- **Will this Hugging Face model run here?** — `ModelCompatibilityChecker` reads a repo's config and GGUF/safetensors headers and reports, for a device preset, whether AuraLocal's pinned runtimes (mlx-swift-lm 3.31.3, llama.cpp b8851) can load it, why not, and its `models.json` entry. Also `aura models search|check|devices` and the Model Finder example app.
- **Multilingual dense RAG (opt-in)** — `AutoEmbeddingProvider(embeddingModelAt:)` swaps TF-IDF for multilingual-e5-small via Core ML; the index re-embeds itself when the provider changes.

---

## Requirements

- **iOS 18+** / **macOS 15+** / **visionOS 2+**
- **Xcode 16.3+** (Swift 6.1: the pinned LocalLLMClient 0.5.0 and mlx-swift-lm 3.31.3 declare tools-version 6.1); **Xcode 26** (iOS 26 / macOS 26 SDK) to build `AuraAppleIntelligence` and `AuraAgents`, which import FoundationModels
- **Swift 6.1+** (C++ interoperability mode enabled — required by llama.cpp)
- `Increased Memory Limit` **and** `Extended Virtual Addressing` entitlements (both required for models > 500 MB — the second one is what makes mmap/layer-streaming actually work on iOS)

> **Note:** The C++ interoperability requirement means all targets that import `AuraCore` must enable `.interoperabilityMode(.Cxx)` in their `Package.swift` `swiftSettings`. In an Xcode app target, set the build setting `SWIFT_OBJC_INTEROP_MODE = objcxx` (as `Examples/ModelFinder/project.yml` does).

---

## Installation

Add via Swift Package Manager (in Xcode's **Add Package** dialog, choose a commit or branch rule, not a version rule):

```
https://github.com/iOSDevC/AuraLocal
```

Or in `Package.swift`, pinned to a commit on `main`:

```swift
// swift-tools-version: 6.0
.package(url: "https://github.com/iOSDevC/AuraLocal", revision: "<commit SHA from main>")
```

> A version requirement (`from:` / `exact:`) cannot resolve. AuraLocal depends on LocalLLMClient by
> `revision:` (its `unsafeFlags` rule out a version pin), and SwiftPM rejects a version-pinned package
> whose dependencies are pinned by revision or branch. `branch: "main"` also resolves, but it moves under you.

If your package imports `AuraCore`, add the C++ interop setting:

```swift
.target(
    name: "MyTarget",
    dependencies: [.product(name: "AuraCore", package: "AuraLocal")],
    swiftSettings: [.interoperabilityMode(.Cxx)]
)
```

### Dependencies

| Dependency | Purpose |
|-----------|---------|
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | MLX GPU inference for `.mlx` models |
| [LocalLLMClient](https://github.com/tattn/LocalLLMClient) | Swift wrapper for llama.cpp — GGUF inference + Metal kernels |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | `Tokenizers` — tokenization for MLX models |

### Modules

| Module | Contents |
|--------|----------|
| `AuraCore` | Core inference, dual-backend engine, models, conversation persistence, hybrid escalation, on-device ML tools, Hugging Face compatibility checker |
| `AuraUI` | SwiftUI views and ViewModels for drop-in UI |
| `AuraVoice` | Turn-based voice (on-device STT + TTS), 100% local |
| `AuraDocs` | RAG document library — PDF, DOCX, text, images · TF-IDF or multilingual-e5-small embeddings |
| `AuraAppleIntelligence` | Apple FoundationModels agents, tools & structured output (iOS 26 / macOS 26) |
| `AuraAgents` | Reusable multi-agent orchestration (`AgentCrew`) with hybrid escalation of its Architect step |
| `AuraImageGen` | FLUX text-to-image + `lora.safetensors` via mflux (**macOS only**) |

---

## Backend Architecture

`BackendRouter` automatically selects the best engine based on model format and device hardware:

```
model.format == .mlx                      →  MLXBackend             (Apple Silicon GPU; not the Simulator)
model.format == .gguf + fits in memory    →  LlamaCppBackend        (full Metal offload, macOS or iOS)
model.format == .gguf + streamingRequired →  LayerStreamingBackend  (mmap; in practice iPhone / iPad)
```

| Backend | Platform | Model Size | Peak RAM | Tokens/sec |
|---------|----------|-----------|---------|------------|
| MLX | iOS + macOS | 258M–35B in the catalog (≤4B on most iPhones) | ~1 – 20 GB | 20–45 (≤4B) |
| llama.cpp (standard) | iOS + macOS, when the model fits | 1.2B–70B | ~1 – 40 GB | 8–20 (Mac) |
| Layer-streaming | iOS, when it does not fit | 7B–14B (32B at the iPhone 17 Pro budget) | ~1 GB for a 7B–8B | 2–6 |

You can inspect which backend a model will use:

```swift
import AuraCore

let backend = BackendRouter.recommendedBackend(for: .llama3_1_8b_gguf)
// → .llamaCpp      (Mac with 16 GB)
// → .layerStreaming (entitled 6 GB iPhone — ~3 GB budget)
// → still .llamaCpp when not even streaming fits: recommendedBackend never reports a refusal.
//   Check HardwareAnalyzer.assess(.llama3_1_8b_gguf).fitLevel == .tooLarge (ModelManager.load throws AuraError.modelTooLarge)
```

---

## Hybrid Inference (local + remote)

AuraLocal is **local-first**: everything runs on-device by default. `AuraLocal.stream()` and
`chat()` run on-device models only; `HybridEscalator` is the remote path. It sends a request to
your own `llama-server`/Ollama box or a cloud model (Anthropic / OpenAI), selected **per request**,
and streams the answer through an `onToken` callback. When a task exceeds the local model it
**escalates**, but only after **shrinking the payload** so the remote (often paid) call sends far
fewer tokens. The policy path is **opt-in and consent-gated**: `routeAndEscalate` returns `nil`
when the request stays local (policy off, no target, a missing key) and throws when consent is
declined or the call fails. In every one of those cases your code keeps the local answer.

### Token savings — the headline feature

Spending as few remote tokens as possible is the whole point. Four mechanisms **stack**:

| Mechanism | What it does | Effect |
|---|---|---|
| **Local-first routing** | Answer on-device whenever the local model suffices | Remote tokens: **0** |
| **Selective-context compression** | Keep only the sentences relevant to the question, within a budget derived from the remote's context window | **Fewer** input tokens when the context exceeds the budget (input-dependent; a context that fits is sent unchanged) |
| **Response cache** | Identical escalations in the same app session replay the prior answer (in-memory, cleared on relaunch) | Repeat calls: **$0** |
| **PII redaction** | Strip secrets before sending (opt-in: `escalate(…, redactPII: true)`) | Smaller + safer payload |

`HybridEscalator.Result` carries `compression` (`originalTokens`, `compressedTokens`, `factor`) and
`usage`, and `CostLedger.shared.records` keeps each call's `costUSD` (`sessionCostUSD` sums them), so
your UI can show a receipt per escalation. The Example app's Hybrid tab shows one and lists each
call's cost in its escalation history.

### Mixed integration at a glance

1. **Local GGUF** answers by default — **0 remote tokens**.
2. The **router** decides local-vs-remote per request (size overflow, local uncertainty, sensitive domain, policy, cost cap).
3. On escalation: **compress → consent** (every cloud request; your LAN box too under `.askEachTime`), then call the chosen remote — *your own* `llama-server`/Ollama first, cloud (BYOK) second. `PIIRedactor` runs only when you call `escalate(to:…, redactPII: true)` yourself; `routeAndEscalate` does not redact.
4. The remote streams back through `HybridEscalator`'s `onToken` callback (cumulative text). If `routeAndEscalate` returns `nil` or throws, keep the local answer: the `nil` check and the `catch` are your code's.

### Detect local providers

```swift
import AuraCore

// Probe a running Ollama (:11434) and llama.cpp llama-server (:8080/v1) concurrently.
let providers = await AuraLocal.detectLocalProviders()   // never throws
for p in providers where p.isAvailable {
    print(p.kind, p.version ?? "", p.models.map(\.name))
}
```

### Escalate a request

```swift
// Discover the best local "bigger model" and send a compressed request to it.
if let target = await HybridEscalator.bestLocalTarget() {
    let result = try await HybridEscalator().escalate(
        to: target,
        systemPrompt: "Answer concisely.",
        context: longContext,          // compressed to the remote's budget
        question: userQuestion,
        redactPII: !target.isLocalNetwork) { partial in
            // cumulative streamed text
        }
    print(result.answer)
    print("sent \(result.compression.compressedTokens) of \(result.compression.originalTokens) tokens",
          "· \(result.usage?.totalTokens ?? 0) remote tokens",
          result.fromCache ? "· (cached)" : "")
}
```

### Auto-routing & policy

An `EscalationPolicy` (per-profile, default `.off`) plus a pure `EscalationRouter`
decide **local vs remote** (rules R1–R7: consent gate, target availability,
reachability, size overflow, local uncertainty + sensitive-domain bias, cost cap).
Cloud escalations ask your `ConsentGate` **on every request**; under `.autoWithConsentMemory`,
your own LAN box escalates without asking. Remembering consent per conversation is up to your
`ConsentGate`. Without `localAnswer`, only size overflow can trigger an escalation.

```swift
let escalator = HybridEscalator()
let result = try await escalator.routeAndEscalate(
    policy: profile.escalation,        // .off, or EscalationPolicy(mode: .askEachTime / .autoWithConsentMemory, allowCloud: true)
    context: history, question: prompt,
    domain: .security,                 // with localAnswer, short answers in security/medicine escalate more readily
    localAnswer: localDraft,           // the local model's draft; enables R5/R6
    consent: myConsentGate)            // your ConsentGate; it receives a compressed preview to approve
// result == nil → the router kept the request local; keep localDraft
```

### Agent orchestration (Architect-step escalation)

`AgentCrew` (in `AuraAgents`) runs Extractor → Reviewer → Architect → Reporter on-device and
escalates only the **Architect** step: when its local draft looks weak (the router's
low-confidence trigger), `HybridEscalator.routeAndEscalate(localAnswer:)` sends it to a bigger
model, reusing the same compression, consent, and cost machinery. Fail-closed: any error or a
*stay-local* decision keeps the local draft.

### What's included

| Area | Type(s) |
|---|---|
| Discovery | `LocalProviderDetector`, `LocalProviderStatus`, `LocalProviderModel` |
| Providers | `RemoteLLMProvider`, `OpenAICompatibleProvider` (llama-server / Ollama / OpenAI), `AnthropicProvider` |
| Transport (internal) | SSE parsing (inside the providers' `stream(_:)`) and a remote `InferenceBackend` used by `HybridEscalator.escalate(to:…)` |
| Routing | `EscalationRouter` (R1–R7), `EscalationPolicy`, `RoutingDecision`, `AskTargetResolver` (one target for a one-shot ask) |
| Compression | `ContextCompressor` + pluggable `SelfInfoScorer` (default `HeuristicScorer`) |
| Privacy & cost | `ConsentGate`, `KeychainStore` (BYOK, this-device-only), `PIIRedactor`, `CostLedger`, `ResponseCache`, `NetworkMonitor` |

> **GitHub Models is retired.** GitHub shut down GitHub Models on 2026-07-30.
> `OpenAICompatibleProvider.gitHubModels(apiKey:)` is now `unavailable` (a compile error naming the
> replacements), `HybridEscalator.cloudTargets` no longer reads the `cloud.github-models` Keychain
> key, and `aura ask` uses your own llama-server/Ollama box, or an Anthropic/OpenAI key when you
> name that provider.

**Privacy & cost:** cloud API keys live only in the Keychain (never in source, files,
or logs). `ConsentGate` receives the target, the projected cost and a compressed preview of the
context, and your app presents it: the Example app ships a sheet, and the library's only built-in
gate, `DenyingConsentGate`, declines everything. `PIIRedactor` strips obvious secrets when you pass
`redactPII: true`; `CostLedger` records per-escalation token usage and cost; `ResponseCache` avoids
paying twice for identical requests within one app session.

> **Limitations.**
> - The consent preview is not byte-identical to the payload. It is compressed with
>   `EscalationPolicy.keepRatio`, while the send is compressed to half the remote's context window
>   (minus `maxTokens`) and adds the question and system prompt. They match only while `keepRatio`
>   is 0.5, the default.
> - Prices are built in only for `cloud.anthropic` and `cloud.openai` (an internal table). Every
>   other provider, including a custom `OpenAICompatibleProvider`, is priced at **$0**, both
>   recorded and projected, so it never trips the cost cap. A response without usage data is also
>   recorded at $0.
> - The cost cap is compared with each request's projected cost, not with the session's running total.

> **Deferred — true self-information compression.** The current scorer is heuristic
> (relevance + recency). A real Selective-Context scorer needs per-token logprobs from
> the local model, but `LocalLLMClient`'s `llama_context` is `package`-private, so it
> can't be reused. A `LlamaCppSelfInfoScorer` would load its own `llama_context` via the
> C API (`llama.h` is available) and run a windowed prefill (à la `perplexity.cpp`) — it
> drops in behind `SelfInfoScorer` without touching callers. Not yet shipped (doubles
> model memory + heavy iOS-side C interop).

See the [Hybrid guide](docs/guide/hybrid.md).

---

## Image Generation (macOS)

`AuraImageGen` generates images with **FLUX** and folds in `lora.safetensors` adapters. It drives
[mflux](https://github.com/filipstrand/mflux) (FLUX on Apple MLX) as a subprocess — so it is **macOS
only** (a Swift package can't embed the Python runtime FLUX needs) and needs mflux installed:

```bash
uv tool install mflux      # one-time; the Example app has a button that runs this for you
```

> **Two real prerequisites** (the Example surfaces both):
> 1. **mflux installed** — the assisted-install button, or the command above.
> 2. **A downloadable model** — the plain `schnell`/`dev` aliases resolve to black-forest-labs'
>    **gated** HuggingFace repos (need a login + accepted license). Point `--model` at an **ungated
>    pre-quantized** repo to skip auth entirely, e.g. `dhairyashil/FLUX.1-schnell-mflux-4bit`.

```swift
import AuraImageGen

let engine = MFluxEngine()                       // isAvailable == false off macOS / if mflux is missing
let result = try await engine.generate(ImageGenRequest(
    prompt: "a red fox in snow, cinematic",
    model: "dhairyashil/FLUX.1-schnell-mflux-4bit",  // ungated, pre-quantized → no HF login
    baseModel: "schnell",                            // architecture for custom repos
    steps: 4,
    loras: [LoRA(url: loraURL, scale: 0.9)]          // your lora.safetensors + weight
))
result.image      // PlatformImage (NSImage)
result.fileURL    // the PNG on disk
```

CLI:

```bash
aura imagegen "a red fox in snow" \
  --model dhairyashil/FLUX.1-schnell-mflux-4bit --base-model schnell \
  --lora /path/to/style.safetensors --lora-scale 0.9 --steps 4
```

### Find a model that fits your machine

Search HuggingFace and filter by what this device can actually run — the Example's Image tab has
**Search HuggingFace…** with an *"Only what runs"* toggle, plus gated/kind badges:

```swift
let hits = try await HuggingFaceSearch.search("flux schnell mflux")
let report = await ModelCompatibilityChecker().check(hits[0].id, on: .thisDevice())
print(report.status.label, report.headline)  // e.g. "Runs, with caveats — mflux support not verified"
```

Fit is judged by **kind**, not raw size: an `.llm` peaks ≈1.15× its weights and can layer-stream, a
`.diffusion` model peaks **≈2.2×** (text encoder + VAE + activations) and cannot. That multiplier is
measured — so a 6.8 GB FLUX reads *too large* with 13 GB free, *good* with ~20 GB and *excellent* from
~25 GB, instead of a naive "it's only 6.8 GB, it fits."

**Measured** on an M1 Pro (32 GB), schnell 4-bit, 1024×1024, 4 steps: **peak memory footprint ≈ 20 GB**
(the *runtime* peak — not the ~9 GB on-disk size), **~25 s/step ≈ 100 s** of diffusion (≈2–3 min/image
with model load). It fits a 32 GB Mac but runs best **exclusively** (it presses the ~21.5 GB Metal wired
limit) — don't co-load an LLM. FLUX does **not** run on iPhone (far too large) or the iOS Simulator
(no Metal GPU); the mflux subprocess is also blocked by the macOS **App Sandbox**, so it serves the
`aura` CLI and non-sandboxed apps (like the Example), not App-Store-sandboxed apps.

See the [Image generation guide](docs/guide/imagegen.md).

---

## On-device ML tools

Not everything needs an LLM. `AuraCore` wraps Apple's machine-learning frameworks as typed tools
that run on-device with no extra package dependencies. The analysis tools use models that ship
with the OS, so they need no download and work offline; BERT transfer learning may first download
OS embedding assets for the text's script.

| Category | Tools |
|---|---|
| Vision | `VisionOCRTool` (text, or lines with boxes), `VisionImageClassificationTool`, `VisionBarcodeTool` (QR + 23 other symbologies), `VisionFaceDetectionTool` (detection only) |
| Language | `NLLanguageIdentificationTool`, `NLEntityRecognitionTool`, `NLSentimentTool`, `NLEmbeddingTool` |
| Audio | `SoundClassificationTool` (303 everyday sounds in an audio file) |
| Custom models | `CoreMLModelTool` (describe and run your own Core ML models), `TextClassifierTool` (Create ML text classifiers), `CoreMLTextEmbeddingTool` (sentence embeddings from a Core ML bundle such as multilingual-e5-small) |
| Training | `TextClassifierTrainer` (train a text classifier on-device with Create ML) |

```swift
import Foundation
import AuraCore

// Read a QR code and the text around it.
let reader = VisionBarcodeTool()
guard await reader.availability().isAvailable else { return }   // false in the Simulator
let scan = try Data(contentsOf: URL(fileURLWithPath: "/path/to/ticket.png"))
let codes = try reader.detectBarcodes(inImageData: scan)
let lines = try VisionOCRTool().recognizeLines(inImageData: scan)
print(codes.compactMap(\.payload), lines.map(\.text))

// Train an expense classifier on-device, then use it.
let report = try await TextClassifierTrainer().train(
    csvAt: URL(fileURLWithPath: "/path/to/expenses.csv"),
    writingModelTo: URL.documentsDirectory.appending(path: "Expenses.mlmodel"),
    algorithm: .transferLearning(.bertEmbedding))
let category = try await TextClassifierTool(modelAt: report.modelURL).classify("Taxi to the airport").label
```

Every tool reports `availability()`; check it before calling, since the execution methods do not.
For example, Vision classification, barcodes and faces report unavailable in the Simulator, and
Create ML training is unavailable in the iOS / visionOS Simulator.
`SystemToolRegistry.discover()` lists the built-in tools with their availability;
`CoreMLModelTool`, `TextClassifierTool` and `CoreMLTextEmbeddingTool` are created per model file or bundle instead. Try them with `aura tools` and `aura ml …`, or in the **ML** tab of the
example app. Full guide, with a train → ship → classify walkthrough and a bring-your-own-model
section: [On-device ML tools](docs/guide/ml-tools.md).

---

## CLI & Binaries

Two ways to run and ship these features beyond the source package.

### `aura` CLI

A headless integration harness (macOS) that drives the hybrid + native-tool features —
handy as a reference and in CI. Build with `scripts/build-cli.sh` (or
`swift build -c release --product aura`).

```
aura providers                      # detect Ollama / llama-server + models
aura tools                          # list on-device ML tools by category, with availability
aura ask "<prompt>" [--provider auto|local|openai|anthropic] [--model <id>] [--base-url <url>]
                                    # ask a bigger model: your llama-server/Ollama unless you name a cloud API
aura ocr <image>                    # extract text via native Vision OCR
aura ml <subcommand> …              # on-device ML: classify-image, barcodes, faces, ocr-lines,
                                    # language, entities, sentiment, similarity, sounds,
                                    # coreml-describe, coreml-predict, train-text, classify-text
aura imagegen "<prompt>" [--lora <path>] …  # FLUX image generation via mflux (macOS)
aura models search|check|devices …  # which Hugging Face models run here, and why not
```

`ask` sends to a running llama-server, else Ollama, and never picks a cloud API on its own.
`--provider openai` / `--provider anthropic` use `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (else the
Keychain accounts `cloud.openai` / `cloud.anthropic`); `--base-url <url> --model <id>` targets any
other OpenAI-compatible server, with an optional `AURA_API_KEY`. All flags:
[CLI guide](docs/guide/cli.md).

### Demo app (`.dmg`)

`scripts/build-app.sh` builds `AuraExample.app` and packages it into
`build/AuraExample.dmg` with a drag-to-**Applications** alias. The app is
development-signed (runs on this Mac); to distribute it, re-sign with a Developer ID
identity and notarize (`xcrun notarytool submit … && xcrun stapler staple …`).
Requires macOS 26 (AgentCrew).

### Model Finder (`Examples/ModelFinder`)

A small iOS 18 / macOS 15 app that searches Hugging Face and shows, per repo, whether AuraLocal's pinned
runtimes can run it on a chosen device (this one, iPhone classes, Macs) and why not — with the exact
`models.json` entry for the ones that run. See [its README](Examples/ModelFinder/README.md) and
[Finding compatible models](docs/guide/models.md#finding-compatible-models).

The same check from code:

```swift
import AuraCore

let report = await ModelCompatibilityChecker().check("mlx-community/Qwen3-4B-4bit", on: .iPhone8GB)
print(report.status.label, "—", report.headline)
for finding in report.blockers { print("blocker:", finding.title) }
if let entry = report.suggestedEntry { print(entry.jsonText()) }   // the models.json entry, when it runs
```

---

## Text Chat

```swift
import AuraCore

// MLX small model — one-liner
let reply = try await AuraLocal.chat("How much did I spend this week?")

// Reusable instance (loads model once — preferred for multiple calls)
let llm = try await AuraLocal.text(.qwen3_1_7b) { progress in
    print(progress) // "Downloading Qwen3 1.7B: 42%"
}
let summary = try await llm.chat("Summarize my expenses")

// Streaming
for try await token in llm.stream("Explain this transaction") {
    print(token, terminator: "")
}

// With system prompt
let answer = try await llm.chat(
    "What is the VAT rate in Mexico?",
    systemPrompt: "You are a personal finance assistant."
)
```

### Using GGUF Large Models

GGUF models are downloaded automatically from HuggingFace on first use. Use `ModelManager` to track progress:

```swift
import AuraCore

// Load via ModelManager — handles GGUF download + backend selection automatically
let llm = try await ModelManager.shared.load(.llama3_1_8b_gguf)

// Observe download progress (in a SwiftUI view: @ObservedObject var manager = ModelManager.shared)
let manager = ModelManager.shared

switch manager.state(for: .llama3_1_8b_gguf) {
case .idle: break                                   // not started
case .downloading(let progress): print(progress)    // human-readable progress text
case .loading: break                                // downloaded, loading into memory
case .ready: break                                  // ready for inference
case .failed(let error): print(error)               // download or load error
}

// Chat — same API regardless of backend
let reply = try await llm.chat("Write a short story about a robot.")

// Streaming — works with all backends
for try await token in llm.stream("Explain quantum computing simply") {
    print(token, terminator: "")
}
```

### Text Models — MLX (GPU, Apple Silicon)

| Model | Size | Best for |
|-------|------|----------|
| `.qwen3_0_6b` | ~400 MB | Ultra-fast responses |
| `.qwen3_1_7b` ⭐ | ~1.0 GB | Balanced (default) |
| `.qwen3_4b` | ~2.5 GB | Higher quality |
| `.gemma3_1b` | ~700 MB | Google alternative |
| `.phi3_5_mini` | ~2.2 GB | Microsoft alternative |
| `.llama3_2_1b` | ~700 MB | Meta, lightweight |
| `.llama3_2_3b` | ~1.8 GB | Meta, higher quality |

### Text Models — GGUF (llama.cpp)

| Model | Disk Size | Smallest iOS preset | Smallest Mac preset | Notes |
|-------|-----------|---------------------|---------------------|-------|
| `.llama3_1_8b_gguf` ⭐ | ~4.7 GB | 4 GB (streaming) | 16 GB | Primary large model |
| `.qwen2_5_7b_gguf` | ~4.4 GB | 4 GB (streaming) | 16 GB | **Download broken**: upstream ships this quant split into shards |
| `.mistral_7b_gguf` | ~4.1 GB | 4 GB (streaming) | 16 GB | Fast, efficient |
| `.phi3_medium_gguf` | ~8.0 GB | 6 GB (streaming) | 16 GB | High quality |
| `.gemma2_9b_gguf` | ~5.5 GB | 4 GB (streaming) | 16 GB | Google Gemma 2 |
| `.qwen2_5_32b_gguf` | ~18.5 GB | 12 GB iPhone 17 Pro (streaming) | 32 GB (streams on 16 GB) | **Download broken**: upstream ships this quant split into shards |
| `.llama3_1_70b_gguf` | ~40.0 GB | Not viable | 64 GB (streams on 32 GB) | Largest in the catalog |

Each cell is the smallest device preset at which `HardwareAnalyzer.assess` rates the model runnable. The
budgets are estimates (except the measured 32 GB Mac), and the iPhone cells assume the memory
entitlements; see [Large Models — GGUF](docs/guide/models.md#large-models--gguf).

> **Tip:** Use `HardwareAnalyzer.compatibleModels()` to get a device-specific list sorted by fit level.

---

## Vision / Image Analysis

```swift
import AuraCore

// One-liner receipt extraction
let json = try await AuraLocal.extractDocument(receiptImage)
// → {"store":"OXXO","date":"2026-03-06","items":[...],"total":125.50,"currency":"MXN"}

// Reusable instance
let vlm = try await AuraLocal.vision(.qwen35_0_8b) { print($0) }

// Free-form image analysis
let description = try await vlm.analyze("What items are on this receipt?", image: photo)

// Streaming with image
for try await token in vlm.streamVision("Describe this image", image: photo) {
    print(token, terminator: "")
}
```

### Vision Models

| Model | Size | Best for |
|-------|------|----------|
| `.qwen35_0_8b` ⭐ | ~625 MB | Default, iPhone |
| `.qwen35_2b` | ~1.7 GB | iPad, higher accuracy |
| `.smolvlm_500m` | ~1.0 GB | Minimum memory |
| `.smolvlm_2b` | ~1.5 GB | SmolVLM, balanced |

---

## OCR & Document Extraction

Specialized models optimized for receipts, invoices, and structured documents.

```swift
import AuraCore

// FastVLM — outputs structured JSON
let ocr = try await AuraLocal.specialized(.fastVLM_0_5b_fp16) { print($0) }
let json = try await ocr.extractDocument(receiptImage)

// Granite Docling — outputs DocTags, converted to Markdown
let docOCR = try await AuraLocal.specialized(.graniteDocling_258m)
let raw = try await docOCR.extractDocument(documentImage)
let markdown = AuraLocal.parseDocTags(raw)
```

### Specialized Models

| Model | Size | Output |
|-------|------|--------|
| `.fastVLM_0_5b_fp16` ⭐ | ~1.25 GB | JSON (receipts) |
| `.fastVLM_1_5b_int8` | ~800 MB | JSON (receipts) |
| `.graniteDocling_258m` | ~631 MB | DocTags → Markdown |
| `.graniteVision_3_3` | ~1.2 GB | Plain text |

---

## Hardware Compatibility

```swift
import AuraCore

// Check fit for the current device
let result = HardwareAnalyzer.assess(.llama3_1_8b_gguf)

switch result.fitLevel {
case .excellent:         break  // runs comfortably
case .good:              break  // runs well
case .marginal:          break  // runs but may be slow
case .streamingRequired: break  // too large for monolithic load; uses layer-streaming
case .tooLarge:          break  // cannot run on this device even with streaming
}

print(result.fitLevel.isRunnable) // true for all except .tooLarge

// Get all compatible models sorted by fit level (best first)
let compatible = HardwareAnalyzer.compatibleModels()

// Custom profile (for UI preview or device picker)
let profile = HardwareProfile(totalMemoryGB: 8.0, availableMemoryGB: 4.0, deviceName: "iPhone 15")
let results = HardwareAnalyzer.compatibleModels(profile: profile)
```

### Memory Budget — Layer-Streaming Mode (7B Q4 on 6 GB iPhone)

| Component | Memory |
|-----------|--------|
| iOS system | ~2.5 GB |
| Available for app | ~1.5 GB |
| Current layer weights (Q4) | ~130 MB |
| Prefetched next layer | ~130 MB |
| Embedding table | ~130 MB |
| KV cache (1024–2048 tokens, GQA, fp16) | ~56–256 MB |
| Activations + overhead | ~80 MB |
| Safety margin | ~250 MB |
| **Total app usage** | **~0.8–1 GB** (incl. the 250 MB margin) |

---

## Background Lifecycle (iOS)

`BackgroundLifecycle` tracks whether an iOS app is in the background. It does **not** stop a running
generation, and nothing in AuraCore reads `isPaused`, so check it before starting one. Touch
`BackgroundLifecycle.shared` at launch so it starts observing. With `aggressiveMemorySaving`,
backgrounding evicts every model except the most recently used.

```swift
import AuraCore

// true while the app is in the background
if BackgroundLifecycle.shared.isPaused {
    // wait before starting a new generation
}

// Evict every model except the most recently used when backgrounded
BackgroundLifecycle.shared.aggressiveMemorySaving = true
```

This is a no-op on macOS where apps are not suspended.

---

## Receipt Scanner Example

```swift
import Foundation
import AuraCore

struct ReceiptData: Codable {
    let store: String
    let date: String
    let items: [Item]
    let subtotal: Double
    let tax: Double
    let total: Double
    let currency: String

    struct Item: Codable {
        let name: String
        let quantity: Int
        let price: Double
    }
}

func scanReceipt(_ image: PlatformImage) async throws -> ReceiptData {
    let json = try await AuraLocal.extractDocument(image)
    return try JSONDecoder().decode(ReceiptData.self, from: Data(json.utf8))
}
```

---

## Conversation Persistence

`ConversationStore` provides a SQLite-backed store (no external dependencies) for persisting chat history. The LLM automatically loads a context window of the most recent turns that fit within the token budget.

```swift
import AuraCore

let store = ConversationStore.shared

// Create a conversation
let conv = try await store.createConversation(model: .qwen3_1_7b, title: "Finance assistant")

// Chat with automatic history — context window managed automatically
let llm = try await AuraLocal.text(.qwen3_1_7b)
let reply  = try await llm.chat("What is 2+2?", in: conv.id)
let reply2 = try await llm.chat("Why?", in: conv.id) // includes previous exchange

// Streaming with history
for try await token in llm.stream("Tell me more", in: conv.id) {
    print(token, terminator: "")
}

// One-liner (creates conversation automatically)
let (greeting, convID) = try await AuraLocal.chat("Hello", model: .qwen3_1_7b)

// List all conversations
let conversations = try await store.allConversations()

// Full-text search across all messages
let results = try await store.search("VAT Mexico")

// Auto-title based on first message
try await llm.autoTitle(conversationID: conv.id)

// Prune and summarize long conversations
try await llm.summarizeAndPrune(conversationID: conv.id)
```

### Context Window Management

When a conversation exceeds the token budget, `summarizeAndPrune` uses the model itself to summarize older turns and replace them with a compact system-level summary — preserving semantic continuity without truncating abruptly.

```swift
// AuraUI's TextChatTab calls this after every reply; with the AuraLocal API, call it yourself
// (a no-op below maxContextTokens)
try await llm.summarizeAndPrune(
    conversationID: conv.id,
    keepLastN: 10,         // always keep the 10 most recent turns
    maxContextTokens: 4096
)
```

---

## Voice Interface

`AuraVoice` provides a turn-based (half-duplex) voice pipeline using only Apple frameworks — no external dependencies, no network calls.

```
Microphone → SFSpeechRecognizer (on-device) → AuraLocal.stream() → AVSpeechSynthesizer
```

Sentences are streamed to TTS **while the LLM is still generating** — the assistant starts speaking after the first complete sentence, not after the full response.

Speech is recognized in one locale: `Config.locale`, else the device's first preferred language, else `en-US`. Each sentence of the reply is language-detected with `NLLanguageRecognizer` to pick the TTS voice (enhanced quality first). A code such as `"es"` becomes the first of the user's preferred languages that starts with it (`"es-MX"` only if listed), otherwise it stays `"es"`.

### Drop-in button

```swift
import AuraVoice

// Minimal — manages its own VoiceSession internally
VoiceButton(llm: llm)

// With external session for full state control
@StateObject var session = VoiceSession(llm: llm)

VoiceButton(session: session)
Text(session.transcript)  // live STT transcript
Text(session.response)    // live LLM response
```

### Full voice chat view

```swift
import AuraVoice

// Complete UI: transcript bubble + response bubble + VoiceButton
VoiceChatView(llm: llm)

// With persistent conversation
VoiceChatView(llm: llm, conversationID: conv.id)
```

### Manual pipeline control

```swift
import AuraVoice

let session = VoiceSession(llm: llm, conversationID: conv.id)

// Request permissions once on launch
let granted = await session.requestPermissions()

// Start — silence detection triggers LLM automatically
try await session.startListening()

// Or stop manually
await session.stopListening()

// Interrupt TTS mid-sentence
session.interrupt()

// Stop recording and TTS (an LLM reply in progress keeps generating and speaking)
session.cancel()
```

### Configuration

```swift
var config = VoiceSession.Config()
config.silenceThreshold     = 1.4    // seconds of silence before triggering LLM
config.maxRecordingDuration = 30     // max recording time in seconds
config.speakingRate         = 0.5    // TTS rate (0–1)
config.maxTokens            = 512    // max LLM tokens per response
config.systemPrompt         = "You are a helpful assistant. Be concise."

let session = VoiceSession(llm: llm, config: config)
```

### VoiceSession States

| State | Meaning |
|-------|---------|
| `.idle` | Ready, waiting for input |
| `.listening` | Recording + live transcription |
| `.thinking(partial:)` | LLM streaming, partial response available |
| `.speaking(sentence:)` | TTS playing current sentence |
| `.error(String)` | Something went wrong |

### Required permissions

Add to your `Info.plist`:

```xml
<key>NSSpeechRecognitionUsageDescription</key>
<string>Used for voice input to the local AI assistant.</string>
<key>NSMicrophoneUsageDescription</key>
<string>Used to capture your voice for the AI assistant.</string>
```

---

## Document Library (RAG)

`AuraDocs` provides a fully local Retrieval-Augmented Generation (RAG) pipeline. Index documents once, then ask questions in natural language. No API keys, no cloud services.

### Supported formats

| Format | Parser |
|--------|--------|
| `.pdf` | PDFKit (text extraction per page) |
| `.docx` | ZIP + XML (no external dependencies) |
| `.txt`, `.md`, `.markdown` | Plain text |
| `.png`, `.jpg`, `.jpeg`, `.heic`, `.tiff` | MLX VLM OCR (needs a `visionLLM` in `configure`; otherwise `unsupportedFormat`) |

### Retrieval pipeline

```
query → embed (TF-IDF, or multilingual-e5-small) → FTS5 top-20 candidates → cosine re-rank top-5 → LLM
```

Two-stage hybrid search: FTS5 for fast keyword recall, cosine similarity for semantic precision. All vectors stored as BLOBs in SQLite — no external vector database required.

**Dense multilingual embeddings (opt-in).** `AutoEmbeddingProvider()` stays TF-IDF: no download.
`AutoEmbeddingProvider(embeddingModelAt: bundleURL)` uses **multilingual-e5-small** on the Neural
Engine (384-dim, Spanish/English and ~100 other languages; 2–20 ms per chunk on an M1 Pro) when its
~225 MB bundle is installed, and TF-IDF otherwise. Build the bundle with
`uv run scripts/embeddings/convert_e5_coreml.py --out <dir>`, call `warmUp()` before use (the first
load compiles the model for the Neural Engine, ~35 s), and mind the 512-token limit per chunk and
the ~95 MB its tokenizer takes in memory. Switching providers re-embeds the stored chunks
automatically. See [Embedding providers](docs/guide/rag.md#embedding-providers).

### Quick start

```swift
import AuraCore
import AuraDocs

// 1. Configure once (e.g. in app startup)
let llm      = try await AuraLocal.text(.qwen3_1_7b)
let embedder = AutoEmbeddingProvider()

let library = DocumentLibrary.shared
await library.configure(embeddingProvider: embedder, llm: llm,
                        visionLLM: try await AuraLocal.vision())   // needed to index images
try await library.open()

// 2. Index documents — progress delivered on @MainActor
try await library.add(url: pdfURL) { progress in
    print(progress) // "Embedding MyDoc: 42%"
}
try await library.add(url: docxURL)
try await library.add(url: imageURL)   // OCR via VLM

// Rebuild TF-IDF weights after indexing
await library.refreshCorpus()

// 3. Ask questions
let answer = try await library.ask("What is the contract amount?")
print(answer.text)

// 4. Inspect sources
for source in answer.sources {
    print("[\(source.documentTitle) p.\(source.pageNumber)] score: \(source.score)")
    print(source.excerpt)
}
```

### Document chat

```swift
import AuraDocs

// DocumentChat keeps a message list and cites sources per message
let chat = DocumentChat(library: library, llm: llm)

let reply1 = try await chat.send("What is the payment schedule?")
let reply2 = try await chat.send("What are the penalties for late payment?") // answered on its own

for msg in chat.messages {
    print(msg.role, msg.text)
    print(msg.sources.map { $0.documentTitle }) // cited documents
}
```

> **Limitation:** `DocumentChat` answers each question independently. Earlier turns are kept in
> `messages` and persisted to `ConversationStore`, but never sent to the model, so a follow-up must
> name its subject.

### Advanced options

```swift
// Custom chunk size and overlap
let library = DocumentLibrary(
    chunkTargetTokens:    512,   // target tokens per chunk
    chunkOverlapFraction: 0.1    // 10% overlap between chunks
)

// Ask with more context
let answer = try await library.ask(
    "Summarize the key obligations",
    topK:             8,      // retrieve 8 chunks (default 5)
    maxContextTokens: 4096,   // context budget for LLM
    systemPrompt:     "You are a legal assistant. Be precise and cite page numbers."
)

// Manage library
let docs = try await library.allDocuments()
try await library.removeDocument(id: doc.id)
```

### Drop-in tab

Add `DocsTab` to any existing `TabView`:

```swift
import AuraDocs

TabView {
    // ... existing tabs
    DocsTab()
        .tabItem { Label("Docs", systemImage: "doc.text.magnifyingglass") }
}
```

---

## Prebuilt SwiftUI Interface

`AuraUI` ships composable tab views — drop them into your **own** `TabView`. (There is no
prebuilt `ContentView`; the one in the Example app is demo code, not part of the library.)

```swift
import SwiftUI
import AuraUI

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                TextChatTab().tabItem { Label("Chat", systemImage: "bubble.left") }
                VisionTab().tabItem  { Label("Vision", systemImage: "photo") }
                OCRTab().tabItem     { Label("OCR", systemImage: "doc.text.viewfinder") }
            }
        }
    }
}
```

For voice, add `AuraVoice` and place `VoiceChatView(llm:)` in a tab (it needs an `AuraLocal` instance).

| Tab | Module | Description |
|-----|--------|-------------|
| **Text** | `AuraUI` | Multi-conversation chat; MLX + GGUF model picker with backend badges |
| **Vision** | `AuraUI` | Image analysis with standard and streaming modes |
| **OCR** | `AuraUI` | Document and receipt extraction |
| **Models** (`ModelSection`) | `AuraUI` | A `List` section, not a tab: download status, backend badge, fit badge per model |
| **Voice** | `AuraVoice` | Turn-based voice chat; TTS voice picked per sentence |
| **Docs** | `AuraDocs` | Document library and RAG chat |

`ModelSection` shows three badge types per model:
- **MLX** (blue) — GPU inference via mlx-swift
- **GGUF** (purple) — Full load via llama.cpp
- **STREAM** (orange) — Layer-streaming via llama.cpp (low RAM mode)

---

## Model Management

`ModelManager` is the recommended way to load models. It prevents redundant downloads, shares instances across tabs, handles GGUF downloads transparently, and manages memory pressure automatically.

```swift
import AuraCore

// Load from anywhere — returns cached instance if already loaded
// For GGUF models, downloads the file from HuggingFace first
let llm = try await ModelManager.shared.load(.qwen3_1_7b)
let largeLLM = try await ModelManager.shared.load(.llama3_1_8b_gguf)

// Observe per-model state (in a SwiftUI view: @ObservedObject var manager = ModelManager.shared)
let manager = ModelManager.shared

switch manager.state(for: .llama3_1_8b_gguf) {
case .idle: break                                // not loaded
case .downloading(let progress): print(progress) // human-readable progress text
case .loading: break                             // file downloaded, loading into memory
case .ready: break                               // ready for inference
case .failed(let error): print(error)            // download or load error
}

// Check which backend a model will use
let backend = manager.recommendedBackend(for: .llama3_1_8b_gguf)
// → .llamaCpp or .layerStreaming depending on device

// Manual eviction
ModelManager.shared.evict(.llama3_1_8b_gguf)
ModelManager.shared.evictAll()
```

### Memory Budget

The LRU cache budget is computed once, when `ModelManager.shared` is created: (available memory − 2 GB)
÷ 1.5 GB, rounded down and clamped to 1…4 models. It is 1 when the OS reports no figure (the process is
at its jetsam limit).

When the OS sends a memory warning (`DispatchSource.makeMemoryPressureSource` + `UIApplication.didReceiveMemoryWarningNotification`), all models except the most recently used are evicted immediately.

---

## Entitlements

Add to your `.entitlements` file for models larger than 500 MB:

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
```

For GGUF models on macOS (models stored outside the sandbox):

```xml
<key>com.apple.security.files.user-selected.read-write</key>
<true/>
```

---

## Concurrency Model

AuraLocal is designed for Swift 6 strict concurrency:

| Type | Isolation | Rationale |
|------|-----------|-----------|
| `AuraLocal` | `@MainActor` | Wraps MLX/llama.cpp callbacks that fire on main thread |
| `AuraEngine` | `@MainActor` | Delegates to `InferenceBackend` protocol |
| `MLXBackend` | `@MainActor` | Owns MLX model container and GPU state |
| `LlamaCppBackend` | `@MainActor` | Owns `LLMSession` from LocalLLMClient |
| `LayerStreamingBackend` | `@MainActor` | Owns streaming session + `MemoryBudgetManager` |
| `MemoryBudgetManager` | `@MainActor` | Reads `os_proc_available_memory()` and publishes pressure state |
| `BackgroundLifecycle` | `@MainActor` | Drives UIApplication notifications and published `isPaused` |
| `ModelManager` | `@MainActor` | `ObservableObject` publishing `@Published` state |
| `GGUFModelDownloader` | `@MainActor` | `ObservableObject` publishing download progress |
| `ConversationStore` | `actor` | Serializes SQLite reads/writes without locks |
| `DocumentLibrary` | `actor` | Coordinates parsing, embedding, and vector store |
| `VoiceSession` | `@MainActor` | Drives `AVAudioEngine` + `SFSpeechRecognizer` on main |
| `Model`, `Turn`, `Conversation` | `Sendable` | Value types safe to pass across isolation boundaries |

All streaming APIs use `AsyncThrowingStream` to bridge inference callbacks to Swift async/await.

---

## Architecture

```
AuraCore
├── InferenceBackend (protocol)
│   ├── MLXBackend          →  MLXLLM / MLXVLM (Apple Silicon GPU, .mlx models)
│   ├── LlamaCppBackend     →  LocalLLMClient → llama.cpp Metal (GGUF that fits in memory)
│   └── LayerStreamingBackend → LocalLLMClient → mmap streaming (GGUF that does not fit)
│
├── BackendRouter           →  selects backend by model.format + HardwareAnalyzer
├── AuraEngine              →  thin delegator to InferenceBackend
├── AuraLocal               →  public facade (.text / .vision / .specialized / .chat)
│
├── HardwareAnalyzer        →  fit levels (excellent / good / marginal / streamingRequired / tooLarge)
│                              GQA-aware KV cache estimates, streaming memory budget
├── MemoryBudgetManager     →  jetsam monitoring, adaptive context, per-generation pressure checks
├── BackgroundLifecycle     →  iOS background flag (isPaused) + optional eviction
│
├── ModelManager            →  LRU cache, memory-pressure eviction, GGUF download orchestration
├── GGUFModelDownloader     →  HuggingFace downloads with resume + @Published progress
│
├── Hybrid/ + Remote/       →  HybridEscalator, EscalationRouter, providers, CostLedger
├── SystemTools/            →  Vision · NaturalLanguage · SoundAnalysis · Core ML · Create ML tools, SystemToolRegistry
├── ModelCompatibility/     →  ModelCompatibilityChecker, DevicePreset, CompatibilityReport
│
├── ConversationStore       →  SQLite-backed chat history (actor)
└── AuraLocal+History       →  context window · auto-title · summarize+prune

AuraUI (optional) — composable views; you supply the TabView
├── TextChatTab   →  TextChatViewModel  →  ConversationStore
│                    Model picker: MLX section + GGUF section
├── VisionTab     →  VisionViewModel
├── OCRTab        →  OCRViewModel
└── ModelSection  →  ModelRow · BackendBadge · FitBadge  (a component, not a full tab)

AuraVoice (optional)
├── VoiceSession          →  SFSpeechRecognizer (on-device STT)
│                         →  AuraLocal.stream() + ConversationStore
│                         →  AVSpeechSynthesizer (on-device TTS)
├── VoiceButton           →  SwiftUI mic button with state animations
└── VoiceChatView         →  Full voice chat UI  (VoiceChatView(llm:))

AuraDocs (optional)
├── DocumentLibrary          →  add() · ask() · allDocuments() · refreshCorpus()
├── DocumentParserDispatcher →  PDF (PDFKit) · DOCX (ZIP+XML) · TXT · Image (VLM OCR)
├── DocumentChunker          →  sliding window · sentence boundaries · overlap
├── AutoEmbeddingProvider    →  TF-IDF sparse (default) · multilingual-e5-small via Core ML (opt-in)
├── VectorStore              →  SQLite BLOB vectors · FTS5 pre-filter · cosine re-rank
├── DocumentChat             →  per-question Q&A · source citations · history in ConversationStore
└── DocsTab                  →  SwiftUI tab · file picker · progress bar · chat sheet

Sources/
├── AuraCore/
│   └── LlamaCpp/           (LlamaCppBackend, LayerStreamingBackend, MemoryBudgetManager,
│                            GGUFModelDownloader, BackgroundLifecycle)
├── AuraUI/
├── AuraVoice/
├── AuraDocs/
├── AuraAgents/
├── AuraAppleIntelligence/
├── AuraImageGen/
├── aura/                   (CLI)
└── AuraExample/

Examples/
└── ModelFinder/

MLX models download automatically and are cached at:
  ~/Library/Caches/huggingface/hub/models--<org>--<repo>/snapshots/main/

GGUF models are downloaded to:
  ~/Library/Caches/models/<org>/<repo>/<file>.gguf
```

---

## License

MIT
