# Changelog

Notable changes to AuraLocal. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Swift package consumers pin AuraLocal by `revision:` (a version requirement cannot resolve; see
[Installation](docs/guide/installation.md#pin-a-commit)), so a tag marks a release and locates the `aura` CLI
tarball; it is not a version you can require.

## [Unreleased] (planned 1.4.0)

Changes since `v1.3.0` (7653c93, 2026-07-08). `v0.1.0` (97ab9d1, 2026-07-10) was tagged two days later on the
same history, so it already contains the first hybrid, SystemTool and `aura` CLI entries below.

### License

- AuraLocal is now licensed under **AGPL-3.0-or-later**, or under a commercial license from the copyright holder
  (af056da). Commits up to and including 1647874, among them every earlier tag (`v0.1.0`, `v1.0.0`–`v1.3.0`),
  remain under the MIT License. See [License](README.md#license).
- Contributions are accepted under the terms in [CONTRIBUTING.md](CONTRIBUTING.md) and must be signed off.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the dependencies' licenses.

### Breaking changes

- `InferenceBackend` has a new requirement, `generate(prompt:systemPrompt:maxTokens:constraint:onToken:)`.
  A custom backend must implement it; one that cannot enforce the constraint throws
  `GenerationConstraintError.unsupportedBackend`.
- `Model.textModels`, `.visionModels`, `.specializedModels`, `.ggufModels`, `.mlxModels`, `.runnableModels`,
  `Model.models(in:)`, `HardwareAnalyzer.compatibleModels()` and `HardwareAnalyzer.compatibleModels(purpose:)`
  leave uncensored models out, so AuraUI's pickers list one only once it is downloaded. `Model.uncensoredModels`
  lists them, `Model.allModels` still has every model, and AuraExample's Catalog shows them behind a toggle.
- `Model.gemma4_1b`, `.gemma4_4b`, `.granite4_compact`, `.lfm25_3_2b`, `.ministral_3b` and `.graniteVision_3_3`
  are `@available(*, unavailable)`, with the reason in the message: none of them ever downloaded a loadable model.
- Tools on an MLX model throw `ModelConfigurationError.toolsUnsupported(.mlx)` from `ModelManager.load` and
  `AuraSession` instead of being dropped silently. GGUF tool calling is unchanged.
- Hybrid API, for code written against `v0.1.0` or a revision before 2026-09-30:
  - `OpenAICompatibleProvider.gitHubModels(apiKey:)` and the `cloudTargets` overload taking `gitHubModelsModel:`
    are `@available(*, unavailable)`: GitHub retired GitHub Models on 2026-07-30.
  - `ConsentGate.requestConsent(target:preview:projectedCostUSD:)` takes `Decimal?` (`nil` when the model is
    unpriced), and the protocol gains `requestConsent(target:preview:offer:)`, which has a default.
  - `CostLedger.Record.costUSD` is `Decimal?`, `RoutingInput.projectedCostUSD` is `Decimal?`, and
    `EscalationReason` has a new case, `.costUnknown`.
  - The static `CostLedger.projectedCost(target:inputTokens:maxOutput:)` is deprecated in favour of the instance
    method.

### Added

- Hybrid local + remote inference: `HybridEscalator`, `EscalationRouter`, `OpenAICompatibleProvider`,
  `AnthropicProvider`, `ConsentGate` with the exact payload, `KeychainStore` (bring your own key),
  `PIIRedactor`, `CostLedger` with per-model prices and a session cap, `ResponseCache` and `ContextCompressor`.
- `AuraAgents`: the `AgentCrew` multi-agent pipeline as a library, with per-step escalation and recall of what
  earlier runs established.
- `AuraImageGen`: Core ML Stable Diffusion on iOS and macOS, including models inspected and adopted from Hugging
  Face (`CoreMLDiffusionInspector`), and FLUX + LoRA through mflux on macOS.
- On-device ML tools: Vision OCR (`VisionOCRTool`, `recognizeLines`), image classification, barcodes and faces;
  language identification, entities and sentiment; sound classification; Core ML models; text classifiers and
  Create ML training; `CoreMLTextEmbeddingTool` and opt-in multilingual-e5-small embeddings in `AuraDocs`.
- The `aura` CLI: `providers`, `tools`, `ask`, `ocr`, `ml`, `models search|check|devices` and `imagegen`.
- `ModelCompatibilityChecker`: whether a Hugging Face repo runs on the pinned runtimes, and why not; Hugging Face
  search with a device-fit filter; the Model Finder example app; the AuraUI Hugging Face browser.
- `ModelDownloads`: resumable GGUF and MLX downloads, gated and private repos, one folder of a repo pinned by
  revision (`HFRepoInfo.revision`, `deleteFolder`), and `DownloadRecords` that survive relaunches.
- GGUF: `GenerationConstraint` (`.json`, `.grammar(gbnf:root:)`) enforced in llama.cpp's sampler, and
  conversations rendered with each model's chat template and real roles.
- `GenerationStats` after each generation, a decode-speed estimate, and `HardwareAnalyzer.assess` judging wall
  clock as well as memory.
- KV cache sizing per layer type and MoE decode by active experts; `Model.maxContextLength`; MLX `maxKVSize` and
  `lastContextOverflow`.
- iOS: generation stops when the app backgrounds (`GenerationInterrupted`), an opt-in `DeviceConditions` guard,
  and models judged against the app's memory limit.
- `AuraSession` enforces an `OutputSchema` by validation and repair.
- `temperature:` and `topP:` on `ModelManager.load` and `AuraLocal.text`, `.vision` and `.specialized`.
- `PrivacyInfo.xcprivacy` manifests for `AuraCore`, `AuraImageGen` and the AuraExample app.
- `VisionDocumentTool` (`system.vision.document`, iOS 26 / macOS 26): Vision's `RecognizeDocumentsRequest` with
  paragraphs, tables and lines with boxes, and the money amounts (as `Decimal` plus currency), dates, contacts and
  links Data Detection finds; `aura ml document <image>` prints them.
- `aura receipt extract|eval`: receipt fields (merchant, date, total, tax, currency) read from a photo by a
  rule-based baseline and, with `--model`, by a local model at temperature 0 under a JSON Schema. Model values are
  checked against the OCR text, and `eval` scores both against a labelled CSV with pass/fail thresholds (exit
  status 1 on a failed gate, 2 when the run cannot finish or an image cannot be read). Currency is reported but not
  scored or checked. The deterministic part lives in the `ReceiptSpike` target, which is not a library product.
- `Model.license`, the license the model's Hugging Face card states, and `Model.hasPermissiveLicense` (Apache-2.0
  or MIT; an unrecorded license is not permissive). The catalog's `license` key is checked and written by
  `scripts/verify_catalog.py`, and `aura models check --entry` fills it in. 46 of the 53 entries record one; the
  other seven model cards declare none. `Model.listedModels` is the catalog
  without its uncensored entries.
- Continuous integration (`.github/workflows/ci.yml`) and a weekly canary that resolves without the lockfile.

### Changed

- MLX is back: `AuraCore` links mlx-swift-lm again and offers both engines.
- LocalLLMClient is pinned by revision `edc39ef` (tag 0.5.0, llama.cpp b8851); mlx-swift-lm from 3.31.3;
  ml-stable-diffusion exact 1.1.1; mlx-swift `0.31.3..<0.31.5`.
- Toolchain floor: Swift 6.1 / Xcode 16.3; Xcode 26 for `AuraAppleIntelligence` and `AuraAgents`; Xcode 26.4 for
  `AuraImageGen` and the `aura` CLI.
- The model catalog is verified against Hugging Face (`scripts/verify_catalog.py`); several entries moved repo
  (Dolphin 8B to `dphn/`, so it downloads again).
- The context window follows the model's KV cost and the memory left after its weights, capped at what the
  weights support; memory on macOS is measured instead of assumed.
- Every GGUF turn starts from an empty KV cache and re-prefills the history.
- `aura ask` uses a local llama-server or Ollama by default, or OpenAI, Anthropic or a custom endpoint when named.

### Fixed

- AuraLocal resolves for integrators (the swift-syntax conflict between LocalLLMClient and mlx-swift-lm).
- AuraCore builds for the iOS Simulator, and GGUF runs there.
- Conversation search after deletes, turns stored in the same millisecond, one reply per prompt when a generation
  stops, and `stream(_:in:)` and Apple Intelligence streams cancelling when the consumer stops.
- Consent shows, caches and prices the exact escalation payload; redaction applies on the policy path; the cost
  cap holds at the consent gate; unknown spend is never priced as $0.
- A loaded model stays ready when another sampling load fails.
- RAG document IDs survive relaunches, and vectors from a provider the index no longer matches are never stored.
- The memory safeguards that were documented but inert (`evictAllButMostRecent`, the iOS OOM guards).
- `scripts/package-cli.sh` finds the build products of newer SwiftPM (`.build/out/Products/Release`), and the
  `aura` tarball now carries the SwiftPM resource bundles and the license texts.

### Removed

- The GitHub Models preset; `cloudTargets` no longer reads the `cloud.github-models` Keychain entry.
- Six catalog entries (see Breaking changes).

### Security

- Cloud API keys live only in the Keychain, this device only.
- The compatibility checker no longer traps on crafted GGUF headers or `config.json` files.
- mlx-swift is held below 0.31.5: 0.31.6 is the version in an open report of an app hanging before `main()` on an
  iPhone running iOS 27 (mlx-swift #485), and 0.31.5 does not build for iOS (#436).

## [0.1.0] - 2026-07-10

First public release, MIT licensed: the `aura` CLI tarball for Homebrew, the hybrid MVP, the SystemTool layer and
the LocalLLMClient revision pin.

[0.1.0]: https://github.com/iOSDevC/AuraLocal/releases/tag/v0.1.0
