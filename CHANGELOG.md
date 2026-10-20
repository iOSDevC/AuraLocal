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
- Tools on an MLX vision or document model throw `ModelConfigurationError.toolsUnsupported(.mlx)` from
  `ModelManager.load` and `AuraSession` instead of being dropped silently. GGUF tool calling is unchanged.
- **Behaviour change (MLX tools):** an MLX text model loaded with tools runs them with the same tool loop as GGUF
  (`toolLoop`, `ToolEvent`, `lastToolTurn`, `{"error": …}` results, one round by default), with calls parsed in the
  format mlx-swift-lm infers from the model's `config.json`. `ModelCapabilities.tools` includes MLX text models
  outside the iOS Simulator.
- **Behaviour change (MLX cancellation):** an MLX call cancelled before it starts, or while it waits for another
  generation on the same weights, throws `CancellationError` without prefilling, as on GGUF, instead of
  prefilling the prompt and returning one token's text.
- **Behaviour change (MLX prompt cache):** an MLX text model keeps the KV cache of its last generation, so the
  next prompt prefills only the tokens it does not share with it (prefill over an 8-turn conversation with Qwen3
  0.6B: 0.46–0.61 s against 1.44–1.80 s). The cache stays in memory until `unload()` or an eviction. A greedy
  reply can differ from a full prefill's after a few dozen tokens, as with any change of prefill split; on a hit
  `lastGenerationStats.promptTime` covers only the tokens prefilled, so `prefillTokensPerSecond` overstates
  prefill speed, and AuraExample's model evaluation no longer shows it. MLX stats are timed by AuraLocal around
  mlx-swift-lm's token iterator instead of read from `GenerateResult`.
- **Behaviour change (GGUF tools):** calls a model makes after `maxToolRounds` rounds (default 1, as many rounds as
  before) are reported as `ToolEvent.roundLimitReached` and in `ToolTurnSummary.skipped` instead of being dropped
  silently; a tool that throws, arguments that do not decode and an unknown tool name reach the model as
  `{"error": …}` instead of failing the turn (`ToolLoopOptions(toolErrors: .failTurn)` restores that); tool output
  reaches the model as JSON with sorted keys instead of `key: value` pairs in random order; with a chat template that
  supports tools, the resumed prompt carries the model's call as an assistant turn and each result as a `tool` turn
  in the template's own format; with other templates each round is sent as one assistant turn, the model's call as
  `<tool_call>` text followed by every result in `<tool_response>` (before, only the results were sent, and only
  the last one as an assistant turn), so templates that require alternating roles, such as Gemma 3's, render it;
  after a result the tool instructions are sent again while another round may run (before, only the first reply's
  prompt carried them, so on a template without tool support a later round had no list of the tools).
- **Behaviour change (GGUF tools):** models asked to write calls as `<tool_call>` JSON (templates without tool
  support, and templates that write calls that way, such as Qwen2.5's) run on AuraLocal's own llama.cpp context and
  tool-call parser instead of LocalLLMClient's `LlamaClient`: every call in a reply runs (LocalLLMClient kept only
  the first per tool name); a `<tool_call>` block that is not a JSON object with a `name` stays in the reply text
  instead of vanishing; arguments reach the tool as JSON with sorted keys; a round goes back to the model as the
  reply it wrote, `<tool_call>` text included; `maxTokens` counts the calls' tokens too; and a reply cut short no
  longer makes the next turn wait for its decode to finish. Other call formats keep the `LlamaClient` path.
- **Behaviour change (GGUF):** when a chat template ends with a newline, the prompt no longer does: AuraLocal drops
  one trailing newline from the template, as Python Jinja (Hugging Face's and llama.cpp's renderers) does, so
  Qwen2.5's prompt ends `<|im_start|>assistant\n` instead of `<|im_start|>assistant\n\n`. Replies at temperature 0
  can differ from before.
- **Behaviour change:** AuraUI's `TextChatTab`, `VisionTab` and `OCRTab` and AuraDocs' `DocsTab` no longer download
  a model without asking. While a model they need is not on disk they show `ModelDownloadPlaceholder` with
  **Download**; models already on disk load as before. `DocsTab` used to download Qwen3 1.7B and FastVLM (~2 GB)
  as soon as it appeared. Each tab takes `autoDownload: Bool = false`; pass `true` to keep the old behaviour.
- Hybrid API, for code written against `v0.1.0` or a revision before 2026-09-30:
  - `OpenAICompatibleProvider.gitHubModels(apiKey:)` and the `cloudTargets` overload taking `gitHubModelsModel:`
    are `@available(*, unavailable)`: GitHub retired GitHub Models on 2026-07-30.
  - `ConsentGate.requestConsent(target:preview:projectedCostUSD:)` takes `Decimal?` (`nil` when the model is
    unpriced), and the protocol gains `requestConsent(target:preview:offer:)`, which has a default.
  - `CostLedger.Record.costUSD` is `Decimal?`, `RoutingInput.projectedCostUSD` is `Decimal?`, and
    `EscalationReason` has a new case, `.costUnknown`.
  - The static `CostLedger.projectedCost(target:inputTokens:maxOutput:)` is deprecated in favour of the instance
    method.
- **Behaviour change:** with an `outputSchema`, no tools and a GGUF model on `LlamaCppBackend`, each `AuraSession`
  generation (repairs included) runs on a dedicated constrained llama.cpp client: replies follow the schema's
  structure by construction, with required properties in `required` order and optional ones by name; objects
  carry only their declared properties, even when `additionalProperties` allows more; each generation pays a
  client build and grammar sampling (1.0–1.35 s against 0.41–0.60 s per reply on Qwen2.5-0.5B, M1 Pro; see
  [Schema grammars](docs/guide/backends.md#schema-grammars)). This includes `aura receipt extract|eval --model`
  with such a model. MLX and layer-streaming keep validation only. `usesSchemaGrammar = false` restores the
  previous path.

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
- `DownloadConflictError.partialChanged(path:)`: a finished `.partial` that another writer removed or resized before
  it was moved into place fails with this instead of a raw file-system error, and nothing is replaced.
- GGUF: `GenerationConstraint` (`.json`, `.grammar(gbnf:root:)`) enforced in llama.cpp's sampler, and
  conversations rendered with each model's chat template and real roles.
- `GenerationStats` after each generation, a decode-speed estimate, and `HardwareAnalyzer.assess` judging wall
  clock as well as memory.
- KV cache sizing per layer type and MoE decode by active experts; `Model.maxContextLength`; MLX `maxKVSize` and
  `lastContextOverflow`.
- iOS: generation stops when the app backgrounds (`GenerationInterrupted`), an opt-in `DeviceConditions` guard,
  and models judged against the app's memory limit.
- `AuraSession` enforces an `OutputSchema` by validation and repair.
- On a GGUF model loaded in full, `AuraSession` decodes `OutputSchema` replies under a GBNF grammar built from the
  schema (profiles without tools); `AuraSession.usesSchemaGrammar` turns it off.
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
- `ModelDownloadPlaceholder` (AuraCore, SwiftUI): stands in for a feature whose models are not on disk, with each
  model's purpose and size, the free space and a warning when it may not fit, **Download**, progress, Cancel and
  Retry, and an `onReady` callback once the files are there. `RequiredModel`, `[RequiredModel].notDownloaded`,
  `OnDemandDownloads` (one transfer at a time through `ModelDownloads`, tracked by file) and
  `Model.approximateSizeBytes` go with it; `DocsTab.neededModels` lists what `DocsTab` loads.
- GGUF tool loop controls: `ToolLoopOptions` (`maxToolRounds`, `toolErrors`) on `AuraLocal.toolLoop` and
  `AuraProfile.toolLoop`, `ToolEvent` through `AuraLocal.onToolEvent` and `AuraSession.onToolEvent`, and
  `ToolTurnSummary` in `lastToolTurn`.
- Tool calling on MLX text models, with the same controls. `ModelManager.load(_:tools:)` runs an MLX tool instance
  on the weights of a cached instance of the same model instead of loading a second copy.

### Changed

- AuraExample's Library and (macOS) Agents tabs ask before downloading their models, like the SDK tabs (see
  Breaking changes).
- MLX is back: `AuraCore` links mlx-swift-lm again and offers both engines.
- LocalLLMClient is pinned by revision `edc39ef` (tag 0.5.0, llama.cpp b8851); mlx-swift-lm from 3.31.3;
  ml-stable-diffusion exact 1.1.1; mlx-swift `0.31.3..<0.31.5`.
- Toolchain floor: Swift 6.1 / Xcode 16.3; Xcode 26 for `AuraAppleIntelligence` and `AuraAgents`; Xcode 26.4 for
  `AuraImageGen` and the `aura` CLI.
- The model catalog is verified against Hugging Face (`scripts/verify_catalog.py`); several entries moved repo
  (Dolphin 8B to `dphn/`, so it downloads again).
- The context window follows the model's KV cost and the memory left after its weights, capped at what the
  weights support; memory on macOS is measured instead of assumed.
- A GGUF turn whose rendered prompt starts with exactly the last turn's prompt and reply, and goes on with a special
  token, decodes only what it adds; any other turn starts from an empty KV cache and re-prefills the history, and so
  does every turn after a reply that was cut. Tool rounds on templates with native tool turns (Qwen2.5's) reuse the
  cache the same way while another round may run; on templates without tool support each round's results join the
  reply's assistant turn, so every round re-prefills. After a ~2,400-token system prompt, Qwen2.5-0.5B's time to the
  first token stayed near 50 ms instead of about 800 ms (M1 Pro). A reused turn keeps the reply's tokens as sampled
  and as llama.cpp computed them one at a time; a full prefill re-tokenizes the reply text (usually, not always, to
  the same tokens) and computes it in one batch, so at temperature 0 a reused turn can now and then pick a different
  token.
- `aura ask` uses a local llama-server or Ollama by default, or OpenAI, Anthropic or a custom endpoint when named.

### Fixed

- AuraLocal resolves for integrators (the swift-syntax conflict between LocalLLMClient and mlx-swift-lm).
- AuraCore builds for the iOS Simulator, and GGUF runs there.
- Conversation search after deletes, turns stored in the same millisecond, one reply per prompt when a generation
  stops, and `stream(_:in:)` and Apple Intelligence streams cancelling when the consumer stops.
- Consent shows, caches and prices the exact escalation payload; redaction applies on the policy path; the cost
  cap holds at the consent gate; unknown spend is never priced as $0.
- A loaded model stays ready when another sampling load fails.
- Two downloads of one model no longer write the same `.partial` files at once, which failed one of them with
  "`<file>.partial` doesn't exist". `ModelDownloads.download` and `downloadFolder`, `GGUFModelDownloader`, the MLX
  load path and `ChunkedDownloader` wait for each other per file or per Hugging Face snapshot, in one process and,
  through a `flock` on a lock file next to the destination, across processes sharing the storage root (the `aura`
  CLI and AuraExample on a Mac). The later call returns the finished model without a transfer.
- `ChunkedDownloader` replaces a file already at the destination instead of failing with "an item with the same
  name already exists" on a single-stream download, and keeps it without a transfer when it hashes to the
  `expectedSHA256` passed.
- `GGUFModelDownloader.cancel()` also ends calls of that downloader still waiting for the file, and a download
  cancelled while it verifies is no longer moved into place after the next download of the file started.
- RAG document IDs survive relaunches, and vectors from a provider the index no longer matches are never stored.
- The memory safeguards that were documented but inert (`evictAllButMostRecent`, the iOS OOM guards).
- `scripts/package-cli.sh` finds the build products of newer SwiftPM (`.build/out/Products/Release`), and the
  `aura` tarball now carries the SwiftPM resource bundles and the license texts.
- GGUF tool turns: a resumed prompt that outgrew the context failed the turn; every round's prompt now drops the
  oldest exchanges, as the first one does.
- GGUF tool instructions listed the tools as JSON with its keys in a different order on every render; the keys are
  sorted now.

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
