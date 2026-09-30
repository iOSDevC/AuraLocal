---
layout: docs
title: AuraCore
parent: API Reference
nav_order: 1
description: "AuraCore API reference — AuraLocal, ModelManager, ConversationStore, HardwareAnalyzer."
---

# AuraCore
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## AuraLocal

The main entry point. All methods are `@MainActor`.

### Factory Methods

```swift
// Load a text-generation model
static func text(
    _ model: Model = .qwen3_1_7b,
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> AuraLocal

// Load a vision model
static func vision(
    _ model: Model = .qwen35_0_8b,
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> AuraLocal

// Load an OCR/document-specialized model
static func specialized(
    _ model: Model = .fastVLM_0_5b_fp16,
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> AuraLocal
```

### One-Liners

```swift
// Static chat — loads the model on each call, returns the reply
static func chat(
    _ prompt: String,
    model: Model = .qwen3_1_7b,
    systemPrompt: String? = nil,
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> String

// Static chat that stores the exchange; conversationID nil creates a new conversation
@discardableResult
static func chat(
    _ prompt: String,
    in conversationID: UUID? = nil,
    model: Model = .qwen3_1_7b,
    systemPrompt: String? = nil,
    store: ConversationStore = .shared,
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> (reply: String, conversationID: UUID)

// Static receipt/document extraction
static func extractDocument(
    _ image: PlatformImage,
    model: Model = .fastVLM_0_5b_fp16,
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> String

// Parse DocTags output from Granite Docling
static func parseDocTags(_ raw: String) -> String
```

### Instance Methods

```swift
// Text generation
func chat(_ prompt: String, systemPrompt: String? = nil, maxTokens: Int = 1024) async throws -> String
func stream(_ prompt: String, systemPrompt: String? = nil, maxTokens: Int = 1024) -> AsyncThrowingStream<String, Error>

// Vision (image nil = text-only)
func analyze(_ prompt: String, image: PlatformImage? = nil, maxTokens: Int = 800) async throws -> String
func streamVision(_ prompt: String, image: PlatformImage? = nil, maxTokens: Int = 800) -> AsyncThrowingStream<String, Error>
func extractDocument(_ image: PlatformImage, maxTokens: Int? = nil) async throws -> String   // nil: 2048 for DocTags models, else 600

// Multi-turn with conversation history
func chat(_ prompt: String, in conversationID: UUID, systemPrompt: String? = nil, maxTokens: Int = 1024,
          maxContextTokens: Int = 3072, store: ConversationStore = .shared) async throws -> String
func stream(_ prompt: String, in conversationID: UUID, systemPrompt: String? = nil, maxTokens: Int = 1024,
            maxContextTokens: Int = 3072, store: ConversationStore = .shared) -> AsyncThrowingStream<String, Error>
```

### History Methods

```swift
// Auto-generate a title from the first message
func autoTitle(conversationID: UUID, store: ConversationStore = .shared) async throws

// Summarize and prune long conversations
func summarizeAndPrune(
    conversationID: UUID,
    keepLastN: Int = 10,
    maxContextTokens: Int = 4096,
    store: ConversationStore = .shared
) async throws
```

### Properties

```swift
var model: Model { get }     // the loaded model
```

Loaded state is queried on `ModelManager`, not on `AuraLocal`:

```swift
// Whether a model's weights are currently held in memory.
ModelManager.shared.isLoaded(_ model: Model) -> Bool
```

---

## ModelManager

Shared singleton for loading and caching models. Preferred over direct `AuraLocal` factory calls — prevents redundant downloads and handles memory pressure.
Guide: [Memory Management]({{ '/guide/memory' | relative_url }}).

```swift
@MainActor
public final class ModelManager: ObservableObject
```

### Loading

```swift
// Load a model — returns cached instance if already loaded
// For GGUF models, downloads the file from HuggingFace first
func load(
    _ model: Model,
    tools: [any LLMTool] = [],            // sessions with tools are built fresh, never cached
    onProgress: (@MainActor (String) -> Void)? = nil
) async throws -> AuraLocal
```

{: .warning }
> `tools` reach only the llama.cpp backends (GGUF models). For an MLX model they are dropped without
> an error, and the model answers without calling them.

### State Observation

```swift
@Published private(set) var states: [Model: ModelLoadState]

func state(for model: Model) -> ModelLoadState

public enum ModelLoadState: Equatable {
    case idle
    case downloading(progress: String)   // human-readable progress text
    case loading
    case ready
    case failed(String)
}
```

### Backend Query

```swift
// Which backend will be used for a model on the current device
func recommendedBackend(for model: Model) -> BackendKind

public enum BackendKind: String, Sendable {
    case mlx
    case llamaCpp
    case layerStreaming
    case remote   // cloud, or the user's own llama-server / Ollama
    case hybrid   // local backend that can escalate single requests
}
// recommendedBackend(for:) returns only .mlx, .llamaCpp or .layerStreaming.
```

### Eviction

```swift
func evict(_ model: Model)
func evictAll()
```

### GGUF Download Progress

```swift
// Detailed download progress for GGUF models
let ggufDownloader: GGUFModelDownloader

// GGUFModelDownloader published properties (read-only):
@Published private(set) var progress: Double           // 0.0–1.0
@Published private(set) var downloadedBytes: Int64
@Published private(set) var totalBytes: Int64
@Published private(set) var isDownloading: Bool
@Published private(set) var isPaused: Bool
@Published private(set) var error: String?
```

---

## ConversationStore

SQLite-backed store for persistent chat history. Actor-isolated.
Guide: [Conversations & History]({{ '/guide/conversations' | relative_url }}).

```swift
public actor ConversationStore
static let shared: ConversationStore
```

### Conversations

```swift
func createConversation(model: Model, title: String = "New conversation") async throws -> Conversation
func allConversations() async throws -> [Conversation]
func conversation(id: UUID) async throws -> Conversation?
func updateTitle(_ title: String, for id: UUID) async throws
func deleteConversation(id: UUID) async throws
```

### Messages

```swift
func turns(for conversationID: UUID) async throws -> [Turn]
func search(_ query: String, limit: Int = 20) async throws -> [Turn]
```

### Types

```swift
public struct Conversation: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    var title: String
    let model: String          // Model.rawValue
    let createdAt: Date
    var updatedAt: Date
    var turnCount: Int
}

public struct Turn: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    let conversationID: UUID
    let role: Role          // .user / .assistant / .system
    let content: String
    let createdAt: Date
    let tokenEstimate: Int

    public enum Role: String, Codable, Sendable { case system, user, assistant }
}
```

---

## AuraSession

A conversation that can switch `AuraProfile`s mid-session; the engine is rebuilt for the new
profile. `AuraSession` keeps no history: each `stream` call is a single turn with no earlier
messages. `conversationID` stays the same across switches, as a key for you to store turns under
(for example in `ConversationStore`).

```swift
@MainActor
public final class AuraSession
init(conversationID: UUID = UUID(),
     profile: AuraProfile,
     makeEngine: @escaping @MainActor (AuraProfile) async throws -> any AuraProfileEngine = AuraSession.liveEngine) async throws

let conversationID: UUID
private(set) var profile: AuraProfile
var maxRepairAttempts: Int = 2     // re-prompts after a reply that breaks profile.outputSchema

func switchProfile(to newProfile: AuraProfile) async throws
func stream(_ prompt: String, maxTokens: Int? = nil) -> AsyncThrowingStream<String, Error>
func cancel()
```

`stream` sends `profile.instructions` as the system prompt and defaults `maxTokens` to
`profile.sampling.maxTokens`. One generation runs at a time: a new `stream` or a `switchProfile`
cancels the previous one and waits for it to stop, also after `cancel()`, which returns without
waiting. A `stream` or `switchProfile` called while a switch is rebuilding the engine waits for it,
so that stream runs on the new profile. After a switch fails to build its engine, `stream` throws
`AuraError.modelNotLoaded` until a later `switchProfile` succeeds.

### AuraProfile

```swift
public struct AuraProfile: Sendable, Identifiable, Equatable {
    let id: String
    var displayName: String
    var instructions: String            // the system prompt
    var model: Model
    var sampling: SamplingParams        // only maxTokens reaches generation
    var tools: [any LLMTool]            // GGUF backends only
    var outputSchema: OutputSchema?
    var escalation: EscalationPolicy    // default .off
}

public enum OutputSchema: Sendable, Equatable, Codable {
    case json(String)                   // a JSON Schema document
}
```

`AuraProfileCatalog.chat(model:)`, `.agent(model:tools:)` and `.securityReview(model:)` build the
common profiles.

### Structured output

With `outputSchema` set, `AuraSession` enforces the schema by **validation and repair**, not
constrained decoding, so the model can still write anything and the session checks what comes
back. The pinned llama.cpp client accepts a GBNF grammar only when a context is created, and
`ModelManager` caches one instance per model; AuraLocal has no JSON-Schema-to-GBNF converter, and
MLX exposes only a raw `LogitProcessor` hook.

1. `init` and `switchProfile` compile the schema before building an engine. A schema that does not
   compile throws `OutputSchemaError` and leaves the current profile and engine untouched. A
   `switchProfile` that fails to build the new engine has already torn the old one down; the
   session recovers on the next successful `switchProfile`.
2. `stream` appends the schema to the system prompt with an instruction to reply with a single
   JSON value, buffers the whole reply, and looks for a conforming JSON value in it (see below).
3. A reply that breaks the schema is sent back with the original prompt and the list of
   violations, up to `maxRepairAttempts` times. Each repair is a full extra generation.
4. The stream yields **one** element, the conforming JSON text, or finishes throwing
   `OutputSchemaError.violations(output:violations:)` for the last reply.

Without a schema, `stream` yields deltas as they are generated.

```swift
let profile = AuraProfile(
    id: "extract", displayName: "Extract",
    instructions: "Extract the person mentioned in the text.",
    model: .qwen3_1_7b,
    sampling: .precise,
    outputSchema: .json(#"""
        {"type": "object",
         "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
         "required": ["name", "age"], "additionalProperties": false}
        """#))

let session = try await AuraSession(profile: profile)
for try await json in session.stream("Ana turned 30 last week.") {
    print(json)   // e.g. {"name": "Ana", "age": 30}
}
```

`AuraLocal.chat` and `AuraLocal.stream` do not read profiles. To check their output, use the
validator directly:

```swift
public struct OutputSchemaValidator: Sendable {
    init(_ schema: OutputSchema) throws(OutputSchemaError)
    func validate(_ json: String) -> [OutputSchemaViolation]                     // empty = conforms
    func conformingJSON(in output: String) throws(OutputSchemaError) -> String  // first candidate that conforms
    static func extractJSON(from output: String) -> String?                      // first candidate that parses
}

public struct OutputSchemaViolation: Sendable, Equatable, CustomStringConvertible {
    let path: String       // JSON pointer (RFC 6901); "" is the whole value
    let message: String
}

public enum OutputSchemaError: Error, LocalizedError, Sendable, Equatable {
    case invalidSchema(String)            // not JSON (a GBNF grammar, say), or a malformed keyword value
    case unsupportedKeywords([String])    // JSON pointers into the schema
    case violations(output: String, violations: [OutputSchemaViolation])
}
```

Supported keywords:

| Keywords | Notes |
|----------|-------|
| `type` | A name or an array of names. `integer` accepts whole numbers, including `3.0` |
| `properties`, `required`, `additionalProperties` | `additionalProperties` is a boolean or a schema |
| `items`, `minItems`, `maxItems` | `items` is one schema; the tuple (array) form is rejected |
| `enum`, `const` | |
| `minLength`, `maxLength` | Counted in Unicode scalars |
| `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` | The exclusive bounds are numbers, not booleans |
| `anyOf`, `oneOf`, `allOf` | |
| `description`, `title`, `$schema`, `examples`, `default`, `$comment` | Accepted and ignored |

Any other keyword (`$ref`, `$defs`, `pattern`, `format`, `patternProperties`, `if`/`then`/`else`,
`not`, …) makes compilation throw `unsupportedKeywords`, so no constraint is skipped silently.
`true` and `false` work as schemas anywhere a schema does.

Candidates, in order: the whole reply; then, with reasoning removed (`<think>…</think>` spans, a
leading block closed by a lone `</think>`, an unclosed `<think>` to the end), the whole reply again,
each ```` ```json ```` (or unlabelled) fenced block, and each top-level balanced `{…}` or `[…]`.
Brackets inside strings are skipped, and values nested in one that parsed are not candidates.
`extractJSON` returns the first candidate that parses. `conformingJSON` and `AuraSession` take the
first that conforms and, when none does, report the violations of the first that parsed, so a
citation such as `[1]` or a draft inside a reasoning block does not hide the answer.

Parsing is strict RFC 8259: trailing commas, comments, single quotes, `NaN` and duplicate member
names make a candidate not JSON, so the text `conformingJSON` returns reads the same in any
conforming parser. Nesting deeper than 512 levels is rejected. Numbers compare as exact decimals:
`9007199254740993` does not equal `9007199254740992`, and `1e400` is a valid number.

{: .warning }
> Validation checks shape, not truth: a conforming reply can still hold wrong values. A small model
> that keeps breaking the schema costs up to `1 + maxRepairAttempts` generations per `stream` call.

---

## HardwareAnalyzer

Assesses whether a catalog `Model` fits this device's memory. All methods are synchronous.
Guide: [Hardware Compatibility Check]({{ '/guide/models' | relative_url }}#hardware-compatibility-check).
For an arbitrary Hugging Face repo, use [`ModelCompatibilityChecker`](#modelcompatibilitychecker).

```swift
public enum HardwareAnalyzer   // namespace for static assessment methods
```

### Assessment

```swift
// Assess a model against the current device
static func assess(
    _ model: Model,
    profile: HardwareProfile = .current()
) -> ModelCompatibility

public struct HardwareProfile: Sendable {
    let totalMemoryGB: Double
    let availableMemoryGB: Double        // convenience: falls back to an estimate for display
    let deviceName: String
    let memoryBandwidthGBs: Double?      // nil on iOS and for unrecognised chips

    static func current() -> HardwareProfile

    /// Bytes the process may still allocate, or nil when the OS cannot tell.
    /// iOS returns nil exactly when the process is at or over its jetsam limit —
    /// safety-critical callers must treat nil as danger, never as "plenty free".
    static func availableMemoryBytes() -> Int?
}

public struct ModelCompatibility: Sendable {
    let model: Model
    let fitLevel: ModelFitLevel
    let requiredMemoryGB: Double
    let availableMemoryGB: Double
    let estimatedDecodeTokensPerSecond: Double?  // nil when chip bandwidth is unknown
    var utilizationPercent: Double               // % of available memory consumed
    var speedLevel: SpeedLevel
}

// The estimated memory footprints are properties on `Model`, not on ModelCompatibility:
//   model.estimatedRuntimeMemoryGB    // full-load (monolithic) footprint in GB
//   model.estimatedStreamingMemoryGB  // layer-streaming footprint in GB
```

### Compatible Models

```swift
// Every model assessed, none filtered out. Order: .tooLarge last; otherwise downloaded first,
// then better fit, then lower memory utilization.
static func compatibleModels(
    from models: [Model] = Model.allModels,
    profile: HardwareProfile = .current()
) -> [ModelCompatibility]
```

### ModelFitLevel

```swift
public enum ModelFitLevel: Comparable, Sendable {
    case excellent          // >40% RAM headroom
    case good               // 20–40% headroom
    case marginal           // <20% headroom
    case streamingRequired  // GGUF only — layer-streaming viable
    case tooLarge           // not runnable on this device

    var isRunnable: Bool    // false only for .tooLarge
    var label: String       // "Excellent", "Good", etc.
    var systemImage: String // SF Symbol name
}
```

---

## ModelCompatibilityChecker

Answers whether AuraLocal's pinned runtimes (`PinnedRuntimes.mlxSwiftLMVersion` and
`PinnedRuntimes.llamaCppBuild`) can load and run a Hugging Face repo on a given device, and if not,
why. It reads the repo listing, `config.json`, the safetensors weight map and the first bytes of
GGUF or safetensors files through HTTP range requests; nothing is downloaded in full. It never
throws: fetch failures (network errors, 401/403, 404) become findings. Unless a blocker was found,
the verdict is `.unknown` when the repo listing, an MLX repo's `config.json` or the GGUF
architecture could not be read.

This is not `ModelCompatibility`, the RAM-fit result `HardwareAnalyzer.assess` returns for catalog
models. Guide: [Finding compatible models]({{ '/guide/models' | relative_url }}#finding-compatible-models).
CLI: `aura models search`, `aura models check`, `aura models devices`.

```swift
let checker = ModelCompatibilityChecker()
let report = await checker.check("mlx-community/Qwen3.5-27B-4bit", on: .mac32GB)
print(report.status.label, report.headline)
```

### Checking

```swift
public struct ModelCompatibilityChecker: Sendable {
    init(transport: URLSession = .shared,
         authorizer: any DownloadAuthorizing = KeychainDownloadAuth(),   // HF token: Keychain account "download.huggingface"
         ggufHeaderBytes: Int = 4 << 20,
         requestTimeout: TimeInterval = 60)

    func check(_ repo: String, on target: DevicePreset = .thisDevice()) async -> CompatibilityReport
    func snapshot(of repo: String) async -> RepoSnapshot      // fetch once…
    static func repoID(from input: String) -> String?         // "owner/repo" from an id or a huggingface.co / hf.co URL
}

public enum CompatibilityEvaluator {
    // …then judge any number of devices without refetching. Pure, no network.
    static func evaluate(_ snapshot: RepoSnapshot, on target: DevicePreset) -> CompatibilityReport
}
```

### Report

```swift
public struct CompatibilityReport: Sendable {
    let repoID: String
    let target: DevicePreset
    let weightFormat: DetectedFormat         // .mlx .gguf .imageGeneration .unconvertedSafetensors .noWeights .unknown
    let modelCategory: Model.Category?       // how AuraLocal would load it; nil when it cannot
    let status: CompatibilityVerdict         // .runnable .runnableWithCaveats .notRunnable .unknown
    let findings: [CompatibilityFinding]     // blockers first, then caveats, then info
    let overview: ArchitectureFacts          // family, layers, trainedContext, kvHeads, headDim, hasVision, quantization
    let weightsBytes: Int64?                 // non-GGUF weights
    let weightsFit: FitEstimate?
    let quantFits: [QuantFit]                // every GGUF quant, smallest first, each with its fit
    let suggestedEntry: CatalogEntry?        // the models.json entry when it can run: entry.jsonText()
    let licenseID: String?
    let licenseName: String?
    let gatedMode: String?

    var blockers: [CompatibilityFinding]
    var caveats: [CompatibilityFinding]
    var headline: String                     // first blocker or caveat, else how it fits
    var bestFit: FitEstimate?                // weightsFit, or the recommended GGUF quant's fit
    var huggingFaceURL: URL?
}

public struct CompatibilityFinding: Sendable, Equatable, Identifiable {
    let rule: String                         // the CompatibilityRule id
    let level: FindingSeverity               // .blocker / .caveat / .info
    let title: String
    let detail: String
}

public struct FitEstimate: Sendable, Equatable {
    let rating: ModelFitLevel
    let requiredGB: Double
    let budgetGB: Double
    let tokensPerSecond: Double?             // nil when the device's bandwidth is unknown
    var summary: String                      // "Good · 12.0 of 20.0 GB"
}
```

`CompatibilityRules.all` lists the rules (`CompatibilityRule` has an `id` and a `summary`).
`GGUFHeaderParser.parse(_:)` reads GGUF v2/v3 metadata from the first bytes of a file into a
`GGUFHeader` (`isComplete` is `false` when the bytes ran out), and `SafetensorsHeader.parse(_:)`
does the same for a safetensors header.

To find repos to check, `HuggingFaceSearch.search(_:limit:sort:tag:session:)` queries the Hugging
Face model search; `tag:` (`"mlx"`, `"gguf"`, …) filters on the server.

### DevicePreset

A device to judge against: this one, measured now, or a fixed class of iPhone or Mac. `budgetGB`
is the memory an app may fill with weights and KV cache. Only `thisDevice()` and `mac32GB` are
measured; every other budget is an estimate and says so in `source`.

```swift
public struct DevicePreset: Sendable, Hashable, Identifiable {
    let id: String                           // "this-device", "iphone-4gb", "mac-32gb", …
    let displayName: String
    let platform: TargetOS                   // .iOS / .macOS
    let totalMemoryGB: Double
    let budgetGB: Double
    let isMeasured: Bool
    let source: String                       // where budgetGB comes from
    let bandwidthGBs: Double?
    var memoryProfile: HardwareProfile       // what HardwareAnalyzer consumes
    var budgetText: String                   // "20.0 GB (measured)" / "≈3.0 GB (estimate)"

    // Without a profile, a Mac is judged by Metal's recommendedMaxWorkingSetSize and an
    // iPhone by the memory the process can still allocate. A given profile is used as is.
    static func thisDevice(profile: HardwareProfile? = nil) -> DevicePreset

    static let iPhone4GB: DevicePreset
    static let iPhone6GB: DevicePreset
    static let iPhone8GB: DevicePreset
    static let iPhone17Pro: DevicePreset
    static let mac16GB: DevicePreset
    static let mac32GB: DevicePreset
    static let mac64GB: DevicePreset
    static let classes: [DevicePreset]                                     // the fixed presets, phones first
    static func all(profile: HardwareProfile? = nil) -> [DevicePreset]     // thisDevice + classes
    static func named(_ id: String, profile: HardwareProfile? = nil) -> DevicePreset?
}
```

---

## Model

`Model` is a value type backed by a JSON model registry (not a `String` enum — there is
no `rawValue` initializer). Named models like `.qwen3_1_7b` are static factory properties.

```swift
public struct Model: Sendable, Identifiable, Codable, CustomStringConvertible

// Collections
static var textModels: [Model]
static var visionModels: [Model]
static var specializedModels: [Model]
static var mlxModels: [Model]
static var ggufModels: [Model]
static var runnableModels: [Model]     // filtered by HardwareAnalyzer

// Metadata
var displayName: String
var approximateSizeMB: Int
var purpose: Purpose
var format: ModelFormat                // .mlx or .gguf
var ggufFilename: String?              // nil for MLX models
var isMacOSRecommended: Bool          // true for models ≥15 GB
var isDownloaded: Bool
var cacheDirectory: URL
```

---

## BackgroundLifecycle (iOS)

Sets `isPaused` when the app enters the background and clears it when the app returns. With
`aggressiveMemorySaving`, it also evicts every loaded model except the most recently used one.
Guide: [Background Lifecycle]({{ '/guide/memory' | relative_url }}#background-lifecycle-ios).

```swift
@MainActor
public final class BackgroundLifecycle
static let shared: BackgroundLifecycle

@Published private(set) var isPaused: Bool     // true while the app is backgrounded
var aggressiveMemorySaving: Bool = false       // evicts all but the most recently used model on background
```

{: .warning }
> It does not pause generation: nothing in AuraLocal reads `isPaused`, so observe
> `BackgroundLifecycle.shared.$isPaused` and stop your own streams. Its notification observers are
> installed when `BackgroundLifecycle.shared` is first accessed, so touch it once at launch.

No-op on macOS.

---

## System tools

On-device Apple frameworks (Vision, Natural Language, Sound Analysis, Core ML, Create ML) wrapped as
`SystemTool`s. Each tool's own API is in the
[On-device ML tools]({{ '/guide/ml-tools' | relative_url }}) guide.

```swift
public protocol SystemTool: Sendable {
    var id: String { get }
    var displayName: String { get }
    var summary: String { get }
    var category: SystemToolCategory { get }                // default .customModel
    func availability() async -> SystemToolAvailability     // .available / .unavailable(reason:); never throws
}

public enum SystemToolRegistry {
    static let all: [any SystemTool]                        // the built-in tools that need no file
    static func discover() async -> [ToolInfo]
    static func availableTools() async -> [ToolInfo]
    static func tool(id: String) -> (any SystemTool)?
}
```

### Text embeddings (Core ML)

`CoreMLTextEmbeddingTool` runs a sentence-embedding bundle such as multilingual-e5-small on-device.
A bundle is a directory holding `embedding-model.json` (`TextEmbeddingManifest`), the `.mlpackage`
or `.mlmodelc`, and the Hugging Face `tokenizer.json` and `tokenizer_config.json`. The tool is not
in `SystemToolRegistry.all`; create it with the bundle's URL. AuraDocs wraps it in
[`CoreMLEmbeddingProvider`]({{ '/api/rag' | relative_url }}#coremlembeddingprovider).
Guide: [Using the model without AuraDocs]({{ '/guide/rag' | relative_url }}#using-the-model-without-auradocs).

```swift
public struct CoreMLTextEmbeddingTool: SystemTool {
    // compiledModelsDirectory nil = Caches/AuraLocal/CompiledModels
    init(bundleAt url: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine, compiledModelsDirectory: URL? = nil)

    // Checks the manifest, the model file and the tokenizer files without loading the model.
    @discardableResult
    static func validateBundle(at url: URL) throws -> TextEmbeddingManifest

    let manifest: TextEmbeddingManifest?     // nil when missing or invalid; availability() says why
    var dimensions: Int                      // 0 without a manifest
    var identifier: String                   // "model_id@revision"; the tool id without a manifest

    func embed(_ text: String, role: TextEmbeddingRole = .passage) async throws -> TextEmbedding
    func embed(_ texts: [String], role: TextEmbeddingRole = .passage) async throws -> [TextEmbedding]
    func tokenize(_ text: String, role: TextEmbeddingRole = .passage) async throws -> TextEmbeddingTokens
    @discardableResult
    func warmUp() async throws -> Duration   // the first Neural Engine load compiles on-device
    func truncatedInputCount() async -> Int  // inputs cut at max_tokens so far
    func unload() async
}

public enum TextEmbeddingRole: String, Sendable, CaseIterable {
    case query     // prefixed with the manifest's query_prefix
    case passage   // prefixed with passage_prefix
    case raw       // used as given
}

public struct TextEmbedding: Sendable, Equatable {
    let vector: [Float]
    let tokenCount: Int                      // <s> and </s> included
    let isTruncated: Bool                    // the text exceeded max_tokens and its tail was dropped
}

public struct TextEmbeddingTokens: Sendable, Equatable {
    let ids: [Int]
    let originalCount: Int                   // length before truncation
    var isTruncated: Bool
}

public struct TextEmbeddingManifest: Codable, Sendable, Equatable {
    static let schemaName: String            // "aura.text-embedding/1"
    static let fileName: String              // "embedding-model.json"
    let modelID: String                      // JSON: model_id
    let revision: String
    let dimensions: Int
    let maxTokens: Int                       // JSON: max_tokens
    let queryPrefix: String                  // JSON: query_prefix
    let passagePrefix: String                // JSON: passage_prefix
    // also schema, modelFile, inputName, outputName, buckets, padTokenID, pooling, normalize, license
    var identifier: String                   // "model_id@revision"
    static func load(fromBundle bundleURL: URL) throws -> TextEmbeddingManifest
}
```

### EmbeddingIndexIdentity

Which embedding model produced the vectors in a stored index. `DocumentLibrary` records it and
re-embeds when the configured provider no longer matches.

```swift
public struct EmbeddingIndexIdentity: Sendable, Equatable {
    let identifier: String
    let dimensions: Int
    init(identifier: String, dimensions: Int)

    static let preTrackingIdentifier: String      // "aura.tfidf-hash/4096"

    enum Reconciliation { case upToDate, adopt, reembed }
    static func reconciliation(
        stored: EmbeddingIndexIdentity?,          // nil for an index written before identities were recorded
        configured: EmbeddingIndexIdentity,
        storedVectorLengths: Set<Int>,
        storedVectorCount: Int
    ) -> Reconciliation
}
```

---

## Hybrid Inference

Optional, **consent-gated** escalation to a bigger local or cloud model. Off by default.
See the [Hybrid Inference guide]({{ '/guide/hybrid' | relative_url }}) for the full flow.

### Local provider detection

```swift
@MainActor
public enum LocalProviderDetector {
    // Probe Ollama (:11434) and llama-server (:8080/v1). Never throws.
    static func detectAll(
        endpoints: [LocalProviderEndpoint] = LocalProviderEndpoint.defaults,
        timeout: TimeInterval = 2.0
    ) async -> [LocalProviderStatus]
}

public struct LocalProviderModel: Sendable, Codable, Identifiable, Hashable {
    let name: String
    let sizeBytes: Int64?                // Ollama: size; llama-server: meta.size
    let quantization: String?            // Ollama only
    let contextLength: Int?              // llama-server only
    let remoteHost: String?              // Ollama cloud models: "https://ollama.com:443"
    var runsLocally: Bool                // false with a remoteHost or a ":cloud" / "-cloud" tag
}
```

### HybridEscalator

```swift
@MainActor
public final class HybridEscalator {
    init(ledger: CostLedger = .shared)   // prices and session spend come from this ledger

    // Route by policy: stays local, escalates, or asks consent per rules R1–R7.
    func routeAndEscalate(
        policy: EscalationPolicy,
        systemPrompt: String? = nil,
        context: String,
        question: String,
        domain: Model.Domain? = nil,
        localContextWindow: Int = 8192,
        localAnswer: String? = nil,        // enables the low-confidence trigger
        consent: any ConsentGate = DenyingConsentGate(),   // asked via requestConsent(target:preview:offer:)
        maxTokens: Int = 1024,
        targets: [RemoteTarget]? = nil,    // nil = candidateTargets(policy:)
        onToken: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> Result?              // nil = the router kept it local; never redacts PII
                                           // throws AuraError.escalationDeclined when the gate declines

    // Direct escalation to a chosen target.
    func escalate(
        to target: RemoteTarget,
        systemPrompt: String? = nil,
        context: String,
        question: String,
        maxTokens: Int = 1024,
        redactPII: Bool = false,           // PIIRedactor on the context and the question
        onToken: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> Result

    // Your llama-server (preferred) or Ollama, with its largest-context model; nil if none runs.
    // Skips models with runsLocally == false (Ollama cloud models forward prompts off the machine).
    static func bestLocalTarget() async -> RemoteTarget?
    // BYOK targets from the Keychain: "cloud.anthropic", then "cloud.openai". [] when !allowCloud.
    static func cloudTargets(
        allowCloud: Bool,
        anthropicModel: String = RemoteTarget.defaultAnthropicModel,   // "claude-sonnet-4-5"
        openAIModel: String = RemoteTarget.defaultOpenAIModel          // "gpt-4o"
    ) -> [RemoteTarget]
    // The LAN box first, then cloudTargets(allowCloud: policy.allowCloud).
    static func candidateTargets(policy: EscalationPolicy) async -> [RemoteTarget]

    public struct Result: Sendable {
        let answer: String
        let compression: CompressionResult
        let usage: TokenUsage?
        let providerName: String
        let redactedPIICount: Int
        let fromCache: Bool
    }
}
```

### EscalationPolicy

```swift
public struct EscalationPolicy: Sendable, Equatable, Codable {
    var mode: Mode                     // .off / .askEachTime / .autoWithConsentMemory
    var allowCloud: Bool
    // Default 1.0. Rule R7: when the request's projection is above $0 (a LAN target never trips it)
    // and CostLedger.sessionCostUSD plus that projection exceeds the cap, the router offers
    // .costCapped, which the default ConsentGate.requestConsent(target:preview:offer:) declines.
    // An unpriced cloud model, or a session with unpriced calls, is offered as .costUnknown.
    var costCapUSDPerSession: Decimal
    // Default 0.5. Sizes only the consent preview; escalate() always budgets 50% of the
    // target's context window minus maxTokens.
    var keepRatio: Double

    init(mode: Mode = .off, allowCloud: Bool = false, costCapUSDPerSession: Decimal = 1.0, keepRatio: Double = 0.5)
    static let off: EscalationPolicy   // the default — never leaves the device
}
```

### EscalationRouter

The pure decision behind `routeAndEscalate`, which fills `RoutingInput` from its target and its
`CostLedger`. Call it yourself to test a policy.

```swift
public enum EscalationRouter {
    static func decide(_ input: RoutingInput) -> RoutingDecision   // rules R1–R7, no I/O
}

public struct RoutingInput: Sendable {
    init(policy: EscalationPolicy, hasCandidateTarget: Bool, candidateIsCloud: Bool,
         online: Bool = true, promptTokens: Int = 0, localContextWindow: Int = 8192,
         localAnswer: String? = nil, domain: Model.Domain? = nil,
         projectedCostUSD: Decimal? = nil,   // nil = unknown; a cloud target is then offered, never free
         sessionSpentUSD: Decimal = 0,       // priced spend already recorded this session
         unpricedRecordCount: Int = 0)       // session calls of unknown cost; > 0 makes the spend a lower bound
}

public enum RoutingDecision: Sendable, Equatable {
    case stayLocal
    case escalate(reason: EscalationReason)
    case offer(reason: EscalationReason)
}

public enum EscalationReason: String, Sendable, Equatable {
    case sizeOverflow, lowConfidence, domainSensitive, userRequested
    case costCapped    // a trigger fired, but sessionSpentUSD + a non-zero projection exceeds the cap
    case costUnknown   // a trigger fired for a cloud target whose projection is nil, or with a
                       // non-zero projection under the cap while unpricedRecordCount > 0
}
```

### Providers & targets

```swift
public protocol RemoteLLMProvider: Sendable { /* id, displayName, retentionNote, stream(_:) */ }

public struct OpenAICompatibleProvider: RemoteLLMProvider {
    init(id: String, displayName: String, baseURL: URL, apiKey: String? = nil,
         streaming: Bool = true, retentionNote: String = "Sent to an OpenAI-compatible endpoint.")
    // Build from a detected local llama-server / Ollama endpoint (Ollama requests go to <root>/v1).
    static func from(_ status: LocalProviderStatus) -> OpenAICompatibleProvider
    // Unavailable: GitHub retired GitHub Models on 2026-07-30. Calling it is a compile error.
    @available(*, unavailable, message: "GitHub retired GitHub Models on 2026-07-30. …")
    static func gitHubModels(apiKey: String) -> OpenAICompatibleProvider
}

public struct RemoteTarget: Sendable {
    let provider: any RemoteLLMProvider
    let modelID: String
    let contextLength: Int?
    let origin: Origin                 // .cloud / .localNetwork(LocalProviderKind)
    init(provider: any RemoteLLMProvider, modelID: String, contextLength: Int? = nil, origin: Origin)
    var isLocalNetwork: Bool

    static let defaultOpenAIModel: String      // "gpt-4o"
    static let defaultAnthropicModel: String   // "claude-sonnet-4-5"
}
```

{: .note }
> GitHub retired GitHub Models on 2026-07-30. `OpenAICompatibleProvider.gitHubModels(apiKey:)` and
> `HybridEscalator.cloudTargets(allowCloud:anthropicModel:openAIModel:gitHubModelsModel:)` are
> `unavailable`, so old callers get a compile error with the migration path. A key stored under
> `"cloud.github-models"` is no longer read and stays in the Keychain until you delete it.

### AskTargetResolver

The provider choice behind `aura ask`, as a pure function: no network, no Keychain access of its own.
Without a `baseURL`, `.auto` / `.local` never return a cloud target; a prompt leaves the machine
only when the caller names `.openAI` / `.anthropic` or passes a base URL. A `baseURL` is
classified by its host alone: loopback or private network is `.localNetwork(.llamaServer)`,
anything else `.cloud` (also under `.auto`).

```swift
public enum AskTargetResolver {
    enum Choice: String, Sendable, CaseIterable {
        case auto, local                 // llama-server, else Ollama (HybridEscalator.bestLocalTarget order)
        case openAI = "openai"           // key: OPENAI_API_KEY, else Keychain "cloud.openai"
        case anthropic                   // key: ANTHROPIC_API_KEY, else Keychain "cloud.anthropic"
    }

    enum Failure: Error, Equatable, LocalizedError {
        case noLocalProvider
        case modelNotServedLocally(model: String, available: [String])
        case missingAPIKey(provider: Choice, environmentVariable: String, keychainAccount: String)
        case conflictingOptions(String)  // baseURL with .openAI/.anthropic, or .local with a public host
        case modelRequired               // baseURL without a model
        case invalidBaseURL(String)
        var isUsageError: Bool           // true for the last three (aura exits 2; otherwise 1)
    }

    static func resolve(
        _ choice: Choice,
        model: String? = nil,             // nil: the provider's default (local: largest context window)
        baseURL: String? = nil,           // any OpenAI-compatible server; key from AURA_API_KEY
        environment: [String: String],
        readKey: (String) -> String?,     // e.g. KeychainStore.read(for:); called only for .openAI/.anthropic
        localProviders: [LocalProviderStatus]   // from LocalProviderDetector.detectAll()
    ) throws(Failure) -> RemoteTarget
}
```

A `baseURL` host that is loopback, private (RFC 1918, IPv6 ULA), link-local, `localhost`, `*.local` or
`*.home.arpa` gets the origin `.localNetwork(.llamaServer)`; any other host is `.cloud`.

### Keys, cost & privacy

```swift
// BYOK API keys — Keychain only (WhenUnlockedThisDeviceOnly, never iCloud-synced).
// Accounts read by AuraLocal: "cloud.anthropic" · "cloud.openai" · "download.huggingface"
// (Hugging Face token for gated repos, KeychainDownloadAuth.huggingFaceAccount; also used by
// ModelCompatibilityChecker). "cloud.github-models" is no longer read (GitHub Models was retired).
public enum KeychainStore {
    static func save(_ key: String, for account: String) throws
    static func read(for account: String) -> String?
    static func delete(for account: String)
    static func hasKey(for account: String) -> Bool
}

// USD per 1M tokens.
public struct TokenPrice: Sendable, Hashable {
    var inputUSDPerMillion: Decimal
    var outputUSDPerMillion: Decimal
    init(inputUSDPerMillion: Decimal, outputUSDPerMillion: Decimal)
}

@MainActor
public final class CostLedger: ObservableObject {
    static let shared: CostLedger      // HybridEscalator's default ledger
    init()                             // starts with the built-in prices

    @Published private(set) var records: [Record]   // per-escalation token + $ accounting, all sessions
    // The totals below cover the records since the last startNewSession().
    var sessionTokens: Int
    var sessionCostUSD: Decimal        // priced records only
    var unpricedRecordCount: Int       // records whose costUSD is nil
    var sessionRecordCount: Int        // priced and unpriced
    func startNewSession()             // keeps records, restarts the totals

    // model nil = every model of the provider; a nil price removes the entry.
    func setPrice(_ price: TokenPrice?, provider: String, model: String? = nil)
    // The model-specific price, else the provider-wide one, else nil.
    func price(provider: String, model: String? = nil) -> TokenPrice?
    // 0 for local-network targets, nil for an unpriced cloud model.
    func projectedCost(target: RemoteTarget, inputTokens: Int, maxOutput: Int) -> Decimal?
    // Deprecated: the static form reads CostLedger.shared's prices.
    static func projectedCost(target: RemoteTarget, inputTokens: Int, maxOutput: Int) -> Decimal?

    func record(provider: String, model: String? = nil, usage: TokenUsage?,
                origin: RemoteTarget.Origin, compressionRatio: Double?)

    struct Record: Identifiable, Sendable {
        let id: UUID
        let provider: String
        let model: String?
        let isLocalNetwork: Bool
        let usage: TokenUsage?
        let costUSD: Decimal?          // nil: a cloud call with no price, or without both token counts
        let compressionRatio: Double?
    }
}

@MainActor
public protocol ConsentGate: Sendable {
    // projectedCostUSD is nil when the target's price is unknown.
    func requestConsent(target: RemoteTarget, preview: CompressionResult,
                        projectedCostUSD: Decimal?) async -> Bool
    // What routeAndEscalate calls. Default: declines .costCapped, passes any other offer to the method above.
    func requestConsent(target: RemoteTarget, preview: CompressionResult,
                        offer: EscalationOffer) async -> Bool
}

public struct EscalationOffer: Sendable, Equatable {
    init(reason: EscalationReason, projectedCostUSD: Decimal?, sessionSpentUSD: Decimal,
         unpricedRecordCount: Int, costCapUSD: Decimal)
    let reason: EscalationReason
    let projectedCostUSD: Decimal?     // nil when the target is unpriced
    let sessionSpentUSD: Decimal       // CostLedger.sessionCostUSD; a lower bound while unpricedRecordCount > 0
    let unpricedRecordCount: Int
    let costCapUSD: Decimal            // policy.costCapUSDPerSession
}
```

A call that throws after it streamed text (a cancel, a dropped stream) is still recorded, with the
usage the provider reported before failing, if any; a cloud call recorded without usage has an
unknown cost. A call that fails before any output is not recorded. A provider response that carries
only one of the two token counts is treated as no usage.

Built-in prices, approximate list prices to override with your plan's: `claude-sonnet-4-5` on
`"cloud.anthropic"` ($3 in / $15 out per 1M tokens) and `gpt-4o` on `"cloud.openai"` ($2.50 / $10).
No provider-wide defaults exist, so any other model starts unpriced.

```swift
let ledger = CostLedger.shared
ledger.setPrice(TokenPrice(inputUSDPerMillion: 0.5, outputUSDPerMillion: 1.5), provider: "my-gateway")
let projected = ledger.projectedCost(target: target, inputTokens: 2_000, maxOutput: 1_024)   // Decimal?
```

Also available: `ContextCompressor` (token-saving compression), `PIIRedactor`,
`ResponseCache`, and `NetworkMonitor`.

### Limitations

- `CostLedger` prices only the two default cloud models out of the box. Any other cloud model is
  unknown (`nil`), not free, until you call `setPrice`, and until then the router offers each
  escalation to it with `.costUnknown` rather than checking the cap. Unpriced records add nothing to
  `sessionCostUSD`, so while `unpricedRecordCount > 0` a priced request under the cap is offered as
  `.costUnknown` too.
- Only `routeAndEscalate` applies the cap. `escalate(to:)` sends without checking it.
- `ResponseCache.shared` is in memory only (64 entries by default, cleared on relaunch). Its key is
  the provider, the model and the user message (compressed context plus question); the system
  prompt is not part of it.
- The consent preview `routeAndEscalate` shows is compressed with `policy.keepRatio`, but
  `escalate` compresses again at a fixed 0.5. With any other `keepRatio`, the context sent differs
  from the preview the user approved.
