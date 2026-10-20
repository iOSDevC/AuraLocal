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
    temperature: Float? = nil,            // nil: backend default; 0 is greedy
    topP: Float? = nil,                   // nil: backend default; 0 < topP <= 1
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> AuraLocal

// Load a vision model
static func vision(
    _ model: Model = .qwen35_0_8b,
    temperature: Float? = nil,
    topP: Float? = nil,
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> AuraLocal

// Load an OCR/document-specialized model
static func specialized(
    _ model: Model = .fastVLM_0_5b_fp16,
    temperature: Float? = nil,
    topP: Float? = nil,
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> AuraLocal
```

`temperature` and `topP` hold for the instance's lifetime and behave as in
[`ModelManager.load`](#loading); an out-of-range value throws `ModelConfigurationError` before loading.

The factories download MLX weights on first use but not GGUF files: an uncached GGUF throws *GGUF file not
found*. They also skip `ModelManager`'s cache and memory guard. Use `ModelManager.shared.load(_:)`, which
downloads, caches and applies the guard, or fetch the file first with [`ModelDownloads`](#modeldownloads).

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
// Text generation (constraint: GGUF on llama.cpp only, see below)
func chat(_ prompt: String, systemPrompt: String? = nil, maxTokens: Int = 1024,
          constraint: GenerationConstraint? = nil) async throws -> String
func stream(_ prompt: String, systemPrompt: String? = nil, maxTokens: Int = 1024,
            constraint: GenerationConstraint? = nil) -> AsyncThrowingStream<String, Error>

// Vision (image nil = text-only)
func analyze(_ prompt: String, image: PlatformImage? = nil, maxTokens: Int = 800) async throws -> String
func streamVision(_ prompt: String, image: PlatformImage? = nil, maxTokens: Int = 800) -> AsyncThrowingStream<String, Error>
func extractDocument(_ image: PlatformImage, maxTokens: Int? = nil) async throws -> String   // nil: 2048 for DocTags models, else 600

// Multi-turn with conversation history
func chat(_ prompt: String, in conversationID: UUID, systemPrompt: String? = nil, maxTokens: Int = 1024,
          maxContextTokens: Int = 3072, store: ConversationStore = .shared) async throws -> String
func stream(_ prompt: String, in conversationID: UUID, systemPrompt: String? = nil, maxTokens: Int = 1024,
            maxContextTokens: Int = 3072, store: ConversationStore = .shared) -> AsyncThrowingStream<String, Error>
// Stopping iteration cancels the generation; what it produced is stored as the reply (see below).
```

When the consumer of `stream(_:in:)` stops early, the generation is cancelled and the conversation is settled as
for a `GenerationInterrupted`: the text generated until then becomes the assistant turn, and the user turn is removed
when there was none. Stopping means cancelling the task that iterates, or leaving the loop once nothing else holds
the stream: a stream kept in a variable or property is cancelled only when it is released, so stop one by cancelling
its task. These writes land after the loop has ended, so `chat(_:in:)` and `stream(_:in:)` on the same instance wait
for the previous `stream(_:in:)` to finish, a stopped one included, before they store their prompt. Streams consumed
outside history (`stream`, `streamVision`) store nothing and cancel their decode the same way.

On a GGUF instance with tools (`ModelManager.load(_:tools:)`, an `AuraProfile` with tools), stopping cuts the reply
and, in history, stores it at once, but llama.cpp keeps decoding until the end of generation or a full context window,
and the next turn on that instance waits for it.

### Constrained Generation

```swift
public enum GenerationConstraint: Sendable, Equatable {
    case json                                          // a JSON object (LocalLLMClient's json.gbnf)
    case grammar(gbnf: String, root: String = "root")  // any GBNF grammar
}

public enum GenerationConstraintError: LocalizedError, Equatable {
    case unsupportedBackend(BackendKind)   // .mlx, .layerStreaming or .remote
    case invalidGrammar(root: String)      // blank, or rejected by llama.cpp
}
```

```swift
let json = try await llm.chat("Return a JSON object with name and age for: Ana, 31.",
                              maxTokens: 128, constraint: .json)
let verdict = try await llm.chat("Is 4 even? Answer yes or no.", maxTokens: 8,
                                 constraint: .grammar(gbnf: #"root ::= "yes" | "no""#))
```

Only `chat(_:systemPrompt:maxTokens:constraint:)` and `stream(_:systemPrompt:maxTokens:constraint:)` take a
constraint. Each constrained call runs on its own short-lived llama.cpp client; see
[Constrained generation]({{ '/guide/backends' | relative_url }}#constrained-generation-gguf-only) for why, what it
costs, and the limits.

{: .warning }
> Only `LlamaCppBackend` enforces a constraint. An MLX, layer-streaming or remote backend throws
> `GenerationConstraintError.unsupportedBackend` instead of answering unconstrained. `maxTokens` or a full context
> window still cuts the output, and a cut JSON object does not parse. A cancelled constrained call throws
> `CancellationError`.

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
var model: Model { get }                           // the loaded model
var lastGenerationStats: GenerationStats? { get }  // the last generation that returned; nil after one that threw

// Opt-in guard checked before and during every generation; nil (default) is off.
// Instances from ModelManager without tools are shared per model and sampling setting, so this applies to every
// caller of that cached instance; a tool-enabled instance is never shared.
var deviceGuard: DeviceConditions?

// Set when the last generation outgrew the MLX KV window (which drops old tokens silently);
// nil when it fit, the model does not run on MLX, its architecture ignores the window (for example
// Qwen3.5 text, Gemma 3/4 and LFM2 keep their own caches, so nothing was dropped),
// or the last generation threw.
var lastContextOverflow: ContextWindowOverflow? { get }
```

### Generation Stats

Token counts and timings of the last generation, labelled with where they come from. Read them after `chat`
returns or a stream finishes:

```swift
for try await delta in llm.stream("Summarize this in one line.") { print(delta, terminator: "") }
if let stats = llm.lastGenerationStats {
    print(stats.source.label, stats.timeToFirstToken, stats.decodeTokensPerSecond ?? 0)
}
```

```swift
public struct GenerationStats: Sendable, Hashable {
    public enum Provenance: String, Sendable, Hashable, CaseIterable {
        case runtime    // the runtime's own counters (MLX GenerateResult); decode timed by AuraLocal
        case measured   // AuraLocal's clock around the stream; output counted per piece; prompt estimated
        case estimated  // counts from text length (UTF-8 bytes ÷ 4)
        var label: String
    }
    let backend: BackendKind
    let source: Provenance
    let promptTokens: Int              // exact only for .runtime
    let generatedTokens: Int           // stop token excluded
    let promptTime: TimeInterval       // prefill through the first token; equals timeToFirstToken unless .runtime
    let timeToFirstToken: TimeInterval // from when the backend starts the request; includes template + tokenization
    let decodeTime: TimeInterval       // first generated token to last
    var decodeTokensPerSecond: Double?  // (generatedTokens − 1) ÷ decodeTime; nil below two tokens
    var prefillTokensPerSecond: Double? // promptTokens ÷ promptTime
}
```

| Backend | `source` | Prompt tokens | Output tokens | Durations |
|---|---|---|---|---|
| `MLXBackend` | `.runtime` | exact | exact | runtime prefill; decode and TTFT from AuraLocal's clock (first token callback to last) |
| `LlamaCppBackend`, no tools; constrained calls | `.measured` | estimated from the rendered prompt | one per streamed piece (one per sampled token in LocalLLMClient 0.5.0) | AuraLocal's clock |
| `LlamaCppBackend` / `LayerStreamingBackend` with tools | `.estimated` | estimated | estimated | AuraLocal's clock |
| `LayerStreamingBackend`, no tools | `.measured` | estimated | one per streamed piece | AuraLocal's clock |
| Remote (`HybridEscalator`) | — | — | — | no stats; provider-reported usage is in `HybridEscalator.Result.usage` |

LocalLLMClient 0.5.0 exposes no tokenizer or prompt count, so llama.cpp prompt counts stay estimates. A constrained
call's clock starts after its dedicated client loads. `InferenceBackend` gained `var lastGenerationStats:
GenerationStats? { get }` and `var lastContextOverflow: ContextWindowOverflow? { get }`, both with a public
protocol-extension default of `nil`. It also requires `generate(prompt:systemPrompt:maxTokens:constraint:onToken:)`
with no default, so a conformer written before constrained generation must add it; throwing
`GenerationConstraintError.unsupportedBackend` is enough. No public API accepts a custom backend.

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
// Load a model — returns cached instance if already loaded with the same sampling
// For GGUF models, downloads the file from HuggingFace first
func load(
    _ model: Model,
    tools: [any LLMTool] = [],            // GGUF only; sessions with tools are built fresh, never cached
    temperature: Float? = nil,            // nil: backend default; 0 is greedy
    topP: Float? = nil,                   // nil: backend default; 0 < topP <= 1
    onProgress: (@MainActor (String) -> Void)? = nil
) async throws -> AuraLocal

public enum ModelConfigurationError: LocalizedError, Equatable, Sendable {
    case toolsUnsupported(BackendKind)    // tools with an MLX model
    case invalidTemperature(Float)        // negative, infinite or NaN
    case invalidTopP(Float)               // outside 0 < topP <= 1, or NaN
}
```

`load` checks `tools` and the sampling values first and throws `ModelConfigurationError` before
downloading anything. `temperature` and `topP` hold for the instance's lifetime: a GGUF builds its sampler when
the context is created. `nil` keeps the backend default (GGUF 0.7 / 0.95; MLX 0.7 for text, 0.1 for vision,
0 for OCR, top-p off). Each model and setting is its own cached instance, a separate load of the model; see
[Sampling]({{ '/guide/backends' | relative_url }}#sampling-temperature-and-top-p).

Before downloading, `load` throws `AuraError.modelTooLarge` when `HardwareAnalyzer` rates the model
`.tooLarge` on its own. It keeps at most `memoryBudget` instances loaded (a model at two settings counts twice)
and evicts the least recently used to make room. On iOS, where `HardwareProfile.current()` reports the app's
memory limit, it also evicts least-recently-used models until the new one is no longer `.tooLarge` beside the
ones that stay
([Memory Management]({{ '/guide/memory' | relative_url }})).

{: .warning }
> `tools` run only on the llama.cpp backends (GGUF models). With an MLX model `load` throws
> `ModelConfigurationError.toolsUnsupported(.mlx)`; MLX has no tool-calling loop yet.

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

    var isActive: Bool { get }           // .downloading or .loading
}
```

`ModelLoadingOverlay(model:)` is a SwiftUI view that shows a spinner and the progress text while that
model's state `isActive`, and nothing otherwise.

The state is per model, not per sampling setting. While another temperature or top-p of a loaded model loads,
the state reports that load, and it returns to `.ready` when that load fails or is cancelled (`cancelLoad`
included) as long as another setting of the model is still cached.

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
func evict(_ model: Model)               // every sampling setting the model is loaded with
func evictAll()
func evictAllButMostRecent()             // also runs on memory pressure or a memory warning
func release(_ instance: AuraLocal)      // unloads one instance; drops the cache entry only if it is the shared one
private(set) var memoryBudget: Int       // max instances kept loaded (1–4; one per model and sampling setting),
                                         // sized once from available memory
```

### GGUF Download Progress

```swift
// Detailed download progress for GGUF models
let ggufDownloader: GGUFModelDownloader

func cancelLoad(_ model: Model)          // also cancels the GGUF download
func pauseDownload()
func resumeDownload()
var isDownloadPaused: Bool { get }
var isDownloading: Bool { get }          // false while paused

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
init(directory: URL? = nil)                   // nil = Application Support/AuraLocal/conversations.sqlite
// open() also runs on first use. On a store from an earlier version it replaces the full-text triggers and
// rebuilds the search index once.
func open() async throws
func close() async
@discardableResult func appendTurn(_ turn: Turn) async throws -> Turn
func deleteTurn(id: UUID) async throws
func turns(for conversationID: UUID) async throws -> [Turn]   // oldest first; insertion order within one millisecond
func contextWindow(for conversationID: UUID, maxTokens: Int = 2048) async throws -> [Turn]  // system turns + newest that fit
func pruneAndSummarize(conversationID: UUID, keepLastN: Int = 10, summary: String) async throws
func stats(for conversationID: UUID) async throws -> ConversationStats  // turnCount, totalTokenEstimate, oldestTurn, newestTurn
func search(_ query: String, limit: Int = 20) async throws -> [Turn]
func searchAgentMemory(_ query: String, limit: Int = 8) async throws -> [Turn]   // "AgentCrew:" conversations only
```

`search` and `searchAgentMemory` return turns containing any of the query's words, best match first (FTS5); each
word is quoted, so FTS5 operators in the query are plain text. `deleteTurn`, `deleteConversation` and
`pruneAndSummarize` remove the turns from the index too. Earlier versions left deleted turns in it, so a search
could return a later turn in their place or end early; opening such a store repairs the index. A read that returns
rows (`search`, `searchAgentMemory`, `turns(for:)`, `contextWindow`, `allConversations`, `conversation(id:)`) and
fails part-way throws `StoreError.sqliteError` instead of returning the rows read so far.

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

    init(id: UUID = UUID(), conversationID: UUID, role: Role, content: String,
         createdAt: Date = Date(), tokenEstimate: Int? = nil)   // nil = about 4 characters per token

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
`profile.sampling.maxTokens`. The default `makeEngine`, `AuraSession.liveEngine`, loads the model through
`ModelManager` with the profile's tools and `sampling.temperature` and `sampling.topP`. `init` and
`switchProfile` throw `ModelConfigurationError` for tools on an MLX model or an out-of-range temperature or top-p
before building or tearing down an engine, whatever `makeEngine` is. One generation runs at a time: a new
`stream` or a `switchProfile` cancels the previous one and waits for it to stop, also after `cancel()`, which returns without
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
    var sampling: SamplingParams        // temperature + topP at load, maxTokens per turn; topK, contextOverride unused
    var tools: [any LLMTool]            // GGUF only; an MLX model throws ModelConfigurationError
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
back. AuraLocal has no JSON-Schema-to-GBNF converter, MLX exposes only a raw `LogitProcessor` hook,
and the `AuraProfileEngine` a session drives takes no constraint. On a GGUF model,
`AuraLocal.chat(_:constraint: .json)` (see [Constrained Generation](#constrained-generation)) guarantees
JSON syntax, but not your schema.

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
    let availableMemoryGB: Double        // what a model may fill; from current(): iOS = memoryLimitBytes(),
                                         // macOS = reclaimable memory; 60% of RAM when the OS cannot tell
    let deviceName: String
    let memoryBandwidthGBs: Double?      // nil on iOS and for unrecognised chips

    static func current() -> HardwareProfile

    /// Bytes the process may still allocate, or nil when the OS cannot tell.
    /// iOS returns nil exactly when the process is at or over its jetsam limit —
    /// safety-critical callers must treat nil as danger, never as "plenty free".
    static func availableMemoryBytes() -> Int?

    /// iOS: the app's memory limit, os_proc_available_memory() + phys_footprint, capped at 74% of RAM.
    /// Does not shrink while a model is loaded. nil when iOS reports unknown, and on macOS.
    static func memoryLimitBytes() -> Int?
}

public struct ModelCompatibility: Sendable {
    let model: Model
    let fitLevel: ModelFitLevel
    let requiredMemoryGB: Double
    let availableMemoryGB: Double
    let estimatedDecodeTokensPerSecond: Double?  // nil when chip bandwidth is unknown; see Benchmarks › Speed estimate.
                                                 // For a mixture of experts, the low end of the range below
    let estimatedDecodeTokensPerSecondRange: ClosedRange<Double>?  // mixture of experts only; nil for dense
    var utilizationPercent: Double               // % of available memory consumed
    var speedLevel: SpeedLevel
}

// The estimated memory footprints are properties on `Model`, not on ModelCompatibility:
//   model.estimatedRuntimeMemoryGB    // full-load (monolithic) footprint in GB
//   model.estimatedStreamingMemoryGB  // layer-streaming footprint in GB
```

### Compatible Models

```swift
// Every model passed in is assessed, none filtered out; the default leaves uncensored models out.
// Order: .tooLarge last; otherwise downloaded first, then better fit, then lower memory utilization.
static func compatibleModels(
    from models: [Model] = Model.listedModels,
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

    // ggufFile: the GGUF whose header is read when the repo lists it; else the smallest whole top-level quant
    func check(_ repo: String, on target: DevicePreset = .thisDevice(), ggufFile: String? = nil) async -> CompatibilityReport
    func snapshot(of repo: String, ggufFile: String? = nil) async -> RepoSnapshot      // fetch once…
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
    let overview: ArchitectureFacts          // family, layers, trainedContext, kvHeads, headDim, hasVision, quantization,
                                             // kvLayers (cache per layer type), experts (MoE routing)
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
    let tokensPerSecond: Double?             // nil when the device's bandwidth is unknown; MoE: the range's low end
    let tokensPerSecondRange: ClosedRange<Double>?  // mixture of experts only; nil for dense
    var summary: String                      // "Good · 12.0 of 20.0 GB"
    var decodeText: String?                  // "~18 tok/s", or "~37–75 tok/s (uncalibrated MoE range)"
}
```

`CompatibilityRules.all` lists the rules (`CompatibilityRule` has an `id` and a `summary`).
`RepoSnapshot.listing` is the parsed repo listing (`HFRepoInfo`); its `revision` is the commit sha the listing
describes, so later fetches can be pinned to it, as `CoreMLDiffusionInspector` does.
`GGUFHeaderParser.parse(_:)` reads GGUF v2/v3 metadata from the first bytes of a file into a
`GGUFHeader` (`isComplete` is `false` when the bytes ran out), and `SafetensorsHeader.parse(_:)`
does the same for a safetensors header.

To find repos to check, `HuggingFaceSearch.search(_:limit:sort:tag:session:)` queries the Hugging
Face model search; `tag:` (`"mlx"`, `"gguf"`, …) filters on the server. `HFPipelineFilter` keeps the hits
whose declared `pipelineTag` it admits (`admits(_:)`); `HFPipelineFilter.chat` admits `text-generation`,
`image-text-to-text` and `any-to-any`, plus repos that declare none (`admitsUndeclared`, `true` by default),
since many GGUF uploads declare none. It reads only the tag: whether the weights load is still the checker's call.

### DevicePreset

A device to judge against: this one, measured now, or a fixed class of iPhone or Mac. `budgetGB`
is the memory an app may fill with weights and KV cache. Only `thisDevice()` and `mac32GB` are
measured (on iOS, `thisDevice()` falls back to an estimate of 60% of RAM when iOS reports its memory
limit as unknown); every other budget is an estimate and says so in `source`.

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
    // iPhone by the app's memory limit (HardwareProfile.memoryLimitBytes()), the figure
    // ModelManager.load checks too. A given profile is used as is.
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

// Collections. allModels has every registered model; the others draw from listedModels, which
// leaves uncensored models out: only uncensoredModels lists those.
static var allModels: [Model]
static var listedModels: [Model]
static var textModels: [Model]
static var visionModels: [Model]
static var specializedModels: [Model]
static var mlxModels: [Model]
static var ggufModels: [Model]
static var runnableModels: [Model]     // filtered by HardwareAnalyzer
static var uncensoredModels: [Model]
static func models(in domain: Domain) -> [Model]

// Unavailable (a compile error that gives the reason): the catalog was checked against Hugging Face
// and these repos hold no loadable weights or do not exist:
// graniteVision_3_3, granite4_compact, gemma4_1b, gemma4_4b, lfm25_3_2b, ministral_3b

// Metadata
var displayName: String
var approximateSizeMB: Int
var purpose: Purpose
var format: ModelFormat                // .mlx or .gguf
var ggufFilename: String?              // nil for MLX models
var isUncensored: Bool
var license: String?                   // the card's license, or its license_name when "other"; nil = not recorded
var hasPermissiveLicense: Bool         // license is apache-2.0 or mit (Model.permissiveLicenses); false when nil
var isMacOSRecommended: Bool          // true for models ≥15 GB
var isDownloaded: Bool                 // in ModelStorage.rootDirectory, or a copy an earlier version left in Caches
var cacheDirectory: URL                // GGUF download folder: <root>/models/<repoID> (not in Caches, despite the name)
var resolvedFileURL: URL?              // GGUF: local file, else the finished download (root, then Caches), else its destination
var kvLayers: KVCacheLayout?           // cache per layer type (sliding, recurrent); nil = all full attention
var experts: ExpertRouting?            // mixture of experts: routedExperts, expertsPerToken, activeSizeMB; nil = dense
var maxContextLength: Int?             // a ceiling the weights impose below the memory budget; nil = none
var capabilities: ModelCapabilities    // see below
```

### ModelCapabilities

`model.capabilities` is an `OptionSet` saying what a model takes and does, read from its catalog category, its
`defaultDocumentPrompt` and the backend `BackendRouter` picks for it. AuraExample's Chat tab shows it as badges and
offers image input and document extraction only where they work.

| Capability | Set when | Enables |
|------------|----------|---------|
| `.text` | category `.text` or `.vision` | a conversation from text alone (`chat`, `stream`, `stream(_:in:)`) |
| `.vision` | category `.vision` or `.visionSpecialized`, MLX format (the llama.cpp backends drop the image) | an image with the prompt (`analyze`, `streamVision`) |
| `.documentExtraction` | an image model with a `defaultDocumentPrompt` | `extractDocument(_:)` |
| `.tools` | a GGUF model (only the llama.cpp backends run a tool loop) | `ModelManager.shared.load(_:tools:…)`; an MLX model throws `ModelConfigurationError.toolsUnsupported(.mlx)` |

### ModelRegistry

The collections above read `ModelRegistry.shared.models`: the bundled catalog, any remote catalog, and models
added at runtime.

```swift
public final class ModelRegistry: @unchecked Sendable
static let shared: ModelRegistry
var models: [Model]                                   // read under the registry's lock
func model(id: String) -> Model?
func model(repoID: String) -> Model?
func register(_ model: Model)                         // adds, or replaces the model with the same id
@discardableResult func unregister(id: String) -> Bool  // only ids register(_:) added; catalog ids stay
func refreshFromRemote(url: URL) async throws
func loadCachedRemote(maxAge: TimeInterval = 86_400)
```

### ModelStorage

Where downloaded weights live: `<Application Support>/AuraLocal/Models`, excluded from backup. Inside the app
container on iOS and in a sandboxed Mac app; `~/Library/Application Support/AuraLocal/Models`, shared, for a
non-sandboxed macOS process (the `aura` CLI, an app without App Sandbox). Guide:
[Where models are stored]({{ '/guide/models' | relative_url }}#where-models-are-stored).

```swift
public enum ModelStorage {
    static var rootDirectory: URL     // first access creates it, excludes it from backup, migrates Caches when sandboxed
    // Renames AuraLocal's downloads out of Caches (never copies, never overwrites); idempotent
    @discardableResult static func migrateLegacyDownloads() -> ModelStorageMigration
}

public struct ModelStorageMigration: Sendable, Equatable {
    let moved: [String]               // paths relative to the roots, now in the storage root
    let conflicts: [String]           // left in Caches: the storage root already has that path
    let failures: [String]            // left in Caches: the rename failed; still read from there
    var isEmpty: Bool
}
```

### ModelDownloads

Downloads a model to the exact path its backend loads from, under `ModelStorage.rootDirectory`, without loading
it. Each file arrives in `Range` chunks appended to `<file>.partial`, which survives a cancel, a network cut or a
relaunch; a partial whose ETag no longer matches restarts from zero, and a file moves into place only when its
size matches and, for an LFS file, when its SHA-256 matches the one Hugging Face publishes. A finished model an
earlier version left in Caches is used from there; a GGUF `.partial` left there moves into the storage root and is
resumed. Downloads of the same file or snapshot run one at a time, also with `GGUFModelDownloader`, the MLX load
path, `ChunkedDownloader` and other processes sharing the storage root; a later call returns the finished model
without a transfer. Guides:
[Downloading without loading]({{ '/guide/models' | relative_url }}#downloading-without-loading),
[Concurrent downloads]({{ '/guide/models' | relative_url }}#concurrent-downloads).

```swift
public enum ModelDownloads {
    static func plan(for model: Model, auth: any DownloadAuthorizing = KeychainDownloadAuth()) async throws -> DownloadPlan
    // Resumes <file>.partial, verifies SHA-256; returns the GGUF file or the MLX snapshot folder.
    // progress: (bytes on disk, total)
    static func download(_ model: Model, auth: any DownloadAuthorizing = KeychainDownloadAuth(),
                         progress: @escaping @Sendable (Int64, Int64) -> Void) async throws -> URL
    static func delete(_ model: Model) throws               // finished and partial files, root and Caches; evict first
    static func bytesOnDisk(for model: Model) -> Int64      // root and Caches
    static func availableDiskBytes() -> Int64?              // download needs the remaining bytes + 20%
    static func verifiedSHA256(of file: URL) -> String?     // what a download verified the file against; nil = unverified

    // One folder of a repo at a pinned revision (e.g. a Core ML pipeline); sends no token by default
    static func downloadFolder(repoID: String, revision: String, folder: String, include: [String],
                               auth: any DownloadAuthorizing = NoDownloadAuth(),
                               progress: @escaping @Sendable (Int64, Int64) -> Void) async throws -> URL
    static func planFolder(repoID: String, revision: String, folder: String, include: [String],
                           auth: any DownloadAuthorizing = NoDownloadAuth()) async throws -> DownloadPlan
    static func localFolder(repoID: String, folder: String) -> URL   // where downloadFolder writes
    static func completedFolder(repoID: String, folder: String, revision: String) -> URL?  // root, else Caches
    static func isFolderComplete(repoID: String, folder: String, revision: String) -> Bool
    static func deleteRepo(_ repoID: String) throws                   // models--org--repo: every folder, finished and partial
    static func deleteFolder(repoID: String, folder: String) throws   // one folder with its partials and markers, then the
                                                                      // parents it leaves empty; other folders stay
}

public struct DownloadPlan: Sendable, Equatable {
    let repoID: String
    let files: [PlannedFile]          // path, sizeBytes
    let totalBytes: Int64
    let bytesOnDisk: Int64            // finished files plus .partial bytes
    var remainingBytes: Int64
}

public enum ModelDownloadError: LocalizedError, Equatable, Sendable {
    case unauthorized(repoID: String), notFound(repoID: String, path: String), notFoundOrPrivate(repoID: String)
    case insufficientSpace(neededBytes: Int64, availableBytes: Int64), http(status: Int)
    case incompleteRepo(repoID: String), sizeMismatch(path: String, expected: Int64, actual: Int64), unsupported(String)
}

// Separate from ModelDownloadError so exhaustive switches over it keep compiling. The file is already deleted.
public enum ModelIntegrityError: LocalizedError, Equatable, Sendable {
    case checksumMismatch(path: String, expected: String, actual: String)   // lowercase hex SHA-256
}

// Separate from ModelDownloadError for the same reason. Nothing was moved into place; download again.
public enum DownloadConflictError: LocalizedError, Equatable, Sendable {
    case partialChanged(path: String)   // the finished .partial was gone or resized by another writer
}
```

Only single-file GGUF quants at the top level of a repo download; a split or subfolder quant, or a model
backed by a local file, throws `.unsupported`. `DownloadedModels.scan()`, `scanSnapshots()` and
`partialFiles()` list the GGUF files, MLX snapshots and `.partial` files on disk, in the storage root and in
Caches; `DownloadedModels.modelsDirectory` is the GGUF root downloads write to and `modelsDirectories` every GGUF
root that is read. `GGUFModelDownloader` (the GGUF path of `ModelManager.load`) and
`ChunkedDownloader.download(from:to:fileName:expectedSHA256:onProgress:)` throw `ModelIntegrityError` too; the
latter checks only a hash you pass. `GGUFModelDownloader` and the MLX load path throw `DownloadConflictError` too.
`ChunkedDownloader` waits for another download to the same destination and, when that one ran in the same
process and fetched the same URL and hash, returns its file without a transfer.

### DownloadRecords

The sized `Model` behind each download from a search result, stored in `UserDefaults` under
`dev.auralocal.downloadRecords`, so the model can be registered again after a relaunch. Files on disk stay the
source of truth. See [Keeping downloads across launches]({{ '/guide/models' | relative_url }}#keeping-downloads-across-launches).

```swift
public struct DownloadRecords: @unchecked Sendable
static let standard: DownloadRecords
init(defaults: UserDefaults = .standard)
static func key(for model: Model) -> String            // "gguf:<repo>/<file>" or "mlx:<repo>"
func remember(_ model: Model)                           // replaces any record for the same file
func forget(_ model: Model)
var models: [Model]
func model(repoID: String, ggufFilename: String?) -> Model?   // nil filename = the MLX snapshot
func contains(_ model: Model) -> Bool                   // by id
@discardableResult func registerDownloaded() -> [Model] // at launch; drops records with nothing on disk
@discardableResult func registerIfDownloaded(_ model: Model) -> Bool
func remove(_ model: Model)                             // forget + ModelRegistry.shared.unregister(id:)
```

A record whose file a registry entry already points at is not registered, so one file never shows twice in the
pickers. Each read-modify-write of the stored dictionary holds a lock, so calls from several threads don't drop
records.

---

## BackgroundLifecycle (iOS)

When the app enters the background it cancels every generation running through `AuraLocal` or
`CoreMLDiffusionEngine`, and until the app returns new ones throw at once. Both throw
`GenerationInterrupted`. With `aggressiveMemorySaving`, it also evicts every loaded model except the
most recently used one. The observers are installed when a model loads or an image generation starts.
Guide: [Background Lifecycle]({{ '/guide/memory' | relative_url }}#background-lifecycle-ios).

```swift
@MainActor
public final class BackgroundLifecycle
static let shared: BackgroundLifecycle

@Published private(set) var isPaused: Bool     // true while the app is backgrounded; generations are refused
var aggressiveMemorySaving: Bool = false       // evicts all but the most recently used model on background
```

Nothing is cancelled on macOS.

### GenerationInterrupted

Thrown when AuraLocal stops a generation itself. A stream yields the text first, then finishes with it.

```swift
public struct GenerationInterrupted: Error, LocalizedError, Sendable, Equatable {
    enum Trigger: Sendable, Equatable {
        case appBackgrounded
        case deviceConditions(DeviceConditions.Assessment)
    }
    let reason: Trigger
    let partialText: String      // what was produced; always truncated, always "" for images
}
```

### DeviceConditions

Opt-in guard: assessed before a generation and every `checkInterval` tokens or denoising steps. At warn
the generation continues and `onWarning` runs; at block it throws `GenerationInterrupted`. The battery is
read only when the app has enabled `UIDevice.isBatteryMonitoringEnabled` (process-wide; AuraLocal never
sets it). Guide: [Device conditions]({{ '/guide/memory' | relative_url }}#device-conditions).

```swift
public struct DeviceConditions: Sendable {
    enum Level: Int, Comparable { case normal, warn, block }
    enum Cause: Equatable { case thermal(ProcessInfo.ThermalState), lowPowerMode, lowBattery(Float) }

    struct Reading: Sendable, Equatable {
        var thermalPressure: ProcessInfo.ThermalState
        var isLowPowerModeEnabled: Bool
        var batteryLevel: Float?             // nil unless battery monitoring is on
        var isCharging: Bool?
        static func system() -> Reading                  // any thread; no battery
        @MainActor static func current() -> Reading      // plus battery when monitoring is on (iOS)
    }

    struct Assessment: Sendable, Equatable {
        let severity: Level
        let causes: [Cause]                  // every cause at warn or block
        let snapshot: Reading
    }

    var warnAtThermalState: ProcessInfo.ThermalState?   // .serious
    var blockAtThermalState: ProcessInfo.ThermalState?  // .critical
    var lowPowerMode: Level                             // .warn
    var warnBelowBatteryLevel: Float?                   // nil: battery ignored
    var blockBelowBatteryLevel: Float?                  // nil
    var checkInterval: Int                              // 8
    var onWarning: (@Sendable (Assessment) -> Void)?    // once per transition into warn

    func assess(_ reading: Reading) -> Assessment
}
```

### ContextWindowOverflow

```swift
public struct ContextWindowOverflow: Sendable, Equatable {
    let promptTokens: Int
    let generatedTokens: Int
    let windowSize: Int
    var droppedTokens: Int { get }   // promptTokens + generatedTokens - windowSize
}
```

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
    // By default probes Ollama (http://localhost:11434) and llama-server (http://127.0.0.1:8080/v1)
    // on this machine; pass LocalProviderEndpoint(kind:baseURL:) for another host. Never throws.
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

    // Route by policy: stays local, escalates, or asks consent per rules R1–R7. Builds one payload
    // (policy.keepRatio, redaction per policy.piiRedaction), shows it to the gate on an offer, and
    // sends exactly it.
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
    ) async throws -> Result?              // nil = the router kept it local
                                           // throws AuraError.escalationDeclined when the gate declines

    // Direct escalation to a chosen target: preparePayload(keepRatio: 0.5, redactPII:), then
    // escalate(_:to:). Checks no policy, consent or cost cap.
    func escalate(
        to target: RemoteTarget,
        systemPrompt: String? = nil,
        context: String,
        question: String,
        maxTokens: Int = 1024,
        redactPII: Bool = false,           // PIIRedactor on the context and the question
        onToken: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> Result

    // The exact request, not sent: with redactPII, redacts the context and the question, then compresses
    // the context to keepRatio × the target's context window (8192 when unknown) − maxTokens, at least 256.
    func preparePayload(
        for target: RemoteTarget,
        systemPrompt: String? = nil,
        context: String,
        question: String,
        maxTokens: Int = 1024,
        keepRatio: Double = 0.5,
        redactPII: Bool = false
    ) -> EscalationPayload
    // Sends a payload exactly as built, without recompressing or redacting; an earlier request with the same
    // provider, model, messages and maxTokens is answered from ResponseCache.shared. Records cost. Checks no
    // policy, consent or cap.
    func escalate(
        _ payload: EscalationPayload,
        to target: RemoteTarget,
        onToken: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> Result

    // A llama-server (preferred) or Ollama on this machine (LocalProviderEndpoint.defaults), with its
    // largest-context model; nil if none runs. Nothing scans the network.
    // Skips models with runsLocally == false (Ollama cloud models forward prompts off the machine).
    static func bestLocalTarget() async -> RemoteTarget?
    // BYOK targets from the Keychain: "cloud.anthropic", then "cloud.openai". [] when !allowCloud.
    static func cloudTargets(
        allowCloud: Bool,
        anthropicModel: String = RemoteTarget.defaultAnthropicModel,   // "claude-sonnet-4-5"
        openAIModel: String = RemoteTarget.defaultOpenAIModel          // "gpt-4o"
    ) -> [RemoteTarget]
    // bestLocalTarget() first, then cloudTargets(allowCloud: policy.allowCloud).
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

public struct EscalationPayload: Sendable, Equatable {
    let systemPrompt: String?            // sent verbatim: never compressed or redacted
    let userMessage: String              // "Context:\n<context>\n\nQuestion: <question>", or the question alone
    let maxTokens: Int
    let redactionEnabled: Bool           // PIIRedactor ran on the context and the question
    let redactedPIICount: Int            // in the question and the whole context, before compression
    let compression: CompressionResult   // keptText is the context as sent
    var messages: [[String: String]]     // role/content, system first; Anthropic sends it as top-level "system"
    var estimatedInputTokens: Int        // system prompt + user message at ~4 chars/token; what routeAndEscalate prices
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
    // Default 0.5. routeAndEscalate compresses the context to keepRatio × the target's context window
    // minus maxTokens; escalate(to:) always uses 0.5.
    var keepRatio: Double
    // Default .cloudOnly. Where routeAndEscalate redacts the context and the question.
    // A policy saved without this key decodes as .cloudOnly.
    var piiRedaction: RedactionScope

    enum RedactionScope: String, Sendable, Codable {
        case cloudOnly                 // cloud targets; your own LAN box gets the text verbatim
        case allTargets
        case never
    }

    init(mode: Mode = .off, allowCloud: Bool = false, costCapUSDPerSession: Decimal = 1.0, keepRatio: Double = 0.5,
         piiRedaction: RedactionScope = .cloudOnly)
    static let off: EscalationPolicy   // the default — never leaves the device
    func redactsPII(for target: RemoteTarget) -> Bool
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
// ModelCompatibilityChecker) · "download.<host>" (bearer token for any other download host).
// "cloud.github-models" is no longer read (GitHub Models was retired).
public enum KeychainStore {
    static func save(_ key: String, for account: String) throws
    static func read(for account: String) -> String?
    static func delete(for account: String)
    static func hasKey(for account: String) -> Bool
}
// KeychainDownloadAuth.saveHuggingFaceToken(_:) (trims; throws on empty), .deleteHuggingFaceToken(),
// .hasHuggingFaceToken

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
    // projectedCostUSD is nil when the target's price is unknown. From routeAndEscalate, preview is the
    // payload's compression: its keptText is the context as sent, without the question or system prompt.
    func requestConsent(target: RemoteTarget, preview: CompressionResult,
                        projectedCostUSD: Decimal?) async -> Bool
    // What routeAndEscalate calls. Default: declines .costCapped, passes any other offer to the method above.
    func requestConsent(target: RemoteTarget, preview: CompressionResult,
                        offer: EscalationOffer) async -> Bool
}

public struct EscalationOffer: Sendable, Equatable {
    init(reason: EscalationReason, projectedCostUSD: Decimal?, sessionSpentUSD: Decimal,
         unpricedRecordCount: Int, costCapUSD: Decimal, payload: EscalationPayload? = nil)
    let reason: EscalationReason
    let projectedCostUSD: Decimal?     // payload's estimated tokens + maxTokens; nil when the target is unpriced
    let sessionSpentUSD: Decimal       // CostLedger.sessionCostUSD; a lower bound while unpricedRecordCount > 0
    let unpricedRecordCount: Int
    let costCapUSD: Decimal            // policy.costCapUSDPerSession
    let payload: EscalationPayload?    // what approval sends; routeAndEscalate always sets it
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
- `ResponseCache.shared` is in memory only (64 entries by default, cleared on relaunch). `HybridEscalator`
  keys it on the provider, the model and the whole payload: system prompt, user message (compressed
  context plus question) and `maxTokens`. `lookup`/`insert` with `prompt:` key on that string alone.
- `PIIRedactor` is pattern-based (emails, `sk-…`/`pk_…`/`rk-…` keys, bearer tokens, Slack and GitHub
  tokens, AWS access key IDs) and covers the context and the question only. The system prompt is
  sent verbatim on every path.
