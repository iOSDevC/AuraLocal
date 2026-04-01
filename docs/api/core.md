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
// Static chat — loads default model, returns reply
static func chat(
    _ prompt: String,
    model: Model = .qwen3_1_7b,
    systemPrompt: String? = nil
) async throws -> String

// Static receipt/document extraction
static func extractDocument(_ image: PlatformImage) async throws -> String

// Parse DocTags output from Granite Docling
static func parseDocTags(_ raw: String) -> String
```

### Instance Methods

```swift
// Text generation
func chat(_ prompt: String, systemPrompt: String? = nil) async throws -> String
func stream(_ prompt: String, systemPrompt: String? = nil) -> AsyncThrowingStream<String, Error>

// Vision
func analyze(_ prompt: String, image: PlatformImage) async throws -> String
func streamVision(_ prompt: String, image: PlatformImage) -> AsyncThrowingStream<String, Error>
func extractDocument(_ image: PlatformImage) async throws -> String

// Multi-turn with conversation history
func chat(_ prompt: String, in conversationID: UUID) async throws -> String
func stream(_ prompt: String, in conversationID: UUID) -> AsyncThrowingStream<String, Error>
```

### History Methods

```swift
// Auto-generate a title from the first message
func autoTitle(conversationID: UUID) async throws

// Summarize and prune long conversations
func summarizeAndPrune(
    conversationID: UUID,
    keepLastN: Int = 10,
    maxContextTokens: Int = 4096
) async throws
```

### Properties

```swift
var model: Model { get }     // the loaded model
var isLoaded: Bool { get }   // whether the backend has weights in memory
```

---

## ModelManager

Shared singleton for loading and caching models. Preferred over direct `AuraLocal` factory calls — prevents redundant downloads and handles memory pressure.

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
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> AuraLocal
```

### State Observation

```swift
@Published var states: [Model: ModelLoadState]

func state(for model: Model) -> ModelLoadState

public enum ModelLoadState {
    case idle
    case downloading(progress: Double)   // 0.0–1.0
    case loading
    case ready
    case failed(String)
}
```

### Backend Query

```swift
// Which backend will be used for a model on the current device
func recommendedBackend(for model: Model) -> BackendKind

public enum BackendKind {
    case mlx
    case llamaCpp
    case layerStreaming
}
```

### Eviction

```swift
func evict(_ model: Model)
func evictAll()
```

### GGUF Download Progress

```swift
// Detailed download progress for GGUF models
var ggufDownloader: GGUFModelDownloader

// GGUFModelDownloader published properties:
@Published var progress: Double           // 0.0–1.0
@Published var bytesDownloaded: Int64
@Published var totalBytes: Int64
@Published var isDownloading: Bool
```

---

## ConversationStore

SQLite-backed store for persistent chat history. Actor-isolated.

```swift
public actor ConversationStore
static let shared: ConversationStore
```

### Conversations

```swift
func createConversation(model: Model, title: String) async throws -> Conversation
func allConversations() async throws -> [Conversation]
func conversation(id: UUID) async throws -> Conversation?
func updateTitle(_ title: String, for id: UUID) async throws
func deleteConversation(id: UUID) async throws
```

### Messages

```swift
func turns(for conversationID: UUID) async throws -> [Turn]
func search(_ query: String) async throws -> [Turn]
```

### Types

```swift
public struct Conversation: Sendable {
    let id: UUID
    let model: Model
    var title: String
    let createdAt: Date
    let updatedAt: Date
}

public struct Turn: Sendable {
    let id: UUID
    let conversationID: UUID
    let role: Role          // .user / .assistant / .system
    let content: String
    let createdAt: Date

    public enum Role: String, Sendable { case user, assistant, system }
}
```

---

## HardwareAnalyzer

Assesses model–device compatibility. All methods are synchronous.

```swift
public struct HardwareAnalyzer
```

### Assessment

```swift
// Assess a model against the current device
static func assess(
    _ model: Model,
    profile: HardwareProfile = .current()
) -> AssessmentResult

public struct AssessmentResult: Sendable {
    let model: Model
    let profile: HardwareProfile
    let fitLevel: ModelFitLevel
    let estimatedRuntimeMemoryGB: Double
    let estimatedStreamingMemoryGB: Double
}
```

### Compatible Models

```swift
// All models that can run on this device, sorted by fit (best first)
static func compatibleModels(
    profile: HardwareProfile = .current()
) -> [(model: Model, result: AssessmentResult)]
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

## Model Enum

```swift
public enum Model: String, CaseIterable, Sendable

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

Pauses inference when app enters background to prevent jetsam termination.

```swift
@MainActor
public final class BackgroundLifecycle
static let shared: BackgroundLifecycle

@Published var isPaused: Bool                  // true when app is backgrounded
var aggressiveMemorySaving: Bool = false       // evicts models on background
```

No-op on macOS.
