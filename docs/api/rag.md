---
layout: docs
title: AuraDocs (RAG)
parent: API Reference
nav_order: 4
description: "AuraDocs API reference — DocumentLibrary, DocumentChat, DocsTab for on-device RAG."
---

# AuraDocs — Document RAG
{: .no_toc }

Fully local Retrieval-Augmented Generation pipeline. Index documents once, ask questions in natural language. Zero external dependencies.

```
query → embed (TF-IDF, or multilingual-e5-small) → FTS5 top-20 → cosine re-rank top-5 → LLM
```

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Supported Formats

| Format | Parser |
|--------|--------|
| `.pdf` | PDFKit |
| `.docx` | ZIP + XML parsing |
| `.txt`, `.md`, `.markdown` | Plain text |
| `.png`, `.jpg`, `.jpeg`, `.heic`, `.tiff` | MLX VLM OCR |

---

## DocumentLibrary

```swift
public actor DocumentLibrary
static let shared: DocumentLibrary
```

### Setup

```swift
func configure(embeddingProvider: any EmbeddingProvider, llm: AuraLocal) async
func open() async throws
```

### Indexing

```swift
func add(
    url: URL,
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws

func refreshCorpus() async  // rebuild TF-IDF weights after batch indexing
```

### Querying

```swift
func ask(
    _ question: String,
    topK: Int = 5,
    maxContextTokens: Int = 2048,
    systemPrompt: String? = nil
) async throws -> Answer

public struct Answer {
    let text: String
    let sources: [Source]
}

public struct Source {
    let documentTitle: String
    let pageNumber: Int
    let excerpt: String
    let score: Float
}
```

### Management

```swift
func allDocuments() async throws -> [IndexedDocument]
func removeDocument(id: UUID) async throws
```

### Custom Options

```swift
// Custom chunk size (default: 512 tokens, 10% overlap)
init(chunkTargetTokens: Int = 512, chunkOverlapFraction: Double = 0.1)
```

---

## DocumentChat

Stateful multi-turn Q&A with source citations.

```swift
@MainActor
public final class DocumentChat: ObservableObject

init(library: DocumentLibrary, llm: AuraLocal)
```

```swift
func send(_ message: String) async throws -> ChatMessage

@Published var messages: [ChatMessage]

public struct ChatMessage {
    let role: Turn.Role
    let text: String
    let sources: [Source]
}
```

---

## Embedding providers

```swift
public protocol EmbeddingProvider: Sendable {
    func embed(_ text: String) async throws -> [Float]
    func embedBatch(_ texts: [String]) async throws -> [[Float]]        // default: embed() in a loop
    var dimensions: Int { get }
    func embedQuery(_ text: String) async throws -> [Float]             // default: embed()
    func embedDocuments(_ texts: [String]) async throws -> [[Float]]    // default: embedBatch()
    var identifier: String { get }                                      // default: type name + "/" + dimensions
}
```

### AutoEmbeddingProvider

```swift
public actor AutoEmbeddingProvider: EmbeddingProvider
init()                                                            // TF-IDF, 4096-dim, no download
init(embeddingModelAt bundleURL: URL?, compiledModelsDirectory: URL? = nil)   // e5 if the bundle is valid, else TF-IDF
static let defaultModelBundleURL: URL                             // Application Support/AuraLocal/embeddings/multilingual-e5-small
nonisolated var usesDenseModel: Bool
nonisolated let denseModelProblem: String?                        // why the bundle is not used
func warmUp() async throws -> Duration                            // @discardableResult; .zero for TF-IDF
func truncatedInputCount() async -> Int
func backendName() -> String
func updateCorpus(texts: [String]) async                          // TF-IDF weights
```

### CoreMLEmbeddingProvider

A Core ML embedding bundle (multilingual-e5-small) with no fallback; `embedQuery` uses the
`query: ` prefix, `embed` / `embedBatch` / `embedDocuments` the `passage: ` prefix.

```swift
public struct CoreMLEmbeddingProvider: EmbeddingProvider
init(bundleAt url: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine, compiledModelsDirectory: URL? = nil) throws
let tool: CoreMLTextEmbeddingTool       // AuraCore
let identifier: String                  // "model_id@revision"
var modelName: String
func warmUp() async throws -> Duration
func truncatedInputCount() async -> Int
```

### Index identity

The store records the provider's `identifier` and `dimensions`; `add` and `ask` re-embed stored
chunks when they change. See [Embedding providers](../guide/rag.md#embedding-providers).

```swift
func indexNeedsReembedding() async throws -> Bool
func reembedAll(onProgress: @escaping @MainActor (String) -> Void = { _ in }) async throws
```

---

## DocsTab

Drop-in SwiftUI tab. Includes file picker, per-document progress, swipe-to-delete, and chat sheet with source citations.

```swift
import AuraDocs

TabView {
    DocsTab()
        .tabItem { Label("Docs", systemImage: "doc.text.magnifyingglass") }
}
```

---

## Progress Stages

`onProgress` is called on `@MainActor` through four stages:

| Stage | Example | % |
|-------|---------|---|
| Parsing | `"Parsing MyDoc.pdf…"` | 5% |
| Chunking | `"Chunking MyDoc…"` | 15% |
| Embedding | `"Embedding MyDoc: 42%"` | 15–100% |
| Done | `"'MyDoc' indexed ✓ (253 chunks)"` | 100% |
