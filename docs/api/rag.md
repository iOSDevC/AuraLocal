---
layout: docs
title: AuraDocs (RAG)
parent: API Reference
nav_order: 4
description: "AuraDocs API reference — DocumentLibrary, DocumentChat, DocsTab for on-device RAG."
---

# AuraDocs — Document RAG
{: .no_toc }

Fully local Retrieval-Augmented Generation pipeline. Index documents once, ask questions in natural language. Adds no packages beyond AuraCore.

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
| `.txt`, `.md`, `.markdown`, `.rtf` | Plain text, read as UTF-8 |
| Source and config files (`.swift`, `.py`, `.js`, `.ts`, `.json`, `.yaml`, …) | Plain text; the title keeps the extension |
| `.png`, `.jpg`, `.jpeg`, `.heic`, `.tiff`, `.bmp` | OCR by the `visionLLM` passed to `configure` |

{: .note }
> Images are indexed only when `configure` received a `visionLLM`; without one, `add` throws
> `DocumentError.unsupportedFormat`. `.rtf` files are not converted, so RTF control words are indexed
> along with the text.

---

## DocumentLibrary

```swift
public actor DocumentLibrary
static let shared: DocumentLibrary
```

### Setup

```swift
// Actor-isolated, so call with await. visionLLM is needed to index image files.
func configure(embeddingProvider: any EmbeddingProvider, llm: AuraLocal, visionLLM: AuraLocal? = nil)
func open() async throws
func close() async
```

### Indexing

```swift
@discardableResult
func add(
    url: URL,
    onProgress: @escaping @MainActor (String) -> Void = { _ in }
) async throws -> IndexedDocument   // returns the existing entry only for a file already added in this app launch

// Rebuilds TF-IDF weights from the stored chunks after batch indexing. Only acts when the
// configured provider is an AutoEmbeddingProvider; a TFIDFEmbeddingProvider passed directly is skipped.
func refreshCorpus() async
```

### Querying

```swift
func ask(
    _ question: String,
    topK: Int = 5,
    maxContextTokens: Int = 2048,
    systemPrompt: String? = nil
) async throws -> DocumentAnswer

public struct DocumentAnswer: Sendable {
    let text: String
    let sources: [SourceReference]

    public struct SourceReference: Sendable {   // DocumentAnswer.SourceReference
        let documentTitle: String
        let pageNumber: Int      // 1-based; 0 when the format has no pages
        let excerpt: String      // first 200 characters of the chunk
        let score: Float
    }
}
```

### Management

```swift
func allDocuments() async throws -> [IndexedDocument]
func removeDocument(id: UUID) async throws

@discardableResult
func export(
    documentID: UUID,
    to destination: URL,                // a directory
    format: ExportFormat = .jsonlGz,    // .jsonlGz or .jsonl
    includeEmbeddings: Bool = false
) async throws -> URL                   // the written file

public struct IndexedDocument: Identifiable, Sendable {
    let id: UUID
    let title: String
    let url: URL
    let chunkCount: Int
    let indexedAt: Date
}
```

### Custom Options

```swift
// Custom location and chunk size (default: 512 tokens, 10% overlap).
// directory nil = Application Support/AuraLocal/docs
init(directory: URL? = nil, chunkTargetTokens: Int = 512, chunkOverlapFraction: Double = 0.1)
```

---

## DocumentChat

Observable Q&A session with source citations.

{: .warning }
> Each `send` is answered on its own: earlier messages are not sent to the model, so a follow-up
> such as "and the second one?" has no context. The question and answer are also appended to a
> "Document chat" conversation in `ConversationStore`.

```swift
@MainActor
public final class DocumentChat: ObservableObject

init(library: DocumentLibrary, llm: AuraLocal, store: ConversationStore = .shared)
```

```swift
@discardableResult
func send(_ question: String, topK: Int = 5) async throws -> DocumentAnswer
func clear()                        // empties messages; the stored conversation is kept

@Published private(set) var messages: [DocumentChatMessage]
@Published private(set) var isThinking: Bool
@Published private(set) var progress: String   // never set by DocumentChat; stays ""

public struct DocumentChatMessage: Identifiable, Sendable {
    let id: UUID
    enum Role { case user, assistant }
    let role: Role
    let text: String
    let sources: [DocumentAnswer.SourceReference]
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
    var identifier: String { get }                                      // default: module.Type + "/" + dimensions; must be stable across launches
}
```

### TFIDFEmbeddingProvider

Hashed TF-IDF: offline, nothing to download. IDF weights live in memory and start empty.

```swift
public actor TFIDFEmbeddingProvider: EmbeddingProvider
init()
static let vocabSize: Int                  // 4096, also `dimensions`
static let vectorSpaceIdentifier: String   // "aura.tfidf-hash/4096", also `identifier`
func updateCorpus(texts: [String])         // IDF weights; embed() weighs unseen terms 1.0
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
let dimensions: Int
let identifier: String                  // "model_id@revision"
var modelName: String
@discardableResult
func warmUp() async throws -> Duration
func truncatedInputCount() async -> Int
```

The init throws when the bundle is missing or invalid; the model itself loads on `warmUp()` or the
first call. The underlying tool is documented in
[AuraCore: Text embeddings]({{ '/api/core' | relative_url }}#text-embeddings-core-ml).

### Index identity

The store records the provider's `identifier` and `dimensions`. When they no longer match the
configured provider, the next `add` or `ask` re-embeds every stored chunk from its text first
(documents are not parsed again). `ask` reports no progress while it does, so call
`indexNeedsReembedding()` and `reembedAll(onProgress:)` up front to show it. An empty index is
adopted without re-embedding, and so is an index written before identities were recorded when the
configured provider is TF-IDF with the same vector length. See
[Embedding providers]({{ '/guide/rag' | relative_url }}#embedding-providers).

```swift
func indexNeedsReembedding() async throws -> Bool
func reembedAll(onProgress: @escaping @MainActor (String) -> Void = { _ in }) async throws
// progress: "Re-embedding 400 chunks…", then "Re-embedding 50/400 chunks: 12%" per batch of 50
@discardableResult
func configureIfNeeded(embeddingProvider: any EmbeddingProvider, llm: AuraLocal,
                       visionLLM: AuraLocal? = nil) -> Bool   // keeps a provider set earlier
```

---

## DocsTab

Drop-in SwiftUI tab over `DocumentLibrary.shared`. Includes a file picker, per-document progress,
swipe to export (leading) or delete (trailing), and a chat sheet with source citations.

On first appearance it loads `.qwen3_1_7b` and `.fastVLM_0_5b_fp16` through `ModelManager`,
downloading them if needed. It uses multilingual-e5-small when a valid bundle is installed at
`AutoEmbeddingProvider.defaultModelBundleURL` (the toolbar's "Import embedding model…" copies one
there), and TF-IDF otherwise. When the provider changed since the index was built, it re-embeds the
index before it becomes ready.

```swift
import AuraDocs

TabView {
    DocsTab()
        .tabItem { Label("Docs", systemImage: "doc.text.magnifyingglass") }
}
```

---

## Progress Stages

`add` calls `onProgress` on `@MainActor` with these messages, in this order. The % column is how
`DocsTab` fills its progress bar, not part of the callback: it maps any `N%` in a message to
15% + N × 0.85.

| Stage | Example | % in DocsTab |
|-------|---------|---|
| Already indexed in this launch (`add` returns the existing entry) | `"'MyDoc.pdf' already indexed."` | 100% |
| Parsing | `"Parsing MyDoc.pdf…"` | 5% |
| Chunking | `"Chunking MyDoc…"` | 15% |
| Re-embedding (only when the stored vectors came from another provider) | `"Re-embedding 400 chunks…"`, then `"Re-embedding 50/400 chunks: 12%"` per batch of 50 | 15–100% |
| Embedding | `"Embedding 253 chunks…"`, then `"Embedding MyDoc: 42%"` per batch of 50 | 15–100% |
| Done | `"'MyDoc' indexed ✓ (253 chunks)"`, or `"'MyDoc' indexed ✓ (253 chunks, 3 cut at the model's token limit)"` when the embedding model truncated chunks | 100% |
