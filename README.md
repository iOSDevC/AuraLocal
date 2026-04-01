<p align="center">
  <img src="https://github.com/iOSDevC/AuraLocal/blob/main/docs/media/logo.png" alt="AuraLocal" width="400">
</p>

# AuraLocal

Lightweight on-device LLM & VLM Swift package for iOS/macOS/visionOS. Run Qwen3, Llama 3, Mistral, Gemma, SmolVLM and more — locally, privately, no API keys. Supports both MLX (small models, Apple Silicon GPU) and llama.cpp/GGUF (large models 7B–70B, including layer-streaming for memory-constrained devices).

---

## v3.0 Highlights

- **Dual-backend inference** — MLX for 0.5B–4B models on GPU; llama.cpp/GGUF for 7B–70B with full Metal acceleration on macOS or layer-streaming on iOS.
- **Layer-streaming mode** — Run 7B–13B models on 6 GB iPhones at ~2–5 tok/s with ≤750 MB peak RAM. OS pages weights from disk on-demand; jetsam limit never approached.
- **`InferenceBackend` protocol** — Clean abstraction over all backends; `BackendRouter` selects the optimal engine automatically based on model format and device RAM.
- **GGUF model catalog** — 7 new large models (Llama 3.1, Qwen 2.5, Mistral, Phi-3, Gemma 2) with automatic download from HuggingFace, resume support, and `@Published` progress.
- **GQA-aware memory estimates** — KV cache calculations account for Grouped-Query Attention (Llama 3.1 8B: 256 MB at FP16/2048 ctx vs ~1 GB with naive full-attention assumption).
- **Unified model management** — `ModelManager.shared.load()` provides LRU caching, in-flight deduplication, automatic memory-pressure eviction, and backend-aware GGUF downloading.
- **Swift 6 concurrency** — All public APIs are `@MainActor`-isolated or `Sendable`, with `actor`-based stores for data-race safety.
- **OOM prevention** — `os_proc_available_memory()` monitoring with `DispatchSource` pressure listeners; `MemoryBudgetManager` performs adaptive context sizing and per-generation jetsam checks.
- **Hybrid RAG pipeline** — FTS5 keyword pre-filter + Accelerate cosine re-ranking, stored in SQLite. Zero external dependencies.

---

## Requirements

- **iOS 18+** / **macOS 15+** / **visionOS 2+**
- **Xcode 16+**
- **Swift 6.0** (C++ interoperability mode enabled — required by llama.cpp)
- `Increased Memory Limit` entitlement (required for all models > 500 MB)

> **Note:** The C++ interoperability requirement means all targets that import `AuraCore` must enable `.interoperabilityMode(.Cxx)` in their `Package.swift` `swiftSettings`.

---

## Installation

Add via Swift Package Manager:

```
https://github.com/iOSDevC/AuraLocal
```

Or in `Package.swift`:

```swift
// swift-tools-version: 6.0
.package(url: "https://github.com/iOSDevC/AuraLocal", branch: "main")
```

If your package imports `AuraCore`, add the C++ interop setting:

```swift
.target(
    name: "MyTarget",
    dependencies: ["AuraCore"],
    swiftSettings: [.interoperabilityMode(.Cxx)]
)
```

### Dependencies

| Dependency | Purpose |
|-----------|---------|
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | MLX GPU inference for small models (≤4B) |
| [LocalLLMClient](https://github.com/tattn/LocalLLMClient) | Swift wrapper for llama.cpp — GGUF inference + Metal kernels |

### Modules

| Module | Contents |
|--------|----------|
| `AuraCore` | Core inference, dual-backend engine, models, conversation persistence |
| `AuraUI` | SwiftUI views and ViewModels for drop-in UI |
| `AuraVoice` | Full-duplex voice interface (STT + TTS), 100% local |
| `AuraDocs` | RAG document library — PDF, DOCX, text, images |

---

## Backend Architecture

`BackendRouter` automatically selects the best engine based on model format and device hardware:

```
model.format == .mlx  →  MLXBackend      (GPU, Apple Silicon, models ≤4B)
model.format == .gguf + fits in RAM →  LlamaCppBackend   (full GPU offload, macOS 8B–70B)
model.format == .gguf + streamingRequired →  LayerStreamingBackend  (mmap, iOS 6–8 GB, 7B–13B)
```

| Backend | Platform | Model Size | Peak RAM | Tokens/sec |
|---------|----------|-----------|---------|------------|
| MLX | iOS + macOS | ≤4B | 500 MB – 3 GB | 20–45 |
| llama.cpp (standard) | macOS | 7B–70B | 4–40 GB | 8–20 |
| Layer-streaming | iOS | 7B–13B | ≤750 MB | 2–6 |

You can inspect which backend a model will use:

```swift
import AuraCore

let backend = BackendRouter.recommendedBackend(for: .llama3_1_8b_gguf)
// → .llamaCpp (on a Mac with 16 GB)
// → .layerStreaming (on a 6 GB iPhone)
```

---

## Text Chat

```swift
import AuraCore

// MLX small model — one-liner
let reply = try await AuraLocal.chat("¿Cuánto gasté esta semana?")

// Reusable instance (loads model once — preferred for multiple calls)
let llm = try await AuraLocal.text(.qwen3_1_7b) { progress in
    print(progress) // "Downloading Qwen3 1.7B: 42%"
}
let reply = try await llm.chat("Summarize my expenses")

// Streaming
for try await token in llm.stream("Explain this transaction") {
    print(token, terminator: "")
}

// With system prompt
let reply = try await llm.chat(
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

// Observe download progress in SwiftUI
@ObservedObject var manager = ModelManager.shared

switch manager.state(for: .llama3_1_8b_gguf) {
case .idle:                         // not started
case .downloading(let progress):    // "Downloading Llama 3.1 8B: 62%"
case .loading:                      // weights loaded, initializing session
case .ready:                        // ready for inference
case .failed(let error):            // download or load error
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

| Model | Disk Size | iOS (streaming) | macOS | Notes |
|-------|-----------|-----------------|-------|-------|
| `.llama3_1_8b_gguf` ⭐ | ~4.7 GB | 6+ GB RAM | 8+ GB RAM | Primary large model |
| `.qwen2_5_7b_gguf` | ~4.4 GB | 6+ GB RAM | 8+ GB RAM | Qwen 2.5 series |
| `.mistral_7b_gguf` | ~4.1 GB | 6+ GB RAM | 8+ GB RAM | Fast, efficient |
| `.phi3_medium_gguf` | ~8.0 GB | 8+ GB RAM | 16+ GB RAM | High quality |
| `.gemma2_9b_gguf` | ~5.4 GB | 8+ GB RAM | 12+ GB RAM | Google Gemma 2 |
| `.qwen2_5_32b_gguf` | ~20.0 GB | Not viable | 48+ GB RAM | Mac Studio/Pro |
| `.llama3_1_70b_gguf` | ~40.0 GB | Not viable | 80+ GB RAM | Mac Pro |

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
case .excellent:         // runs comfortably
case .good:              // runs well
case .marginal:          // runs but may be slow
case .streamingRequired: // too large for monolithic load; uses layer-streaming
case .tooLarge:          // cannot run on this device even with streaming
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
| KV cache (Q4, 1024 tokens, GQA) | ~32 MB |
| Activations + overhead | ~80 MB |
| Safety margin | ~250 MB |
| **Total app usage** | **≤750 MB** |

---

## Background Lifecycle (iOS)

`BackgroundLifecycle` automatically pauses inference when the app enters background on iOS, preventing jetsam termination due to active Metal GPU buffers:

```swift
import AuraCore

// Check if inference is paused (app in background)
if BackgroundLifecycle.shared.isPaused {
    // wait before starting a new generation
}

// Enable aggressive memory saving (evicts models when backgrounded)
BackgroundLifecycle.shared.aggressiveMemorySaving = true
```

This is a no-op on macOS where apps are not suspended.

---

## Receipt Scanner Example

```swift
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
let (reply, convID) = try await AuraLocal.chat("Hello", model: .qwen3_1_7b)

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
// Called automatically during chat if conversation exceeds 4096 tokens
try await llm.summarizeAndPrune(
    conversationID: conv.id,
    keepLastN: 10,         // always keep the 10 most recent turns
    maxContextTokens: 4096
)
```

---

## Voice Interface

`AuraVoice` provides a full-duplex voice pipeline using only Apple frameworks — no external dependencies, no network calls.

```
Microphone → SFSpeechRecognizer (on-device) → AuraLocal.stream() → AVSpeechSynthesizer
```

Sentences are streamed to TTS **while the LLM is still generating** — the assistant starts speaking after the first complete sentence, not after the full response.

Language is detected automatically per utterance using `NLLanguageRecognizer` and mapped to the best available system voice with region (e.g. `"es"` → `"es-MX"`).

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

// Cancel everything
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
| `.png`, `.jpg`, `.jpeg`, `.heic`, `.tiff` | MLX VLM OCR |

### Retrieval pipeline

```
query → TF-IDF embed → FTS5 top-20 candidates → cosine re-rank top-5 → LLM
```

Two-stage hybrid search: FTS5 for fast keyword recall, cosine similarity for semantic precision. All vectors stored as BLOBs in SQLite — no external vector database required.

### Quick start

```swift
import AuraCore
import AuraDocs

// 1. Configure once (e.g. in app startup)
let llm      = try await AuraLocal.text(.qwen3_1_7b)
let embedder = AutoEmbeddingProvider()

let library = DocumentLibrary.shared
await library.configure(embeddingProvider: embedder, llm: llm)
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

### Stateful document chat

```swift
import AuraDocs

// DocumentChat maintains conversation history and cites sources per message
let chat = DocumentChat(library: library, llm: llm)

let reply1 = try await chat.send("What is the payment schedule?")
let reply2 = try await chat.send("And the penalties for late payment?") // context-aware

for msg in chat.messages {
    print(msg.role, msg.text)
    print(msg.sources.map { $0.documentTitle }) // cited documents
}
```

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

`AuraUI` provides a ready-to-use tabbed interface. Add `AuraVoice` to unlock the Voice tab.

```swift
import SwiftUI
import AuraUI
import AuraVoice  // enables Voice tab

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

| Tab | Module | Description |
|-----|--------|-------------|
| **Text** | `AuraUI` | Multi-conversation chat; MLX + GGUF model picker with backend badges |
| **Vision** | `AuraUI` | Image analysis with standard and streaming modes |
| **OCR** | `AuraUI` | Document and receipt extraction |
| **Models** | `AuraUI` | Browser: all models, download status, backend badge, fit level badge |
| **Voice** | `AuraVoice` | Full-duplex voice chat with auto language detection |
| **Docs** | `AuraDocs` | Document library and RAG chat |

The **Models** tab shows three badge types per model:
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

// Observe per-model state in SwiftUI
@ObservedObject var manager = ModelManager.shared

switch manager.state(for: .llama3_1_8b_gguf) {
case .idle:                      // not loaded
case .downloading(let progress): // "Downloading Llama 3.1 8B GGUF: 42%"
case .loading:                   // file downloaded, loading into memory/session
case .ready:                     // ready for inference
case .failed(let error):         // download or load error
}

// Check which backend a model will use
let backend = manager.recommendedBackend(for: .llama3_1_8b_gguf)
// → .llamaCpp or .layerStreaming depending on device

// Manual eviction
ModelManager.shared.evict(.llama3_1_8b_gguf)
ModelManager.shared.evictAll()
```

### Memory Budget

The LRU cache size adapts to the device:

| Device RAM | Budget | Behavior |
|-----------|--------|----------|
| < 4 GB | 1 model | Evicts on every model switch |
| 4–6 GB | 1–2 models | iPhone 15, base iPad |
| 8+ GB | 2–4 models | iPad Pro, Mac |

When the OS sends a memory warning (`DispatchSource.makeMemoryPressureSource` + `UIApplication.didReceiveMemoryWarningNotification`), all models except the most recently used are evicted immediately.

---

## Entitlements

Add to your `.entitlements` file for models larger than 500 MB:

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
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
│   ├── MLXBackend          →  MLXLLM / MLXVLM (GPU, Apple Silicon, ≤4B models)
│   ├── LlamaCppBackend     →  LocalLLMClient → llama.cpp Metal (GGUF, 7B–70B, macOS)
│   └── LayerStreamingBackend → LocalLLMClient → mmap streaming (GGUF, 7B–13B, iOS)
│
├── BackendRouter           →  selects backend by model.format + HardwareAnalyzer
├── AuraEngine              →  thin delegator to InferenceBackend
├── AuraLocal               →  public facade (.text / .vision / .specialized / .chat)
│
├── HardwareAnalyzer        →  fit levels (excellent / good / marginal / streamingRequired / tooLarge)
│                              GQA-aware KV cache estimates, streaming memory budget
├── MemoryBudgetManager     →  jetsam monitoring, adaptive context, per-generation pressure checks
├── BackgroundLifecycle     →  iOS app lifecycle, pauses inference in background
│
├── ModelManager            →  LRU cache, memory-pressure eviction, GGUF download orchestration
├── GGUFModelDownloader     →  HuggingFace downloads with resume + @Published progress
│
├── ConversationStore       →  SQLite-backed chat history (actor)
└── AuraLocal+History       →  context window · auto-title · summarize+prune

AuraUI (optional)
├── ContentView  (TabView)
├── TextChatTab   →  TextChatViewModel  →  ConversationStore
│                    Model picker: MLX section + GGUF section
├── VisionTab     →  VisionViewModel
├── OCRTab        →  OCRViewModel
└── ModelsTab     →  ModelSection · ModelRow · BackendBadge · FitBadge

AuraVoice (optional)
├── VoiceSession          →  SFSpeechRecognizer (on-device STT)
│                         →  AuraLocal.stream() + ConversationStore
│                         →  AVSpeechSynthesizer (on-device TTS)
├── VoiceButton           →  SwiftUI mic button with state animations
├── VoiceChatView         →  Full voice chat UI
└── VoiceTab              →  Tab for AuraUI ContentView

AuraDocs (optional)
├── DocumentLibrary          →  add() · ask() · allDocuments() · refreshCorpus()
├── DocumentParserDispatcher →  PDF (PDFKit) · DOCX (ZIP+XML) · TXT · Image (VLM OCR)
├── DocumentChunker          →  sliding window · sentence boundaries · overlap
├── AutoEmbeddingProvider    →  TF-IDF sparse (local, no download)
├── VectorStore              →  SQLite BLOB vectors · FTS5 pre-filter · cosine re-rank
├── DocumentChat             →  stateful Q&A · source citations · ConversationStore
└── DocsTab                  →  SwiftUI tab · file picker · progress bar · chat sheet

Sources/
├── AuraCore/
│   └── LlamaCpp/           (LlamaCppBackend, LayerStreamingBackend, MemoryBudgetManager,
│                            GGUFModelDownloader, BackgroundLifecycle)
├── AuraUI/
├── AuraVoice/
├── AuraDocs/
└── AuraExample/

MLX models download automatically and are cached at:
  ~/Library/Caches/models/<org>/<repo>/

GGUF models are downloaded to:
  ~/Library/Caches/gguf/<model-name>.gguf
```

---

## License

MIT
