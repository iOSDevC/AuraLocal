---
layout: docs
title: Document RAG
parent: Guide
nav_order: 7
description: "Index PDFs, Word docs, and images locally with AuraDocs and ask questions in natural language."
---

# Document RAG
{: .no_toc }

`AuraDocs` provides a fully local Retrieval-Augmented Generation pipeline. Index documents once — PDF, DOCX, images — and ask questions in natural language. No API keys, no cloud services, no external vector databases.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Retrieval Pipeline

```
query
  → embedding: TF-IDF (default, no download) or multilingual-e5-small (opt-in, Core ML)
    → FTS5 keyword pre-filter (top 20 candidates)
      → Accelerate cosine re-rank (top 5)
        → LLM with retrieved context
```

Two-stage hybrid search: FTS5 for fast keyword recall, cosine similarity for semantic precision. All vectors stored as BLOBs in a single SQLite database. The vectors, TF-IDF or e5, only re-rank the up to 20 chunks that share at least one word with the question; every chunk is scored by cosine alone only when FTS5 finds no keyword match at all. Cross-language retrieval therefore depends on keyword overlap (see [Limits and costs](#limits-and-costs)). See [Embedding providers](#embedding-providers) for choosing the vectors.

---

## Quick Start

```swift
import AuraCore
import AuraDocs

// 1. Setup
let llm      = try await AuraLocal.text(.qwen3_1_7b)
let vlm      = try await AuraLocal.vision(.qwen35_0_8b)  // optional: pass to enable image OCR
let embedder = AutoEmbeddingProvider()
let library  = DocumentLibrary.shared

await library.configure(embeddingProvider: embedder, llm: llm, visionLLM: vlm)
try await library.open()

// 2. Index documents
try await library.add(url: pdfURL) { progress in   // @discardableResult -> IndexedDocument
    print(progress)  // "Embedding Contract: 67%"
}
try await library.add(url: imageURL)  // needs the visionLLM passed to configure(); without one it throws DocumentError.unsupportedFormat

// TF-IDF only: updates the in-memory IDF weights used for later queries and new chunks.
// Stored vectors keep theirs (reembedAll() re-applies them), and each call counts the whole
// corpus again, so repeated calls skew the weights.
await library.refreshCorpus()

// 3. Ask
let answer = try await library.ask("What is the total contract value?")
print(answer.text)
for source in answer.sources {
    print("[\(source.documentTitle) p.\(source.pageNumber)] \(source.excerpt)")
}
```

{: .warning }
> Re-adding a file skips it only within the same launch: document IDs come from the path's
> `hashValue`, which Swift seeds per process, so after a relaunch `add(url:)` indexes the same file
> again. Check `allDocuments()` by `url` before re-adding.

---

## Embedding providers

| Provider | Vectors | Needs | Good at |
|---|---|---|---|
| `AutoEmbeddingProvider()` — **default** | TF-IDF, 4096-dim hashed sparse | nothing | exact words; works everywhere with no download |
| `AutoEmbeddingProvider(embeddingModelAt:)` — **opt-in** | multilingual-e5-small, 384-dim dense | a ~225 MB model bundle | meaning across wording and languages (≈100 languages, Spanish and English included) |
| `CoreMLEmbeddingProvider(bundleAt:)` | same model, no fallback | the bundle | when you want an error instead of TF-IDF |
| your own `EmbeddingProvider` | anything | — | remote APIs, other models |

### multilingual-e5-small (opt-in)

```swift
import Foundation
import AuraCore
import AuraDocs

let bundleURL = AutoEmbeddingProvider.defaultModelBundleURL   // or wherever you installed it
let embedder = AutoEmbeddingProvider(embeddingModelAt: bundleURL)
if let problem = embedder.denseModelProblem {
    print("Using TF-IDF: \(problem)")   // bundle missing or invalid
}
try await embedder.warmUp()            // compile + load before you report "ready"
print(await embedder.backendName())     // "multilingual-e5-small (Core ML, on-device)"

let library = DocumentLibrary.shared
await library.configure(embeddingProvider: embedder, llm: llm)
try await library.open()
```

`AutoEmbeddingProvider(embeddingModelAt:)` checks the bundle on disk (manifest, model file, tokenizer
files) without loading anything, and falls back to TF-IDF when it is unusable; `denseModelProblem`
says why. `AutoEmbeddingProvider()` is unchanged and never uses a model.

e5 is asymmetric: `DocumentLibrary` embeds chunks with `embedDocuments` (prefix `passage: `) and
questions with `embedQuery` (prefix `query: `), which is what the model was trained on.

Call `warmUp()` before telling the user the library is ready. It compiles the model, loads it and
runs every sequence length once. **The first load compiles the model for the Neural Engine, which
took ~35 s on an M1 Pro**; loading it again from the compiled cache and running every length took
~0.17 s. A `warmUp()` that throws means the model cannot run here: fall back to
`AutoEmbeddingProvider()`, which is what `DocsTab` does.

### Limits and costs

- **512 tokens per chunk**, counting `<s>` and `</s>`. Longer text keeps its first 510 tokens and its
  closing `</s>`; the rest is not embedded. `DocumentLibrary`'s default chunks (~2,000 characters)
  measured ~380 tokens for Spanish prose, but 2,000 characters of Chinese came to 1,181 tokens,
  of numbers and dates 937, of Swift code 891: such chunks lose about half their text. `add`
  reports cuts: `"'Contract' indexed ✓ (87 chunks, 3 cut at the model's token limit)"`. For dense
  text, lower `DocumentLibrary(chunkTargetTokens:)` (e.g. to 200).
- **Speed** (M1 Pro, Neural Engine, release build, one text per call, tokenization included):
  ~2.4 ms at 26 tokens, ~4 ms at 114, ~8 ms at 246, ~20 ms at 512; tokenization is 20–45% of
  that. A 380-token chunk costs under 20 ms, so 1,000 chunks index in about 20 s.
- **Memory**: loading the tokenizer (a 250,000-entry vocabulary, 0.6 s) grew the process footprint
  by ~95 MB; the fp16 weights are file-backed and added only ~5 MB, not the 225 MB on disk. The
  compiled model is cached under `Caches/AuraLocal/CompiledModels` (another 225 MB); if the system
  purges it, the next load compiles again.
- It runs on `.cpuAndNeuralEngine` by default. `.all` measured slower: Core ML moves the embedding
  lookup to the GPU, where it also competes with the LLM.
- Measured on macOS only. It has not been run on an iPhone, iPad or Vision Pro yet.
- Retrieval is still hybrid: e5 re-ranks the FTS5 keyword candidates and scores every chunk only
  when no keyword matches at all. So cross-language matches are fragile: asking
  "¿Cuántas proteínas debe comer una mujer al día?" over a Spanish and an English document, a common
  word matched a Spanish chunk, and the English passage about protein was never scored, although a
  cosine-only search ranked it first (0.865).
- Tokenization runs in Swift (swift-transformers) and matched the Python tokenizer on Spanish and
  English text. It approximates the Python normalizer for some rare characters (orphan combining
  marks, fullwidth forms, ligatures), so those can tokenize slightly differently.

### The model bundle

A bundle is a folder. Its name becomes the tool's `id` (`coreml.text-embedding.<folder name>`), but the
index identity comes from the manifest (`model_id@revision`):

```
multilingual-e5-small/
  embedding-model.json            manifest
  MultilingualE5Small.mlpackage   the encoder (or a compiled .mlmodelc)
  tokenizer.json                  Hugging Face tokenizer
  tokenizer_config.json
  special_tokens_map.json
```

`embedding-model.json` (schema `aura.text-embedding/1`; unknown keys are ignored):

| Key | e5 value | Meaning |
|---|---|---|
| `schema` | `"aura.text-embedding/1"` | format version |
| `model_id`, `revision` | `"intfloat/multilingual-e5-small"`, `"614241f6…"` | together they are the provider `identifier` |
| `model_file` | `"MultilingualE5Small.mlpackage"` | file name inside the bundle |
| `input_name`, `output_name` | `"input_ids"`, `"last_hidden_state"` | Core ML feature names |
| `buckets` | `[64, 128, 256, 512]` | sequence lengths the model accepts, ascending |
| `pad_token_id` | `1` | right-padding id; padded positions are left out of the mean |
| `pooling`, `normalize` | `"mean"`, `true` | mean over non-padding tokens, then L2 |
| `dimensions` | `384` | vector length |
| `query_prefix`, `passage_prefix` | `"query: "`, `"passage: "` | prepended per role |
| `max_tokens` | `512` | longest input, `<s>`/`</s>` included |
| `license` | `"MIT"` | the model's license |

A missing or invalid manifest, model file or tokenizer file makes the provider unusable with a
reason naming it, e.g. `Invalid embedding-model.json: missing “pad_token_id”.`

**Build the bundle** with the conversion script (macOS, [uv](https://docs.astral.sh/uv/)):

```sh
uv run scripts/embeddings/convert_e5_coreml.py --out /path/to/multilingual-e5-small
```

It downloads the pinned model revision, converts it for the Neural Engine, refuses to write the
bundle unless Core ML matches sentence-transformers (cosine ≥ 0.999 on `CPU_ONLY` and
`CPU_AND_NE`), and cleans up its downloads. Details: [`scripts/embeddings/README.md`](https://github.com/iOSDevC/AuraLocal/blob/main/scripts/embeddings/README.md).

**Install it** by copying the folder to `AutoEmbeddingProvider.defaultModelBundleURL`
(`Application Support/AuraLocal/embeddings/multilingual-e5-small`), or with **Import embedding
model…** in `DocsTab`.

### Using the model without AuraDocs

`CoreMLTextEmbeddingTool` (in `AuraCore`, listed with the other [on-device ML tools]({{ '/guide/ml-tools' | relative_url }})) is the tool underneath, usable for semantic search,
deduplication or clustering:

```swift
import Foundation
import AuraCore

let e5 = CoreMLTextEmbeddingTool(bundleAt: bundleURL)   // id "coreml.text-embedding.multilingual-e5-small"
guard await e5.availability().isAvailable else { return }
try await e5.warmUp()

let question = try await e5.embed("¿Cuánta proteína necesita una mujer al día?", role: .query)
let passages = try await e5.embed(["Las mujeres adultas necesitan unos 46 g de proteína al día.",
                                    "El contrato vence el 3 de marzo."], role: .passage)
let scores = passages.map { passage in zip(question.vector, passage.vector).reduce(0) { $0 + $1.0 * $1.1 } }
print(scores, passages.map(\.isTruncated))   // vectors are unit length: dot product = cosine
```

### Switching providers re-embeds the index

The vector store records which provider produced its vectors (`identifier` + `dimensions`, in a
`metadata` table added to existing databases on open). Before `add` or `ask`, `DocumentLibrary`
compares it with the configured provider:

- same provider → nothing to do;
- different provider → every stored chunk is re-embedded from its stored text, in batches of 50;
  documents are not parsed again;
- a database from before this tracking → adopted as is only when the configured provider is TF-IDF
  (the only built-in provider then) and the vector lengths match, so TF-IDF indexes keep working with
  `AutoEmbeddingProvider()`; with any other provider it is re-embedded once, since a different model
  of the same width cannot be told apart.

An interrupted re-embed is redone on next use. If `configure` switches providers while an `add` or
`ask` is embedding, that call embeds again with the new provider instead of mixing vector spaces.
`ask` re-embeds silently, so to show progress, do it up front:

```swift
if try await library.indexNeedsReembedding() {
    try await library.reembedAll { message in
        print(message)   // "Re-embedding 50/400 chunks: 12%"
    }
}
```

Custom providers get `embedQuery` / `embedDocuments` (defaulting to `embed` / `embedBatch`) and an
`identifier` (defaulting to the module-qualified type name plus `dimensions`, stable across launches
even for a `private` type). Give yours an explicit `identifier` such as `model@revision` and change it
whenever its vectors change (another model, another version).

Two screens sharing one library should not both call `configure`: the second call switches the
vector space under the first. `configureIfNeeded(embeddingProvider:llm:visionLLM:)` keeps a provider
that is already set.

---

## Supported Formats

| Format | How it's parsed |
|--------|----------------|
| PDF | PDFKit text extraction per page |
| DOCX | ZIP + XML (no external libs) |
| TXT, MD, Markdown, RTF | Plain text (RTF is read as raw text, markup included) |
| Source and config files (Swift, Obj-C, C/C++, Rust, Go, Python, Ruby, JS/TS, shell, PHP, Lua, Java, Kotlin, Gradle, C#, JSON, YAML, TOML, XML, plist, SQL, HTML, CSS/SCSS) | Plain text; the title keeps the extension |
| PNG, JPG, JPEG, HEIC, TIFF, BMP | OCR via the `visionLLM:` passed to `configure()`; without one, `add(url:)` throws `DocumentError.unsupportedFormat` |

---

## Document Chat

```swift
import AuraDocs

let chat = DocumentChat(library: library, llm: llm)

let r1 = try await chat.send("What is the payment schedule?")   // send(_:topK:) -> DocumentAnswer
let r2 = try await chat.send("What are the late payment penalties?")
// r2 is answered from the documents alone: each send retrieves independently; earlier turns
// are saved to ConversationStore but not given to the model

for msg in chat.messages {
    print("[\(msg.role)] \(msg.text)")
    msg.sources.forEach { print("  Source: \($0.documentTitle)") }
}
```

{: .note }
> `DocumentChat` keeps the transcript (`messages`) but answers each question independently, so write
> follow-ups as complete questions ("What are the late payment penalties?", not "And the penalties?").

---

## Advanced Options

```swift
// Custom chunking
let library = DocumentLibrary(
    chunkTargetTokens: 512,       // tokens per chunk
    chunkOverlapFraction: 0.10    // 10% overlap between chunks
)

// Ask with more context
let answer = try await library.ask(
    "Summarize the indemnification clause",
    topK: 8,
    maxContextTokens: 4096,
    systemPrompt: "You are a legal assistant. Cite specific clauses."
)

// Manage documents
let docs = try await library.allDocuments()
try await library.removeDocument(id: doc.id)
```

---

## Drop-in Tab

```swift
import AuraDocs

TabView {
    DocsTab()
        .tabItem { Label("Docs", systemImage: "doc.text.magnifyingglass") }
}
```

`DocsTab` includes:
- Multi-file picker
- Per-document indexing progress with percentage
- Swipe-to-delete
- Full chat sheet with expandable source citations
- multilingual-e5-small when its bundle is installed at `Application Support/AuraLocal/embeddings/multilingual-e5-small`
  (warmed up before the tab reports ready), TF-IDF otherwise; the backend in use is shown under the list
- **Import embedding model…** (toolbar): pick a bundle folder, it is copied into place and the index
  is re-embedded with progress

---

## Progress Stages

| Stage | Example message | Progress |
|-------|----------------|---------|
| Parsing | `"Parsing Contract.pdf…"` | 5% |
| Chunking | `"Chunking Contract…"` | 15% |
| Re-embedding (only after a provider change) | `"Re-embedding 50/400 chunks: 12%"` | before parsing |
| Embedding | `"Embedding Contract: 67%"` | 15–100% |
| Complete | `"'Contract' indexed ✓ (87 chunks)"`, plus `", 3 cut at the model's token limit"` when e5 truncated chunks | 100% |
