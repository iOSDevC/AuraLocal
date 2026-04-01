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
  → TF-IDF sparse embedding (local, no download)
    → FTS5 keyword pre-filter (top 20 candidates)
      → Accelerate cosine re-rank (top 5)
        → LLM with retrieved context
```

Two-stage hybrid search: FTS5 for fast keyword recall, cosine similarity for semantic precision. All vectors stored as BLOBs in a single SQLite database.

---

## Quick Start

```swift
import AuraCore
import AuraDocs

// 1. Setup
let llm      = try await AuraLocal.text(.qwen3_1_7b)
let embedder = AutoEmbeddingProvider()
let library  = DocumentLibrary.shared

await library.configure(embeddingProvider: embedder, llm: llm)
try await library.open()

// 2. Index documents
try await library.add(url: pdfURL) { progress in
    print(progress)  // "Embedding Contract: 67%"
}
try await library.add(url: imageURL)  // OCR via VLM automatically

await library.refreshCorpus()  // rebuild TF-IDF after batch indexing

// 3. Ask
let answer = try await library.ask("What is the total contract value?")
print(answer.text)
for source in answer.sources {
    print("[\(source.documentTitle) p.\(source.pageNumber)] \(source.excerpt)")
}
```

---

## Supported Formats

| Format | How it's parsed |
|--------|----------------|
| PDF | PDFKit text extraction per page |
| DOCX | ZIP + XML (no external libs) |
| TXT, MD, Markdown | Plain text |
| PNG, JPG, HEIC, TIFF | MLX VLM OCR (requires a vision model) |

---

## Stateful Document Chat

```swift
import AuraDocs

let chat = DocumentChat(library: library, llm: llm)

let r1 = try await chat.send("What is the payment schedule?")
let r2 = try await chat.send("What are the late payment penalties?")
// r2 has context from r1

for msg in chat.messages {
    print("[\(msg.role)] \(msg.text)")
    msg.sources.forEach { print("  Source: \($0.documentTitle)") }
}
```

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

---

## Progress Stages

| Stage | Example message | Progress |
|-------|----------------|---------|
| Parsing | `"Parsing Contract.pdf…"` | 5% |
| Chunking | `"Chunking Contract…"` | 15% |
| Embedding | `"Embedding Contract: 67%"` | 15–100% |
| Complete | `"'Contract' indexed ✓ (87 chunks)"` | 100% |
