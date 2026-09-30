---
layout: docs
title: AuraUI
parent: API Reference
nav_order: 2
description: "AuraUI API reference — prebuilt SwiftUI tabs for text chat, vision and OCR, plus a model list section."
---

# AuraUI
{: .no_toc }

Prebuilt SwiftUI views. Import `AuraUI` and compose the public tab views — `TextChatTab`, `VisionTab`, `OCRTab` — into your own `TabView`. The library ships no top-level `ContentView`; you own the container. (`ContentView` exists only in the bundled `AuraExample` demo app, not in the `AuraUI` library.)

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Composing the tabs

`AuraUI` ships individual tab views. Compose them into your own `TabView` — you own the top-level container. (Add a Voice tab yourself by importing `AuraVoice` and placing its `VoiceChatView`.)

```swift
import SwiftUI
import AuraUI

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                TextChatTab()
                    .tabItem { Label("Text", systemImage: "text.bubble") }
                VisionTab()
                    .tabItem { Label("Vision", systemImage: "eye") }
                OCRTab()
                    .tabItem { Label("OCR", systemImage: "doc.viewfinder") }
            }
        }
    }
}
```

---

## Tabs

| Tab | Module | Description |
|-----|--------|-------------|
| **Text** | `AuraUI` | Multi-conversation chat with streaming. MLX + GGUF model picker. |
| **Vision** | `AuraUI` | Image analysis — Standard and Stream modes. |
| **OCR** | `AuraUI` | Receipt and document extraction with FastVLM or Granite Docling. |
| **Models** | `AuraUI` (`ModelSection`) | No tab type. `ModelSection` is a `Section` of model rows with download status, backend, fit and speed badges; place it in your own `List`. |
| **Voice** | `AuraVoice` | No tab type. Place `VoiceChatView(llm:)` in your own tab for turn-based voice chat. See [AuraVoice]({{ '/api/voice' | relative_url }}). |
| **Docs** | `AuraDocs` | `DocsTab`: document library and RAG chat. Requires `AuraDocs` import. See [AuraDocs]({{ '/api/rag' | relative_url }}). |

---

## Model Badges

`ModelSection` rows show per-model badges. (The Text tab's model picker shows none: it groups models into MLX and GGUF sections and marks the downloaded ones.)

| Badge | Meaning |
|-------|---------|
| <span class="badge badge-mlx">MLX</span> | GPU inference via mlx-swift |
| <span class="badge badge-gguf">GGUF</span> | Full load via llama.cpp |
| <span class="badge badge-stream">STREAM</span> | Layer-streaming via llama.cpp (low-RAM mode) |

Fit level badges show device compatibility (label and SF Symbol from `ModelFitLevel`):

| Label | Symbol | Color | Fit level |
|-------|--------|-------|-----------|
| Excellent | `checkmark.seal.fill` | green | >40% RAM headroom |
| Good | `checkmark.circle.fill` | blue | 20–40% headroom |
| Marginal | `exclamationmark.triangle.fill` | orange | <20% headroom |
| Streaming | `arrow.down.circle.fill` | purple | Layer-streaming required |
| Too Large | `xmark.octagon.fill` | red | Not runnable on this device; the row is dimmed |

A speed badge (`~N tok/s`) follows when the chip's memory bandwidth is known; it is omitted otherwise.

---

## Individual Components

Use individual tabs and components directly:

```swift
import AuraUI
import AuraCore

// Prebuilt tab views
TextChatTab()
VisionTab()
OCRTab()

// Models browser component: build a Section from a [Model] array.
// (There is no `ModelsTab` in AuraUI — compose `ModelSection` inside your own List/Form.)
ModelSection(
    title: "Text",
    icon: "text.bubble",
    color: .green,
    models: Model.textModels
) { model in
    // handle the model's "Test" tap
}
```
