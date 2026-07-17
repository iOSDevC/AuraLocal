---
layout: docs
title: AuraUI
parent: API Reference
nav_order: 2
description: "AuraUI API reference — prebuilt SwiftUI tabs for text chat, vision, OCR, models browser, and voice."
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
| **Models** | `AuraUI` | Model browser with download status, backend badges, and fit level badges. |
| **Voice** | `AuraVoice` | Full-duplex voice chat. Requires `AuraVoice` import. |
| **Docs** | `AuraDocs` | Document library and RAG chat. Requires `AuraDocs` import. |

---

## Model Badges

The Models tab and Text tab picker show per-model badges:

| Badge | Meaning |
|-------|---------|
| <span class="badge badge-mlx">MLX</span> | GPU inference via mlx-swift |
| <span class="badge badge-gguf">GGUF</span> | Full load via llama.cpp |
| <span class="badge badge-stream">STREAM</span> | Layer-streaming via llama.cpp (low-RAM mode) |

Fit level badges show device compatibility:

| Badge | Fit Level |
|-------|-----------|
| ✅ Excellent | >40% RAM headroom |
| ✅ Good | 20–40% headroom |
| ⚠️ Marginal | <20% headroom |
| 🟠 Streaming | Layer-streaming required |
| ❌ Too Large | Not runnable on this device |

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
