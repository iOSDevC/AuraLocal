---
layout: docs
title: AuraUI
parent: API Reference
nav_order: 2
description: "AuraUI API reference — prebuilt SwiftUI tabs for text chat, vision, OCR, models browser, and voice."
---

# AuraUI
{: .no_toc }

Drop-in SwiftUI interface. Import `AuraUI` (and optionally `AuraVoice`) and add `ContentView` to your `WindowGroup`.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## ContentView

Top-level tabbed interface. All tabs are included automatically.

```swift
import SwiftUI
import AuraUI
import AuraVoice  // adds Voice tab

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
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

Use individual tabs and components if you don't want the full `ContentView`:

```swift
import AuraUI

// Individual tabs
TextChatTab()
VisionTab()
OCRTab()
ModelsTab()
```
