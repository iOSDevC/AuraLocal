---
layout: docs
title: AuraUI
parent: API Reference
nav_order: 2
description: "AuraUI API reference — prebuilt SwiftUI tabs for text chat, vision and OCR, a model list section and a Hugging Face model browser."
---

# AuraUI
{: .no_toc }

Prebuilt SwiftUI views. Import `AuraUI` and compose the public tab views — `TextChatTab`, `VisionTab`, `OCRTab` — into your own `TabView`. The library ships no top-level `ContentView`; you own the container. (`ContentView` exists only in the bundled `AuraExample` demo app, not in the `AuraUI` library. That app no longer uses these three tabs: its Chat tab is one multimodal chat built on `Model.capabilities`; see [The example apps]({{ '/guide/models' | relative_url }}#the-example-apps).)

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
| **Text** | `AuraUI` | Multi-conversation chat with streaming. Model menu with a **Downloaded** section first, then MLX and GGUF. `TextChatTab(offersHuggingFaceSearch: true)` adds **Find on Hugging Face…** at the top of the menu. |
| **Vision** | `AuraUI` | Image analysis — Standard and Stream modes. Model picker with **Downloaded** and **Available** sections. |
| **OCR** | `AuraUI` | Receipt and document extraction with FastVLM or Granite Docling. |
| **Models** | `AuraUI` (`ModelSection`) | No tab type. `ModelSection` is a `Section` of model rows with download (through `ModelDownloads`: the paths the backends load, the Hugging Face token, a free-space check, cancel and resume), delete, and backend, fit and speed badges; place it in your own `List`. A row does not start a file that the [Hugging Face browser](#hugging-face-browser) is downloading, and its delete also unregisters a model added from search. |
| **Voice** | `AuraVoice` | No tab type. Place `VoiceChatView(llm:)` in your own tab for turn-based voice chat. See [AuraVoice]({{ '/api/voice' | relative_url }}). |
| **Docs** | `AuraDocs` | `DocsTab`: document library and RAG chat. Requires `AuraDocs` import. See [AuraDocs]({{ '/api/rag' | relative_url }}). |

---

## Model Badges

`ModelSection` rows show per-model badges. (The Text tab's model picker shows none: it lists downloaded models first, then MLX and GGUF sections, and marks the downloaded ones.)

The Text and Vision pickers read `ModelDownloadCenter.shared.revision`, so they redraw when a download finishes or a model is deleted, and a selection that pointed at a deleted model added from search resets. A deleted catalog model stays selected and, like any catalog model that isn't downloaded, downloads again on the next send. In the Text menu's **Downloaded** section a model that can't load now is disabled with the reason: *too large now* (`HardwareAnalyzer` rates it *Too Large*, or *Streaming* for MLX) or *device only* (MLX on the Simulator). The Vision picker leaves out downloaded models rated *Too Large*. Selecting a model never loads it; loading happens on send or run, behind `ModelManager`'s memory guard.

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

A speed badge follows when the chip's memory bandwidth is known: `~N tok/s`, or `~N–M tok/s` for a mixture of
experts (an uncalibrated range). It is omitted otherwise, including on iPhone, where AuraLocal knows no bandwidth.

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

---

## Hugging Face browser

Search Hugging Face, see which repos run on a device and why, download one and try it. Six public types; none
brings its own navigation container, so they fit a `NavigationStack`, a `NavigationSplitView` or a sheet.

| Type | What it is |
|------|------------|
| `HFModelSearch` | `@MainActor @Observable` state: `init(query:devices:compatibleOnly:pipelineFilter:)` and `search()`. Fetched repo snapshots are cached; changing the device re-evaluates them offline. `pipelineFilter` (an `HFPipelineFilter` from AuraCore, such as `.chat`; `nil` by default) hides results whose declared pipeline it does not admit, and the list's controls name the pipelines it keeps. |
| `HFModelSearchList` | `init(search:selection:)`. Search field, format filter (starts on GGUF on the Simulator), sort, device picker (only with more than one device), **Compatible only**, and a verdict badge per row. Selecting a row sets `selection` to its repo id. |
| `HFRepoDetailView` | `init(repoID:search:)`. Fetches the repo if needed and shows the findings, per-quant fits, license and gating, plus a **Download** section for repos rated runnable: quant picker, size and free space, progress, Cancel/Resume, **Try it** (load, one 32-token reply, unload). A key button and the gated error open the token form. |
| `DownloadedModelsList` | `init()`. Everything on disk under the model storage root and the Caches folder earlier versions used, finished or paused, with Try it, Resume, Cancel and Delete (unload every model reading the file, files, record, registry), plus the keep-awake switch. Try it and Resume are off for a file whose size is unknown (no record or catalog entry). A paused `.partial` whose finished file is already on disk (left in Caches by an older build) shows as **Leftover**: no Resume, and Delete removes only that partial. Sets the title "Downloads". |
| `ModelDownloadCenter` | `ModelDownloadCenter.shared`: the single running download. `revision` changes whenever models are downloaded, deleted or registered; `keepsScreenAwake`; `isBusy`. `isDownloading(_:)` says whether this center or a `ModelSection` row is writing a model's file; `delete(_:)` unloads every model reading that file, removes its finished and partial files, forgets its record and unregisters a model added from search (catalog models stay registered), and throws while a download writes the file. It records each download started from a search result with `DownloadRecords` and registers it when it finishes; resuming a catalog file does not record it (see [Keeping downloads across launches]({{ '/guide/models' | relative_url }}#keeping-downloads-across-launches)). |
| `HuggingFaceTokenSection` | `init()`. Two `Form` sections that save, show and remove the read token for gated repos (`KeychainDownloadAuth`). |

```swift
import SwiftUI
import AuraCore
import AuraUI

// In a stack (iPhone): judge repos against this device only and hide the ones that won't run.
struct FindModels: View {
    @State private var search = HFModelSearch(query: "qwen", devices: [DevicePreset.thisDevice()],
                                              compatibleOnly: true)
    @State private var selection: String?

    var body: some View {
        NavigationStack {
            HFModelSearchList(search: search, selection: $selection)
                .navigationTitle("Hugging Face")
                .navigationDestination(item: $selection) { HFRepoDetailView(repoID: $0, search: search) }
        }
        .task { search.search() }
    }
}

// In a split view, with every device preset to compare against:
struct FindModelsSplit: View {
    @State private var search = HFModelSearch()
    @State private var selection: String?

    var body: some View {
        NavigationSplitView {
            HFModelSearchList(search: search, selection: $selection)
        } detail: {
            if let selection { HFRepoDetailView(repoID: selection, search: search) }
        }
        .task { search.search() }
    }
}

// Elsewhere:
NavigationStack { DownloadedModelsList() }
Form { HuggingFaceTokenSection() }

// A chat's model finder: text-generation, image-text-to-text and any-to-any repos, plus undeclared ones.
let chatSearch = HFModelSearch(query: "qwen", devices: [DevicePreset.thisDevice()],
                               compatibleOnly: true, pipelineFilter: .chat)
```

Call `DownloadRecords.standard.registerDownloaded()` once at launch (for example in your `App`'s `init`) so models
downloaded in an earlier run are in the pickers before the first view draws.

Limits: one download at a time; no background downloads (about 30 s of background time, then Resume continues from
the partial file); `ModelDownloadCenter` uses `UIApplication.shared` on iOS, so it is not
app-extension-safe; on the Simulator MLX can't run and "this device" is the Mac's memory.
