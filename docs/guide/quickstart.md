---
layout: docs
title: Quick Start
parent: Guide
nav_order: 0
description: "Get Tonamox running in your iOS or macOS app in under 5 minutes."
---

# Quick Start
{: .no_toc }

Get on-device LLM inference running in your app in under 5 minutes.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Add the Package

You need Xcode 16.3 or later (Swift 6.1).

In Xcode: **File → Add Package Dependencies** → enter `https://github.com/iOSDevC/Tonamox`, set **Dependency Rule** to **Commit** and paste a commit SHA from `main`. Then, in your app target's **Build Settings**, set **C++ and Objective-C Interoperability** to **C++ / Objective-C++** (`SWIFT_OBJC_INTEROP_MODE = objcxx`). Every target that imports `TonamoxCore`, or a module built on it, needs it.

In `Package.swift`:
```swift
// swift-tools-version: 6.0
platforms: [.iOS(.v18), .macOS(.v15)],   // Tonamox's minimums (visionOS: .v2)
dependencies: [
    .package(url: "https://github.com/iOSDevC/Tonamox", revision: "<commit SHA from development>"),
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [.product(name: "TonamoxCore", package: "Tonamox")],
        swiftSettings: [.interoperabilityMode(.Cxx)]  // required
    )
]
```

Tonamox depends on LocalLLMClient by revision, so SwiftPM rejects a version requirement (`from:` or `exact:`) on Tonamox. Pin a commit SHA from `main` instead. `branch: "main"` also resolves, but it moves with every push.

---

## 2. Add the Entitlements

In Xcode, open your target's **Signing & Capabilities** tab and add **Increased Memory Limit** and **Extended Virtual Addressing**. Or add to your `.entitlements`:

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
```

---

## 3. Your First Chat

```swift
import TonamoxCore

// Simplest possible usage — one line
let reply = try await Tonamox.chat("Hello, what can you do?")
print(reply)
```

The first call downloads `Qwen3 1.7B` (~1 GB) automatically and caches it for subsequent launches.

{: .note }
> The default model is MLX, and MLX needs a Metal GPU. In the iOS Simulator the load fails with *MLX models can't run on the iOS Simulator*. Run on a device or a Mac, or in the Simulator load a GGUF model with `ModelManager.shared.load(_:)`, which downloads it first and runs it on the CPU there.

---

## 4. Reusable Instance (Recommended)

Load the model once and reuse it for multiple calls:

```swift
import Combine
import TonamoxCore

@MainActor
class MyViewModel: ObservableObject {
    @Published var response = ""
    private var llm: Tonamox?

    func setup() async throws {
        llm = try await Tonamox.text(.qwen3_1_7b) { progress in
            print(progress)  // "Downloading Qwen3 1.7B: 42%"
        }
    }

    func ask(_ question: String) async throws {
        guard let llm else { return }
        for try await token in llm.stream(question) {
            response += token
        }
    }
}
```

---

## 5. Drop-in SwiftUI Interface

Add `TonamoxUI` for ready-made chat, vision, and OCR views. Compose the public tab views in your own `TabView`:

```swift
import SwiftUI
import TonamoxUI

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                TextChatTab()
                    .tabItem { Label("Chat", systemImage: "text.bubble") }
                VisionTab()
                    .tabItem { Label("Vision", systemImage: "eye") }
                OCRTab()
                    .tabItem { Label("OCR", systemImage: "doc.text.viewfinder") }
            }
        }
    }
}
```

The tabs download nothing on their own: while the selected model is not on this device they show a placeholder with
its size, the free space and a **Download** button, and continue once it is downloaded. Pass `autoDownload: true` to
download on the first send or run instead.

`TextChatTab(offersHuggingFaceSearch: true)` starts the model menu with **Find on Hugging Face…**, which searches,
checks and downloads models for this device. To list those downloads again after a relaunch, call
`DownloadRecords.standard.registerDownloaded()` (`import TonamoxCore`) in your `App`'s `init`, before the first view
reads the model lists. See [The example apps]({{ '/guide/models' | relative_url }}#the-example-apps).

---

## Next Steps

- [Installation details →]({{ '/guide/installation' | relative_url }})
- [Choose the right model →]({{ '/guide/models' | relative_url }})
- [Understand backends →]({{ '/guide/backends' | relative_url }})
- [API reference →]({{ '/api/core' | relative_url }})
