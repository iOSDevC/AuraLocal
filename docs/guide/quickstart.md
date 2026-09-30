---
layout: docs
title: Quick Start
parent: Guide
nav_order: 0
description: "Get AuraLocal running in your iOS or macOS app in under 5 minutes."
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

In Xcode: **File → Add Package Dependencies** → enter `https://github.com/iOSDevC/AuraLocal`, set **Dependency Rule** to **Commit** and paste a commit SHA from `main`. Then, in your app target's **Build Settings**, set **C++ and Objective-C Interoperability** to **C++ / Objective-C++** (`SWIFT_OBJC_INTEROP_MODE = objcxx`). Every target that imports `AuraCore`, or a module built on it, needs it.

In `Package.swift`:
```swift
// swift-tools-version: 6.0
dependencies: [
    .package(url: "https://github.com/iOSDevC/AuraLocal", revision: "<commit SHA from main>"),
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [.product(name: "AuraCore", package: "AuraLocal")],
        swiftSettings: [.interoperabilityMode(.Cxx)]  // required
    )
]
```

AuraLocal depends on LocalLLMClient by revision, so SwiftPM rejects a version requirement (`from:` or `exact:`) on AuraLocal. Pin a commit SHA from `main` instead. `branch: "main"` also resolves, but it moves with every push.

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
import AuraCore

// Simplest possible usage — one line
let reply = try await AuraLocal.chat("Hello, what can you do?")
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
import AuraCore

@MainActor
class MyViewModel: ObservableObject {
    @Published var response = ""
    private var llm: AuraLocal?

    func setup() async throws {
        llm = try await AuraLocal.text(.qwen3_1_7b) { progress in
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

Add `AuraUI` for ready-made chat, vision, and OCR views. Compose the public tab views in your own `TabView`:

```swift
import SwiftUI
import AuraUI

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

---

## Next Steps

- [Installation details →]({{ '/guide/installation' | relative_url }})
- [Choose the right model →]({{ '/guide/models' | relative_url }})
- [Understand backends →]({{ '/guide/backends' | relative_url }})
- [API reference →]({{ '/api/core' | relative_url }})
