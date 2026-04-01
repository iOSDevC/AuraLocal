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

In Xcode: **File → Add Package Dependencies** → enter `https://github.com/iOSDevC/AuraLocal`

In `Package.swift`:
```swift
// swift-tools-version: 6.0
dependencies: [
    .package(url: "https://github.com/iOSDevC/AuraLocal", branch: "main"),
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [.product(name: "AuraCore", package: "AuraLocal")],
        swiftSettings: [.interoperabilityMode(.Cxx)]  // required
    )
]
```

---

## 2. Add the Entitlement

In Xcode, open your target's **Signing & Capabilities** tab, add **Increased Memory Limit**. Or add to your `.entitlements`:

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
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

---

## 4. Reusable Instance (Recommended)

Load the model once and reuse it for multiple calls:

```swift
import AuraCore

@MainActor
class MyViewModel: ObservableObject {
    @Published var response = ""
    private var llm: AuraLocal?

    func setup() async throws {
        llm = try await AuraLocal.text(.qwen3_1_7b) { progress in
            print(progress)  // "Fetching model: 42%"
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

Add `AuraUI` for a complete tabbed interface with zero configuration:

```swift
import SwiftUI
import AuraUI

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()  // includes Text, Vision, OCR, Models tabs
        }
    }
}
```

---

## Next Steps

- [Installation details →]({% link guide/installation.md %})
- [Choose the right model →]({% link guide/models.md %})
- [Understand backends →]({% link guide/backends.md %})
- [API reference →]({% link api/core.md %})
