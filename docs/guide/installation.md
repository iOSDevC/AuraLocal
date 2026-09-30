---
layout: docs
title: Installation
parent: Guide
nav_order: 1
description: "Add AuraLocal to your iOS, macOS, or visionOS project via Swift Package Manager."
---

# Installation
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Requirements

| Requirement | Minimum |
|-------------|---------|
| iOS | 18.0+ |
| macOS | 15.0+ |
| visionOS | 2.0+ |
| Xcode | 16.3+ (Swift 6.1); Xcode 26 for `AuraAppleIntelligence` and `AuraAgents` |
| Swift | 6.1 toolchain (your manifest can stay at tools-version 6.0) |

{: .important }
> AuraLocal requires a **Swift 6.1** toolchain, because its pinned dependencies declare tools-version 6.1, and **C++ interoperability mode**, because the llama.cpp backend (`LocalLLMClient`) contains C++ headers. Every target that imports `AuraCore`, or a module built on it, must enable C++ interop. `AuraAppleIntelligence` and `AuraAgents` import FoundationModels, so they build only with the iOS 26 / macOS 26 SDK.

---

## Swift Package Manager

### Xcode (recommended)

1. Open your project in Xcode
2. **File → Add Package Dependencies…**
3. Enter the repository URL:
   ```
   https://github.com/iOSDevC/AuraLocal
   ```
4. Under **Dependency Rule** choose **Commit** and paste a commit SHA from `main` (or choose **Branch** `main` to follow it). Version rules do not resolve; see [Pin a commit](#pin-a-commit).
5. Add the modules you need to your target
6. In your app target's **Build Settings**, set **C++ and Objective-C Interoperability** to **C++ / Objective-C++** (`SWIFT_OBJC_INTEROP_MODE = objcxx`). Every target that imports `AuraCore`, or a module built on it, needs it.

### Package.swift

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MyApp",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    dependencies: [
        .package(url: "https://github.com/iOSDevC/AuraLocal", revision: "<commit SHA from main>"),
    ],
    targets: [
        .target(
            name: "MyApp",
            dependencies: [
                .product(name: "AuraCore", package: "AuraLocal"),
                .product(name: "AuraUI",   package: "AuraLocal"),   // optional
                .product(name: "AuraVoice", package: "AuraLocal"),  // optional
                .product(name: "AuraDocs", package: "AuraLocal"),   // optional
            ],
            swiftSettings: [
                // Required — LocalLLMClient (llama.cpp) contains C++ headers
                .interoperabilityMode(.Cxx),
            ]
        ),
    ]
)
```

{: .warning }
> The `.interoperabilityMode(.Cxx)` setting is **mandatory**. Without it the build fails with `module 'AuraCore' was built with C++ interoperability enabled, but current compilation does not enable C++ interoperability` (the message names the AuraLocal module you import).

### Pin a commit

AuraLocal depends on LocalLLMClient by revision, so SwiftPM rejects a version requirement (`from:` or `exact:`) on AuraLocal. Put a commit SHA from `main` in `revision:`. `branch: "main"` also resolves, but it moves with every push.

---

## Dependencies

AuraLocal pulls in three dependencies automatically — you do not need to add them manually:

| Package | Purpose |
|---------|---------|
| [LocalLLMClient](https://github.com/tattn/LocalLLMClient) 0.5.0, pinned by revision | Swift wrapper for llama.cpp (build b8851): GGUF inference |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) 3.31.3 or later | MLX GPU inference for `.mlx` models |
| [swift-transformers](https://github.com/huggingface/swift-transformers) 1.3.0 or later | Tokenizers for MLX models and for the Core ML text-embedding tool (multilingual-e5) |

---

## Entitlements

Add the following to your `.entitlements` file. Without it the OS will terminate your app when loading models larger than ~500 MB.

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
```

For macOS apps using GGUF models stored outside the sandbox:

```xml
<key>com.apple.security.files.user-selected.read-write</key>
<true/>
```

---

## Permissions (Voice)

If you use `AuraVoice`, add these keys to `Info.plist`:

```xml
<key>NSSpeechRecognitionUsageDescription</key>
<string>Used for voice input to the local AI assistant.</string>
<key>NSMicrophoneUsageDescription</key>
<string>Used to capture your voice for the AI assistant.</string>
```

---

## Verify the Install

```swift
import AuraCore

// Should print the display name and size of a model
print(Model.qwen3_1_7b.displayName)       // "Qwen3 1.7B"
print(Model.qwen3_1_7b.approximateSizeMB) // 1000

// Check hardware compatibility
let fit = HardwareAnalyzer.assess(.llama3_1_8b_gguf)
print(fit.fitLevel.label) // "Streaming" on 6 GB iPhone, "Good" on Mac
```

---

## Modules Reference

Import only what you need:

```swift
import AuraCore   // needed by every module except AuraAppleIntelligence

import AuraUI     // prebuilt SwiftUI tabs
import AuraVoice  // voice pipeline
import AuraDocs   // RAG document library

import AuraAgents             // AgentCrew multi-agent pipeline (iOS 26 / macOS 26)
import AuraAppleIntelligence  // FoundationModels agents; no AuraCore dependency (iOS 26 / macOS 26)
import AuraImageGen           // FLUX text-to-image via mflux (works on macOS only)
```

The `aura` command-line tool is a separate executable product. See [CLI]({{ '/guide/cli' | relative_url }}).
