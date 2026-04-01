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
| Xcode | 16.0+ |
| Swift | 6.0 |

{: .important }
> AuraLocal requires **Swift 6.0** and **C++ interoperability mode** because the llama.cpp backend (`LocalLLMClient`) contains C++ headers. All targets that import `AuraCore` must enable this.

---

## Swift Package Manager

### Xcode (recommended)

1. Open your project in Xcode
2. **File → Add Package Dependencies…**
3. Enter the repository URL:
   ```
   https://github.com/iOSDevC/AuraLocal
   ```
4. Select **branch: main** (or a specific version tag)
5. Add the modules you need to your target

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
        .package(url: "https://github.com/iOSDevC/AuraLocal", branch: "main"),
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
> The `.interoperabilityMode(.Cxx)` setting is **mandatory**. Without it you will get the compile error: `LocalLLMClientC needs to be compiled in C++ interoperability mode`.

---

## Dependencies

AuraLocal pulls in two dependencies automatically — you do not need to add them manually:

| Package | Purpose |
|---------|---------|
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | MLX GPU inference for models ≤4B |
| [LocalLLMClient](https://github.com/tattn/LocalLLMClient) | Swift wrapper for llama.cpp (GGUF, 7B–70B) |

---

## Entitlements

Add the following to your `.entitlements` file. Without it the OS will terminate your app when loading models larger than ~500 MB.

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
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

Import only what you need — each module is independent:

```swift
import AuraCore   // always needed

import AuraUI     // prebuilt SwiftUI tabs
import AuraVoice  // voice pipeline
import AuraDocs   // RAG document library
```
