// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AuraLocal",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
        .visionOS(.v2)
    ],
    products: [
        .library(name: "AuraCore",              targets: ["AuraCore"]),
        .library(name: "AuraUI",                targets: ["AuraUI"]),
        .library(name: "AuraVoice",             targets: ["AuraVoice"]),
        .library(name: "AuraDocs",              targets: ["AuraDocs"]),
        .library(name: "AuraAppleIntelligence", targets: ["AuraAppleIntelligence"]),
        .library(name: "AuraAgents",            targets: ["AuraAgents"]),
        .library(name: "AuraImageGen",          targets: ["AuraImageGen"]),
        .executable(name: "aura", targets: ["aura"]),
    ],
    dependencies: [
        // GGUF (llama.cpp) + MLX local client, pinned to the SHA of tag 0.5.0.
        // `exact: "0.5.0"` is rejected: LocalLLMClientLlamaC carries unsafeFlags, which
        // SwiftPM forbids in version-pinned deps but allows in revision-pinned ones.
        //
        // 0.5.0 is what makes AuraLocal resolvable at all. It swapped mlx-swift-lm
        // `branch:"main"` for `from: "3.31.3"`, so SwiftPM can backtrack past the
        // swift-syntax 602..<604 floor mlx-swift-lm adopted in 3.31.4 and settle on
        // 3.31.3. A branch pin cannot backtrack, so every resolve without our lockfile
        // failed — which is what downstream integrators got, since SwiftPM honours only
        // the ROOT package's Package.resolved.
        .package(
            url: "https://github.com/tattn/LocalLLMClient.git",
            revision: "edc39ef2ffc1cef9cf856b0788de8d331f776c2e"   // tag 0.5.0
        ),

        // MLX (Apple Silicon). Both are ALREADY in the resolve — LocalLLMClient pulls them
        // in itself — so declaring them adds zero packages; it only lets AuraCore link what
        // it already downloads. Ranges mirror LocalLLMClient's own so they unify instead of
        // conflicting; `from:` (not a branch) is what lets SwiftPM backtrack off the 602 floor.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm",      from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        // MARK: - Core
        .target(
            name: "AuraCore",
            dependencies: [
                .product(name: "LocalLLMClient",      package: "LocalLLMClient"),
                .product(name: "LocalLLMClientLlama", package: "LocalLLMClient"),
                // Lights up MLXBackend.swift + the 32 MLX catalog entries, both of which
                // sit behind `#if canImport(MLXLLM)` and were dark while these were absent.
                .product(name: "MLXLLM",      package: "mlx-swift-lm"),
                .product(name: "MLXVLM",      package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers",  package: "swift-transformers"),
            ],
            path: "Sources/AuraCore",
            resources: [
                .process("Resources")
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ]
        ),

        // MARK: - UI
        .target(
            name: "AuraUI",
            dependencies: [
                "AuraCore",
                "AuraVoice",
            ],
            path: "Sources/AuraUI",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Voice
        .target(
            name: "AuraVoice",
            dependencies: ["AuraCore"],
            path: "Sources/AuraVoice",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Docs (RAG)
        .target(
            name: "AuraDocs",
            dependencies: ["AuraCore"],
            path: "Sources/AuraDocs",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Apple Intelligence Agents
        // Requires iOS 26+ / macOS 26+ with Apple Intelligence enabled.
        // No dependency on AuraCore — standalone module using FoundationModels.
            .target(
                name: "AuraAppleIntelligence",
                dependencies: [],
                path: "Sources/AuraAppleIntelligence",
                swiftSettings: [
                    .enableUpcomingFeature("StrictConcurrency")
                ]
            ),

        // MARK: - Agents (multi-agent orchestration)
        // Reusable agent pipeline (AgentCrew) on FoundationModels + per-step hybrid
        // escalation. Sits above the base modules; gated to iOS 26 / macOS 26 in-code.
        .target(
            name: "AuraAgents",
            dependencies: ["AuraCore", "AuraAppleIntelligence", "AuraDocs"],
            path: "Sources/AuraAgents",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Image Generation (macOS-only)
        // Drives mflux (FLUX on MLX) as a subprocess for text-to-image + lora.safetensors.
        // The target compiles on every platform (iOS gets non-functional stubs that throw), so
        // depending on it never breaks an iOS/simulator build; it only *works* on macOS.
        .target(
            name: "AuraImageGen",
            dependencies: ["AuraCore"],
            path: "Sources/AuraImageGen",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - CLI
        // `aura` — a headless integration harness that drives the hybrid + tools
        // features (provider detection, GitHub Models escalation, native OCR).
        // Cxx interop is required because it links AuraCore (llama.cpp).
        .executableTarget(
            name: "aura",
            dependencies: ["AuraCore", "AuraImageGen"],
            path: "Sources/aura",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Example App
        // AuraExample is a macOS/iOS app (built via Sources/AuraExample/AuraExample.xcodeproj),
        // not a library product. It requires macOS 26 (AgentCrew) and is intentionally NOT a
        // SwiftPM target here, so `swift build`/`swift test` stay green on the macOS-15 package.

        // MARK: - Tests
        .testTarget(
            name: "AuraCoreTests",
            dependencies: ["AuraCore"],
            path: "Tests/AuraCoreTests",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .testTarget(
            name: "AuraImageGenTests",
            dependencies: ["AuraImageGen"],
            path: "Tests/AuraImageGenTests",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        )
    ]
)
