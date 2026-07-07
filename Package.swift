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
    ],
    dependencies: [
        // GGUF (llama.cpp) + Ollama-capable local client. mlx-swift-lm and swift-transformers were removed:
        // they served ONLY the MLX backend, and mlx-swift-lm@main pins swift-syntax 602..<604 which is
        // incompatible with LocalLLMClient's 600..<601 (hard resolve failure). Target models are GGUF/Ollama,
        // so the MLX path is compiled out via `#if canImport(MLXLLM)`.
        // NOTE: kept on branch:"main" deliberately. A hard `revision:` pin re-triggers an unsatisfiable
        // resolve (localllmclient wants swift-syntax 600..<601 while a transitive mlx-swift-lm wants 602..<604)
        // — the same conflict that made us drop the MLX products. Tool-calling now depends on this package's
        // internals, so revisit pinning (e.g. a fork that drops the MLX product deps) before relying on it.
        .package(
            url: "https://github.com/tattn/LocalLLMClient.git",
            branch: "main"
        ),
    ],
    targets: [
        // MARK: - Core
        .target(
            name: "AuraCore",
            dependencies: [
                .product(name: "LocalLLMClient",      package: "LocalLLMClient"),
                .product(name: "LocalLLMClientLlama", package: "LocalLLMClient"),
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
        )
    ]
)
