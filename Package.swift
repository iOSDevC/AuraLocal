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
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            branch: "main"
        ),
        .package(
            url: "https://github.com/tattn/LocalLLMClient.git",
            branch: "main"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            branch: "main"
        ),
    ],
    targets: [
        // MARK: - Core
        .target(
            name: "AuraCore",
            dependencies: [
                .product(name: "MLXVLM",          package: "mlx-swift-lm"),
                .product(name: "MLXLLM",          package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon",     package: "mlx-swift-lm"),
                .product(name: "LocalLLMClient",      package: "LocalLLMClient"),
                .product(name: "LocalLLMClientLlama", package: "LocalLLMClient"),
                .product(name: "Tokenizers",          package: "swift-transformers"),
            ],
            path: "Sources/AuraCore",
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
        .target(
            name: "AuraExample",
            dependencies: [
                "AuraCore",
                "AuraUI",
                "AuraVoice",
                "AuraDocs",
                "AuraAppleIntelligence",
            ],
            path: "Sources/AuraExample",
            exclude: ["Package.swift"],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Tests
        .testTarget(
            name: "AuraCoreTests",
            dependencies: ["AuraCore"],
            path: "Tests/AuraCoreTests",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        )
    ]
)
