// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Tonamox",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
        .visionOS(.v2)
    ],
    products: [
        .library(name: "TonamoxCore",              targets: ["TonamoxCore"]),
        .library(name: "TonamoxUI",                targets: ["TonamoxUI"]),
        .library(name: "TonamoxVoice",             targets: ["TonamoxVoice"]),
        .library(name: "TonamoxDocs",              targets: ["TonamoxDocs"]),
        .library(name: "TonamoxAppleIntelligence", targets: ["TonamoxAppleIntelligence"]),
        .library(name: "TonamoxAgents",            targets: ["TonamoxAgents"]),
        .library(name: "TonamoxImageGen",          targets: ["TonamoxImageGen"]),
        .executable(name: "tonamox", targets: ["tonamox"]),
    ],
    dependencies: [
        // GGUF (llama.cpp) + MLX local client, pinned to the SHA of tag 0.5.0.
        // `exact: "0.5.0"` is rejected: LocalLLMClientLlamaC carries unsafeFlags, which
        // SwiftPM forbids in version-pinned deps but allows in revision-pinned ones.
        //
        // 0.5.0 is what makes Tonamox resolvable at all. It swapped mlx-swift-lm
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
        // in itself — so declaring them adds zero packages; it only lets TonamoxCore link what
        // it already downloads. Ranges mirror LocalLLMClient's own so they unify instead of
        // conflicting; `from:` (not a branch) is what lets SwiftPM backtrack off the 602 floor.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm",      from: "3.31.3"),
        // mlx-swift-lm 3.31.3 allows mlx-swift 0.31.3..<0.32.0, so a resolve without our lockfile took
        // 0.31.6, the version in the open pre-main launch hang report (mlx-swift #485); 0.31.5 does not
        // build for iOS (#436). 0.31.4 vendors the same Cmlx sources as 0.31.3.
        .package(url: "https://github.com/ml-explore/mlx-swift",         "0.31.3"..<"0.31.5"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // Also already resolved: LocalLLMClient renders chat templates with it. The GGUF backends render the
        // same templates themselves, so every turn can start from an empty KV cache with real roles.
        .package(url: "https://github.com/huggingface/swift-jinja", from: "2.3.5"),

        // Core ML Stable Diffusion for TonamoxImageGen. `exact:` because upstream main pins
        // swift-transformers 0.1.8, which conflicts with the 1.3.0 floor above; 1.1.1 depends only on
        // swift-argument-parser (for its CLI).
        .package(url: "https://github.com/apple/ml-stable-diffusion", exact: "1.1.1"),
    ],
    targets: [
        // MARK: - Core
        .target(
            name: "TonamoxCore",
            dependencies: [
                .product(name: "LocalLLMClient",      package: "LocalLLMClient"),
                .product(name: "LocalLLMClientLlama", package: "LocalLLMClient"),
                // Lights up MLXBackend.swift + the 32 MLX catalog entries, both of which
                // sit behind `#if canImport(MLXLLM)` and were dark while these were absent.
                .product(name: "MLXLLM",      package: "mlx-swift-lm"),
                .product(name: "MLXVLM",      package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                // Already linked through MLXLLM; naming it makes the mlx-swift range bind downstream, where a
                // dependency no target uses is not resolved.
                .product(name: "MLX",         package: "mlx-swift"),
                .product(name: "Tokenizers",  package: "swift-transformers"),
                .product(name: "Jinja",       package: "swift-jinja"),
            ],
            path: "Sources/TonamoxCore",
            resources: [
                .process("Resources")
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ]
        ),

        // MARK: - UI
        .target(
            name: "TonamoxUI",
            dependencies: [
                "TonamoxCore",
                "TonamoxVoice",
            ],
            path: "Sources/TonamoxUI",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Voice
        .target(
            name: "TonamoxVoice",
            dependencies: ["TonamoxCore"],
            path: "Sources/TonamoxVoice",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Docs (RAG)
        .target(
            name: "TonamoxDocs",
            dependencies: ["TonamoxCore"],
            path: "Sources/TonamoxDocs",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Apple Intelligence Agents
        // Requires iOS 26+ / macOS 26+ with Apple Intelligence enabled.
        // No dependency on TonamoxCore — standalone module using FoundationModels.
            .target(
                name: "TonamoxAppleIntelligence",
                dependencies: [],
                path: "Sources/TonamoxAppleIntelligence",
                swiftSettings: [
                    .enableUpcomingFeature("StrictConcurrency")
                ]
            ),

        // MARK: - Agents (multi-agent orchestration)
        // Reusable agent pipeline (AgentCrew) on FoundationModels + per-step hybrid
        // escalation. Sits above the base modules; gated to iOS 26 / macOS 26 in-code.
        .target(
            name: "TonamoxAgents",
            dependencies: ["TonamoxCore", "TonamoxAppleIntelligence", "TonamoxDocs"],
            path: "Sources/TonamoxAgents",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Image Generation
        // On device (iOS + macOS): Core ML Stable Diffusion. macOS only: FLUX + LoRA through the
        // mflux subprocess. The target compiles everywhere; visionOS gets no StableDiffusion
        // (1.1.1 declares no visionOS), so its engines throw `.unsupportedPlatform` there.
        .target(
            name: "TonamoxImageGen",
            dependencies: [
                "TonamoxCore",
                .product(name: "StableDiffusion", package: "ml-stable-diffusion",
                         condition: .when(platforms: [.iOS, .macOS, .macCatalyst])),
            ],
            path: "Sources/TonamoxImageGen",
            resources: [.process("PrivacyInfo.xcprivacy")],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Recibo spike (not a product)
        // The deterministic half of the Recibo spike: the rule-based baseline, the prompt, the OCR anchors and the
        // scoring that `tonamox receipt` drives. Recibo itself lives in its own repository.
        .target(
            name: "ReceiptSpike",
            dependencies: ["TonamoxCore"],
            path: "Sources/ReceiptSpike",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - CLI
        // `tonamox` — a headless integration harness that drives the hybrid + tools
        // features (provider detection, local-first `tonamox ask` escalation, native OCR).
        // Cxx interop is required because it links TonamoxCore (llama.cpp).
        .executableTarget(
            name: "tonamox",
            dependencies: ["TonamoxCore", "TonamoxImageGen", "ReceiptSpike"],
            path: "Sources/tonamox",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),

        // MARK: - Example App
        // TonamoxExample is a macOS/iOS app (built via Sources/TonamoxExample/TonamoxExample.xcodeproj),
        // not a library product. It requires macOS 26 (AgentCrew) and is intentionally NOT a
        // SwiftPM target here, so `swift build`/`swift test` stay green on the macOS-15 package.

        // MARK: - Tests
        .testTarget(
            name: "TonamoxCoreTests",
            dependencies: ["TonamoxCore"],
            path: "Tests/TonamoxCoreTests",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .testTarget(
            name: "TonamoxDocsTests",
            dependencies: ["TonamoxDocs", "TonamoxCore"],
            path: "Tests/TonamoxDocsTests",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .testTarget(
            name: "ReceiptSpikeTests",
            dependencies: ["ReceiptSpike", "TonamoxCore"],
            path: "Tests/ReceiptSpikeTests",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .testTarget(
            name: "TonamoxImageGenTests",
            dependencies: ["TonamoxImageGen"],
            path: "Tests/TonamoxImageGenTests",
            resources: [.copy("Fixtures")],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        )
    ]
)
