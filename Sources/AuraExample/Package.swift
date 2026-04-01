// swift-tools-version: 5.9
import PackageDescription

// MARK: - AgentCrew — Standalone Example App
//
// Demonstrates the full AuraLocal suite working together:
//
//   AuraAppleIntelligence  →  AIAgent, AIPipeline, Tool calling (FoundationModels)
//   AuraDocs               →  DocumentLibrary, RAG search, JSONL.GZ export
//   AuraCore               →  ModelManager, ConversationStore, AuraLocal inference
//
// Four-agent pipeline: Extractor → Reviewer → Architect → Reporter
// Shared state via ConversationStore (SQLite blackboard between agents)
// Document grounding via DocumentQueryTool → DocumentLibrary.ask()
// Final export as JSONL.GZ via DocumentExporter
//
// Requirements:
//   - iOS 26+ / macOS 26+
//   - Apple Intelligence enabled on device
//   - Xcode 26+
//
// Setup:
//   1. Place this package next to AuraLocal/  (sibling directory)
//   2. Open Package.swift in Xcode 26
//   3. Select AgentCrewExample scheme
//   4. Run on iPhone with Apple Intelligence enabled

let package = Package(
    name: "AgentCrewExample",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),
    ],
    dependencies: [
        .package(path: "../AuraLocal")
    ],
    targets: [
        .executableTarget(
            name: "AgentCrewExample",
            dependencies: [
                .product(name: "AuraCore",              package: "AuraLocal"),
                .product(name: "AuraDocs",              package: "AuraLocal"),
                .product(name: "AuraAppleIntelligence", package: "AuraLocal"),
            ],
            path: "Sources/AgentCrewExample",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency")
            ]
        )
    ]
)
