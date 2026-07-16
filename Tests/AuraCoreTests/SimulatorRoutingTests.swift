import XCTest
@testable import AuraCore

/// The codebase gated five whole files, two routers and a download path on
/// "GGUF is not supported on the iOS Simulator. Use MLX models instead."
/// Both halves of that sentence are false, and the cost was that AuraCore did not
/// COMPILE for the simulator at all — so the fatalError enforcing it was never even
/// reached. The truth, verified against the artifacts on disk:
///   - llama.xcframework ships an `ios-arm64_x86_64-simulator` slice, and upstream
///     LocalLLMClientLlama/Model.swift sets `n_gpu_layers = 0` there → GGUF runs on CPU.
///   - mlx-swift documents that MLX cannot run on the simulator (no Metal GPU family).
/// So the advice was exactly inverted: it recommended the engine that crashes and
/// disclaimed the one that works.
@MainActor
final class SimulatorRoutingTests: XCTestCase {

    /// GGUF must route to a real engine on every platform, simulator included.
    func testGGUFRoutesToARealEngineEverywhere() {
        let profile = HardwareProfile(totalMemoryGB: 16, availableMemoryGB: 8, deviceName: "test")
        let kind = BackendRouter.recommendedBackend(for: .llama3_1_8b_gguf, profile: profile)
        XCTAssertTrue([.llamaCpp, .layerStreaming].contains(kind),
                      "GGUF must pick a llama.cpp engine, got \(kind)")
    }

    /// A GGUF model must never be refused for being on the simulator.
    func testGGUFIsNeverRefusedForBeingOnSimulator() {
        let profile = HardwareProfile(totalMemoryGB: 16, availableMemoryGB: 8, deviceName: "test")
        let backend = BackendRouter.selectBackend(for: .llama3_1_8b_gguf, profile: profile)
        XCTAssertFalse(backend is UnavailableBackend,
                       "GGUF is runnable on every Apple platform we ship, simulator included")
    }

    #if targetEnvironment(simulator)
    /// MLX has no Metal GPU here. It must fail with a clear message, not crash mid-load —
    /// and the message must point at GGUF, which actually works.
    func testMLXFailsClearlyOnSimulatorInsteadOfCrashing() async {
        let profile = HardwareProfile(totalMemoryGB: 16, availableMemoryGB: 8, deviceName: "sim")
        let backend = BackendRouter.selectBackend(for: .qwen3_0_6b, profile: profile)
        XCTAssertTrue(backend is UnavailableBackend, "MLX cannot run on the simulator")
    }
    #else
    /// Off the simulator, an MLX model must reach the MLX engine — not be quietly rerouted.
    func testMLXReachesTheMLXEngineOffSimulator() {
        XCTAssertEqual(BackendRouter.recommendedBackend(for: .qwen3_0_6b), .mlx)
    }
    #endif
}
