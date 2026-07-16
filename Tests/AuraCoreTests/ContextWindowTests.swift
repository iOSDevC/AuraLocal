import XCTest
@testable import AuraCore

/// Locks the fix behind "the model fails on several files": the context window was
/// HARDCODED per platform (2048 iOS / 8192 macOS), collapsing every model to the
/// same tier regardless of its real KV cost. It is now derived from the model's own
/// GQA-aware KV math and the memory left after weights.
///
/// (The companion fix — teaching the RAG library to index code files — lives in
/// AuraDocs, which this target can't import: adding it pulls swift-numerics and
/// breaks the Cxx-interop test build. It is compile-verified only.)
final class ContextWindowTests: XCTestCase {

    // MARK: - KV-aware context window

    /// The core of the fix: at the SAME memory, a KV-light model must get a much
    /// larger window than a KV-heavy one. The old hardcode gave both the same tier.
    func testContextWindowIsModelAware() {
        // dolphin3_qwen25_1_5b: 990 MB, 28 layers x 2 kv-heads x 128 dim  → ~28 KB/token
        // llama3_1_8b:        4700 MB, 32 layers x 8 kv-heads x 128 dim  → ~131 KB/token
        let light = HardwareAnalyzer.recommendedContextWindow(for: .dolphin3_qwen25_1_5b_gguf, availableGB: 6)
        let heavy = HardwareAnalyzer.recommendedContextWindow(for: .llama3_1_8b_gguf, availableGB: 6)

        XCTAssertGreaterThan(light, heavy,
            "a KV-light model must earn a larger window than a KV-heavy one at the same memory")
        XCTAssertGreaterThan(light, 8192, "the KV-light model should exceed the old hardcoded tier")
    }

    /// More memory must never shrink the window.
    func testContextWindowScalesWithMemory() {
        let tight = HardwareAnalyzer.recommendedContextWindow(for: .llama3_1_8b_gguf, availableGB: 6)
        let roomy = HardwareAnalyzer.recommendedContextWindow(for: .llama3_1_8b_gguf, availableGB: 12)
        XCTAssertGreaterThanOrEqual(roomy, tight)
    }

    /// Weights alone exceed memory → degrade to the floor, never a negative/huge value.
    func testContextWindowFloorsWhenWeightsExceedMemory() {
        let ctx = HardwareAnalyzer.recommendedContextWindow(for: .llama3_1_70b_gguf, availableGB: 3)
        XCTAssertEqual(ctx, 1024)
    }

    /// Models without KV metadata (kvHeads == 0 — user-supplied GGUFs, MLX entries)
    /// fall back to the platform tier instead of inviting dishonest math.
    func testContextWindowFallsBackWithoutKVMetadata() throws {
        let user = try XCTUnwrap(Model.fromURL("https://example.com/models/custom.gguf"))
        XCTAssertEqual(user.kvHeads, 0, "precondition: user-supplied models carry no KV metadata")
        let ctx = HardwareAnalyzer.recommendedContextWindow(for: user, availableGB: 16)
        #if os(macOS)
        XCTAssertTrue([4096, 8192].contains(ctx), "expected the macOS platform tier, got \(ctx)")
        #else
        XCTAssertTrue([1024, 2048].contains(ctx))
        #endif
    }

}
