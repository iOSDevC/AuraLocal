import XCTest
@testable import AuraCore

/// Always-on: the arithmetic behind `GenerationStats`, how each backend labels its numbers, and that a conformer
/// written before `lastGenerationStats` existed still compiles. No model needed.
@MainActor
final class GenerationStatsTests: XCTestCase {

    private func stats(source: GenerationStats.Provenance = .runtime, prompt: Int = 100, generated: Int = 11,
                       promptTime: TimeInterval = 0.5, decodeTime: TimeInterval = 2) -> GenerationStats {
        GenerationStats(backend: .mlx, source: source, promptTokens: prompt, generatedTokens: generated,
                        promptTime: promptTime, timeToFirstToken: 0.6, decodeTime: decodeTime)
    }

    // MARK: - Rates

    func testDecodeRateExcludesTheFirstTokenWhichBelongsToPrefill() throws {
        XCTAssertEqual(try XCTUnwrap(stats(generated: 11, decodeTime: 2).decodeTokensPerSecond), 5, accuracy: 1e-9)
    }

    func testDecodeRateIsNilBelowTwoTokensOrWithoutDecodeTime() {
        XCTAssertNil(stats(generated: 1).decodeTokensPerSecond)
        XCTAssertNil(stats(generated: 0).decodeTokensPerSecond)
        XCTAssertNil(stats(decodeTime: 0).decodeTokensPerSecond)
    }

    func testPrefillRateIsPromptTokensOverPromptTime() throws {
        XCTAssertEqual(try XCTUnwrap(stats(prompt: 100, promptTime: 0.5).prefillTokensPerSecond), 200, accuracy: 1e-9)
        XCTAssertNil(stats(prompt: 0).prefillTokensPerSecond)
        XCTAssertNil(stats(promptTime: 0).prefillTokensPerSecond)
    }

    func testEstimateIsUTF8BytesOverFourRoundedUp() {
        XCTAssertEqual(GenerationStats.estimatedTokens(in: ""), 0)
        XCTAssertEqual(GenerationStats.estimatedTokens(in: "a"), 1)
        XCTAssertEqual(GenerationStats.estimatedTokens(in: String(repeating: "a", count: 9)), 3)
        XCTAssertEqual(GenerationStats.estimatedTokens(in: String(repeating: "ñ", count: 4)), 2)
    }

    func testDurationConvertsToSeconds() {
        XCTAssertEqual(GenerationStats.seconds(.milliseconds(1500)), 1.5, accuracy: 1e-12)
    }

    // MARK: - StreamMeter

    func testMeasuredStreamCountsOnePerPieceAndTimesFromTheFirst() {
        var meter = StreamMeter()
        meter.recordPiece()
        meter.recordPiece()
        meter.recordPiece()
        let result = meter.stats(backend: .llamaCpp, source: .measured, prompt: "12345678", output: "a much longer reply")
        XCTAssertEqual(result.backend, .llamaCpp)
        XCTAssertEqual(result.source, .measured)
        XCTAssertEqual(result.generatedTokens, 3, "pieces, not text length")
        XCTAssertEqual(result.promptTokens, 2, "prompt is estimated")
        XCTAssertEqual(result.promptTime, result.timeToFirstToken)
        XCTAssertGreaterThanOrEqual(result.timeToFirstToken, 0)
        XCTAssertGreaterThanOrEqual(result.decodeTime, 0)
    }

    func testEstimatedStreamCountsTheOutputText() {
        var meter = StreamMeter()
        meter.recordPiece()
        let result = meter.stats(backend: .layerStreaming, source: .estimated, prompt: "", output: "12345678")
        XCTAssertEqual(result.generatedTokens, 2)
        XCTAssertEqual(result.promptTokens, 0)
    }

    func testStreamWithoutPiecesReportsNoDecode() {
        let result = StreamMeter().stats(backend: .llamaCpp, source: .measured, prompt: "hi", output: "")
        XCTAssertEqual(result.generatedTokens, 0)
        XCTAssertEqual(result.decodeTime, 0)
        XCTAssertNil(result.decodeTokensPerSecond)
    }

    func testToolCallingStreamsAreLabelledEstimated() {
        XCTAssertEqual(GenerationStats.Provenance.llamaStream(toolsRegistered: false), .measured)
        XCTAssertEqual(GenerationStats.Provenance.llamaStream(toolsRegistered: true), .estimated)
    }

    // MARK: - Backends

    func testBackendsStartWithoutStats() throws {
        let gguf = Model.dolphin3_qwen25_1_5b_gguf
        XCTAssertNil(LlamaCppBackend(model: gguf).lastGenerationStats)
        XCTAssertNil(LayerStreamingBackend(model: gguf).lastGenerationStats)
        let target = RemoteTarget(provider: SilentProvider(), modelID: "fake", contextLength: 8192, origin: .cloud)
        XCTAssertNil(RemoteBackend(target: target).lastGenerationStats)
    }

    func testFailedGenerationLeavesNoStats() async {
        let backend = LlamaCppBackend(model: .dolphin3_qwen25_1_5b_gguf)
        _ = try? await backend.generate(prompt: "x", systemPrompt: nil, maxTokens: 8) { _ in }
        XCTAssertNil(backend.lastGenerationStats)
    }

    func testEngineForwardsTheBackendsStats() async throws {
        let backend = ReportingBackend()
        let engine = AuraEngine(model: .dolphin3_qwen25_1_5b_gguf, backend: backend)
        XCTAssertNil(engine.lastGenerationStats)
        _ = try await engine.generate(prompt: "x") { _ in }
        XCTAssertEqual(engine.lastGenerationStats, ReportingBackend.report)
    }

    func testEngineClearsStatsWhenACallThrows() async throws {
        let backend = ReportingBackend()
        let engine = AuraEngine(model: .dolphin3_qwen25_1_5b_gguf, backend: backend)
        _ = try await engine.generate(prompt: "x") { _ in }
        XCTAssertNotNil(engine.lastGenerationStats)
        backend.failNext = true
        _ = try? await engine.generate(prompt: "x") { _ in }
        XCTAssertNil(engine.lastGenerationStats, "the previous call's numbers must not survive a failed one")
    }

    func testConformerWithoutStatsCompilesAndReportsNil() {
        XCTAssertNil(LegacyBackend().lastGenerationStats)
    }
}

// MARK: - Fixtures

/// Implements only what `InferenceBackend` required before `lastGenerationStats`.
@MainActor
private final class LegacyBackend: InferenceBackend {
    var isLoaded: Bool { true }
    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {}
    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String { "" }
    func generate(prompt: String, systemPrompt: String?, maxTokens: Int, constraint: GenerationConstraint,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String { "" }
    func generate(prompt: String, image: PlatformImage?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String { "" }
    func generate(messages: [[String: String]], maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String { "" }
    func unload() {}
}

@MainActor
private final class ReportingBackend: InferenceBackend {
    static let report = GenerationStats(backend: .llamaCpp, source: .measured, promptTokens: 3, generatedTokens: 4,
                                        promptTime: 0.1, timeToFirstToken: 0.1, decodeTime: 0.3)
    private(set) var lastGenerationStats: GenerationStats?
    var failNext = false
    var isLoaded: Bool { true }
    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {}
    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        if failNext { throw AuraError.modelNotLoaded }
        lastGenerationStats = Self.report
        return "ok"
    }
    func generate(prompt: String, systemPrompt: String?, maxTokens: Int, constraint: GenerationConstraint,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String { "" }
    func generate(prompt: String, image: PlatformImage?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String { "" }
    func generate(messages: [[String: String]], maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String { "" }
    func unload() {}
}

private struct SilentProvider: RemoteLLMProvider {
    let id = "test.silent"
    let displayName = "Silent"
    let retentionNote = "test"

    func stream(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}
