import XCTest
@testable import TonamoxCore

/// Needs AURA_GGUF_TEST_MODEL: proves a real llama.cpp run reports non-zero, correctly labelled stats.
@MainActor
final class GGUFGenerationStatsTests: XCTestCase {

    private static let modelVariable = "AURA_GGUF_TEST_MODEL"
    private static let budget = 48

    func testChatReportsMeasuredStats() async throws {
        let llm = try await loadModel()
        let reply = try await llm.chat("Count from one to twenty in words.", maxTokens: Self.budget)
        XCTAssertFalse(reply.isEmpty)
        try assertMeasured(try XCTUnwrap(llm.lastGenerationStats), label: "chat")
    }

    func testStreamReportsStatsOnceItFinishes() async throws {
        let llm = try await loadModel()
        var reply = ""
        for try await delta in llm.stream("Count from one to twenty in words.", maxTokens: Self.budget) { reply += delta }
        XCTAssertFalse(reply.isEmpty)
        try assertMeasured(try XCTUnwrap(llm.lastGenerationStats), label: "stream")
    }

    func testConstrainedCallReportsMeasuredStats() async throws {
        let llm = try await loadModel()
        _ = try await llm.chat("Return a JSON object with the key \"n\" set to 7.", maxTokens: Self.budget,
                               constraint: .json)
        try assertMeasured(try XCTUnwrap(llm.lastGenerationStats), label: "constrained")
    }

    // MARK: - Helpers

    private func assertMeasured(_ stats: GenerationStats, label: String) throws {
        XCTAssertEqual(stats.backend, .llamaCpp)
        XCTAssertEqual(stats.source, .measured)
        XCTAssertGreaterThan(stats.promptTokens, 0)
        XCTAssertGreaterThan(stats.generatedTokens, 1)
        XCTAssertLessThanOrEqual(stats.generatedTokens, Self.budget)
        XCTAssertGreaterThan(stats.timeToFirstToken, 0)
        XCTAssertEqual(stats.promptTime, stats.timeToFirstToken)
        XCTAssertGreaterThan(stats.decodeTime, 0)
        let decode = try XCTUnwrap(stats.decodeTokensPerSecond)
        XCTAssertGreaterThan(decode, 0)
        print(String(format: "%@: %d prompt (est.) / %d generated tokens, TTFT %.3f s, decode %.1f tok/s",
                     label, stats.promptTokens, stats.generatedTokens, stats.timeToFirstToken, decode))
    }

    private func loadModel() async throws -> Tonamox {
        guard let path = ProcessInfo.processInfo.environment[Self.modelVariable], !path.isEmpty else {
            throw XCTSkip("Set \(Self.modelVariable) to a local .gguf file to run the live stats tests.")
        }
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("\(Self.modelVariable) points at a missing file: \(path)")
        }
        let model = try XCTUnwrap(Model.from(.localFile(URL(fileURLWithPath: path))))
        let llm = Tonamox(model: model)
        XCTAssertTrue(llm.engine.backend is LlamaCppBackend)
        try await llm.engine.load { _ in }
        addTeardownBlock { @MainActor in
            llm.engine.unload()
            await (llm.engine.backend as? LlamaCppBackend)?.awaitRelease()
        }
        return llm
    }
}
