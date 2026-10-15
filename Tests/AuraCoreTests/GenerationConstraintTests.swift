import XCTest
@testable import AuraCore

/// Always-on: the context-window arithmetic for dedicated constrained clients, the backends that must refuse a
/// constraint instead of dropping it, and the gate that keeps decodes one at a time. No model needed.
@MainActor
final class GenerationConstraintTests: XCTestCase {

    private let gguf = Model.dolphin3_qwen25_1_5b_gguf

    // MARK: - Context window

    func testWindowIsPromptBytesPlusBudgetPlusTemplateRoundedUpTo256() {
        let prompt = String(repeating: "a", count: 100)   // 100 + 128 + 256 = 484
        XCTAssertEqual(LlamaCppBackend.constrainedContextWindow(forPrompt: prompt, maxTokens: 128, cap: 8192), 512)
    }

    func testWindowCountsUTF8BytesNotCharacters() {
        let prompt = String(repeating: "ñ", count: 100)   // 200 bytes + 100 + 256 = 556
        XCTAssertEqual(LlamaCppBackend.constrainedContextWindow(forPrompt: prompt, maxTokens: 100, cap: 8192), 768)
    }

    func testWindowAlreadyOnA256BoundaryIsNotPaddedFurther() {
        XCTAssertEqual(LlamaCppBackend.constrainedContextWindow(forPrompt: "", maxTokens: 0, cap: 8192), 256)
        let prompt = String(repeating: "a", count: 256)
        XCTAssertEqual(LlamaCppBackend.constrainedContextWindow(forPrompt: prompt, maxTokens: 0, cap: 8192), 512)
    }

    func testWindowNeverExceedsTheCap() {
        let prompt = String(repeating: "x", count: 50_000)
        XCTAssertEqual(LlamaCppBackend.constrainedContextWindow(forPrompt: prompt, maxTokens: 1024, cap: 4096), 4096)
    }

    func testNegativeBudgetCountsAsZero() {
        XCTAssertEqual(LlamaCppBackend.constrainedContextWindow(forPrompt: "", maxTokens: -5, cap: 8192), 256)
    }

    func testUnboundedBudgetIsCappedInsteadOfOverflowing() {
        XCTAssertEqual(LlamaCppBackend.constrainedContextWindow(forPrompt: "hi", maxTokens: .max, cap: 8192), 8192)
    }

    // MARK: - Mapping onto LocalLLMClient

    func testJSONMapsToTheBundledGrammar() {
        guard case .json = GenerationConstraint.json.llamaFormat else { return XCTFail("expected .json") }
    }

    func testCustomGrammarKeepsTextAndRoot() {
        let format = GenerationConstraint.grammar(gbnf: #"answer ::= "yes""#, root: "answer").llamaFormat
        guard case .grammar(let gbnf, let root) = format else { return XCTFail("expected .grammar") }
        XCTAssertEqual(gbnf, #"answer ::= "yes""#)
        XCTAssertEqual(root, "answer")
    }

    func testGrammarRootDefaultsToRoot() {
        XCTAssertEqual(GenerationConstraint.grammar(gbnf: "x"), .grammar(gbnf: "x", root: "root"))
    }

    // MARK: - Backends that cannot enforce a constraint throw

    #if canImport(MLXLLM)
    func testMLXBackendThrowsEvenBeforeLoading() async throws {
        let backend = MLXBackend(model: try mlxModel())
        await assertThrows(.unsupportedBackend(.mlx)) {
            _ = try await backend.generate(prompt: "x", systemPrompt: nil, maxTokens: 8, constraint: .json) { _ in }
        }
    }

    func testAuraLocalChatSurfacesTheMLXRefusal() async throws {
        let llm = AuraLocal(model: try mlxModel())
        await assertThrows(.unsupportedBackend(.mlx)) { _ = try await llm.chat("x", constraint: .json) }
    }

    func testAuraLocalStreamSurfacesTheMLXRefusal() async throws {
        let llm = AuraLocal(model: try mlxModel())
        await assertThrows(.unsupportedBackend(.mlx)) {
            for try await _ in llm.stream("x", constraint: .grammar(gbnf: #"root ::= "a""#)) {}
        }
    }

    private func mlxModel() throws -> Model {
        try XCTUnwrap(ModelRegistry.shared.models.first { $0.format == .mlx }, "catalog has no MLX entry")
    }
    #endif

    func testLayerStreamingBackendThrows() async {
        let backend = LayerStreamingBackend(model: gguf)
        await assertThrows(.unsupportedBackend(.layerStreaming)) {
            _ = try await backend.generate(prompt: "x", systemPrompt: nil, maxTokens: 8, constraint: .json) { _ in }
        }
    }

    func testRemoteBackendThrowsWithoutCallingTheProvider() async {
        let target = RemoteTarget(provider: UnconstrainedProvider(), modelID: "fake", contextLength: 8192, origin: .cloud)
        let backend = RemoteBackend(target: target)
        await assertThrows(.unsupportedBackend(.remote)) {
            _ = try await backend.generate(prompt: "x", systemPrompt: nil, maxTokens: 8, constraint: .json) { _ in }
        }
    }

    func testUnconstrainedCallsStillReachTheRemoteProvider() async throws {
        let target = RemoteTarget(provider: UnconstrainedProvider(), modelID: "fake", contextLength: 8192, origin: .cloud)
        let reply = try await AuraEngine(model: gguf, backend: RemoteBackend(target: target))
            .generate(prompt: "x", constraint: nil) { _ in }
        XCTAssertEqual(reply, UnconstrainedProvider.reply)
    }

    // MARK: - llama.cpp backend guards

    func testUnloadedLlamaCppBackendReportsNotLoaded() async {
        let backend = LlamaCppBackend(model: gguf)
        do {
            _ = try await backend.generate(prompt: "x", systemPrompt: nil, maxTokens: 8, constraint: .json) { _ in }
            XCTFail("expected modelNotLoaded")
        } catch AuraError.modelNotLoaded {
        } catch {
            XCTFail("expected modelNotLoaded, got \(error)")
        }
    }

    /// A blank grammar reaches llama.cpp as "no grammar" and would silently drop the constraint.
    func testEmptyGrammarIsRejectedBeforeAnyModelWork() async {
        let backend = LlamaCppBackend(model: gguf)
        await assertThrows(.invalidGrammar(root: "answer")) {
            _ = try await backend.generate(prompt: "x", systemPrompt: nil, maxTokens: 8,
                                           constraint: .grammar(gbnf: " \n", root: "answer")) { _ in }
        }
    }

    func testErrorDescriptionsNameTheBackendAndTheRoot() {
        let unsupported = GenerationConstraintError.unsupportedBackend(.mlx).errorDescription ?? ""
        XCTAssertTrue(unsupported.contains("mlx") && unsupported.contains("GGUF"), unsupported)
        let invalid = GenerationConstraintError.invalidGrammar(root: "answer").errorDescription ?? ""
        XCTAssertTrue(invalid.contains("'answer'"), invalid)
    }

    // MARK: - DecodeGate

    func testGateRunsBodiesOneAtATimeInArrivalOrder() async {
        let gate = DecodeGate()
        let log = EventLog()
        var tasks: [Task<Void, Never>] = []
        for name in ["a", "b", "c"] {
            tasks.append(overlappingDecode(named: name, gate: gate, log: log))
            await Task.yield()
        }
        for task in tasks { await task.value }
        XCTAssertEqual(log.events, ["a+", "a-", "b+", "b-", "c+", "c-"])
    }

    private func overlappingDecode(named name: String, gate: DecodeGate, log: EventLog) -> Task<Void, Never> {
        Task { @MainActor in
            await gate.run {
                log.events.append("\(name)+")
                try? await Task.sleep(for: .milliseconds(20))
                log.events.append("\(name)-")
            }
        }
    }

    func testGateIsReleasedWhenABodyThrows() async throws {
        let gate = DecodeGate()
        do {
            try await gate.run { throw CancellationError() }
            XCTFail("expected the body's error")
        } catch is CancellationError {}
        let value = await gate.run { 42 }
        XCTAssertEqual(value, 42)
    }

    // MARK: - Helpers

    private func assertThrows(
        _ expected: GenerationConstraintError,
        file: StaticString = #filePath, line: UInt = #line,
        _ body: () async throws -> Void
    ) async {
        do {
            try await body()
            XCTFail("expected \(expected), got a result", file: file, line: line)
        } catch let error as GenerationConstraintError {
            XCTAssertEqual(error, expected, file: file, line: line)
        } catch {
            XCTFail("expected \(expected), got \(error)", file: file, line: line)
        }
    }
}

@MainActor
private final class EventLog {
    var events: [String] = []
}

/// Answers without any constraint, so a backend that forwarded a constrained call would return text, not throw.
private struct UnconstrainedProvider: RemoteLLMProvider {
    static let reply = "free text"
    let id = "test.unconstrained"
    let displayName = "Unconstrained"
    let retentionNote = "test"

    func stream(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(.token(Self.reply))
            continuation.finish()
        }
    }
}
