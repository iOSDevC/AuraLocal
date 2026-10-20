import XCTest
@testable import AuraCore

// A minimal tool defined using only AuraCore's re-exported surface (no `@ToolArguments` macro). File-scope so
// its `Arguments` stays within the nesting limit. `x`/`y` mirror a real pixel-coordinate tool API.
// swiftlint:disable identifier_name
private struct TapTool: LLMTool {
    let name = "tap"
    let description = "Tap the screen at a pixel coordinate."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let x: Int
        let y: Int
        static var argumentsSchema: LLMToolArgumentsSchema {
            ["x": .integer(description: "X pixel"), "y": .integer(description: "Y pixel")]
        }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        ToolOutput(data: ["ok": true, "x": arguments.x, "y": arguments.y])
    }
}
// swiftlint:enable identifier_name

/// A backend that runs a tool loop and reports a fixed summary; `failNext` makes the next call throw.
@MainActor
private final class ToolHostBackend: InferenceBackend, ToolLoopHost {
    static let turn = ToolTurnSummary(rounds: 1, results: [], skipped: [])
    var toolLoop = ToolLoopOptions.default
    var onToolEvent: (@MainActor (ToolEvent) -> Void)?
    private(set) var lastToolTurn: ToolTurnSummary?
    var failNext = false
    var onGenerate: (@MainActor () -> Void)?
    var isLoaded: Bool { true }
    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {}
    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        onGenerate?()
        lastToolTurn = nil
        if failNext { throw AuraError.modelNotLoaded }
        lastToolTurn = Self.turn
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

/// GGUF function-calling plumbing: a host-defined `LLMTool` threads from `AuraLocal(model:tools:)` /
/// `ModelManager.load(tools:)` all the way to the llama.cpp backend. The model-driven tool loop runs live in
/// `GGUFMultiTurnTests` (needs AURA_GGUF_TEST_MODEL); here we verify the wiring and that a tool declared
/// against AuraCore's re-exported API is well-formed.
@MainActor
final class ToolCallingTests: XCTestCase {

    private var ggufModel: Model {
        Model.fromURL("https://huggingface.co/o/r/resolve/main/tiny.Q4_K_M.gguf")!
    }

    /// Read tool names off whichever llama backend the router picked.
    private func toolNames(_ backend: any InferenceBackend) -> [String]? {
        (backend as? LlamaCppBackend)?.registeredToolNames
            ?? (backend as? LayerStreamingBackend)?.registeredToolNames
    }

    func testToolThreadsIntoLlamaBackends() {
        let tap = TapTool()
        XCTAssertEqual(LlamaCppBackend(model: ggufModel, tools: [tap]).registeredToolNames, ["tap"])
        XCTAssertEqual(LayerStreamingBackend(model: ggufModel, tools: [tap]).registeredToolNames, ["tap"])
        // Default is no tools (plain chat path).
        XCTAssertEqual(LlamaCppBackend(model: ggufModel).registeredToolNames, [])
    }

    func testRouterThreadsToolsToTheSelectedBackend() {
        let backend = BackendRouter.selectBackend(for: ggufModel, tools: [TapTool()])
        XCTAssertEqual(toolNames(backend), ["tap"])
    }

    func testAuraLocalThreadsToolsEndToEnd() {
        let aura = AuraLocal(model: ggufModel, tools: [TapTool()])
        XCTAssertEqual(toolNames(aura.engine.backend), ["tap"])
        // No tools → the backend registers none.
        XCTAssertEqual(toolNames(AuraLocal(model: ggufModel).engine.backend), [])
    }

    func testToolDeclarationIsWellFormed() async throws {
        // The schema advertises exactly the declared arguments…
        XCTAssertEqual(Set(TapTool.Arguments.argumentsSchema.keys), ["x", "y"])
        // …and calling the tool returns structured output.
        let output = try await TapTool().call(arguments: .init(x: 5, y: 9))
        XCTAssertEqual(output.data["ok"] as? Bool, true)
        XCTAssertEqual(output.data["x"] as? Int, 5)
    }

    func testToolLoopSettingsReachTheLlamaBackend() throws {
        let aura = AuraLocal(model: ggufModel, tools: [TapTool()])
        let backend = try XCTUnwrap(aura.engine.backend as? LlamaCppBackend)
        XCTAssertEqual(backend.toolLoop, .default)
        var received: [ToolEvent] = []
        aura.toolLoop = ToolLoopOptions(maxToolRounds: 3, toolErrors: .failTurn)
        aura.onToolEvent = { received.append($0) }

        XCTAssertEqual(backend.toolLoop, ToolLoopOptions(maxToolRounds: 3, toolErrors: .failTurn))
        XCTAssertEqual(aura.toolLoop, backend.toolLoop)
        let skipped = ToolEvent.roundLimitReached(skipped: [])
        backend.onToolEvent?(skipped)
        XCTAssertEqual(received, [skipped])
        XCTAssertNil(aura.lastToolTurn, "no turn has run")
        aura.onToolEvent = nil
        XCTAssertNil(backend.onToolEvent)
    }

    func testToolLoopSettingsReachTheLayerStreamingBackend() throws {
        let backend = LayerStreamingBackend(model: ggufModel, tools: [TapTool()])
        let engine = AuraEngine(model: ggufModel, backend: backend)
        engine.toolLoop = ToolLoopOptions(maxToolRounds: 0)
        var received = 0
        engine.onToolEvent = { _ in received += 1 }

        XCTAssertEqual(backend.toolLoop.maxToolRounds, 0)
        backend.onToolEvent?(.roundLimitReached(skipped: []))
        XCTAssertEqual(received, 1)
        XCTAssertNil(engine.lastToolTurn)
    }

    func testDefaultsKeepOneRoundAndReportErrorsToTheModel() {
        XCTAssertEqual(ToolLoopOptions.default.maxToolRounds, 1)
        XCTAssertEqual(ToolLoopOptions.defaultMaxToolRounds, 1)
        XCTAssertEqual(ToolLoopOptions.default.toolErrors, .reportToModel)
        XCTAssertEqual(AuraLocal(model: ggufModel, tools: [TapTool()]).toolLoop, .default)
    }

    func testLastToolTurnIsClearedWhenACallStartsAndNilAfterOneThrows() async throws {
        let backend = ToolHostBackend()
        let engine = AuraEngine(model: ggufModel, backend: backend)
        let llm = AuraLocal(engine: engine)

        _ = try await llm.chat("go")
        XCTAssertEqual(llm.lastToolTurn, ToolHostBackend.turn)

        var seenAtStart: ToolTurnSummary?? = .none
        backend.onGenerate = { seenAtStart = .some(engine.lastToolTurn) }
        backend.failNext = true
        do {
            _ = try await llm.chat("again")
            XCTFail("the backend did not throw")
        } catch {}
        XCTAssertEqual(seenAtStart, .some(nil), "cleared when the call started")
        XCTAssertNil(llm.lastToolTurn)
    }
}
