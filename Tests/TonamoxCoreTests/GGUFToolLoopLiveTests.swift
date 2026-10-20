import Synchronization
import XCTest
@testable import TonamoxCore
@testable import LocalLLMClientLlama

/// Needs AURA_GGUF_TEST_MODEL (a small chat GGUF). A small model's choices vary, so each test asserts the loop's
/// invariants, prints its transcript and events, and skips the model-dependent checks when no tool was called.
@MainActor
final class GGUFToolLoopLiveTests: XCTestCase {

    private static let modelVariable = "AURA_GGUF_TEST_MODEL"
    private static let facts = [
        "My name is Ximena and I keep bees on a rooftop in Lima.", "I have three hives.",
        "The hives are painted blue.", "I harvest honey in March.", "My neighbour is called Tomás.",
    ]
    private static let padding = " Please answer kindly, keep it practical, and mention one detail I could try this week."
    private static let probe: [LlamaChatMessage] = [.system("Be brief."), .user("Name one primary color.")]
    private static let plainChatML = """
    {% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' \
    + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}
    """

    // MARK: - Tests

    func testDefaultLoopThroughTheTonamoxAPI() async throws {
        let log = CallLog()
        let llm = try await loadModel(tools: [EchoTool(log: log)])
        let events = EventLog()
        llm.onToolEvent = { events.append($0) }

        let reply = try await llm.chat("Use the echo tool to repeat the word pineapple.", maxTokens: 128)
        let turn = try XCTUnwrap(llm.lastToolTurn, "a GGUF instance with tools reports its tool turn")
        let engine = try toolEngine(of: llm)
        report("default loop", reply: reply, turn: turn, events: events, prompt: try lastPrompt(of: engine))
        assertInvariants(turn, events: events, limit: 1, log: log)

        if engine.renderer.template?.contains("<tool_call>") == true {
            XCTAssertEqual(log.calls, ["echo:pineapple"], reply)
            XCTAssertEqual(turn.results.first?.output, #"{"text":"pineapple"}"#)
            XCTAssertEqual(turn.rounds, 1)
            if engine.renderer.template?.contains("tools") == true {
                let prompt = try lastPrompt(of: engine)
                XCTAssertTrue(prompt.contains("<tool_response>\n{\"text\":\"pineapple\"}\n</tool_response>"), prompt)
                XCTAssertFalse(prompt.contains("<|im_start|>assistant\n<tool_response>"),
                               "a template with tools gets the native tool turn")
            }
        }
        let next = try await llm.chat("Name one primary color.", systemPrompt: "Be brief.", maxTokens: 24)
        XCTAssertFalse(next.isEmpty)
    }

    /// Qwen's tokens with a ChatML template that has no tool support, so its rounds go out in the text form.
    func testTextFormRoundsOnATemplateWithoutTools() async throws {
        let url = try modelURL()
        let metadata = try GGUFChatMetadata.read(from: url)
        let log = CallLog()
        let tools: [any LLMTool] = [EchoTool(log: log)]
        let renderer = LlamaPromptRenderer(template: Self.plainChatML, specialTokens: metadata.specialTokens,
                                           tools: tools.map { AnyLLMTool($0) })
        let engine = try await makeToolEngine(tools: tools, renderer: renderer)
        let events = EventLog()
        let reply = try await engine.respond(
            to: [.user("Use the echo tool to repeat the word pineapple.")], maxTokens: 128,
            toolLoop: context(.default, events), shouldStop: nil) { _ in }
        let turn = try XCTUnwrap(reply.tools)
        report("text form", reply: reply.text, turn: turn, events: events, prompt: reply.prompt)
        assertInvariants(turn, events: events, limit: 1, log: log)
        try await waitUntilIdle(engine)
        guard turn.results.first != nil else { throw XCTSkip("the model made no tool call") }

        // The round's call is the reply as the model wrote it, the <tool_call> block included.
        let call = try XCTUnwrap(reply.history.first { $0.role == .assistant }?.content)
        XCTAssertTrue(call.contains("<tool_call>"), call)
        XCTAssertTrue(reply.prompt.contains("<|im_start|>assistant\n" + call + "\n<tool_response>\n"), reply.prompt)
        XCTAssertEqual(reply.prompt.components(separatedBy: "<|im_start|>assistant\n").count - 1, 2,
                       "the round is one assistant turn, then the generation prompt")
        XCTAssertFalse(reply.prompt.contains("<|im_start|>tool"), "the result is sent in the round's assistant turn")
    }

    /// Without tool support the instructions are the only list of tools, so they stay while another round may run.
    func testTextFormRoundsKeepTheToolListWhileRoundsRemain() async throws {
        let url = try modelURL()
        let metadata = try GGUFChatMetadata.read(from: url)
        let log = CallLog()
        let tools: [any LLMTool] = [SecretTool(log: log), EchoTool(log: log)]
        let renderer = LlamaPromptRenderer(template: Self.plainChatML, specialTokens: metadata.specialTokens,
                                           tools: tools.map { AnyLLMTool($0) })
        let engine = try await makeToolEngine(tools: tools, renderer: renderer)
        let events = EventLog()
        let reply = try await engine.respond(
            to: [.user("First call get_secret to learn the secret word. Then call echo with that word.")],
            maxTokens: 192, toolLoop: context(ToolLoopOptions(maxToolRounds: 3), events), shouldStop: nil) { _ in }
        let turn = try XCTUnwrap(reply.tools)
        report("text form, 3 rounds", reply: reply.text, turn: turn, events: events, prompt: reply.prompt)
        assertInvariants(turn, events: events, limit: 3, log: log)
        try await waitUntilIdle(engine)
        guard turn.rounds > 0 else { throw XCTSkip("the model made no tool call") }
        let listsTools = reply.prompt.contains("If you decide to invoke")
            && reply.prompt.contains(#""description":"Return the secret word.""#)
        XCTAssertEqual(listsTools, turn.rounds < 3, "the last prompt lists the tools exactly when a round could still run")
    }

    func testChainUpToThreeRounds() async throws {
        let log = CallLog()
        let engine = try await makeToolEngine(tools: [SecretTool(log: log), EchoTool(log: log)])
        let events = EventLog()
        let options = ToolLoopOptions(maxToolRounds: 3)
        let reply = try await engine.respond(
            to: [.user("First call get_secret to learn the secret word. Then call echo with that word. "
                       + "Finally tell me the word.")],
            maxTokens: 192, toolLoop: context(options, events), shouldStop: nil) { _ in }
        let turn = try XCTUnwrap(reply.tools)
        report("chain", reply: reply.text, turn: turn, events: events, prompt: reply.prompt)
        assertInvariants(turn, events: events, limit: 3, log: log)
        try await waitUntilIdle(engine)
    }

    func testZeroRoundsReportsTheCallsAsSkipped() async throws {
        let log = CallLog()
        let engine = try await makeToolEngine(tools: [EchoTool(log: log)])
        let events = EventLog()
        let reply = try await engine.respond(
            to: [.user("Use the echo tool to repeat the word pineapple.")], maxTokens: 128,
            toolLoop: context(ToolLoopOptions(maxToolRounds: 0), events), shouldStop: nil) { _ in }
        let turn = try XCTUnwrap(reply.tools)
        report("zero rounds", reply: reply.text, turn: turn, events: events, prompt: reply.prompt)
        assertInvariants(turn, events: events, limit: 0, log: log)
        try await waitUntilIdle(engine)
        guard !turn.skipped.isEmpty else { throw XCTSkip("the model made no tool call") }

        XCTAssertEqual(turn.rounds, 0)
        XCTAssertEqual(log.calls, [], "nothing runs at limit 0")
        XCTAssertEqual(events.all, [.roundLimitReached(skipped: turn.skipped)])
        XCTAssertEqual(Set(turn.skipped.map(\.round)), [1])
    }

    func testAThrowingToolIsReportedAndTheTurnAnswers() async throws {
        let log = CallLog()
        let engine = try await makeToolEngine(tools: [FailTool(log: log)])
        let events = EventLog()
        let reply = try await engine.respond(
            to: [.user("What is the weather in Lima? Use the get_weather tool.")], maxTokens: 160,
            toolLoop: context(.default, events), shouldStop: nil) { _ in }
        let turn = try XCTUnwrap(reply.tools)
        report("throwing tool", reply: reply.text, turn: turn, events: events, prompt: reply.prompt)
        assertInvariants(turn, events: events, limit: 1, log: log)
        try await waitUntilIdle(engine)
        guard let first = turn.results.first else { throw XCTSkip("the model made no tool call") }

        XCTAssertTrue(first.isError)
        XCTAssertTrue(first.output.hasPrefix(#"{"error":"#), first.output)
        XCTAssertTrue(reply.prompt.contains(first.output), "the error reached the model")
        XCTAssertTrue(!reply.text.isEmpty || !turn.skipped.isEmpty, "the turn went on after the failure")
    }

    func testFailTurnRethrowsTheToolsError() async throws {
        let log = CallLog()
        let engine = try await makeToolEngine(tools: [FailTool(log: log)])
        let events = EventLog()
        do {
            let reply = try await engine.respond(
                to: [.user("What is the weather in Lima? Use the get_weather tool.")], maxTokens: 160,
                toolLoop: context(ToolLoopOptions(toolErrors: .failTurn), events), shouldStop: nil) { _ in }
            try await waitUntilIdle(engine)
            print("fail turn: no call; reply \(reply.text)")
            throw XCTSkip("the model made no tool call")
        } catch is FailTool.Failure {
            print("fail turn events: \(events.all)")
            XCTAssertEqual(log.calls.count, 1)
            guard case .finished(let result)? = events.all.last else { return XCTFail("\(events.all)") }
            XCTAssertTrue(result.isError)
        }
        try await waitUntilIdle(engine)
        let next = try await engine.respond(to: Self.probe, maxTokens: 24, shouldStop: nil) { _ in }
        XCTAssertFalse(next.text.isEmpty)
        try await waitUntilIdle(engine)
    }

    func testCancellingDuringAToolStopsTheTurnAndTheNextIsHealthy() async throws {
        let log = CallLog()
        let engine = try await makeToolEngine(tools: [SlowTool(log: log)])
        let events = EventLog()
        let loop = ToolLoopContext(options: .default) { event in
            events.append(event)
            if case .started = event { withUnsafeCurrentTask { $0?.cancel() } }
        }
        let start = ContinuousClock.now
        let turn = Task { @MainActor in
            try await engine.respond(to: [.user("Look up the population of Lima with the lookup tool.")],
                                     maxTokens: 128, toolLoop: loop, shouldStop: nil) { _ in }
        }
        let outcome = await turn.result
        let elapsed = ContinuousClock.now - start
        print("cancel mid-tool: \(events.all), \(elapsed)")
        switch outcome {
        case .success(let reply):
            try await waitUntilIdle(engine)
            print("cancel mid-tool: no call; reply \(reply.text)")
            throw XCTSkip("the model made no tool call")
        case .failure(let error):
            XCTAssertTrue(error is CancellationError, "\(error)")
            XCTAssertEqual(log.calls.count, 1, "the tool started")
            XCTAssertEqual(log.finished, 0, "the tool's sleep was cut")
            guard case .started? = events.all.last else { return XCTFail("\(events.all)") }
        }

        let next = try await engine.respond(to: Self.probe, maxTokens: 24, shouldStop: nil) { _ in }
        XCTAssertFalse(next.text.isEmpty)
        try await waitUntilIdle(engine)
    }

    func testTwentyTurnsAtContext1024WithTwoRounds() async throws {
        let log = CallLog()
        let engine = try await makeToolEngine(tools: [EchoTool(log: log), SecretTool(log: log)], context: 1024)
        let options = ToolLoopOptions(maxToolRounds: 2)
        var history: [LlamaChatMessage] = [.system("Be brief.")]
        var trimmedTurns = 0
        var roundsRun = 0
        let events = EventLog()
        for turn in 0..<20 {
            history.append(.user(Self.request(forTurn: turn)))
            events.removeAll()
            let before = log.calls.count
            let reply = try await engine.respond(to: history, maxTokens: 48, toolLoop: context(options, events),
                                                 shouldStop: nil) { _ in }
            let summary = try XCTUnwrap(reply.tools)
            assertInvariants(summary, events: events, limit: 2, log: log, alreadyRan: before)
            roundsRun += summary.rounds
            if Self.userTurns(in: reply.history) < Self.userTurns(in: history) { trimmedTurns += 1 }
            XCTAssertEqual(reply.history.first, history.first, "trimming keeps the system prompt")
            history.append(.assistant(reply.text))
        }
        print("20 tool turns at 1024: \(trimmedTurns) trimmed, \(roundsRun) rounds, calls \(log.calls)")
        XCTAssertGreaterThan(trimmedTurns, 0, "the history never outgrew 1024 tokens, so trimming went untested")
        XCTAssertEqual(engine.rebuildCount, 0, "every failed prefill was flushed without a new client")
        try await waitUntilIdle(engine)
    }

    // MARK: - Invariants

    /// `alreadyRan` counts tool runs from earlier turns sharing `log`.
    private func assertInvariants(_ turn: ToolTurnSummary, events: EventLog, limit: Int, log: CallLog,
                                  alreadyRan: Int = 0, file: StaticString = #filePath, line: UInt = #line) {
        let cap = max(0, limit)
        XCTAssertLessThanOrEqual(turn.rounds, cap, file: file, line: line)
        if !turn.skipped.isEmpty {
            XCTAssertEqual(turn.rounds, cap, "skips happen only at the limit", file: file, line: line)
        }
        XCTAssertEqual(events.finishedCount, turn.results.count, file: file, line: line)
        // A tool's own code runs only once its arguments decode and the name is known.
        let ran = turn.results.filter { !$0.isError || $0.output.contains("failed:") }.count
        XCTAssertEqual(log.calls.count - alreadyRan, ran, file: file, line: line)
        let rounds = turn.results.map(\.call.round)
        XCTAssertEqual(rounds, rounds.sorted(), "round numbers never decrease", file: file, line: line)
        if let first = rounds.first { XCTAssertEqual(first, 1, file: file, line: line) }
        XCTAssertEqual(Set(turn.skipped.map(\.round)).count, turn.skipped.isEmpty ? 0 : 1, file: file, line: line)
    }

    private func report(_ name: String, reply: String, turn: ToolTurnSummary, events: EventLog, prompt: String) {
        print("""
        ---- \(name): \(turn.rounds) round(s), \(turn.results.count) result(s), \(turn.skipped.count) skipped
        events: \(events.all)
        reply: \(reply)
        last prompt:
        \(prompt)
        ----
        """)
    }

    private static func request(forTurn turn: Int) -> String {
        let fact = facts[turn % facts.count] + padding
        return turn % 4 == 0 ? fact + " Also use the echo tool to repeat the word turn\(turn)." : fact
    }

    private static func userTurns(in history: [LlamaChatMessage]) -> Int {
        history.filter { $0.role == .user }.count
    }

    // MARK: - Helpers

    private func context(_ options: ToolLoopOptions, _ events: EventLog) -> ToolLoopContext {
        ToolLoopContext(options: options) { events.append($0) }
    }

    private func modelURL() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment[Self.modelVariable], !path.isEmpty else {
            throw XCTSkip("Set \(Self.modelVariable) to a local .gguf file to run the GGUF tool loop tests.")
        }
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("\(Self.modelVariable) points at a missing file: \(path)")
        }
        return URL(fileURLWithPath: path)
    }

    /// The engine a load picks for the model's call format.
    private func makeToolEngine(tools: [any LLMTool], context: Int = 4096,
                                renderer custom: LlamaPromptRenderer? = nil) async throws -> any GGUFTurnEngine {
        let url = try modelURL()
        let settings = LlamaClient.Parameter(context: context, batch: min(context, 2048), temperature: 0, topP: 0.95)
        var renderer = await GGUFTurnEngines.renderer(for: url, tools: tools)
        if let custom { renderer = custom }
        let engine = try await GGUFTurnEngines.engine(url: url, parameter: settings, renderer: renderer)
        print("tool engine: \(type(of: engine))")
        return engine
    }

    private func loadModel(tools: [any LLMTool]) async throws -> Tonamox {
        let url = try modelURL()
        let model = try XCTUnwrap(Model.from(.localFile(url)))
        let llm = Tonamox(model: model, sampling: SamplingSettings(temperature: 0, topP: 0.95), tools: tools)
        XCTAssertTrue(llm.engine.backend is LlamaCppBackend)
        try await llm.engine.load { _ in }
        addTeardownBlock { @MainActor in
            llm.engine.unload()
            await (llm.engine.backend as? LlamaCppBackend)?.awaitRelease()
        }
        return llm
    }

    private func toolEngine(of llm: Tonamox) throws -> any GGUFTurnEngine {
        let backend = try XCTUnwrap(llm.engine.backend as? LlamaCppBackend)
        return try XCTUnwrap(backend.turnEngine)
    }

    private func lastPrompt(of engine: any GGUFTurnEngine) throws -> String {
        try engine.renderer.render(engine.lastHistory)
    }

    private func waitUntilIdle(_ engine: any GGUFTurnEngine) async throws {
        try await waitUntil(timeout: .seconds(60)) { engine.isIdle }
    }
}

// MARK: - Fixtures

@MainActor
private final class EventLog {
    private(set) var all: [ToolEvent] = []
    func append(_ event: ToolEvent) { all.append(event) }
    func removeAll() { all.removeAll() }
    var finishedCount: Int {
        all.filter { if case .finished = $0 { true } else { false } }.count
    }
}

private final class CallLog: Sendable {
    private let recorded = Mutex<[String]>([])
    private let completed = Mutex(0)

    var calls: [String] { recorded.withLock { $0 } }
    var finished: Int { completed.withLock { $0 } }
    func record(_ call: String) { recorded.withLock { $0.append(call) } }
    func complete() { completed.withLock { $0 += 1 } }
}

private struct EchoTool: LLMTool {
    let name = "echo"
    let description = "Repeat the given text back."
    let log: CallLog
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let text: String
        static var argumentsSchema: LLMToolArgumentsSchema { ["text": .string(description: "Text to repeat")] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        log.record("echo:\(arguments.text)")
        return ToolOutput(data: ["text": arguments.text])
    }
}

private struct SecretTool: LLMTool {
    let name = "get_secret"
    let description = "Return the secret word."
    let log: CallLog
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        log.record("get_secret")
        return ToolOutput(data: ["secret": "kiwi"])
    }
}

private struct FailTool: LLMTool {
    struct Failure: LocalizedError {
        var errorDescription: String? { "the weather service is offline" }
    }
    let name = "get_weather"
    let description = "Get the current weather for a city."
    let log: CallLog
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let city: String
        static var argumentsSchema: LLMToolArgumentsSchema { ["city": .string(description: "City name")] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        log.record("get_weather:\(arguments.city)")
        throw Failure()
    }
}

private struct SlowTool: LLMTool {
    let name = "lookup"
    let description = "Look up a fact about a city. Takes a few seconds."
    let log: CallLog
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let query: String
        static var argumentsSchema: LLMToolArgumentsSchema { ["query": .string(description: "What to look up")] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        log.record("lookup:\(arguments.query)")
        try await Task.sleep(for: .seconds(2))
        log.complete()
        return ToolOutput(data: ["answer": "about ten million"])
    }
}
