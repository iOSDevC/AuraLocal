#if canImport(MLXLLM) && !targetEnvironment(simulator)
import Synchronization
import XCTest
@testable import TonamoxCore

/// Needs AURA_MLX_TEST_MODEL: the catalog id or repo id of an MLX text model with a tool-calling chat template
/// (e.g. `mlx-community/Qwen3-0.6B-4bit`). The first run downloads it into the storage root. A small model's choices
/// vary, so the tool tests assert the loop's invariants, print their transcripts, and skip the model-dependent checks
/// when no tool was called. Prompts end in `/no_think` so Qwen3 skips its reasoning.
@MainActor
final class MLXToolLoopLiveTests: XCTestCase {

    private static let modelVariable = "AURA_MLX_TEST_MODEL"
    private static let greedy = SamplingSettings(temperature: 0, topP: 1)
    private static let facts = [
        "My name is Ximena and I keep bees on a rooftop in Lima.", "I have three hives.",
        "The hives are painted blue.", "I harvest honey in March.", "My neighbour is called Tomás.",
        "What colour are my hives?", "How many hives do I have?", "When do I harvest?",
    ]

    // MARK: - Tool loop

    func testOneRoundThroughTheTonamoxAPI() async throws {
        let log = CallLog()
        let llm = try await loadModel(tools: [EchoTool(log: log)])
        let events = EventLog()
        llm.onToolEvent = { events.append($0) }

        let reply = try await llm.chat("Use the echo tool to repeat the word pineapple. /no_think", maxTokens: 256)
        let turn = try XCTUnwrap(llm.lastToolTurn, "an MLX instance with tools reports its tool turn")
        report("one round", reply: reply, turn: turn, events: events)
        assertInvariants(turn, events: events, limit: 1, log: log)
        XCTAssertNotNil(llm.lastGenerationStats)
        guard let first = turn.results.first else { throw XCTSkip("the model made no tool call") }

        XCTAssertEqual(log.calls, ["echo:pineapple"], reply)
        XCTAssertEqual(first.output, #"{"text":"pineapple"}"#)
        XCTAssertEqual(turn.rounds, 1)
        XCTAssertEqual(events.all.first, .started(first.call))
        XCTAssertFalse(reply.contains("<tool_call>"), "the parser takes the call out of the visible text")
        let next = try await llm.chat("Name one primary color. /no_think", systemPrompt: "Be brief.", maxTokens: 64)
        XCTAssertFalse(next.isEmpty)
    }

    func testTwoRoundsChainTheirResults() async throws {
        let log = CallLog()
        let llm = try await loadModel(tools: [CityIDTool(log: log), WeatherTool(log: log)])
        llm.toolLoop = ToolLoopOptions(maxToolRounds: 3)
        let events = EventLog()
        llm.onToolEvent = { events.append($0) }

        let reply = try await llm.chat(Self.chainPrompt, maxTokens: 768)
        let turn = try XCTUnwrap(llm.lastToolTurn)
        report("two rounds", reply: reply, turn: turn, events: events)
        assertInvariants(turn, events: events, limit: 3, log: log)
        guard turn.rounds >= 2 else { throw XCTSkip("the model ran \(turn.rounds) round(s), not a chain") }

        XCTAssertEqual(turn.results.first?.call.round, 1)
        XCTAssertEqual(turn.results.first?.output, #"{"id":7}"#)
        let second = try XCTUnwrap(turn.results.first { $0.call.round == 2 })
        XCTAssertEqual(second.call.name, "weather")
        XCTAssertEqual(second.call.arguments, #"{"id":7}"#, "round 2 used round 1's result")
        XCTAssertEqual(second.output, #"{"forecast":"sunny"}"#)
    }

    func testTheDefaultLimitReportsALaterRoundAsSkipped() async throws {
        let log = CallLog()
        let llm = try await loadModel(tools: [CityIDTool(log: log), WeatherTool(log: log)])
        let events = EventLog()
        llm.onToolEvent = { events.append($0) }

        let reply = try await llm.chat(Self.chainPrompt, maxTokens: 768)
        let turn = try XCTUnwrap(llm.lastToolTurn)
        report("round limit", reply: reply, turn: turn, events: events)
        assertInvariants(turn, events: events, limit: 1, log: log)
        guard !turn.skipped.isEmpty else { throw XCTSkip("the model asked for no second round") }

        XCTAssertEqual(turn.rounds, 1)
        XCTAssertEqual(events.all.last, .roundLimitReached(skipped: turn.skipped))
        XCTAssertEqual(Set(turn.skipped.map(\.round)), [2])
        XCTAssertEqual(log.calls.count, turn.results.count, "skipped calls never run")
    }

    func testZeroRoundsReportsTheCallsAsSkipped() async throws {
        let log = CallLog()
        let llm = try await loadModel(tools: [EchoTool(log: log)])
        llm.toolLoop = ToolLoopOptions(maxToolRounds: 0)
        let events = EventLog()
        llm.onToolEvent = { events.append($0) }

        let reply = try await llm.chat("Use the echo tool to repeat the word pineapple. /no_think", maxTokens: 256)
        let turn = try XCTUnwrap(llm.lastToolTurn)
        report("zero rounds", reply: reply, turn: turn, events: events)
        assertInvariants(turn, events: events, limit: 0, log: log)
        guard !turn.skipped.isEmpty else { throw XCTSkip("the model made no tool call") }

        XCTAssertEqual(turn.rounds, 0)
        XCTAssertEqual(log.calls, [], "nothing runs at limit 0")
        XCTAssertEqual(events.all, [.roundLimitReached(skipped: turn.skipped)])
        XCTAssertEqual(Set(turn.skipped.map(\.round)), [1])
    }

    func testAThrowingToolIsReportedAndTheTurnAnswers() async throws {
        let log = CallLog()
        let llm = try await loadModel(tools: [FailTool(log: log)])
        let events = EventLog()
        llm.onToolEvent = { events.append($0) }

        let reply = try await llm.chat("What is the weather in Lima? Use the get_weather tool. /no_think",
                                       maxTokens: 320)
        let turn = try XCTUnwrap(llm.lastToolTurn)
        report("throwing tool", reply: reply, turn: turn, events: events)
        assertInvariants(turn, events: events, limit: 1, log: log)
        guard let first = turn.results.first else { throw XCTSkip("the model made no tool call") }

        XCTAssertTrue(first.isError)
        XCTAssertEqual(first.output, #"{"error":"Tool \"get_weather\" failed: the weather service is offline"}"#)
        XCTAssertTrue(!reply.isEmpty || !turn.skipped.isEmpty, "the turn went on after the failure")
    }

    func testFailTurnRethrowsTheToolsError() async throws {
        let log = CallLog()
        let llm = try await loadModel(tools: [FailTool(log: log)])
        llm.toolLoop = ToolLoopOptions(toolErrors: .failTurn)
        let events = EventLog()
        llm.onToolEvent = { events.append($0) }
        do {
            let reply = try await llm.chat("What is the weather in Lima? Use the get_weather tool. /no_think",
                                           maxTokens: 320)
            print("fail turn: no call; reply \(reply)")
            throw XCTSkip("the model made no tool call")
        } catch is FailTool.Failure {
            print("fail turn events: \(events.all)")
            XCTAssertEqual(log.calls.count, 1)
            guard case .finished(let result)? = events.all.last else { return XCTFail("\(events.all)") }
            XCTAssertTrue(result.isError)
            XCTAssertNil(llm.lastToolTurn, "a turn that threw reports no summary")
        }
        let next = try await llm.chat("Name one primary color. /no_think", maxTokens: 64)
        XCTAssertFalse(next.isEmpty)
    }

    func testCancellingDuringAToolStopsTheTurnAndTheNextIsHealthy() async throws {
        let log = CallLog()
        let llm = try await loadModel(tools: [SlowTool(log: log)])
        let events = EventLog()
        llm.onToolEvent = { event in
            events.append(event)
            if case .started = event { withUnsafeCurrentTask { $0?.cancel() } }
        }
        let start = ContinuousClock.now
        let turn = Task { @MainActor in
            try await llm.chat("Look up the population of Lima with the lookup tool. /no_think", maxTokens: 256)
        }
        let outcome = await turn.result
        print("cancel mid-tool: \(events.all), \(ContinuousClock.now - start)")
        switch outcome {
        case .success(let reply):
            print("cancel mid-tool: no call; reply \(reply)")
            throw XCTSkip("the model made no tool call")
        case .failure(let error):
            XCTAssertTrue(error is CancellationError, "\(error)")
            XCTAssertEqual(log.calls.count, 1, "the tool started")
            XCTAssertEqual(log.finished, 0, "the tool's sleep was cut")
            guard case .started? = events.all.last else { return XCTFail("\(events.all)") }
        }
        llm.onToolEvent = nil
        let next = try await llm.chat("Name one primary color. /no_think", maxTokens: 64)
        XCTAssertFalse(next.isEmpty)
    }

    func testACallCancelledBeforeItStartsThrowsWithoutDecoding() async throws {
        let log = CallLog()
        let agent = try await loadBackend(tools: [EchoTool(log: log)])
        let plain = try XCTUnwrap(agent.sharingWeights(sampling: Self.greedy, tools: []))
        let nothing = MLXPromptCache.ReuseStats()
        for backend in [agent, plain] {
            let outcome = await cancelledBeforeItStarts(backend)
            XCTAssertThrowsError(try outcome.get()) { XCTAssertTrue($0 is CancellationError, "\($0)") }
            XCTAssertEqual(backend.promptCacheStats, nothing, "nothing was prefilled")
            XCTAssertNil(backend.lastGenerationStats)
        }
        XCTAssertNil(agent.lastToolTurn)
        XCTAssertEqual(log.calls, [])
    }

    func testAToolTurnCancelledWhileQueuedBehindAnotherGenerationThrowsWithoutDecoding() async throws {
        let log = CallLog()
        let plain = try await loadBackend()
        let agent = try XCTUnwrap(plain.sharingWeights(sampling: Self.greedy, tools: [EchoTool(log: log)]))
        let progress = BusyProgress()
        let busy = Task { @MainActor in
            defer { progress.finished = true }
            return try await plain.generate(prompt: "Count from one to one hundred in words. /no_think",
                                            systemPrompt: nil, maxTokens: 200) { _ in progress.started = true }
        }
        for _ in 0..<2_000 where !progress.started {
            try await Task.sleep(for: .milliseconds(5))
        }
        let queued = Task { @MainActor in
            try await agent.generate(prompt: "Use the echo tool to repeat the word pineapple. /no_think",
                                     systemPrompt: nil, maxTokens: 64) { _ in }
        }
        try await Task.sleep(for: .milliseconds(50))
        let busyWhenCancelled = progress.started && !progress.finished
        queued.cancel()
        let outcome = await queued.result
        let busyReply = try await busy.value
        print("cancel while queued: busy \(busyWhenCancelled); queued \(outcome); cache \(agent.promptCacheStats)")
        guard busyWhenCancelled else { throw XCTSkip("the other generation was not running at the cancel") }

        XCTAssertThrowsError(try outcome.get()) { XCTAssertTrue($0 is CancellationError, "\($0)") }
        XCTAssertEqual(agent.promptCacheStats, MLXPromptCache.ReuseStats(), "the queued turn prefilled nothing")
        XCTAssertNil(agent.lastToolTurn)
        XCTAssertEqual(log.calls, [])
        XCTAssertFalse(busyReply.isEmpty)
    }

    func testModelManagerRunsAToolInstanceOnTheCachedWeights() async throws {
        let model = try testModel()
        let plain = try await ModelManager.shared.load(model, temperature: 0)
        defer { ModelManager.shared.evict(model) }
        let log = CallLog()
        let agent = try await ModelManager.shared.load(model, tools: [EchoTool(log: log)], temperature: 0)
        defer { ModelManager.shared.release(agent) }
        let plainBackend = try XCTUnwrap(plain.engine.backend as? MLXBackend)
        let agentBackend = try XCTUnwrap(agent.engine.backend as? MLXBackend)

        XCTAssertTrue(agentBackend.sharesWeights(with: plainBackend), "the tool instance loaded no second copy")
        XCTAssertEqual(agentBackend.registeredToolNames, ["echo"])
        XCTAssertEqual(plainBackend.registeredToolNames, [])
        let reply = try await agent.chat("Use the echo tool to repeat the word pineapple. /no_think", maxTokens: 256)
        print("shared weights: \(reply) \(log.calls)")
        XCTAssertNotNil(agent.lastToolTurn)
        let plainReply = try await plain.chat("Name one primary color. /no_think", maxTokens: 64)
        XCTAssertFalse(plainReply.isEmpty)
    }

    // MARK: - Prompt cache reuse

    /// When every reused token was prefilled, a hit leaves the cache exactly as a full prefill split at the reused
    /// token would: the reply is bit-identical.
    /// An unsplit full prefill can still differ, as any change of prefill split can (fp16 rounding), so the exact
    /// reference reproduces the split with `prefillStepSize`.
    func testAHitMatchesAFullPrefillSplitAtTheSameToken() async throws {
        let reusing = try await loadBackend()
        let first: [[String: String]] = [["role": "system", "content": "Answer in one short sentence."],
                                         ["role": "user", "content": "I keep bees on a rooftop in Lima. /no_think"]]
        _ = try await reusing.generate(messages: first, maxTokens: 1) { _ in }
        let cached = try XCTUnwrap(reusing.lastGenerationStats).promptTokens
        let second = first + [["role": "assistant", "content": "That sounds lovely."],
                              ["role": "user", "content": "How many hives should I start with? /no_think"]]
        let reply = try await reusing.generate(messages: second, maxTokens: 64) { _ in }
        let reuse = reusing.promptCacheStats

        let split = try XCTUnwrap(reusing.sharingWeights(sampling: Self.greedy, tools: []))
        split.reusesPromptCache = false
        split.parameters.prefillStepSize = cached
        let expected = try await split.generate(messages: second, maxTokens: 64) { _ in }
        let prompt = try XCTUnwrap(split.lastGenerationStats).promptTokens
        print("split at \(cached) of \(prompt) tokens; cache \(reuse)\nreply: \(reply)")

        XCTAssertEqual(reuse.hits, 1)
        XCTAssertEqual(reuse.reusedTokens, cached, "the second prompt reused exactly the first (its reply was trimmed)")
        XCTAssertLessThanOrEqual(prompt - cached, cached, "the reference prefills in the same two pieces")
        XCTAssertEqual(reply, expected)
    }

    /// A multi-turn conversation: the hit rate and the prefill time it saves against a backend that never reuses.
    func testReusingThePromptCacheAcrossAConversation() async throws {
        let full = try await loadBackend()
        full.reusesPromptCache = false
        let reusing = try XCTUnwrap(full.sharingWeights(sampling: Self.greedy, tools: []))
        var history: [[String: String]] = [["role": "system", "content": "Answer in one short sentence."]]
        var fullPrefill: TimeInterval = 0
        var reusedPrefill: TimeInterval = 0
        var sameReplies = 0
        for (turn, fact) in Self.facts.enumerated() {
            history.append(["role": "user", "content": fact + " /no_think"])
            let expected = try await full.generate(messages: history, maxTokens: 48) { _ in }
            let fullStats = try XCTUnwrap(full.lastGenerationStats)
            let reply = try await reusing.generate(messages: history, maxTokens: 48) { _ in }
            let stats = try XCTUnwrap(reusing.lastGenerationStats)
            XCTAssertEqual(stats.promptTokens, fullStats.promptTokens, "the whole prompt is counted either way")
            if turn == 0 { XCTAssertEqual(reply, expected, "a miss is a full prefill") }
            if reply == expected { sameReplies += 1 }
            fullPrefill += fullStats.promptTime
            reusedPrefill += stats.promptTime
            print(Self.turnLine(turn, full: fullStats, reusing: stats, same: reply == expected))
            // Both continue from the full prefill's reply, so every turn compares the same prompt.
            history.append(["role": "assistant", "content": expected])
        }
        let reuse = reusing.promptCacheStats
        let hitRate = Double(reuse.hits) / Double(max(1, reuse.hits + reuse.misses))
        let reusedShare = Double(reuse.reusedTokens) / Double(max(1, reuse.reusedTokens + reuse.prefilledTokens))
        print(String(format: """
            prompt cache over %d turns: %d hits, %d misses (%.0f%%), %d tokens reused, %d prefilled (%.0f%% reused); \
            prefill %.1f ms full vs %.1f ms reusing (saved %.1f ms); %d of %d greedy replies identical
            """, Self.facts.count, reuse.hits, reuse.misses, hitRate * 100, reuse.reusedTokens,
                     reuse.prefilledTokens, reusedShare * 100, fullPrefill * 1000, reusedPrefill * 1000,
                     (fullPrefill - reusedPrefill) * 1000, sameReplies, Self.facts.count))
        XCTAssertEqual(reuse.misses, 1, "only the first prompt, with nothing cached, misses")
        XCTAssertEqual(reuse.hits, Self.facts.count - 1)
        XCTAssertGreaterThan(reuse.reusedTokens, reuse.prefilledTokens)
        XCTAssertEqual(full.promptCacheStats, MLXPromptCache.ReuseStats(), "the full backend never leases a cache")
    }

    /// A prompt that leaves the cached conversation early reuses only the tokens it shares.
    func testADivergingPromptReusesOnlyItsSharedPrefix() async throws {
        let reusing = try await loadBackend()
        let first: [[String: String]] = [["role": "system", "content": "Answer in one short sentence."],
                                         ["role": "user", "content": "Name a fruit. /no_think"]]
        let other: [[String: String]] = [["role": "system", "content": "You are a poet."],
                                         ["role": "user", "content": "Name a colour. /no_think"]]
        _ = try await reusing.generate(messages: first, maxTokens: 32) { _ in }
        let before = reusing.promptCacheStats
        let reply = try await reusing.generate(messages: other, maxTokens: 32) { _ in }
        let after = reusing.promptCacheStats
        let prompt = try XCTUnwrap(reusing.lastGenerationStats).promptTokens
        let reused = after.reusedTokens - before.reusedTokens
        print("diverging prompt: reused \(reused) of \(prompt) tokens; reply \(reply)")

        XCTAssertGreaterThan(reused, 0, "the template's opening tokens are shared")
        XCTAssertLessThan(reused, 8, "nothing after the system turn's first words is shared")
        XCTAssertEqual(after.prefilledTokens - before.prefilledTokens, prompt - reused)
        XCTAssertFalse(reply.isEmpty)
    }

    /// The reply after a round prefills only what the round added to the prompt.
    func testToolRoundsReuseThePromptOfTheReplyBefore() async throws {
        let log = CallLog()
        let reusing = try await loadBackend(tools: [EchoTool(log: log)])
        let reply = try await reusing.generate(prompt: "Use the echo tool to repeat the word pineapple. /no_think",
                                               systemPrompt: nil, maxTokens: 256) { _ in }
        let turn = try XCTUnwrap(reusing.lastToolTurn)
        let reuse = reusing.promptCacheStats
        print("tool rounds: \(turn.rounds) round(s); cache \(reuse); reply \(reply)")
        guard turn.rounds == 1 else { throw XCTSkip("the model made no tool call") }

        XCTAssertEqual(reuse.misses, 1, "only the first reply's prompt finds nothing cached")
        XCTAssertEqual(reuse.hits, 1, "the reply after the round reused the first prompt")
        XCTAssertGreaterThan(reuse.reusedTokens, reuse.prefilledTokens / 2,
                             "the first prompt is most of the second: \(reuse)")
    }

    // MARK: - Invariants

    private func assertInvariants(_ turn: ToolTurnSummary, events: EventLog, limit: Int, log: CallLog,
                                  file: StaticString = #filePath, line: UInt = #line) {
        let cap = max(0, limit)
        XCTAssertLessThanOrEqual(turn.rounds, cap, file: file, line: line)
        if !turn.skipped.isEmpty {
            XCTAssertEqual(turn.rounds, cap, "skips happen only at the limit", file: file, line: line)
        }
        XCTAssertEqual(events.finishedCount, turn.results.count, file: file, line: line)
        // A tool's own code runs only once its arguments decode and the name is known.
        let ran = turn.results.filter { !$0.isError || $0.output.contains("failed:") }.count
        XCTAssertEqual(log.calls.count, ran, file: file, line: line)
        let rounds = turn.results.map(\.call.round)
        XCTAssertEqual(rounds, rounds.sorted(), "round numbers never decrease", file: file, line: line)
        if let first = rounds.first { XCTAssertEqual(first, 1, file: file, line: line) }
        XCTAssertEqual(Set(turn.skipped.map(\.round)).count, turn.skipped.isEmpty ? 0 : 1, file: file, line: line)
    }

    private func cancelledBeforeItStarts(_ backend: MLXBackend) async -> Result<String, any Error> {
        let call = Task { @MainActor in
            try await backend.generate(prompt: "Use the echo tool to repeat the word pineapple. /no_think",
                                       systemPrompt: nil, maxTokens: 64) { _ in }
        }
        call.cancel()
        return await call.result
    }

    private func report(_ name: String, reply: String, turn: ToolTurnSummary, events: EventLog) {
        print("""
        ---- \(name): \(turn.rounds) round(s), \(turn.results.count) result(s), \(turn.skipped.count) skipped
        events: \(events.all)
        reply: \(reply)
        ----
        """)
    }

    private static func turnLine(_ turn: Int, full: GenerationStats, reusing: GenerationStats, same: Bool) -> String {
        String(format: "turn %d: %d prompt tokens, prefill %.1f ms full vs %.1f ms reusing%@", turn,
               reusing.promptTokens, full.promptTime * 1000, reusing.promptTime * 1000,
               same ? "" : " (greedy reply differs)")
    }

    private static let chainPrompt = "What is the weather in Lima? The weather tool needs the city id, so call "
        + "lookup_city_id first. /no_think"

    // MARK: - Helpers

    private func testModel() throws -> Model {
        guard let value = ProcessInfo.processInfo.environment[Self.modelVariable], !value.isEmpty else {
            throw XCTSkip("Set \(Self.modelVariable) to an MLX text model's catalog id or repo id "
                          + "(e.g. mlx-community/Qwen3-0.6B-4bit) to run the MLX tool loop tests.")
        }
        guard let model = ModelRegistry.shared.models.first(where: {
            $0.format == .mlx && $0.category == .text && ($0.id == value || $0.repoID == value)
        }) else {
            throw XCTSkip("\(Self.modelVariable)=\(value) is not an MLX text model in the catalog")
        }
        return model
    }

    private func loadModel(tools: [any LLMTool]) async throws -> Tonamox {
        let llm = Tonamox(model: try testModel(), sampling: Self.greedy, tools: tools)
        XCTAssertTrue(llm.engine.backend is MLXBackend)
        try await llm.engine.load { _ in }
        addTeardownBlock { @MainActor in llm.engine.unload() }
        return llm
    }

    private func loadBackend(tools: [any LLMTool] = []) async throws -> MLXBackend {
        let backend = MLXBackend(model: try testModel(), sampling: Self.greedy, tools: tools)
        try await backend.load { _ in }
        addTeardownBlock { @MainActor in backend.unload() }
        return backend
    }
}

// MARK: - Fixtures

@MainActor
private final class EventLog {
    private(set) var all: [ToolEvent] = []
    func append(_ event: ToolEvent) { all.append(event) }
    var finishedCount: Int {
        all.filter { if case .finished = $0 { true } else { false } }.count
    }
}

@MainActor
private final class BusyProgress {
    var started = false
    var finished = false
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

private struct CityIDTool: LLMTool {
    let name = "lookup_city_id"
    let description = "Return the numeric id of a city. The weather tool needs it."
    let log: CallLog
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let city: String
        static var argumentsSchema: LLMToolArgumentsSchema { ["city": .string(description: "City name")] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        log.record("lookup_city_id:\(arguments.city)")
        return ToolOutput(data: ["id": 7])
    }
}

private struct WeatherTool: LLMTool {
    let name = "weather"
    let description = "Return the forecast for a city id from lookup_city_id."
    let log: CallLog
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let id: Int
        static var argumentsSchema: LLMToolArgumentsSchema { ["id": .integer(description: "City id")] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        log.record("weather:\(arguments.id)")
        return ToolOutput(data: ["forecast": "sunny"])
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
#endif
