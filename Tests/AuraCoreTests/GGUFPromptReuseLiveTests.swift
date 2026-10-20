import Synchronization
import XCTest
@testable import AuraCore
@testable import LocalLLMClientLlama

/// Needs AURA_GGUF_TEST_MODEL (a small chat GGUF). Each test runs the same turns on an engine that reuses the KV cache
/// and on one that prefills every prompt in full, at temperature 0, and prints what reuse saved.
@MainActor
final class GGUFPromptReuseLiveTests: XCTestCase {

    private static let modelVariable = "AURA_GGUF_TEST_MODEL"
    private static let system = "You are a friendly assistant. Answer in one short sentence."
    private static let questions = [
        "My name is Ximena and I keep bees on a rooftop in Lima.", "How many hives should a beginner start with?",
        "What colour should I paint them?", "When is the best month to harvest honey there?",
        "My neighbour Tomás is worried about stings. What can I tell him?", "Where could I sell the honey?",
        "How should I store it?", "Name one flower bees like.", "What do bees drink?", "What is my name?",
    ]
    private static let padding = " Please answer kindly, keep it practical, and mention one detail I could try this week."

    // MARK: - Tests

    func testReuseMatchesAFullPrefillOverTenTurns() async throws {
        let reusing = try await makeEngine(context: 4096)
        let baseline = try await makeEngine(context: 4096, reuses: false)
        var history: [LlamaChatMessage] = [.system(Self.system)]
        var rows: [Row] = []
        for question in Self.questions {
            history.append(.user(question))
            let fresh = try await measuredTurn(history, on: baseline, maxTokens: 96)
            let reused = try await measuredTurn(history, on: reusing, maxTokens: 96)
            XCTAssertEqual(reused.text, fresh.text, "turn \(rows.count + 1)")
            XCTAssertEqual(reused.promptTokens, fresh.promptTokens, "turn \(rows.count + 1): the cache holds the prompt")
            rows.append(Self.row(rows.count + 1, full: fresh, reused: reused))
            history.append(.assistant(fresh.text))
        }
        print(Row.table(rows, title: "10 turns at 4096, \(try modelURL().lastPathComponent), reuse vs full prefill"))
        XCTAssertEqual(reusing.reusedPrefills, rows.filter(\.reused.wasReused).count)
        XCTAssertGreaterThanOrEqual(reusing.reusedPrefills, Self.questions.count - 1 - rows.filter(\.full.cut).count,
                                    "every turn after one that ended by itself reuses the cache")
        XCTAssertEqual(baseline.reusedPrefills, 0)
    }

    /// The case reuse is for: a long system prompt (about 2,000 tokens of notes) that a full prefill decodes again
    /// on every turn.
    func testALongConversationPrefillsOnlyEachNewTurn() async throws {
        let reusing = try await makeEngine(context: 4096)
        let baseline = try await makeEngine(context: 4096, reuses: false)
        let colours = ["blue", "white", "yellow", "green", "red", "grey", "orange", "pink"]
        let notes = (1...80).map { hive in
            "Note \(hive): hive \(hive) is painted \(colours[hive % colours.count]), holds about "
                + "\(20 + hive % 30) thousand bees and gave \(hive % 9 + 3) kilograms of honey last March."
        }
        let system = "You answer questions about Ximena's rooftop apiary in Lima in one short sentence, using these notes.\n"
            + notes.joined(separator: "\n")
        let questions = ["What colour is hive 7?", "How many bees does hive 12 hold?", "How much honey did hive 30 give?",
                         "Which colour is hive 41?", "What about hive 42?", "How much honey did hive 55 give?",
                         "How many bees are in hive 63?", "What colour is hive 70?", "And hive 71?",
                         "Which hive did I ask about first?"]
        var history: [LlamaChatMessage] = [.system(system)]
        var rows: [Row] = []
        var diverged: [Int] = []
        for question in questions {
            history.append(.user(question))
            let fresh = try await measuredTurn(history, on: baseline, maxTokens: 64)
            let reused = try await measuredTurn(history, on: reusing, maxTokens: 64)
            if reused.text != fresh.text { diverged.append(rows.count + 1) }
            rows.append(Self.row(rows.count + 1, full: fresh, reused: reused))
            history.append(.assistant(fresh.text))
        }
        print(Row.table(rows, title: "10 turns after a ~2k-token system prompt, \(try modelURL().lastPathComponent)")
              + "\nreplies that differ from the full prefill: \(diverged)")
        XCTAssertGreaterThanOrEqual(reusing.reusedPrefills, questions.count - 1 - rows.filter(\.full.cut).count)
        XCTAssertEqual(diverged, [], "see testAReusedPromptTokenizesLikeTheWholePrompt")
    }

    func testACutTurnForcesAFullPrefillAndChangesNothing() async throws {
        let reusing = try await makeEngine(context: 2048)
        let baseline = try await makeEngine(context: 2048, reuses: false)
        var history: [LlamaChatMessage] = [.system(Self.system), .user(Self.questions[0])]
        for (index, budget) in [96, 3, 96, 96].enumerated() {
            let fresh = try await measuredTurn(history, on: baseline, maxTokens: budget)
            let reused = try await measuredTurn(history, on: reusing, maxTokens: budget)
            XCTAssertEqual(reused.text, fresh.text, "turn \(index + 1)")
            switch index {
            case 1: XCTAssertTrue(reused.wasReused, "the turn after one that ended by itself")
            case 2: XCTAssertFalse(reused.wasReused, "the turn after a cut prefills in full")
            default: break
            }
            history += [.assistant(fresh.text), .user(Self.questions[index + 1])]
        }
    }

    func testAStoppedOrCancelledTurnForcesAFullPrefill() async throws {
        let engine = try await makeEngine(context: 2048)
        var history: [LlamaChatMessage] = [.system(Self.system), .user(Self.questions[0])]
        let first = try await engine.respond(to: history, maxTokens: 96, shouldStop: nil) { _ in }
        history += [.assistant(first.text), .user("Write a long story about a lighthouse.")]
        let stopped = try await engine.respond(to: history, maxTokens: 200, shouldStop: { $0 == 5 }) { _ in }
        XCTAssertGreaterThan(engine.lastReusedBytes, 0, "the second turn extended the first")
        history += [.assistant(stopped.text), .user(Self.questions[1])]
        let answered = try await engine.respond(to: history, maxTokens: 96, shouldStop: nil) { _ in }
        XCTAssertEqual(engine.lastReusedBytes, 0, "a turn stopped by shouldStop leaves nothing to reuse")

        history += [.assistant(answered.text), .user("Write another long story.")]
        let call = Task { @MainActor in
            try await engine.respond(to: history, maxTokens: 300, shouldStop: nil) { _ in
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        let cancelled = try await call.value
        XCTAssertGreaterThan(engine.lastReusedBytes, 0, "the cancelled turn extended the one before")
        _ = try await engine.respond(to: history + [.assistant(cancelled.text), .user(Self.questions[2])],
                                     maxTokens: 64, shouldStop: nil) { _ in }
        XCTAssertEqual(engine.lastReusedBytes, 0, "a cancelled turn leaves nothing to reuse")
    }

    /// A turn that trims more of the oldest history than the turn before changes its prompt's start, so it cannot
    /// extend the cache; one that drops the same exchanges again still can.
    func testWindowTrimmingFallsBackToAFullPrefill() async throws {
        let reusing = try await makeEngine(context: 1024)
        let baseline = try await makeEngine(context: 1024, reuses: false)
        var history: [LlamaChatMessage] = [.system(Self.system)]
        var reusedTurns = 0
        var trimmedTurns = 0
        var dropped = 0
        var diverged: [Int] = []
        for turn in 0..<22 {
            history.append(.user(Self.questions[turn % Self.questions.count] + Self.padding))
            let fresh = try await baseline.respond(to: history, maxTokens: 48, shouldStop: nil) { _ in }
            let reply = try await reusing.respond(to: history, maxTokens: 48, shouldStop: nil) { _ in }
            XCTAssertFalse(reply.text.isEmpty, "turn \(turn + 1) returned nothing")
            let droppedNow = history.count - reply.history.count
            if droppedNow > 0 { trimmedTurns += 1 }
            let reused = reusing.lastReusedBytes > 0
            if reused { reusedTurns += 1 }
            XCTAssertFalse(droppedNow > dropped && reused, "turn \(turn + 1) trimmed further and still reused")
            dropped = droppedNow
            if reply.text != fresh.text { diverged.append(turn + 1) }
            // Both continue from the same history, so a divergence does not carry over.
            history.append(.assistant(fresh.text))
        }
        print("22 turns at 1024: \(reusedTurns) reused, \(trimmedTurns) trimmed, "
              + "replies that differ from the full prefill: \(diverged)")
        XCTAssertGreaterThan(trimmedTurns, 0, "the history never outgrew 1024 tokens, so trimming went untested")
        XCTAssertGreaterThan(reusedTurns, 0)
        XCTAssertEqual(reusing.rebuildCount, 0, "no turn needed a new context")
    }

    /// The boundary argument, checked with the model's own tokenizer: each turn's prompt tokenizes to the turn
    /// before's prompt, then the reply's tokens, then the new part on its own. Different replies with reuse can then
    /// only come from llama.cpp computing the reply's cache one token at a time instead of in one batch.
    func testAReusedPromptTokenizesLikeTheWholePrompt() async throws {
        try await assertReusedPromptsTokenizeLikeWholePrompts(context: 2048, maxTokens: 96, padding: "", turns: 6)
        // The first prompts of testWindowTrimmingFallsBackToAFullPrefill, whose turn 5 can reply differently.
        try await assertReusedPromptsTokenizeLikeWholePrompts(context: 1024, maxTokens: 48, padding: Self.padding,
                                                              turns: 5)
    }

    private func assertReusedPromptsTokenizeLikeWholePrompts(context size: Int, maxTokens: Int, padding: String,
                                                             turns: Int) async throws {
        let engine = try await makeEngine(context: size)
        let context = try XCTUnwrap(engine.contextForTesting)
        var history: [LlamaChatMessage] = [.system(Self.system)]
        var held: (text: String, tokens: [Int32])?
        let reuse = GGUFPromptReuse(boundaries: ["<|im_end|>", "<end_of_turn>", "<|eot_id|>"])
        for question in Self.questions.prefix(turns) {
            history.append(.user(question + padding))
            let prompt = try engine.renderer.render(history)
            let whole = Self.tokens(of: prompt, in: context)
            if let held {
                let suffix = try XCTUnwrap(reuse.suffix(of: prompt, after: held.text),
                                           "the prompt extends the last one and its reply")
                XCTAssertEqual(whole, held.tokens + Self.tokens(of: suffix, in: context), question)
            }
            var pieces: [String] = []
            let reply = try await engine.respond(to: history, maxTokens: maxTokens, shouldStop: nil) { pieces.append($0) }
            XCTAssertLessThan(pieces.count, maxTokens, "the reply ended by itself")
            let replyTokens = pieces.flatMap { Self.tokens(of: $0, in: context) }
            XCTAssertEqual(replyTokens.count, pieces.count, "each piece is one sampled token's text")
            try await engine.waitForDecoder()
            XCTAssertEqual(Self.cachedTokens(in: context), whole.count + pieces.count, "the cache holds the prompt and reply")
            held = (prompt + reply.text, whole + replyTokens)
            history.append(.assistant(reply.text))
        }
    }

    func testToolRoundsReuseTheCacheWhileRoundsRemain() async throws {
        let log = CallLog()
        let tools: [any LLMTool] = [SecretTool(log: log), EchoTool(log: log)]
        let reusing = try await makeEngine(context: 4096, tools: tools)
        let baseline = try await makeEngine(context: 4096, tools: tools, reuses: false)
        let request: [LlamaChatMessage] = [.user("First call get_secret to learn the secret word. "
                                                 + "Then call echo with that word. Finally tell me the word.")]
        let options = ToolLoopOptions(maxToolRounds: 3)
        let fresh = try await timed { try await baseline.respond(to: request, maxTokens: 256,
                                                                   toolLoop: ToolLoopContext(options: options, onEvent: nil),
                                                                   shouldStop: nil) { _ in } }
        let freshCalls = log.calls
        let reused = try await timed { try await reusing.respond(to: request, maxTokens: 256,
                                                                   toolLoop: ToolLoopContext(options: options, onEvent: nil),
                                                                   shouldStop: nil) { _ in } }
        let turn = try XCTUnwrap(reused.value.tools)
        print("""
        ---- tool chain: \(turn.rounds) round(s), calls \(log.calls), reused prefills \(reusing.reusedPrefills)
        full prefill \(fresh.elapsed), reuse \(reused.elapsed)
        reply: \(reused.value.text)
        last prompt:
        \(reused.value.prompt)
        ----
        """)
        XCTAssertEqual(reused.value.text, fresh.value.text)
        XCTAssertEqual(reused.value.tools, fresh.value.tools.map { Self.withIDs(of: turn, applied: $0) })
        XCTAssertEqual(Array(log.calls.dropFirst(freshCalls.count)), freshCalls)
        guard turn.rounds > 0 else { throw XCTSkip("the model made no tool call") }
        // The reply after the last allowed round drops the tool instructions, so its prompt changes and misses.
        let expected = turn.rounds < options.maxToolRounds ? turn.rounds : turn.rounds - 1
        XCTAssertEqual(reusing.reusedPrefills, expected)
        XCTAssertEqual(baseline.reusedPrefills, 0)
    }

    // MARK: - Measuring

    private struct Turn {
        let text: String
        let elapsed: Duration
        let firstPiece: Duration
        /// Tokens in the KV cache once the prompt was decoded, and how many of them this turn decoded.
        let promptTokens: Int
        let decodedTokens: Int
        let wasReused: Bool
        let cut: Bool
    }

    private struct Row {
        let index: Int
        let full: Turn
        let reused: Turn

        static func table(_ rows: [Row], title: String) -> String {
            let header = "turn | prompt tok | prefilled full / reuse | wall full / reuse (ms) | first piece full / reuse (ms)"
            let lines = rows.map { row in
                "\(row.index) | \(row.full.promptTokens) | \(row.full.decodedTokens) / \(row.reused.decodedTokens) | "
                    + "\(milliseconds(row.full.elapsed)) / \(milliseconds(row.reused.elapsed)) | "
                    + "\(milliseconds(row.full.firstPiece)) / \(milliseconds(row.reused.firstPiece))"
            }
            let saved = rows.map { $0.full.decodedTokens - $0.reused.decodedTokens }.reduce(0, +)
            let full = rows.map { milliseconds($0.full.elapsed) }.reduce(0, +)
            let reused = rows.map { milliseconds($0.reused.elapsed) }.reduce(0, +)
            return ([title, header] + lines + ["prefill tokens saved: \(saved); wall: \(full) ms full, \(reused) ms reuse"])
                .joined(separator: "\n")
        }

        private static func milliseconds(_ duration: Duration) -> Int {
            Int(duration.components.seconds * 1000 + duration.components.attoseconds / 1_000_000_000_000_000)
        }
    }

    private static func row(_ index: Int, full: Turn, reused: Turn) -> Row {
        Row(index: index, full: full, reused: reused)
    }

    /// Token counts come from the KV position: after a turn that ended by itself it holds the prompt and one token
    /// per piece.
    private func measuredTurn(_ history: [LlamaChatMessage], on engine: LlamaTurnEngine,
                              maxTokens: Int) async throws -> Turn {
        try await engine.waitForDecoder()
        let before = Int(try XCTUnwrap(engine.contextForTesting?.position))
        let clock = ContinuousClock()
        let start = clock.now
        var first: Duration?
        var pieces = 0
        let reply = try await engine.respond(to: history, maxTokens: maxTokens, shouldStop: nil) { _ in
            if first == nil { first = clock.now - start }
            pieces += 1
        }
        let elapsed = clock.now - start
        try await engine.waitForDecoder()
        let after = Int(try XCTUnwrap(engine.contextForTesting?.position))
        let reused = engine.lastReusedBytes > 0
        let cut = pieces >= maxTokens
        // A cut leaves its last token queued, not in the cache.
        let prompt = after - pieces + (cut ? 1 : 0)
        return Turn(text: reply.text, elapsed: elapsed, firstPiece: first ?? elapsed, promptTokens: prompt,
                    decodedTokens: reused ? prompt - before : prompt, wasReused: reused, cut: cut)
    }

    private static func cachedTokens(in context: LocalLLMClientLlama.Context) -> Int {
        Int(context.position)
    }

    private static func tokens(of text: String, in context: LocalLLMClientLlama.Context) -> [Int32] {
        [Int32](text, addBos: false, special: true, vocab: context.vocab)
    }

    private func timed<T>(_ body: () async throws -> T) async throws -> (value: T, elapsed: Duration) {
        let clock = ContinuousClock()
        let start = clock.now
        let value = try await body()
        return (value, clock.now - start)
    }

    /// Call ids are fresh UUIDs per parse, so a summary compares with another once it carries the same ids.
    private static func withIDs(of reference: ToolTurnSummary, applied other: ToolTurnSummary) -> ToolTurnSummary {
        let ids = reference.results.map(\.call.id) + reference.skipped.map(\.id)
        let results = zip(other.results, ids).map { result, id in
            ToolCallResult(call: ToolInvocation(id: id, round: result.call.round, name: result.call.name,
                                                arguments: result.call.arguments),
                           output: result.output, isError: result.isError)
        }
        let skipped = zip(other.skipped, ids.dropFirst(other.results.count)).map { call, id in
            ToolInvocation(id: id, round: call.round, name: call.name, arguments: call.arguments)
        }
        return ToolTurnSummary(rounds: other.rounds, results: results, skipped: skipped)
    }

    // MARK: - Helpers

    private func modelURL() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment[Self.modelVariable], !path.isEmpty else {
            throw XCTSkip("Set \(Self.modelVariable) to a local .gguf file to run the GGUF prompt reuse tests.")
        }
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("\(Self.modelVariable) points at a missing file: \(path)")
        }
        return URL(fileURLWithPath: path)
    }

    private func makeEngine(context: Int, tools: [any LLMTool] = [], reuses: Bool = true) async throws
        -> LlamaTurnEngine {
        let url = try modelURL()
        let parameter = LlamaClient.Parameter(context: context, batch: min(context, 2048), temperature: 0, topP: 0.95)
        let renderer = await GGUFTurnEngines.renderer(for: url, tools: tools)
        guard let engine = try await GGUFTurnEngines.engine(url: url, parameter: parameter, renderer: renderer)
            as? LlamaTurnEngine else {
            throw XCTSkip("this model's tool-call format runs on LocalLLMClient's client, which does not reuse the cache")
        }
        engine.reusesPromptCache = reuses
        return engine
    }
}

// MARK: - Fixtures

private final class CallLog: Sendable {
    private let recorded = Mutex<[String]>([])

    var calls: [String] { recorded.withLock { $0 } }
    func record(_ call: String) { recorded.withLock { $0.append(call) } }
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
