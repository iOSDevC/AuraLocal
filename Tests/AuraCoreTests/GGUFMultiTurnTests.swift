import Synchronization
import XCTest
@testable import AuraCore
@testable import LocalLLMClientLlama

/// Needs AURA_GGUF_TEST_MODEL (a small chat GGUF). Before each turn started from an empty KV cache, the cache grew
/// every turn until the context overflowed (around turn 8 at 1024 tokens) and the process aborted a few turns later.
@MainActor
final class GGUFMultiTurnTests: XCTestCase {

    private static let modelVariable = "AURA_GGUF_TEST_MODEL"

    private static let facts = [
        "My name is Ximena and I keep bees on a rooftop in Lima.",
        "I have three hives.", "The hives are painted blue.", "I harvest honey in March.",
        "My neighbour is called Tomás.", "I sell honey at a Saturday market.",
    ]
    private static let padding = " Please answer kindly, keep it practical, and mention one detail I could try this week."
    private static let probe: [LlamaChatMessage] = [.system("Be brief."), .user("Name one primary color.")]

    // MARK: - Parity with LocalLLMClient

    #if DEBUG
    /// LocalLLMClient's renderer helper exists only in debug builds.
    func testTemplateTokensAndRenderingMatchLocalLLMClient() throws {
        let url = try modelURL()
        let metadata = try GGUFChatMetadata.read(from: url)
        let context = try LocalLLMClientLlama.Context(url: url, parameter: .init(context: 256))
        XCTAssertEqual(metadata.template, context.model.chatTemplate)
        XCTAssertEqual(metadata.specialTokens, StandardLlamaDecoder().extractSpecialTokens(from: context.model))

        let processor = MessageProcessorFactory.createAutoProcessor(chatTemplate: context.model.chatTemplate)
        let renderer = LlamaPromptRenderer(metadata: metadata, tools: [])
        // The template as AuraLocal patches it (`LlamaPromptRenderer.patched`); the rest must match.
        let expected = try processor.renderAndExtractChunks(
            messages: [.system("Be brief."), .user("Hi"), .assistant("Hello!"), .user("Again")],
            template: try XCTUnwrap(renderer.template),
            specialTokens: metadata.specialTokens
        ).rendered
        let rendered = try renderer.render([.system("Be brief."), .user("Hi"), .assistant("Hello!"), .user("Again")])
        XCTAssertEqual(rendered, expected)
    }
    #endif

    // MARK: - Engine

    func testTwentyTwoTurnsAtContext1024NeverOverflow() async throws {
        let engine = try await makeEngine(context: 1024)
        var history: [LlamaChatMessage] = [.system("You are a friendly assistant. Answer in one or two sentences.")]
        var trimmedTurns = 0
        for turn in 0..<22 {
            history.append(.user(Self.facts[turn % Self.facts.count] + Self.padding))
            let reply = try await engine.respond(to: history, maxTokens: 48, shouldStop: nil) { _ in }
            XCTAssertFalse(reply.text.isEmpty, "turn \(turn + 1) returned nothing")
            if reply.history.count < history.count { trimmedTurns += 1 }
            XCTAssertEqual(reply.history.first, history.first, "trimming keeps the system prompt")
            XCTAssertEqual(reply.history.last, history.last, "trimming keeps the new user turn")
            history.append(.assistant(reply.text))
        }
        print("22 turns at 1024: \(trimmedTurns) trimmed")
        XCTAssertGreaterThan(trimmedTurns, 0, "the history never outgrew 1024 tokens, so trimming went untested")
        XCTAssertEqual(engine.rebuildCount, 0, "no turn needed a new context")
    }

    /// A turn whose prompt does not extend what the cache holds (each probe here) starts from an empty cache.
    func testATurnThatDoesNotExtendTheCacheStartsFromAnEmptyKVCacheAndBatch() async throws {
        let engine = try await makeEngine(context: 2048)
        let fresh = try await promptLength(of: Self.probe, on: engine)

        var history: [LlamaChatMessage] = [.system("Be brief.")]
        for fact in Self.facts.prefix(4) {
            history.append(.user(fact))
            history.append(.assistant(try await engine.respond(to: history, maxTokens: 32, shouldStop: nil) { _ in }.text))
            let after = try await position(of: engine)
            XCTAssertGreaterThan(after, fresh, "the cache held this turn's prompt and reply")
        }
        // Cut by the budget, then by `shouldStop`: each leaves its last sampled token queued in the batch.
        _ = try await engine.respond(to: [.user("Count from one to fifty.")], maxTokens: 3, shouldStop: nil) { _ in }
        _ = try await engine.respond(to: [.user("List ten fruits.")], maxTokens: 50, shouldStop: { $0 == 4 }) { _ in }

        let again = try await promptLength(of: Self.probe, on: engine)
        XCTAssertEqual(again, fresh, "the KV cache held \(again - fresh) stale positions")
    }

    func testTheLastTurnStillSeesTheFirstTurn() async throws {
        let engine = try await makeEngine(context: 4096, temperature: 0)
        var history: [LlamaChatMessage] = [.system("You are a helpful assistant. Answer briefly.")]
        for fact in Self.facts {
            history.append(.user(fact))
            history.append(.assistant(try await engine.respond(to: history, maxTokens: 40, shouldStop: nil) { _ in }.text))
        }
        history.append(.user("What is my name? Answer with the name only."))
        let reply = try await engine.respond(to: history, maxTokens: 16, shouldStop: nil) { _ in }
        XCTAssertEqual(reply.history.count, history.count, "nothing was trimmed at 4096")
        XCTAssertTrue(reply.text.contains("Ximena"), reply.text)
    }

    func testAMaxTokensCutLeavesTheNextTurnLikeAFreshOne() async throws {
        let question: [LlamaChatMessage] = [.system("Be brief."), .user("What is the capital of France?")]
        let freshEngine = try await makeEngine(context: 2048, temperature: 0)
        let expected = try await freshEngine.respond(to: question, maxTokens: 24, shouldStop: nil) { _ in }.text

        let engine = try await makeEngine(context: 2048, temperature: 0)
        let cut = try await engine.respond(to: [.user("Write a long poem about the sea.")], maxTokens: 5,
                                           shouldStop: nil) { _ in }
        XCTAssertFalse(cut.text.isEmpty)
        let reply = try await engine.respond(to: question, maxTokens: 24, shouldStop: nil) { _ in }
        XCTAssertEqual(reply.text, expected)
    }

    func testACancelledTurnLeavesTheNextTurnHealthy() async throws {
        let engine = try await makeEngine(context: 2048)
        let fresh = try await promptLength(of: Self.probe, on: engine)
        let counter = PieceCounter()
        let call = Task { @MainActor in
            try await engine.respond(to: [.user("Write a long story about a lighthouse.")], maxTokens: 400,
                                     shouldStop: nil) { _ in
                counter.pieces += 1
                if counter.pieces == 10 { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        let partial = try await call.value
        XCTAssertLessThan(counter.pieces, 400, "cancellation stopped the reply")
        XCTAssertFalse(partial.text.isEmpty)

        let next = try await engine.respond(to: Self.probe, maxTokens: 24, shouldStop: nil) { _ in }
        XCTAssertFalse(next.text.isEmpty)
        let again = try await promptLength(of: Self.probe, on: engine)
        XCTAssertEqual(again, fresh, "the KV cache held \(again - fresh) stale positions")
    }

    func testAnInterruptedGenerationLeavesTheNextTurnHealthy() async throws {
        let notifications = NotificationCenter()
        let (background, foreground) = (Notification.Name("kv.background"), Notification.Name("kv.foreground"))
        let lifecycle = BackgroundLifecycle(notificationCenter: notifications, didEnterBackground: background,
                                            willEnterForeground: foreground)
        let model = try localModel()
        let backend = LlamaCppBackend(model: model)
        let auraEngine = AuraEngine(model: model, backend: backend, lifecycle: lifecycle)
        try await auraEngine.load { _ in }
        addTeardownBlock { @MainActor in
            backend.unload()
            await backend.awaitRelease()
        }
        let engine = try XCTUnwrap(backend.turnEngine as? LlamaTurnEngine)
        let fresh = try await promptLength(of: Self.probe, on: engine)

        let counter = PieceCounter()
        let call = Task { @MainActor in
            try await auraEngine.generate(prompt: "Write a long story about a lighthouse.", maxTokens: 400) { _ in
                counter.pieces += 1
                if counter.pieces == 5 { notifications.post(name: background, object: nil) }
            }
        }
        do {
            _ = try await call.value
            XCTFail("backgrounding did not interrupt the generation")
        } catch let stop as GenerationInterrupted {
            XCTAssertEqual(stop.reason, .appBackgrounded)
            XCTAssertFalse(stop.partialText.isEmpty)
        }
        notifications.post(name: foreground, object: nil)

        let next = try await auraEngine.generate(prompt: "Name one primary color.", systemPrompt: "Be brief.",
                                                 maxTokens: 24) { _ in }
        XCTAssertFalse(next.isEmpty)
        let again = try await promptLength(of: Self.probe, on: engine)
        XCTAssertEqual(again, fresh, "the KV cache held \(again - fresh) stale positions")
    }

    // MARK: - Public API

    func testChatInAConversationKeepsRolesAndRecallsTurnOne() async throws {
        let llm = try await loadModel()
        let store = try temporaryStore()
        let conversation = try await store.createConversation(model: llm.engine.model)
        let system = "You are a helpful assistant. Answer in at most two sentences."

        var replies: [String] = []
        for fact in Self.facts {
            replies.append(try await llm.chat(fact, in: conversation.id, systemPrompt: system, maxTokens: 48,
                                              store: store))
        }
        let last = try await llm.chat("What is my name? Answer with the name only.", in: conversation.id,
                                      systemPrompt: system, maxTokens: 16, store: store)
        let engine = try turnEngine(of: llm)
        XCTAssertEqual(engine.lastHistory.first, .system(system))
        XCTAssertEqual(engine.lastHistory.dropFirst().first, .user(Self.facts[0]))
        // A whitespace-only reply is stored but never sent.
        let sent = replies.filter { !LlamaChatNormalizer.isBlank($0) }
        XCTAssertEqual(engine.lastHistory.filter { $0.role == .assistant }.map(\.content), sent,
                       "every earlier reply reached the model as an assistant turn")
        XCTAssertTrue(last.contains("Ximena"), last)
    }

    func testCancelledStreamLeavesTheNextChatHealthy() async throws {
        let llm = try await loadModel()
        let engine = try turnEngine(of: llm)
        let fresh = try await promptLength(of: Self.probe, on: engine)
        let stream = Task { @MainActor in
            var received = ""
            for try await delta in llm.stream("Write a long story about a lighthouse.", maxTokens: 400) {
                received += delta
                if received.count > 20 { withUnsafeCurrentTask { $0?.cancel() } }
            }
            return received
        }
        let received = try await stream.value
        XCTAssertGreaterThan(received.count, 20)
        let after = try await llm.chat("Name one primary color.", maxTokens: 24)
        XCTAssertFalse(after.isEmpty)
        let again = try await promptLength(of: Self.probe, on: engine)
        XCTAssertEqual(again, fresh, "the KV cache held \(again - fresh) stale positions")
    }

    func testStoppingAHistoryStreamStopsItsDecodeAndKeepsThePartialReply() async throws {
        let llm = try await loadModel()
        let store = try temporaryStore()
        let conversation = try await store.createConversation(model: llm.engine.model).id
        var shown = ""
        for try await delta in llm.stream("Write a long story about a lighthouse.", in: conversation,
                                          systemPrompt: "Be brief.", maxTokens: 400, store: store) {
            shown += delta
            if shown.count > 20 { break }
        }

        let next = try await llm.chat("Name one primary color.", in: conversation, maxTokens: 24, store: store)
        XCTAssertFalse(next.isEmpty)
        let turns = try await store.turns(for: conversation)
        XCTAssertEqual(turns.map(\.role), [.system, .user, .assistant, .user, .assistant])
        let partial = try XCTUnwrap(turns.dropFirst(2).first).content
        print("stopped history stream: \(shown.count) characters shown, \(partial.count) stored")
        XCTAssertTrue(partial.hasPrefix(shown), "\(partial) does not start with \(shown)")
        // Left to its 400-token budget, the story runs well past a thousand characters.
        XCTAssertLessThan(partial.count, shown.count + 200, partial)
        let engine = try turnEngine(of: llm)
        XCTAssertEqual(engine.lastHistory.map(\.role), [.system, .user, .assistant, .user])
    }

    func testAConstrainedCallBetweenChatTurnsLeavesTheConversationIntact() async throws {
        let llm = try await loadModel()
        let store = try temporaryStore()
        let conversation = try await store.createConversation(model: llm.engine.model)
        let first = try await llm.chat(Self.facts[0], in: conversation.id, systemPrompt: "Be brief.", maxTokens: 48,
                                       store: store)
        XCTAssertFalse(first.isEmpty)

        let json = try await llm.chat("Return a JSON object with the key \"n\" set to 7.", systemPrompt: "Be brief.",
                                      maxTokens: 64, constraint: .json)
        XCTAssertTrue((try? JSONSerialization.jsonObject(with: Data(json.utf8))) is [String: Any], json)

        let second = try await llm.chat("What is my name? Answer with the name only.", in: conversation.id,
                                        maxTokens: 16, store: store)
        XCTAssertFalse(second.isEmpty)
        let engine = try turnEngine(of: llm)
        XCTAssertEqual(engine.lastHistory.map(\.role), [.system, .user, .assistant, .user])
        XCTAssertFalse(engine.lastHistory.contains { $0.content.contains("JSON") })
    }

    func testStatsEstimateTheRenderedPrompt() async throws {
        let llm = try await loadModel()
        _ = try await llm.chat("Name one primary color.", systemPrompt: "Be brief.", maxTokens: 24)
        let stats = try XCTUnwrap(llm.lastGenerationStats)
        let rendered = try turnEngine(of: llm).renderer.render(LlamaChatMessage.list(prompt: "Name one primary color.",
                                                                                     systemPrompt: "Be brief."))
        XCTAssertEqual(stats.source, .measured)
        XCTAssertEqual(stats.promptTokens, GenerationStats.estimatedTokens(in: rendered))
        XCTAssertGreaterThan(stats.generatedTokens, 0)
        XCTAssertLessThanOrEqual(stats.generatedTokens, 24)
        XCTAssertGreaterThan(stats.timeToFirstToken, 0)
    }

    func testLayerStreamingTurnsAlsoStartClean() async throws {
        let backend = LayerStreamingBackend(model: try localModel())
        try await backend.load { _ in }
        addTeardownBlock { @MainActor in
            backend.unload()
            await backend.awaitRelease()
        }
        var messages = [Self.message("system", "Be brief.")]
        for fact in Self.facts.prefix(3) {
            messages.append(Self.message("user", fact))
            let reply = try await backend.generate(messages: messages, maxTokens: 32) { _ in }
            XCTAssertFalse(reply.isEmpty)
            messages.append(Self.message("assistant", reply))
        }
        XCTAssertEqual(backend.lastGenerationStats?.backend, .layerStreaming)
    }

    // MARK: - Tools

    func testAToolCallIsExecutedAndTheNextTurnIsHealthy() async throws {
        let engine = try await makeToolEngine(temperature: 0)
        let calls = CallLog()
        EchoTool.log = calls
        let reply = try await engine.respond(to: [.user("Use the echo tool to repeat the word pineapple.")],
                                             maxTokens: 128, shouldStop: nil) { _ in }
        // Only templates with a native tool-call format are expected to call it.
        if engine.renderer.template?.contains("<tool_call>") == true {
            XCTAssertEqual(calls.texts, ["pineapple"], reply.text)
            XCTAssertEqual(reply.history.last?.role, .tool, "the resumed prompt carried the tool output")
        }
        let next = try await engine.respond(to: [.user("Name one primary color.")], maxTokens: 24,
                                            shouldStop: nil) { _ in }
        XCTAssertFalse(next.text.isEmpty)
        try await waitUntilIdle(engine)
    }

    /// Each cut reply decodes on to end of generation before the next turn starts; a small window bounds that wait.
    func testToolTurnsCutShortLeaveTheNextTurnHealthy() async throws {
        let engine = try await makeToolEngine(context: 1024)
        let requests = (0..<3).map { [LlamaChatMessage.user("Write a long story number \($0).")] }
        for request in requests {
            let cut = try await engine.respond(to: request, maxTokens: 4, shouldStop: nil) { _ in }
            XCTAssertFalse(cut.text.isEmpty)
        }
        let start = ContinuousClock.now
        let reply = try await engine.respond(to: Self.probe, maxTokens: 32, shouldStop: nil) { _ in }
        print("tools: the turn after a cut waited \(ContinuousClock.now - start) in all")
        XCTAssertFalse(reply.text.isEmpty)
        try await waitUntilIdle(engine)
    }

    func testToolTurnsSurviveTwentyTurnsAtContext1024() async throws {
        let engine = try await makeToolEngine(context: 1024)
        var history: [LlamaChatMessage] = [.system("Be brief.")]
        var trimmedTurns = 0
        for turn in 0..<20 {
            history.append(.user(Self.facts[turn % Self.facts.count] + Self.padding))
            let reply = try await engine.respond(to: history, maxTokens: 32, shouldStop: nil) { _ in }
            if reply.history.count < history.count { trimmedTurns += 1 }
            history.append(.assistant(reply.text))
        }
        XCTAssertGreaterThan(trimmedTurns, 0, "the history never outgrew 1024 tokens, so trimming went untested")
        XCTAssertEqual(engine.rebuildCount, 0, "every failed prefill was flushed without a new client")
        try await waitUntilIdle(engine)
    }

    // MARK: - Helpers

    private func modelURL() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment[Self.modelVariable], !path.isEmpty else {
            throw XCTSkip("Set \(Self.modelVariable) to a local .gguf file to run the multi-turn GGUF tests.")
        }
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("\(Self.modelVariable) points at a missing file: \(path)")
        }
        return URL(fileURLWithPath: path)
    }

    private static func message(_ role: String, _ content: String) -> [String: String] {
        ["role": role, "content": content]
    }

    private func parameter(context: Int, temperature: Float) -> LlamaClient.Parameter {
        LlamaClient.Parameter(context: context, batch: min(context, 2048), temperature: temperature, topP: 0.95)
    }

    private func makeEngine(context: Int, temperature: Float = 0.7) async throws -> LlamaTurnEngine {
        let url = try modelURL()
        let core = try await LlamaTurnCore.load(url: url, parameter: parameter(context: context, temperature: temperature))
        return LlamaTurnEngine(renderer: await GGUFTurnEngines.renderer(for: url, tools: []), core: core)
    }

    private func makeToolEngine(context: Int = 4096, temperature: Float = 0.7) async throws -> LlamaToolTurnEngine {
        let url = try modelURL()
        let settings = parameter(context: context, temperature: temperature)
        let renderer = await GGUFTurnEngines.renderer(for: url, tools: [EchoTool()])
        let client = try await LlamaToolTurnEngine.makeClient(url: url, parameter: settings, tools: renderer.tools)
        return LlamaToolTurnEngine(url: url, parameter: settings, renderer: renderer, client: client)
    }

    /// The KV position once the turn's decode has stopped.
    private func position(of engine: LlamaTurnEngine) async throws -> Int32 {
        try await engine.waitForDecoder()
        return try XCTUnwrap(engine.contextForTesting?.position)
    }

    /// With a one-token budget the sampled token stays queued, so the cache holds exactly the prefilled prompt.
    private func promptLength(of messages: [LlamaChatMessage], on engine: LlamaTurnEngine) async throws -> Int32 {
        _ = try await engine.respond(to: messages, maxTokens: 1, shouldStop: nil) { _ in }
        return try await position(of: engine)
    }

    private func waitUntilIdle(_ engine: any GGUFTurnEngine) async throws {
        try await waitUntil(timeout: .seconds(60)) { engine.isIdle }
    }

    /// Resolves the URL first: an `XCTSkip` thrown inside `XCTUnwrap` is reported as a failure.
    private func localModel() throws -> AuraCore.Model {
        let url = try modelURL()
        return try XCTUnwrap(Model.from(.localFile(url)))
    }

    private func loadModel() async throws -> AuraLocal {
        let llm = AuraLocal(model: try localModel())
        XCTAssertTrue(llm.engine.backend is LlamaCppBackend)
        try await llm.engine.load { _ in }
        addTeardownBlock { @MainActor in
            llm.engine.unload()
            await (llm.engine.backend as? LlamaCppBackend)?.awaitRelease()
        }
        return llm
    }

    private func turnEngine(of llm: AuraLocal) throws -> LlamaTurnEngine {
        let backend = try XCTUnwrap(llm.engine.backend as? LlamaCppBackend)
        return try XCTUnwrap(backend.turnEngine as? LlamaTurnEngine)
    }

    private func temporaryStore() throws -> ConversationStore {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("gguf-multiturn-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return ConversationStore(directory: directory)
    }
}

@MainActor
private final class PieceCounter {
    var pieces = 0
}

private struct EchoTool: LLMTool {
    let name = "echo"
    let description = "Repeat the given text back."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let text: String
        static var argumentsSchema: LLMToolArgumentsSchema { ["text": .string(description: "Text to repeat")] }
    }
    nonisolated(unsafe) static var log: CallLog?

    func call(arguments: Arguments) async throws -> ToolOutput {
        Self.log?.record(arguments.text)
        return ToolOutput(data: ["text": arguments.text])
    }
}

private final class CallLog: Sendable {
    private let recorded = Mutex<[String]>([])

    var texts: [String] { recorded.withLock { $0 } }
    func record(_ text: String) { recorded.withLock { $0.append(text) } }
}
