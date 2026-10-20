import Foundation
import LocalLLMClient
import LocalLLMClientLlama
import Synchronization

// MARK: - GGUFTurnEngine

/// Runs one GGUF chat turn at a time over a whole conversation, on a model loaded once. A turn whose prompt extends
/// exactly what the KV cache holds decodes only the new part (``GGUFPromptReuse``); any other turn starts from an empty
/// KV cache and decode batch, so it never inherits tokens from the turn before.
///
/// Not reentrant: the backends serialise turns with a ``DecodeGate``. A turn that returns early (cancelled, or
/// stopped by `shouldStop`) can leave its decode finishing in the background; the next turn waits for it.
@MainActor
protocol GGUFTurnEngine: AnyObject {
    var renderer: LlamaPromptRenderer { get }
    /// No decode of this engine is running and no LocalLLMClient task holds its model, so dropping the engine
    /// frees the model right away.
    var isIdle: Bool { get }
    /// Messages the last prefill rendered, after normalisation and trimming.
    var lastHistory: [LlamaChatMessage] { get }
    /// Contexts or clients rebuilt (a model reload) because a failed prefill could not be flushed.
    var rebuildCount: Int { get }

    /// Streams the reply to `messages` through `onPiece`. Stops after `maxTokens` pieces, when `shouldStop` returns
    /// `true` for the pieces so far, or when the task is cancelled; then it returns the text so far. An engine with
    /// tools runs them as `toolLoop` says, with every round sharing `maxTokens`.
    func respond(
        to messages: [LlamaChatMessage],
        maxTokens: Int,
        toolLoop: ToolLoopContext,
        shouldStop: ((Int) -> Bool)?,
        onPiece: (String) -> Void
    ) async throws -> GGUFTurnReply
}

struct GGUFTurnReply: Sendable {
    let text: String
    /// The last prompt decoded, as rendered after any trimming; empty when the turn stopped before prefilling.
    let prompt: String
    /// The messages `prompt` was rendered from.
    let history: [LlamaChatMessage]
    /// What the tool loop did; `nil` for an engine without tools.
    var tools: ToolTurnSummary? = nil
    /// The text the last prefill decoded: the end of `prompt` when the KV cache already held the rest; `nil` means
    /// all of `prompt`.
    var prefilled: String? = nil
}

extension GGUFTurnEngine {
    /// A turn with the default tool loop.
    func respond(
        to messages: [LlamaChatMessage],
        maxTokens: Int,
        shouldStop: ((Int) -> Bool)?,
        onPiece: (String) -> Void
    ) async throws -> GGUFTurnReply {
        try await respond(to: messages, maxTokens: maxTokens, toolLoop: .standard, shouldStop: shouldStop,
                          onPiece: onPiece)
    }

    /// A turn timed with a ``StreamMeter``: `onToken` receives the accumulated text, and the stats estimate the
    /// prompt that was actually decoded.
    func measuredTurn(
        _ conversation: [LlamaChatMessage],
        maxTokens: Int,
        backend: BackendKind,
        source: GenerationStats.Provenance,
        toolLoop: ToolLoopContext = .standard,
        shouldStop: ((Int) -> Bool)? = nil,
        onToken: (String) -> Void
    ) async throws -> (text: String, stats: GenerationStats, tools: ToolTurnSummary?) {
        var meter = StreamMeter()
        var fullText = ""
        let reply = try await respond(to: conversation, maxTokens: maxTokens, toolLoop: toolLoop,
                                      shouldStop: shouldStop) { piece in
            meter.recordPiece()
            fullText += piece
            onToken(fullText)
        }
        let stats = meter.stats(backend: backend, source: source, prompt: reply.prefilled ?? reply.prompt,
                                output: reply.text)
        return (reply.text, stats, reply.tools)
    }
}

@MainActor
enum GGUFTurnEngines {
    /// Loads the model once, off the main actor. Without tools, or with tools the model writes as `<tool_call>` JSON,
    /// the engine drives a bare `Context` and keeps its KV cache between turns and rounds; other tool-call formats
    /// need LocalLLMClient's parser, which only a `LlamaClient` owns.
    static func load(url: URL, parameter: LlamaClient.Parameter, tools: [any LLMTool]) async throws
        -> any GGUFTurnEngine {
        let metadata = await metadata(of: url)
        let renderer = renderer(for: metadata, tools: tools)
        return try await engine(url: url, parameter: parameter, renderer: renderer,
                                boundaries: metadata?.controlTokenTexts ?? [])
    }

    /// The engine ``load(url:parameter:tools:)`` builds for `renderer`.
    static func engine(url: URL, parameter: LlamaClient.Parameter, renderer: LlamaPromptRenderer,
                       boundaries: [String]? = nil) async throws -> any GGUFTurnEngine {
        if renderer.tools.isEmpty || renderer.asksForTaggedToolCalls {
            let core = try await LlamaTurnCore.load(url: url, parameter: parameter, boundaries: boundaries)
            return LlamaTurnEngine(renderer: renderer, core: core)
        }
        let client = try await LlamaToolTurnEngine.makeClient(url: url, parameter: parameter, tools: renderer.tools)
        return LlamaToolTurnEngine(url: url, parameter: parameter, renderer: renderer, client: client)
    }

    /// The template and special tokens llama.cpp itself uses. Metadata this parser rejects but llama.cpp loads
    /// falls back to a labelled transcript rather than failing the load.
    nonisolated static func renderer(for url: URL, tools: [any LLMTool]) async -> LlamaPromptRenderer {
        renderer(for: await metadata(of: url), tools: tools)
    }

    nonisolated static func renderer(for metadata: GGUFChatMetadata?, tools: [any LLMTool]) -> LlamaPromptRenderer {
        guard let metadata else { return LlamaPromptRenderer(template: nil, tools: tools.map { AnyLLMTool($0) }) }
        return LlamaPromptRenderer(metadata: metadata, tools: tools)
    }

    /// Read off the main actor: finding the special tokens walks the whole vocabulary.
    nonisolated static func metadata(of url: URL) async -> GGUFChatMetadata? {
        try? GGUFChatMetadata.read(from: url)
    }

    /// Waits for `task`, or throws `CancellationError` as soon as the caller is cancelled while `task` runs on.
    nonisolated static func finish(of task: Task<Void, Never>) async throws {
        let (finished, signal) = AsyncStream<Void>.makeStream()
        Task {
            await task.value
            signal.finish()
        }
        for await _ in finished {}
        try Task.checkCancellation()
    }
}

/// Whether a turn's background decode is running: written by that decode, read on the main actor.
final class DecodeActivity: Sendable {
    private let running = Atomic<Bool>(false)

    var isRunning: Bool { running.load(ordering: .acquiring) }

    func begin() { running.store(true, ordering: .releasing) }
    func end() { running.store(false, ordering: .releasing) }
}

// MARK: - LlamaTurnEngine

/// Drives the model's one `Context` with `.plain` semantics: when the rendered prompt extends what the KV cache holds
/// (``GGUFPromptReuse``) decode only the rest, otherwise reset the KV cache and batch and prefill the whole
/// conversation; then sample. The decode runs in a task of its own, so a cancelled caller returns at once even while
/// LocalLLMClient has the generator parked (it pauses on `willResignActive`).
///
/// With tools the model writes its calls as `<tool_call>` JSON (``LlamaPromptRenderer/asksForTaggedToolCalls``) and
/// ``ToolCallTagParser`` reads them. Each reply that asks for tools is a round: ``ToolLoopDriver`` runs its calls,
/// and the reply exactly as the model wrote it, tags included, joins the conversation with one tool turn per result,
/// so the next reply's prompt extends what the KV cache holds.
@MainActor
final class LlamaTurnEngine: GGUFTurnEngine {
    let renderer: LlamaPromptRenderer
    /// Never reassigned; a `var` only so ``isIdle`` can check that no decode task still holds it.
    private var core: LlamaTurnCore
    private var decoder: Task<Void, Never>?
    private let activity = DecodeActivity()
    private(set) var lastHistory: [LlamaChatMessage] = []
    private(set) var rebuildCount = 0
    /// Off, every prompt is prefilled in full; for comparisons.
    var reusesPromptCache = true
    /// The text the KV cache holds, known only after a reply that ended by itself (end of generation or a full
    /// window); `nil` after a cut, cancelled or failed one.
    private var resident: String?
    /// Prefills that decoded only the end of their prompt, and the prompt bytes the last prefill found in the cache.
    private(set) var reusedPrefills = 0
    private(set) var lastReusedBytes = 0

    /// Pieces across every round of a turn, which share `maxTokens`, and the text a reader sees.
    private struct Budget {
        let limit: Int
        var produced = 0
        var text = ""

        var remaining: Int { limit - produced }
    }

    private struct Streamed {
        let history: [LlamaChatMessage]
        let prompt: String
        let prefilled: String?
        /// Everything the model wrote, tool-call tags included.
        let raw: String
        /// The calls a reply that ended by itself made; `nil` for a cut reply.
        let calls: [LLMToolCall]?
    }

    init(renderer: LlamaPromptRenderer, core: LlamaTurnCore) {
        self.renderer = renderer
        self.core = core
    }

    var isIdle: Bool { !activity.isRunning && isKnownUniquelyReferenced(&core) }

    /// The context, for tests that read its KV position while the engine is idle.
    var contextForTesting: LocalLLMClientLlama.Context? { core.currentContext }

    func respond(
        to messages: [LlamaChatMessage],
        maxTokens: Int,
        toolLoop: ToolLoopContext,
        shouldStop: ((Int) -> Bool)?,
        onPiece: (String) -> Void
    ) async throws -> GGUFTurnReply {
        try await waitForDecoder()
        try Task.checkCancellation()

        var budget = Budget(limit: maxTokens)
        var history = LlamaChatNormalizer.normalize(messages)
        guard !renderer.tools.isEmpty else {
            let reply = try await stream(history, toolRoundsRemain: false, budget: &budget, shouldStop: shouldStop,
                                         onPiece: onPiece)
            return GGUFTurnReply(text: budget.text, prompt: reply.prompt, history: reply.history,
                                 prefilled: reply.prefilled)
        }

        let limit = max(0, toolLoop.options.maxToolRounds)
        var rounds = 0
        var replies = 0
        var last: Streamed?
        let driver = ToolLoopDriver(tools: renderer.tools, context: toolLoop)
        let summary = try await driver.run(
            reply: {
                if replies > 0 {
                    try await waitForDecoder()
                    try Task.checkCancellation()
                }
                replies += 1
                let streamed = try await stream(history, toolRoundsRemain: rounds < limit, budget: &budget,
                                                shouldStop: shouldStop, onPiece: onPiece)
                history = streamed.history
                last = streamed
                return streamed.calls
            },
            record: { _, results in
                rounds += 1
                history += Self.roundMessages(reply: last?.raw ?? "", results: results)
            }
        )
        return GGUFTurnReply(text: budget.text, prompt: last?.prompt ?? "", history: last?.history ?? [],
                             tools: summary, prefilled: last?.prefilled)
    }

    /// One round as the conversation records it: the reply as the model wrote it, then one tool turn per result.
    static func roundMessages(reply: String, results: [ToolCallResult]) -> [LlamaChatMessage] {
        [.assistant(reply)] + results.map { .tool($0.output) }
    }

    /// One reply, decoded by the core in a task of its own. With tools, the pieces pass through a
    /// ``ToolCallTagParser`` and only the text outside calls reaches `onPiece`.
    private func stream(
        _ messages: [LlamaChatMessage],
        toolRoundsRemain: Bool,
        budget: inout Budget,
        shouldStop: ((Int) -> Bool)?,
        onPiece: (String) -> Void
    ) async throws -> Streamed {
        let held = reusesPromptCache ? resident : nil
        resident = nil
        let request = LlamaTurnCore.Request(messages: messages, maxTokens: budget.remaining, renderer: renderer,
                                            toolRoundsRemain: toolRoundsRemain, resident: held)
        let (events, sink) = AsyncThrowingStream<LlamaTurnCore.Event, any Error>.makeStream()
        let core = self.core
        let activity = self.activity
        activity.begin()
        let decoder = Task.detached(priority: Task.currentPriority) {
            await core.run(request, into: sink)
            activity.end()
        }
        self.decoder = decoder
        defer { decoder.cancel() }

        var parser = renderer.tools.isEmpty ? nil : ToolCallTagParser()
        var raw = ""
        var prefilled: LlamaTurnCore.Prefill?
        var completed = false
        reading: for try await event in events {
            switch event {
            case .prefilled(let report):
                prefilled = report
                record(report)
            case .piece(let piece):
                raw += piece
                budget.produced += 1
                if parser == nil {
                    show(piece, in: &budget, onPiece: onPiece)
                } else if let visible = parser?.consume(piece), !visible.isEmpty {
                    show(visible, in: &budget, onPiece: onPiece)
                }
                if let shouldStop, shouldStop(budget.produced) { break reading }
            case .completed:
                completed = true
            }
        }
        let ended = completed && !Task.isCancelled
        if let rest = parser?.finish(includingOpenCall: ended), !rest.isEmpty { show(rest, in: &budget, onPiece: onPiece) }
        if ended, let prefilled { resident = prefilled.prompt + raw }
        return Streamed(history: prefilled?.history ?? [], prompt: prefilled?.prompt ?? "",
                        prefilled: prefilled?.decoded, raw: raw, calls: ended ? parser?.calls ?? [] : nil)
    }

    private func record(_ report: LlamaTurnCore.Prefill) {
        lastHistory = report.history
        if report.rebuilt { rebuildCount += 1 }
        lastReusedBytes = report.reused
        if report.reused > 0 { reusedPrefills += 1 }
    }

    private func show(_ text: String, in budget: inout Budget, onPiece: (String) -> Void) {
        budget.text += text
        onPiece(text)
    }

    /// Waits for the decode a cut or cancelled turn left running.
    func waitForDecoder() async throws {
        guard let decoder else { return }
        try await GGUFTurnEngines.finish(of: decoder)
        if self.decoder == decoder { self.decoder = nil }
    }
}

// MARK: - LlamaTurnCore

/// The context of a ``LlamaTurnEngine`` and what it has learned about prompt sizes. Only the current turn's decode
/// task touches it: a turn starts after the previous one's task has finished.
final class LlamaTurnCore: @unchecked Sendable {
    private let modelFile: URL
    private let loadParameters: LlamaClient.Parameter
    private var liveContext: LocalLLMClientLlama.Context?
    private var fitter: GGUFPromptFitter
    private let reuse: GGUFPromptReuse

    struct Request: Sendable {
        let messages: [LlamaChatMessage]
        let maxTokens: Int
        let renderer: LlamaPromptRenderer
        /// Passed to ``LlamaPromptRenderer/render(_:toolRoundsRemain:)``.
        var toolRoundsRemain = false
        /// The text the KV cache holds, when the reply before ended by itself; `nil` starts from an empty cache.
        var resident: String? = nil
    }

    struct Prefill: Sendable {
        let history: [LlamaChatMessage]
        let prompt: String
        /// The context had to be rebuilt first.
        let rebuilt: Bool
        /// Bytes at the start of `prompt` the KV cache already held; 0 after a full prefill.
        let reused: Int

        var decoded: String {
            reused == 0 ? prompt : String(decoding: prompt.utf8.dropFirst(reused), as: UTF8.self)
        }
    }

    enum Event: Sendable {
        case prefilled(Prefill)
        case piece(String)
        /// The reply ended by itself (end of generation or a full window): the KV cache holds exactly the prompt and
        /// every piece, and no token is left queued.
        case completed
    }

    private init(url: URL, parameter: LlamaClient.Parameter, reuse: GGUFPromptReuse) throws {
        modelFile = url
        loadParameters = parameter
        liveContext = try LocalLLMClientLlama.Context(url: url, parameter: parameter)
        fitter = GGUFPromptFitter(windowSize: parameter.context)
        self.reuse = reuse
    }

    /// Loads the model off the main actor. `boundaries` are the vocabulary's special-token texts, read from the file
    /// when `nil`.
    static func load(url: URL, parameter: LlamaClient.Parameter, boundaries: [String]? = nil) async throws
        -> LlamaTurnCore {
        let texts = boundaries ?? (try? GGUFChatMetadata.read(from: url))?.controlTokenTexts ?? []
        return try LlamaTurnCore(url: url, parameter: parameter, reuse: GGUFPromptReuse(boundaries: texts))
    }

    var currentContext: LocalLLMClientLlama.Context? { liveContext }

    func run(_ request: Request, into events: AsyncThrowingStream<Event, any Error>.Continuation) async {
        do {
            // A turn cancelled before it got here (e.g. while queued) skips the prefill too.
            guard !Task.isCancelled else { return events.finish() }
            let (context, report) = try prepared(request)
            events.yield(.prefilled(report))

            var produced = 0
            let pieces = LocalLLMClientLlama.Generator(context: context)
            for try await piece in pieces {
                events.yield(.piece(piece))
                produced += 1
                if produced >= request.maxTokens || Task.isCancelled { break }
            }
            // A break leaves the last sampled token queued; the generator ending on its own does not.
            if produced < request.maxTokens && !Task.isCancelled { events.yield(.completed) }
            events.finish()
        } catch {
            events.finish(throwing: error)
        }
    }

    /// In 0.5.0 `Context.clear()` empties only the KV cache. A reply cut short leaves its last sampled token queued
    /// in the batch, and a prefill that overflowed leaves a whole chunk; decoding into the emptied cache flushes it,
    /// and the second clear drops what that wrote. An empty batch decodes nothing.
    static func reset(_ context: LocalLLMClientLlama.Context) throws {
        context.clear()
        try context.decode()
        context.clear()
    }

    /// The context with the request's prompt decoded: only its new part when the cache holds the rest, otherwise in
    /// full after a reset.
    private func prepared(_ request: Request) throws -> (context: LocalLLMClientLlama.Context, report: Prefill) {
        let render = { (messages: [LlamaChatMessage]) throws -> String in
            try request.renderer.render(messages, toolRoundsRemain: request.toolRoundsRemain)
        }
        let (history, prompt) = try fitter.fitted(LlamaChatNormalizer.normalize(request.messages),
                                                  replyBudget: request.maxTokens, render: render)
        if let resident = request.resident, let context = liveContext,
           let suffix = reuse.suffix(of: prompt, after: resident), Self.decoded(suffix, into: context) {
            return (context, Prefill(history: history, prompt: prompt, rebuilt: false, reused: resident.utf8.count))
        }
        let (context, rebuilt) = try preparedContext()
        let fitted = try prefill(history, prompt: prompt, budget: request.maxTokens, render: render, into: context)
        return (context, Prefill(history: fitted.history, prompt: fitted.prompt, rebuilt: rebuilt, reused: 0))
    }

    /// A failure (an overflowed window among them) can leave part of `suffix` decoded and a chunk queued; the full
    /// prefill that follows resets both.
    private static func decoded(_ suffix: String, into context: LocalLLMClientLlama.Context) -> Bool {
        (try? context.decode(text: suffix)) != nil
    }

    /// A context whose reset fails is replaced, which reloads the model; the old one is freed first.
    private func preparedContext() throws -> (context: LocalLLMClientLlama.Context, rebuilt: Bool) {
        if let liveContext {
            do {
                try Self.reset(liveContext)
                return (liveContext, false)
            } catch {
                self.liveContext = nil
            }
        }
        let fresh = try LocalLLMClientLlama.Context(url: modelFile, parameter: loadParameters)
        liveContext = fresh
        return (fresh, true)
    }

    private func prefill(_ fitted: [LlamaChatMessage], prompt fittedPrompt: String, budget: Int,
                         render: ([LlamaChatMessage]) throws -> String,
                         into context: LocalLLMClientLlama.Context) throws -> (history: [LlamaChatMessage], prompt: String) {
        var (history, prompt) = (fitted, fittedPrompt)
        while true {
            do {
                try context.decode(text: prompt)
                return (history, prompt)
            } catch {
                guard fitter.learn(from: error, promptBytes: prompt.utf8.count) else { throw error }
                do {
                    try Self.reset(context)
                } catch {
                    liveContext = nil
                    throw error
                }
                guard let shorter = try fitter.shortened(history, replyBudget: budget, render: render) else { throw error }
                (history, prompt) = shorter
            }
        }
    }
}

// MARK: - LlamaToolTurnEngine

/// The tool-calling turn for call formats other than `<tool_call>` JSON (``LlamaPromptRenderer/asksForTaggedToolCalls``):
/// LocalLLMClient's parser on a `LlamaClient`, fed `.plain` prompts (which clear the KV cache) rendered with real roles.
/// Each reply that asks for tools is a round: ``ToolLoopDriver`` runs its calls, the round (the reply's calls and
/// their results) joins the conversation, and the whole conversation is prefilled again for the next reply, up to
/// ``ToolLoopOptions/maxToolRounds`` rounds.
///
/// `LlamaClient.responseStream` decodes in a task nobody can cancel. A reply cut by `maxTokens`, `shouldStop` or
/// cancellation returns at once, but the next turn waits until that decode reaches end of generation or fills the
/// window, so it never decodes on top of it.
@MainActor
final class LlamaToolTurnEngine: GGUFTurnEngine {
    let renderer: LlamaPromptRenderer
    private let modelFile: URL
    private let loadParameters: LlamaClient.Parameter
    private var client: LlamaClient?
    private var fitter: GGUFPromptFitter
    /// Reads the current stream to its end, so finishing means the client's decode has stopped.
    private var drain: Task<Void, Never>?
    private let activity = DecodeActivity()
    /// Messages the last stream rendered, after normalisation and trimming.
    private(set) var lastHistory: [LlamaChatMessage] = []
    /// Clients replaced (a model reload) because a failed prefill could not be flushed.
    private(set) var rebuildCount = 0

    private enum Outcome {
        case finished([LLMToolCall])
        case cut
    }

    /// Text and pieces across every round of a turn, which share `maxTokens`.
    private struct Progress {
        let limit: Int
        var text = ""
        var produced = 0
        /// The visible text of the current round's reply.
        var roundText = ""
    }

    init(url: URL, parameter: LlamaClient.Parameter, renderer: LlamaPromptRenderer, client: LlamaClient) {
        modelFile = url
        loadParameters = parameter
        self.renderer = renderer
        self.client = client
        fitter = GGUFPromptFitter(windowSize: parameter.context)
    }

    nonisolated static func makeClient(url: URL, parameter: LlamaClient.Parameter,
                                       tools: [AnyLLMTool]) async throws -> LlamaClient {
        try await LocalLLMClient.llama(url: url, parameter: parameter, tools: tools.map(\.underlyingTool))
    }

    var isIdle: Bool {
        guard !activity.isRunning else { return false }
        return client == nil || isKnownUniquelyReferenced(&client)
    }

    func respond(
        to messages: [LlamaChatMessage],
        maxTokens: Int,
        toolLoop: ToolLoopContext,
        shouldStop: ((Int) -> Bool)?,
        onPiece: (String) -> Void
    ) async throws -> GGUFTurnReply {
        try await waitForDrain()
        try Task.checkCancellation()

        var progress = Progress(limit: maxTokens)
        var history = LlamaChatNormalizer.normalize(messages)
        var sent: (history: [LlamaChatMessage], prompt: String) = (history, "")
        var replies = 0
        let limit = max(0, toolLoop.options.maxToolRounds)
        var rounds = 0
        let driver = ToolLoopDriver(tools: renderer.tools, context: toolLoop)
        let summary = try await driver.run(
            reply: {
                if replies > 0 {
                    try await waitForDrain()
                    try Task.checkCancellation()
                }
                replies += 1
                progress.roundText = ""
                let streamed = try await stream(history, toolRoundsRemain: rounds < limit, progress: &progress,
                                                shouldStop: shouldStop, onPiece: onPiece)
                history = streamed.history
                sent = (streamed.history, streamed.prompt)
                guard case .finished(let calls) = streamed.outcome else { return nil }
                return calls
            },
            record: { calls, results in
                rounds += 1
                history += Self.roundMessages(text: progress.roundText, calls: calls, results: results)
            }
        )
        return GGUFTurnReply(text: progress.text, prompt: sent.prompt, history: sent.history, tools: summary)
    }

    /// One round as the conversation records it: the reply with its calls, then one tool turn per result.
    static func roundMessages(text: String, calls: [LLMToolCall], results: [ToolCallResult]) -> [LlamaChatMessage] {
        // The parser leaves the newlines between calls as text; kept, they would precede the calls in the prompt.
        let visible = LlamaChatNormalizer.isBlank(text) ? "" : text
        return [.assistant(visible, toolCalls: calls)] + results.map { .tool($0.output) }
    }

    /// One `responseStream`, trimmed to the window. Returns the history it sent, after any trimming, and the prompt
    /// that rendered; trimming drops whole exchanges before the newest user turn, never the rounds after it.
    private func stream(
        _ messages: [LlamaChatMessage],
        toolRoundsRemain: Bool,
        progress: inout Progress,
        shouldStop: ((Int) -> Bool)?,
        onPiece: (String) -> Void
    ) async throws -> (outcome: Outcome, history: [LlamaChatMessage], prompt: String) {
        let renderer = self.renderer
        let render = { (messages: [LlamaChatMessage]) throws -> String in
            try renderer.render(messages, toolRoundsRemain: toolRoundsRemain)
        }
        var (history, prompt) = try fitter.fitted(messages, replyBudget: progress.limit, render: render)
        let chunks: AsyncThrowingStream<StreamingChunk, any Error>
        while true {
            let client = try await liveClient()
            do {
                chunks = try await client.responseStream(from: .plain(prompt))
                break
            } catch {
                await discardQueuedTokens(of: client)
                guard fitter.learn(from: error, promptBytes: prompt.utf8.count),
                      let shorter = try fitter.shortened(history, replyBudget: progress.limit,
                                                         render: render) else { throw error }
                (history, prompt) = shorter
            }
        }
        lastHistory = history

        let (relay, sink) = AsyncThrowingStream<StreamingChunk, any Error>.makeStream()
        let activity = self.activity
        activity.begin()
        drain = Task.detached(priority: Task.currentPriority) {
            do {
                for try await chunk in chunks { sink.yield(chunk) }
                sink.finish()
            } catch {
                sink.finish(throwing: error)
            }
            activity.end()
        }

        var calls: [LLMToolCall] = []
        do {
            for try await chunk in relay {
                switch chunk {
                case .text(let piece):
                    progress.text += piece
                    progress.roundText += piece
                    progress.produced += 1
                    onPiece(piece)
                    if progress.produced >= progress.limit || shouldStop?(progress.produced) == true {
                        return (.cut, history, prompt)
                    }
                case .toolCall(let call):
                    calls.append(call)
                }
            }
        } catch {
            // A decode that failed mid-reply can leave its token queued, like a failed prefill.
            if let client { await discardQueuedTokens(of: client) }
            throw error
        }
        return (Task.isCancelled ? .cut : .finished(calls), history, prompt)
    }

    private func waitForDrain() async throws {
        guard let drain else { return }
        try await GGUFTurnEngines.finish(of: drain)
        if self.drain == drain { self.drain = nil }
    }

    private func liveClient() async throws -> LlamaClient {
        if let client { return client }
        let fresh = try await Self.makeClient(url: modelFile, parameter: loadParameters, tools: renderer.tools)
        rebuildCount += 1
        client = fresh
        return fresh
    }

    /// A failed prefill leaves its chunk queued in the client's batch. The client keeps its `Context` private, so it
    /// is read by reflection; if that fails (a LocalLLMClient upgrade renamed it), the client is replaced instead.
    private func discardQueuedTokens(of failed: LlamaClient) async {
        if let context = Mirror(reflecting: failed).descendant("context") as? LocalLLMClientLlama.Context {
            let flushed = await Task.detached { (try? LlamaTurnCore.reset(context)) != nil }.value
            if flushed { return }
        }
        client = nil
    }
}
