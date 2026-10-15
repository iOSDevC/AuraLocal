import Foundation
import LocalLLMClient
import LocalLLMClientLlama
import Synchronization

// MARK: - GGUFTurnEngine

/// Runs one GGUF chat turn at a time over a whole conversation, on a model loaded once. Every turn starts from an
/// empty KV cache and decode batch, so it never inherits tokens from the turn before.
///
/// Not reentrant: the backends serialise turns with a ``DecodeGate``. A turn that returns early (cancelled, or
/// stopped by `shouldStop`) can leave its decode finishing in the background; the next turn waits for it.
@MainActor
protocol GGUFTurnEngine: AnyObject {
    var renderer: LlamaPromptRenderer { get }
    /// No decode of this engine is running and no LocalLLMClient task holds its model, so dropping the engine
    /// frees the model right away.
    var isIdle: Bool { get }

    /// Streams the reply to `messages` through `onPiece`. Stops after `maxTokens` pieces, when `shouldStop` returns
    /// `true` for the pieces so far, or when the task is cancelled; then it returns the text so far.
    func respond(
        to messages: [LlamaChatMessage],
        maxTokens: Int,
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
}

extension GGUFTurnEngine {
    /// A turn timed with a ``StreamMeter``: `onToken` receives the accumulated text, and the stats estimate the
    /// prompt that was actually decoded.
    func measuredTurn(
        _ conversation: [LlamaChatMessage],
        maxTokens: Int,
        backend: BackendKind,
        source: GenerationStats.Provenance,
        shouldStop: ((Int) -> Bool)? = nil,
        onToken: (String) -> Void
    ) async throws -> (text: String, stats: GenerationStats) {
        var meter = StreamMeter()
        var fullText = ""
        let reply = try await respond(to: conversation, maxTokens: maxTokens, shouldStop: shouldStop) { piece in
            meter.recordPiece()
            fullText += piece
            onToken(fullText)
        }
        return (reply.text, meter.stats(backend: backend, source: source, prompt: reply.prompt, output: reply.text))
    }
}

@MainActor
enum GGUFTurnEngines {
    /// Loads the model once, off the main actor. Tools need LocalLLMClient's tool-call parser, which only a
    /// `LlamaClient` owns; without tools the engine drives a bare `Context`.
    static func load(url: URL, parameter: LlamaClient.Parameter, tools: [any LLMTool]) async throws
        -> any GGUFTurnEngine {
        let renderer = await renderer(for: url, tools: tools)
        if tools.isEmpty {
            return LlamaTurnEngine(renderer: renderer, core: try await LlamaTurnCore.load(url: url, parameter: parameter))
        }
        let client = try await LlamaToolTurnEngine.makeClient(url: url, parameter: parameter, tools: renderer.tools)
        return LlamaToolTurnEngine(url: url, parameter: parameter, renderer: renderer, client: client)
    }

    /// The template and special tokens llama.cpp itself uses. Metadata this parser rejects but llama.cpp loads
    /// falls back to a labelled transcript rather than failing the load.
    nonisolated static func renderer(for url: URL, tools: [any LLMTool]) async -> LlamaPromptRenderer {
        guard let metadata = try? GGUFChatMetadata.read(from: url) else {
            return LlamaPromptRenderer(template: nil, tools: tools.map { AnyLLMTool($0) })
        }
        return LlamaPromptRenderer(metadata: metadata, tools: tools)
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

/// Drives the model's one `Context` with `.plain` semantics: reset the KV cache and batch, prefill the rendered
/// conversation, sample. The decode runs in a task of its own, so a cancelled caller returns at once even while
/// LocalLLMClient has the generator parked (it pauses on `willResignActive`).
@MainActor
final class LlamaTurnEngine: GGUFTurnEngine {
    let renderer: LlamaPromptRenderer
    /// Never reassigned; a `var` only so ``isIdle`` can check that no decode task still holds it.
    private var core: LlamaTurnCore
    private var decoder: Task<Void, Never>?
    private let activity = DecodeActivity()
    /// Messages the last prefill rendered, after normalisation and trimming.
    private(set) var lastHistory: [LlamaChatMessage] = []
    /// Contexts rebuilt (a model reload) because a reset failed.
    private(set) var rebuildCount = 0

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
        shouldStop: ((Int) -> Bool)?,
        onPiece: (String) -> Void
    ) async throws -> GGUFTurnReply {
        try await waitForDecoder()
        try Task.checkCancellation()

        let request = LlamaTurnCore.Request(messages: messages, maxTokens: maxTokens, renderer: renderer)
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

        var text = ""
        var produced = 0
        var prefilled: LlamaTurnCore.Prefill?
        reading: for try await event in events {
            switch event {
            case .prefilled(let report):
                prefilled = report
                lastHistory = report.history
                if report.rebuilt { rebuildCount += 1 }
            case .piece(let piece):
                text += piece
                produced += 1
                onPiece(piece)
                if let shouldStop, shouldStop(produced) { break reading }
            }
        }
        return GGUFTurnReply(text: text, prompt: prefilled?.prompt ?? "", history: prefilled?.history ?? [])
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

    struct Request: Sendable {
        let messages: [LlamaChatMessage]
        let maxTokens: Int
        let renderer: LlamaPromptRenderer
    }

    struct Prefill: Sendable {
        let history: [LlamaChatMessage]
        let prompt: String
        /// The context had to be rebuilt first.
        let rebuilt: Bool
    }

    enum Event: Sendable {
        case prefilled(Prefill)
        case piece(String)
    }

    private init(url: URL, parameter: LlamaClient.Parameter) throws {
        modelFile = url
        loadParameters = parameter
        liveContext = try LocalLLMClientLlama.Context(url: url, parameter: parameter)
        fitter = GGUFPromptFitter(windowSize: parameter.context)
    }

    /// Loads the model off the main actor.
    static func load(url: URL, parameter: LlamaClient.Parameter) async throws -> LlamaTurnCore {
        try LlamaTurnCore(url: url, parameter: parameter)
    }

    var currentContext: LocalLLMClientLlama.Context? { liveContext }

    func run(_ request: Request, into events: AsyncThrowingStream<Event, any Error>.Continuation) async {
        do {
            // A turn cancelled before it got here (e.g. while queued) skips the prefill too.
            guard !Task.isCancelled else { return events.finish() }
            let (context, rebuilt) = try preparedContext()
            let (history, prompt) = try prefill(request, into: context)
            events.yield(.prefilled(Prefill(history: history, prompt: prompt, rebuilt: rebuilt)))

            var produced = 0
            let pieces = LocalLLMClientLlama.Generator(context: context)
            for try await piece in pieces {
                events.yield(.piece(piece))
                produced += 1
                if produced >= request.maxTokens || Task.isCancelled { break }
            }
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

    private func prefill(_ request: Request,
                         into context: LocalLLMClientLlama.Context) throws -> (history: [LlamaChatMessage], prompt: String) {
        var (history, prompt) = try fitter.fitted(LlamaChatNormalizer.normalize(request.messages),
                                                  replyBudget: request.maxTokens, render: request.renderer.render)
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
                guard let shorter = try fitter.shortened(history, replyBudget: request.maxTokens,
                                                         render: request.renderer.render) else { throw error }
                (history, prompt) = shorter
            }
        }
    }
}

// MARK: - LlamaToolTurnEngine

/// The tool-calling turn: LocalLLMClient's parser on a `LlamaClient`, fed `.plain` prompts (which clear the KV cache)
/// rendered with real roles, and the call → execute → resume loop `LLMSession` runs.
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

    /// Text and pieces across the first stream and the resumed one, which share `maxTokens`.
    private struct Progress {
        let limit: Int
        var text = ""
        var produced = 0
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
        shouldStop: ((Int) -> Bool)?,
        onPiece: (String) -> Void
    ) async throws -> GGUFTurnReply {
        try await waitForDrain()
        try Task.checkCancellation()

        var progress = Progress(limit: maxTokens)
        let first = try await stream(LlamaChatNormalizer.normalize(messages), trims: true, progress: &progress,
                                     shouldStop: shouldStop, onPiece: onPiece)
        guard case .finished(let calls) = first.outcome, !calls.isEmpty else {
            return GGUFTurnReply(text: progress.text, prompt: first.prompt, history: first.history)
        }

        var resumed = first.history
        for call in calls {
            resumed.append(.tool(try await execute(call)))
        }
        try await waitForDrain()
        try Task.checkCancellation()
        let second = try await stream(resumed, trims: false, progress: &progress, shouldStop: shouldStop,
                                      onPiece: onPiece)
        return GGUFTurnReply(text: progress.text, prompt: second.prompt, history: second.history)
    }

    /// One `responseStream`. Returns the history it sent, after any trimming, and the prompt that rendered.
    private func stream(
        _ messages: [LlamaChatMessage],
        trims: Bool,
        progress: inout Progress,
        shouldStop: ((Int) -> Bool)?,
        onPiece: (String) -> Void
    ) async throws -> (outcome: Outcome, history: [LlamaChatMessage], prompt: String) {
        var (history, prompt) = trims
            ? try fitter.fitted(messages, replyBudget: progress.limit, render: renderer.render)
            : (messages, try renderer.render(messages))
        let chunks: AsyncThrowingStream<StreamingChunk, any Error>
        while true {
            let client = try await liveClient()
            do {
                chunks = try await client.responseStream(from: .plain(prompt))
                break
            } catch {
                await discardQueuedTokens(of: client)
                guard trims, fitter.learn(from: error, promptBytes: prompt.utf8.count),
                      let shorter = try fitter.shortened(history, replyBudget: progress.limit,
                                                         render: renderer.render) else { throw error }
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

    /// Formats the output the way `LLMSession` does, so the resumed prompt matches its tool loop.
    private func execute(_ call: LLMToolCall) async throws -> String {
        guard let tool = renderer.tools.first(where: { $0.name == call.name }) else {
            throw LLMError.invalidParameter(reason: "Tool '\(call.name)' not found")
        }
        let output = try await tool.call(argumentsJSON: call.arguments)
        return output.data.map { "\($0.key): \($0.value)" }.joined(separator: ", ")
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
