import Foundation
import LocalLLMClient
import LocalLLMClientLlama
import Synchronization

// MARK: - LlamaCppBackend

/// Inference backend powered by llama.cpp via LocalLLMClient.
///
/// Loads the full GGUF model into memory and runs inference using llama.cpp's
/// optimized Metal compute kernels. Best for GGUF models that fit in device RAM.
///
/// For models too large for monolithic loading, see ``LayerStreamingBackend``.
@MainActor
final class LlamaCppBackend: InferenceBackend {

    // MARK: - State

    private var session: LLMSession?
    /// The session's client. LocalLLMClient also holds it in its own decode tasks, so `unload()` hands it to
    /// ``releasingClient`` and drops it only once those tasks have let go.
    private var sessionClient: LlamaClient?
    /// A client `unload()` dropped while a LocalLLMClient task may still hold it; see ``awaitRelease(timeout:)``.
    private var releasingClient: LlamaClient?
    /// The GGUF the session loaded; dedicated constrained clients map the same file.
    private var loadedFileURL: URL?
    /// Orders calls into the session and dedicated clients. It cannot wait for decodes LocalLLMClient keeps
    /// running in its own tasks after a call returns (cancelled prefill, tools cut by maxTokens).
    private let decodeLock = DecodeGate()
    private let model: Model
    private let temperature: Float
    private let tools: [any LLMTool]
    private(set) var lastGenerationStats: GenerationStats?

    var isLoaded: Bool { session != nil }
    /// Tool names registered for function-calling — a testable view of what was threaded in (empty = none).
    var registeredToolNames: [String] { tools.map(\.name) }

    // MARK: - Init

    init(model: Model, temperature: Float? = nil, tools: [any LLMTool] = []) {
        self.model = model
        self.temperature = temperature ?? 0.7
        self.tools = tools
    }

    // MARK: - InferenceBackend

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        guard session == nil else { return }

        guard let filename = model.ggufFilename else {
            throw AuraError.invalidResponse("Model \(model.displayName) has no GGUF filename")
        }

        // In-place user file (localFileURL), else the download destination under cacheDirectory.
        let ggufURL = model.localFileURL ?? model.cacheDirectory.appendingPathComponent(filename)

        guard FileManager.default.fileExists(atPath: ggufURL.path) else {
            throw AuraError.invalidResponse(
                "GGUF file not found. Download \(model.displayName) first."
            )
        }

        onProgress("Loading \(model.displayName)...")

        let parameter = LlamaClient.Parameter(
            context: contextSize(),
            numberOfThreads: threadCount(),
            batch: batchSize(),
            temperature: temperature,
            topP: 0.95
        )

        let built = ClientSlot()
        let localModel = RetainingLlamaModel(fileURL: ggufURL, configuration: parameter, built: built)

        // Passing `tools` enables automatic function-calling: `session.streamResponse` routes through the
        // tool-calling path, so the existing generate loop transparently invokes tools and feeds results back.
        let newSession = LLMSession(model: localModel, tools: tools)
        try await newSession.prewarm()
        session = newSession
        sessionClient = built.take()
        loadedFileURL = ggufURL

        onProgress("\(model.displayName) ready")
    }

    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard session != nil else { throw AuraError.modelNotLoaded }
        let fullPrompt = Self.flattenedPrompt(prompt, systemPrompt: systemPrompt)

        return try await decodeLock.run {
            // A stream started from a cancelled task returns at once but leaves its prefill running on the context.
            try Task.checkCancellation()
            guard let session else { throw AuraError.modelNotLoaded }

            // Stateless per call: callers (e.g. AuraLocal's `AIProvider` contract) pass the full conversation
            // as `prompt`, so clear any accumulated session history first — otherwise the model re-ingests
            // prior turns on top of the transcript (O(N²) growth → context overflow → truncated/empty replies).
            session.messages = []
            lastGenerationStats = nil

            var fullText = ""
            var tokenCount = 0
            var meter = StreamMeter()
            try await session.streamReply(to: fullPrompt) { partial in
                meter.recordPiece()
                fullText += partial
                tokenCount += 1
                onToken(fullText)
                return tokenCount < maxTokens   // honor the caller's token budget
            }

            lastGenerationStats = meter.stats(backend: .llamaCpp, source: .llamaStream(toolsRegistered: !tools.isEmpty),
                                              prompt: fullPrompt, output: fullText)
            return fullText
        }
    }

    // Pinned LocalLLMClient never resets its sampler (contrib/localllmclient/0003): each call gets its own client.
    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        constraint: GenerationConstraint,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        if case .grammar(let gbnf, let root) = constraint,
           gbnf.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw GenerationConstraintError.invalidGrammar(root: root)
        }
        guard session != nil, loadedFileURL != nil else { throw AuraError.modelNotLoaded }

        let fullPrompt = Self.flattenedPrompt(prompt, systemPrompt: systemPrompt)
        let window = Self.constrainedContextWindow(forPrompt: fullPrompt, maxTokens: maxTokens, cap: contextSize())
        let parameter = LlamaClient.Parameter(
            context: window,
            numberOfThreads: threadCount(),
            batch: min(window, 2048),
            temperature: temperature,
            topP: 0.95,
            options: .init(responseFormat: constraint.llamaFormat)
        )

        return try await decodeLock.run {
            try Task.checkCancellation()
            // unload() (e.g. ModelManager.evict) may have run while this call was queued.
            guard session != nil, let loadedFileURL else { throw AuraError.modelNotLoaded }
            lastGenerationStats = nil
            let reply = try await Self.generateOnDedicatedClient(
                url: loadedFileURL,
                parameter: parameter,
                constraint: constraint,
                prompt: fullPrompt,
                maxTokens: maxTokens,
                onToken: onToken
            )
            lastGenerationStats = reply.stats
            return reply.text
        }
    }

    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        // llama.cpp text-only — ignore image
        try await generate(
            prompt: prompt,
            systemPrompt: nil,
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    func generate(
        messages: [[String: String]],
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        var systemPrompt: String?
        var userMessages: [String] = []

        for msg in messages {
            let role = msg["role"] ?? "user"
            let content = msg["content"] ?? ""
            if role == "system" {
                systemPrompt = content
            } else {
                userMessages.append(content)
            }
        }

        let prompt = userMessages.joined(separator: "\n")
        return try await generate(
            prompt: prompt,
            systemPrompt: systemPrompt,
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    func unload() {
        session = nil
        loadedFileURL = nil
        guard let sessionClient else { return }
        releasingClient = sessionClient
        self.sessionClient = nil
        Task { [weak self] in await self?.awaitRelease() }
    }

    /// Frees the context `unload()` dropped once no LocalLLMClient task holds it (a decode cut short by the token
    /// budget finishes its token first), waiting at most `timeout`. The free then happens here, on the main actor,
    /// before this returns: a process that exits with a context alive aborts in `ggml_metal_device_free`.
    func awaitRelease(timeout: Duration = .seconds(5)) async {
        let deadline = ContinuousClock.now + timeout
        while releasingClient != nil, !isKnownUniquelyReferenced(&releasingClient), ContinuousClock.now < deadline {
            do {
                try await Task.sleep(for: .milliseconds(1))
            } catch {
                break
            }
        }
        releasingClient = nil
    }

    // MARK: - Constrained generation

    /// Tokens reserved for the chat template around the prompt (role markers, a default system prompt).
    static let templateTokenAllowance = 256

    /// KV window for a dedicated constrained client. A token never covers less than one UTF-8 byte, so the
    /// prompt's byte count bounds its token count without loading a tokenizer. Terms are clamped to `cap`
    /// first so `maxTokens: .max` cannot overflow.
    static func constrainedContextWindow(forPrompt prompt: String, maxTokens: Int, cap: Int) -> Int {
        let needed = min(prompt.utf8.count, cap) + min(max(maxTokens, 0), cap) + templateTokenAllowance
        let padded = (needed + 255) / 256 * 256   // llama.cpp pads n_ctx to 256 anyway
        return min(padded, cap)
    }

    static func flattenedPrompt(_ prompt: String, systemPrompt: String?) -> String {
        guard let systemPrompt else { return prompt }
        return "System: \(systemPrompt)\n\nUser: \(prompt)"
    }

    /// The generator runs in its own task because LocalLLMClient parks it on `willResignActive` until the app is
    /// active again, and a cancelled caller must not wait for that. On a normal end or the token budget this waits
    /// for the decode to stop, so the client (context, KV cache, model handle) is freed before it returns; when
    /// cancelled it returns at once and a parked decode finishes one token after the app becomes active.
    nonisolated private static func generateOnDedicatedClient(
        url: URL,
        parameter: LlamaClient.Parameter,
        constraint: GenerationConstraint,
        prompt: String,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> (text: String, stats: GenerationStats) {
        let client = try await LocalLLMClient.llama(url: url, parameter: parameter)
        if case .grammar(_, let root) = constraint, !grammarSamplerExists(in: client) {
            throw GenerationConstraintError.invalidGrammar(root: root)
        }

        var fullText = ""
        var tokenCount = 0
        // Started after the client loads: `textStream` renders, tokenizes and prefills before it returns.
        var meter = StreamMeter()
        let pieces = try client.textStream(from: .chat([.user(prompt)]))
        let (stream, continuation) = AsyncThrowingStream<String, any Error>.makeStream()
        let decoder = Task {
            do {
                for try await piece in pieces { continuation.yield(piece) }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        defer { decoder.cancel() }
        // Iterating the stream returns at once when this task is cancelled, even while the decoder is parked.
        for try await piece in stream where tokenCount < maxTokens {
            meter.recordPiece()
            fullText += piece
            tokenCount += 1
            await onToken(fullText)
            if tokenCount >= maxTokens { decoder.cancel() }
        }
        // Cancellation ends the stream early; the partial text would not satisfy the constraint.
        try Task.checkCancellation()
        return (fullText, meter.stats(backend: .llamaCpp, source: .measured, prompt: prompt, output: fullText))
    }

    /// 0.5.0 chains a nil sampler when llama.cpp rejects a GBNF and crashes on the first token, so read its
    /// private state first. If an upgrade renames it, this fails closed.
    nonisolated static func grammarSamplerExists(in client: LlamaClient) -> Bool {
        guard let sampler = Mirror(reflecting: client).descendant("context", "grammer") else { return false }
        let optional = Mirror(reflecting: sampler)
        return optional.displayStyle != .optional || !optional.children.isEmpty
    }

    // MARK: - Platform Configuration

    /// KV-aware window for THIS model — a hardcoded platform tier used to collapse
    /// every model to 2048 (iOS) / 8192 (macOS) no matter what it supported.
    private func contextSize() -> Int {
        HardwareAnalyzer.recommendedContextWindow(for: model)
    }

    /// Tokens submitted per `llama_decode` while prefilling.
    ///
    /// LocalLLMClient chunks the prompt by this value (its `Decoder` strides over the token array),
    /// so the 512 default splits a 4 000-token prompt into eight round trips. The staging buffer
    /// costs roughly 30 bytes per token — ~60 KB at 2048 — so context length is the only real
    /// ceiling worth respecting.
    ///
    /// Note this is NOT llama.cpp's `n_ubatch`, which governs the GPU-side prefill split and is the
    /// knob published benchmarks tune; LocalLLMClient does not expose it. The win here is fewer
    /// round trips, not a larger GPU batch, so expect a smaller effect than those numbers suggest.
    private func batchSize() -> Int {
        min(contextSize(), 2048)
    }

    private func threadCount() -> Int {
        #if os(macOS)
        return min(ProcessInfo.processInfo.activeProcessorCount, 8)
        #else
        return min(ProcessInfo.processInfo.activeProcessorCount, 4)
        #endif
    }
}

// MARK: - Session model

/// `LLMSession.LocalModel.llama(url:parameter:)` that also hands the client it builds to the backend.
private struct RetainingLlamaModel: LLMSession.Model {
    let fileURL: URL
    let configuration: LlamaClient.Parameter
    let built: ClientSlot

    var makeClient: @Sendable ([AnyLLMTool]) async throws -> AnyLLMClient {
        { [fileURL, configuration, built] tools in
            let client = try await LocalLLMClient.llama(url: fileURL, parameter: configuration,
                                                        tools: tools.map(\.underlyingTool))
            built.store(client)
            return AnyLLMClient(client)
        }
    }

    func prewarm() async throws {}
}

/// Carries the client out of LocalLLMClient's loading task.
private final class ClientSlot: Sendable {
    private let stored = Mutex<LlamaClient?>(nil)

    func store(_ client: LlamaClient) {
        stored.withLock { $0 = client }
    }

    func take() -> LlamaClient? {
        stored.withLock { slot in
            defer { slot = nil }
            return slot
        }
    }
}

// MARK: - Streaming a reply

extension LLMSession {
    /// Streams the reply to `prompt`, passing each piece to `onPiece` until it returns `false`, the reply ends or
    /// the task is cancelled.
    ///
    /// LocalLLMClient decodes in its own tasks and cancels them only when the stream is released, not when the loop
    /// breaks; the outer task then appends the reply to `messages`. So the stream lives only in
    /// ``read(_:onPiece:)``, and an early stop waits for that append, so it cannot land in the next call's
    /// `messages`. The wait is bounded, because a failed decode appends nothing, and a cancelled caller does not
    /// wait. The inner task can still be finishing its token; `unload()` waits for that before freeing.
    @MainActor
    func streamReply(to prompt: String, onPiece: (String) -> Bool) async throws {
        let stoppedEarly = try await Self.read(streamResponse(to: prompt), onPiece: onPiece)
        guard stoppedEarly else { return }
        let deadline = ContinuousClock.now + .seconds(2)
        while messages.last?.role != .assistant, ContinuousClock.now < deadline {
            do {
                try await Task.sleep(for: .milliseconds(2))
            } catch {
                return
            }
        }
    }

    /// `true` when `onPiece` stopped the stream.
    @MainActor
    private static func read(_ stream: AsyncThrowingStream<String, any Error>,
                             onPiece: (String) -> Bool) async throws -> Bool {
        for try await piece in stream {
            guard !Task.isCancelled else { return false }
            if !onPiece(piece) { return true }
        }
        return false
    }
}

// MARK: - GenerationConstraint → LocalLLMClient

extension GenerationConstraint {
    var llamaFormat: LlamaClient.ResponseFormat {
        switch self {
            case .json:
                return .json
            case .grammar(let gbnf, let root):
                return .grammar(gbnf: gbnf, root: root)
        }
    }
}
