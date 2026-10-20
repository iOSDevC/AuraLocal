import Foundation
import LocalLLMClient
import LocalLLMClientLlama

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

    private var engine: (any GGUFTurnEngine)?
    /// An engine `unload()` dropped while a decode may still hold its model; see ``awaitRelease(timeout:)``.
    private var releasingEngine: (any GGUFTurnEngine)?
    /// The GGUF the engine loaded; dedicated constrained clients map the same file.
    private var loadedFileURL: URL?
    /// Orders turns and dedicated constrained clients. A turn cut short can leave its decode finishing in the
    /// background; the engine waits for it before the next turn touches the context.
    private let decodeLock = DecodeGate()
    private let model: Model
    /// Fixed at load: the sampler chain is built with the context. Dedicated constrained clients use it too.
    let sampling: SamplingSettings
    private let tools: [any LLMTool]
    private(set) var lastGenerationStats: GenerationStats?
    var toolLoop = ToolLoopOptions.default
    var onToolEvent: (@MainActor (ToolEvent) -> Void)?
    private(set) var lastToolTurn: ToolTurnSummary?

    var isLoaded: Bool { engine != nil }
    /// Tool names registered for function-calling — a testable view of what was threaded in (empty = none).
    var registeredToolNames: [String] { tools.map(\.name) }
    /// The loaded engine, for tests.
    var turnEngine: (any GGUFTurnEngine)? { engine }

    // MARK: - Init

    init(model: Model, sampling: SamplingSettings? = nil, tools: [any LLMTool] = []) {
        self.model = model
        self.sampling = sampling ?? .defaults(for: model)
        self.tools = tools
    }

    // MARK: - InferenceBackend

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        guard engine == nil else { return }

        guard let filename = model.ggufFilename else {
            throw AuraError.invalidResponse("Model \(model.displayName) has no GGUF filename")
        }

        // In-place user file, else the download (storage root first, then a copy left in Caches).
        let ggufURL = model.resolvedFileURL ?? model.cacheDirectory.appendingPathComponent(filename)

        guard FileManager.default.fileExists(atPath: ggufURL.path) else {
            throw AuraError.invalidResponse(
                "GGUF file not found. Download \(model.displayName) first."
            )
        }

        onProgress("Loading \(model.displayName)...")

        let parameter = loadParameter()

        // With `tools` the engine runs the tool loop (`toolLoop`), so the generate calls below invoke tools and feed
        // their results back transparently.
        engine = try await GGUFTurnEngines.load(url: ggufURL, parameter: parameter, tools: tools)
        loadedFileURL = ggufURL

        onProgress("\(model.displayName) ready")
    }

    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        try await turn(LlamaChatMessage.list(prompt: prompt, systemPrompt: systemPrompt), maxTokens: maxTokens,
                       onToken: onToken)
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
        guard let engine, loadedFileURL != nil else { throw AuraError.modelNotLoaded }

        let fullPrompt = try engine.renderer.withoutTools.render(
            LlamaChatMessage.list(prompt: prompt, systemPrompt: systemPrompt)
        )
        let window = Self.constrainedContextWindow(forPrompt: fullPrompt, maxTokens: maxTokens, cap: contextSize())
        let parameter = LlamaClient.Parameter(
            context: window,
            numberOfThreads: threadCount(),
            batch: min(window, 2048),
            temperature: sampling.temperature,
            topP: sampling.topP,
            options: .init(responseFormat: constraint.llamaFormat)
        )

        return try await decodeLock.run {
            try Task.checkCancellation()
            // unload() (e.g. ModelManager.evict) may have run while this call was queued.
            guard self.engine != nil, let loadedFileURL else { throw AuraError.modelNotLoaded }
            lastGenerationStats = nil
            lastToolTurn = nil
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
        try await turn(LlamaChatMessage.list(from: messages), maxTokens: maxTokens, onToken: onToken)
    }

    /// One turn over the whole conversation, with its roles.
    private func turn(
        _ conversation: [LlamaChatMessage],
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard engine != nil else { throw AuraError.modelNotLoaded }
        return try await decodeLock.run {
            try Task.checkCancellation()
            guard let engine else { throw AuraError.modelNotLoaded }
            lastGenerationStats = nil
            lastToolTurn = nil
            let loop = ToolLoopContext(options: toolLoop, onEvent: onToolEvent)
            let reply = try await engine.measuredTurn(
                conversation,
                maxTokens: maxTokens,
                backend: .llamaCpp,
                source: .llamaStream(toolsRegistered: !tools.isEmpty),
                toolLoop: loop,
                onToken: onToken
            )
            lastGenerationStats = reply.stats
            lastToolTurn = reply.tools
            return reply.text
        }
    }

    func unload() {
        loadedFileURL = nil
        guard let engine else { return }
        self.engine = nil
        releasingEngine = engine
        Task { [weak self] in await self?.awaitRelease() }
    }

    /// Frees the model `unload()` dropped once no decode holds it, waiting at most `timeout`; within it the free
    /// happens here, on the main actor. Past it the model is freed only when that decode ends: with tools on a
    /// `LlamaClient` it runs to the end of generation or of the context, and one parked in the background waits for
    /// the app to be active.
    /// A process that exits before then aborts in `ggml_metal_device_free`.
    func awaitRelease(timeout: Duration = .seconds(5)) async {
        let deadline = ContinuousClock.now + timeout
        while let releasingEngine, !releasingEngine.isIdle, ContinuousClock.now < deadline {
            do {
                try await Task.sleep(for: .milliseconds(1))
            } catch {
                break
            }
        }
        releasingEngine = nil
    }

    // MARK: - Constrained generation

    /// Headroom on top of the rendered prompt and the budget.
    static let templateTokenAllowance = 256

    /// KV window for a dedicated constrained client. A token never covers less than one UTF-8 byte, so the
    /// prompt's byte count bounds its token count without loading a tokenizer. Terms are clamped to `cap`
    /// first so `maxTokens: .max` cannot overflow.
    static func constrainedContextWindow(forPrompt prompt: String, maxTokens: Int, cap: Int) -> Int {
        let needed = min(prompt.utf8.count, cap) + min(max(maxTokens, 0), cap) + templateTokenAllowance
        let padded = (needed + 255) / 256 * 256   // llama.cpp pads n_ctx to 256 anyway
        return min(padded, cap)
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
        // Started after the client loads: `textStream` tokenizes and prefills before it returns.
        var meter = StreamMeter()
        // A fresh client, so `.plain` (which clears only the KV cache) also starts from an empty batch.
        let pieces = try client.textStream(from: .plain(prompt))
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

    /// The loaded context's parameters. llama.cpp's temperature sampler keeps only the top token at 0, so 0 is greedy.
    func loadParameter() -> LlamaClient.Parameter {
        LlamaClient.Parameter(
            context: contextSize(),
            numberOfThreads: threadCount(),
            batch: batchSize(),
            temperature: sampling.temperature,
            topP: sampling.topP
        )
    }

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

extension LlamaCppBackend: ToolLoopHost {}

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
