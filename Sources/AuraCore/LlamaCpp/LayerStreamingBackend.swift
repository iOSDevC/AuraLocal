import Foundation
import LocalLLMClient
import LocalLLMClientLlama

// MARK: - LayerStreamingBackend

/// Inference backend that uses mmap-based streaming for models too large for monolithic loading.
///
/// Configures llama.cpp with minimal GPU layer offload and relies on the OS virtual
/// memory system to page weights from disk on-demand. Peak memory stays at ~500-750 MB
/// regardless of total model size, enabling 7B-13B models on 6-8 GB iOS devices.
///
/// **Trade-off:** Slower than full-GPU inference (2-5 tok/s for 7B Q4) due to
/// CPU computation and disk page faults.
///
/// On **macOS**, this backend is rarely selected since Macs typically have 16-96 GB RAM.
@MainActor
final class LayerStreamingBackend: InferenceBackend {

    // MARK: - State

    private var engine: (any GGUFTurnEngine)?
    /// An engine `unload()` dropped while a decode may still hold its model; see ``awaitRelease(timeout:)``.
    private var releasingEngine: (any GGUFTurnEngine)?
    /// Turns would otherwise overlap on the one context.
    private let decodeLock = DecodeGate()
    private let model: Model
    let sampling: SamplingSettings
    private let memoryManager: MemoryBudgetManager
    private let tools: [any LLMTool]
    private(set) var lastGenerationStats: GenerationStats?
    var toolLoop = ToolLoopOptions.default
    var onToolEvent: (@MainActor (ToolEvent) -> Void)?
    private(set) var lastToolTurn: ToolTurnSummary?

    var isLoaded: Bool { engine != nil }
    /// Tool names registered for function-calling — a testable view of what was threaded in (empty = none).
    var registeredToolNames: [String] { tools.map(\.name) }

    // MARK: - Init

    init(model: Model, sampling: SamplingSettings? = nil, tools: [any LLMTool] = []) {
        self.model = model
        self.sampling = sampling ?? .defaults(for: model)
        self.memoryManager = MemoryBudgetManager(safetyMarginMB: 250)
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

        onProgress("Preparing \(model.displayName) (streaming mode)...")

        let parameter = loadParameter()
        engine = try await GGUFTurnEngines.load(url: ggufURL, parameter: parameter, tools: tools)

        onProgress("\(model.displayName) ready (streaming mode)")
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

    /// A dedicated second context is exactly what a streaming-sized device cannot afford.
    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        constraint: GenerationConstraint,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        throw GenerationConstraintError.unsupportedBackend(.layerStreaming)
    }

    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
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

    private func turn(
        _ conversation: [LlamaChatMessage],
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard engine != nil else { throw AuraError.modelNotLoaded }
        return try await decodeLock.run {
            try Task.checkCancellation()
            guard let engine else { throw AuraError.modelNotLoaded }

            memoryManager.refresh()
            // Adapt max tokens based on memory pressure
            let safeMaxTokens = memoryManager.isUnderPressure ? min(maxTokens, 256) : maxTokens
            let memoryManager = memoryManager

            lastGenerationStats = nil
            lastToolTurn = nil
            let loop = ToolLoopContext(options: toolLoop, onEvent: onToolEvent)
            let reply = try await engine.measuredTurn(
                conversation,
                maxTokens: safeMaxTokens,
                backend: .layerStreaming,
                source: .llamaStream(toolsRegistered: !tools.isEmpty),
                toolLoop: loop,
                // Stop early if memory becomes critical
                shouldStop: { produced in produced % 32 == 0 && memoryManager.isUnderPressure },
                onToken: onToken
            )
            lastGenerationStats = reply.stats
            lastToolTurn = reply.tools
            return reply.text
        }
    }

    func unload() {
        guard let engine else { return }
        self.engine = nil
        releasingEngine = engine
        Task { [weak self] in await self?.awaitRelease() }
    }

    /// Frees the model `unload()` dropped once no decode holds it, waiting at most `timeout`; see
    /// ``LlamaCppBackend/awaitRelease(timeout:)``.
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

    // MARK: - Adaptive Configuration

    /// Configured for minimal memory: small context, CPU-only, mmap.
    func loadParameter() -> LlamaClient.Parameter {
        LlamaClient.Parameter(
            context: adaptiveContextLength(),
            numberOfThreads: streamingThreadCount(),
            // Streaming runs on memory-constrained devices and pages weights from disk, so a huge
            // prefill chunk buys little and lengthens the window where memory can spike. Modest
            // bump over the 512 default, still bounded by the (already small) context.
            batch: min(adaptiveContextLength(), 1024),
            temperature: sampling.temperature,
            topP: sampling.topP
        )
    }

    private func adaptiveContextLength() -> Int {
        let base: Int
        #if os(macOS)
        base = 4096
        #else
        base = 2048
        #endif
        return memoryManager.recommendedContextLength(baseContext: base)
    }

    private func streamingThreadCount() -> Int {
        #if os(macOS)
        return min(ProcessInfo.processInfo.activeProcessorCount, 6)
        #else
        return min(ProcessInfo.processInfo.activeProcessorCount, 3)
        #endif
    }
}

extension LayerStreamingBackend: ToolLoopHost {}
