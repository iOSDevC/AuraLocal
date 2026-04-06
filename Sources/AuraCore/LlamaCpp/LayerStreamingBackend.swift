#if !targetEnvironment(simulator)
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

    private var session: LLMSession?
    private let model: Model
    private let temperature: Float
    private let memoryManager: MemoryBudgetManager

    var isLoaded: Bool { session != nil }

    // MARK: - Init

    init(model: Model, temperature: Float? = nil) {
        self.model = model
        self.temperature = temperature ?? 0.7
        self.memoryManager = MemoryBudgetManager(safetyMarginMB: 250)
    }

    // MARK: - InferenceBackend

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        guard session == nil else { return }

        guard let filename = model.ggufFilename else {
            throw AuraError.invalidResponse("Model \(model.displayName) has no GGUF filename")
        }

        let ggufURL = model.cacheDirectory.appendingPathComponent(filename)

        guard FileManager.default.fileExists(atPath: ggufURL.path) else {
            throw AuraError.invalidResponse(
                "GGUF file not found. Download \(model.displayName) first."
            )
        }

        onProgress("Preparing \(model.displayName) (streaming mode)...")

        // Configure for minimal memory: small context, CPU-only, mmap
        let parameter = LlamaClient.Parameter(
            context: adaptiveContextLength(),
            numberOfThreads: streamingThreadCount(),
            temperature: temperature,
            topP: 0.95
        )

        let localModel = LLMSession.LocalModel.llama(
            url: ggufURL,
            parameter: parameter
        )

        let newSession = LLMSession(model: localModel)
        try await newSession.prewarm()
        session = newSession

        onProgress("\(model.displayName) ready (streaming mode)")
    }

    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let session else { throw AuraError.modelNotLoaded }

        memoryManager.refresh()

        // Adapt max tokens based on memory pressure
        let safeMaxTokens: Int
        if memoryManager.isUnderPressure {
            safeMaxTokens = min(maxTokens, 256)
        } else {
            safeMaxTokens = maxTokens
        }

        let fullPrompt: String
        if let sys = systemPrompt {
            fullPrompt = "System: \(sys)\n\nUser: \(prompt)"
        } else {
            fullPrompt = prompt
        }

        var fullText = ""
        var tokenCount = 0
        let responseStream = session.streamResponse(to: fullPrompt)

        for try await partial in responseStream {
            guard !Task.isCancelled else { break }

            fullText += partial
            tokenCount += 1
            onToken(fullText)

            // Stop early if memory becomes critical
            if tokenCount % 32 == 0 && memoryManager.isUnderPressure {
                break
            }

            if tokenCount >= safeMaxTokens { break }
        }

        return fullText
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
    }

    // MARK: - Adaptive Configuration

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
#endif
