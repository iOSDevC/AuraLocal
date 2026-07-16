#if !targetEnvironment(simulator)
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

    private var session: LLMSession?
    private let model: Model
    private let temperature: Float
    private let tools: [any LLMTool]

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
            temperature: temperature,
            topP: 0.95
        )

        let localModel = LLMSession.LocalModel.llama(
            url: ggufURL,
            parameter: parameter
        )

        // Passing `tools` enables automatic function-calling: `session.streamResponse` routes through the
        // tool-calling path, so the existing generate loop transparently invokes tools and feeds results back.
        let newSession = LLMSession(model: localModel, tools: tools)
        try await newSession.prewarm()
        session = newSession

        onProgress("\(model.displayName) ready")
    }

    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let session else { throw AuraError.modelNotLoaded }

        let fullPrompt: String
        if let sys = systemPrompt {
            fullPrompt = "System: \(sys)\n\nUser: \(prompt)"
        } else {
            fullPrompt = prompt
        }

        // Stateless per call: callers (e.g. AuraLocal's `AIProvider` contract) pass the full conversation
        // as `prompt`, so clear any accumulated session history first — otherwise the model re-ingests
        // prior turns on top of the transcript (O(N²) growth → context overflow → truncated/empty replies).
        session.messages = []

        var fullText = ""
        var tokenCount = 0
        let responseStream = session.streamResponse(to: fullPrompt)
        for try await partial in responseStream {
            guard !Task.isCancelled else { break }
            fullText += partial
            tokenCount += 1
            onToken(fullText)
            if tokenCount >= maxTokens { break }   // honor the caller's token budget
        }

        return fullText
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
    }

    // MARK: - Platform Configuration

    /// KV-aware window for THIS model — a hardcoded platform tier used to collapse
    /// every model to 2048 (iOS) / 8192 (macOS) no matter what it supported.
    private func contextSize() -> Int {
        HardwareAnalyzer.recommendedContextWindow(for: model)
    }

    private func threadCount() -> Int {
        #if os(macOS)
        return min(ProcessInfo.processInfo.activeProcessorCount, 8)
        #else
        return min(ProcessInfo.processInfo.activeProcessorCount, 4)
        #endif
    }
}
#endif
