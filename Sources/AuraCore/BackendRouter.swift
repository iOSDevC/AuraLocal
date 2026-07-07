import Foundation
import LocalLLMClientCore

// MARK: - ModelFormat

/// The weight format of a model, determining which backend can load it.
public enum ModelFormat: String, Sendable, Codable {
    /// MLX safetensors format — loaded by ``MLXBackend``.
    case mlx
    /// GGUF quantized format — loaded by ``LlamaCppBackend`` or ``LayerStreamingBackend``.
    case gguf
}

// MARK: - BackendRouter

/// Selects the appropriate ``InferenceBackend`` for a model based on its format
/// and the device's hardware profile.
///
/// Routing logic:
/// - **MLX format** → always ``MLXBackend`` (current behavior)
/// - **GGUF format + fits in RAM** → ``LlamaCppBackend`` (standard full-load)
/// - **GGUF format + too large** → ``LayerStreamingBackend`` (layer-by-layer)
///
/// On macOS, the router is more aggressive with standard mode since Macs
/// typically have 16-96 GB of unified memory.
@MainActor
public enum BackendRouter {

    /// Select the best backend for the given model and device.
    static func selectBackend(
        for model: Model,
        temperature: Float? = nil,
        profile: HardwareProfile = .current(),
        tools: [any LLMTool] = []
    ) -> any InferenceBackend {
        switch model.format {
        case .mlx:
            #if canImport(MLXLLM)
            return MLXBackend(model: model, temperature: temperature)
            #else
            // MLX products were dropped to fix a swift-syntax conflict; MLX models are filtered out of the
            // registry, so this is a defensive fallback that fails clearly if one is selected anyway.
            return UnavailableBackend(
                "This is an MLX model, but the MLX backend isn't built. Use a GGUF or Ollama model.")
            #endif

        case .gguf:
            #if !targetEnvironment(simulator)
            let assessment = HardwareAnalyzer.assess(model, profile: profile)

            switch assessment.fitLevel {
            case .excellent, .good, .marginal:
                // Model fits in RAM — use standard llama.cpp load
                return LlamaCppBackend(model: model, temperature: temperature, tools: tools)

            case .streamingRequired:
                // Model too large for monolithic load — use layer streaming
                return LayerStreamingBackend(model: model, temperature: temperature, tools: tools)

            case .tooLarge:
                // Even streaming can't help (e.g. 70B on iPhone)
                // Return llama.cpp anyway; it will fail with a clear error at load time
                return LlamaCppBackend(model: model, temperature: temperature, tools: tools)
            }
            #else
            fatalError("GGUF models are not supported on the iOS Simulator. Use MLX models instead.")
            #endif
        }
    }

    /// Returns the ``BackendKind`` that would be selected for a model.
    /// Useful for UI to display backend info before loading.
    public static func recommendedBackend(
        for model: Model,
        profile: HardwareProfile = .current()
    ) -> BackendKind {
        switch model.format {
        case .mlx:
            return .mlx
        case .gguf:
            #if !targetEnvironment(simulator)
            let assessment = HardwareAnalyzer.assess(model, profile: profile)
            return assessment.fitLevel == .streamingRequired ? .layerStreaming : .llamaCpp
            #else
            return .llamaCpp  // placeholder — GGUF not available on simulator
            #endif
        }
    }
}

// MARK: - UnavailableBackend

/// A no-op ``InferenceBackend`` for a format whose engine isn't built into this configuration (currently
/// MLX, compiled out when the mlx-swift products aren't a direct dependency). It never loads and fails with
/// a clear message, so a stray MLX model degrades gracefully instead of crashing the router.
@MainActor
final class UnavailableBackend: InferenceBackend {
    private let message: String
    init(_ message: String) { self.message = message }

    var isLoaded: Bool { false }

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        throw AuraError.invalidResponse(message)
    }

    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        throw AuraError.invalidResponse(message)
    }

    func generate(prompt: String, image: PlatformImage?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        throw AuraError.invalidResponse(message)
    }

    func generate(messages: [[String: String]], maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        throw AuraError.invalidResponse(message)
    }

    func unload() {}
}
