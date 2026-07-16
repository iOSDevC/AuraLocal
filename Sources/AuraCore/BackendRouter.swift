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
            #if targetEnvironment(simulator)
            // MLX needs a Metal GPU family the simulator does not provide; upstream mlx-swift
            // documents that MLX cannot run there. Fail clearly instead of crashing mid-load.
            return UnavailableBackend(
                "MLX models can't run on the iOS Simulator (no Metal GPU). Use a GGUF model here, "
                + "or run on a device.")
            #elseif canImport(MLXLLM)
            return MLXBackend(model: model, temperature: temperature)
            #else
            return UnavailableBackend(
                "This is an MLX model, but the MLX backend isn't linked. Add the MLX products or "
                + "use a GGUF model.")
            #endif

        case .gguf:
            // Runs on the simulator too: llama.xcframework ships an ios-arm64_x86_64-simulator
            // slice and upstream sets n_gpu_layers = 0 there, so it falls back to CPU inference.
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
            let assessment = HardwareAnalyzer.assess(model, profile: profile)
            return assessment.fitLevel == .streamingRequired ? .layerStreaming : .llamaCpp
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
