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
/// - **MLX format** → ``MLXBackend``; with tools, only for a text model: a vision or document model gets a backend
///   whose load throws ``ModelConfigurationError/toolsUnsupported(_:)``
/// - **GGUF format + fits in RAM** → ``LlamaCppBackend`` (standard full-load)
/// - **GGUF format + too large** → ``LayerStreamingBackend`` (layer-by-layer)
///
/// On macOS, the router is more aggressive with standard mode since Macs
/// typically have 16-96 GB of unified memory.
@MainActor
public enum BackendRouter {

    /// Select the best backend for the given model and device. `nil` sampling takes the backend defaults.
    static func selectBackend(
        for model: Model,
        sampling: SamplingSettings? = nil,
        profile: HardwareProfile = .current(),
        tools: [any LLMTool] = []
    ) -> any InferenceBackend {
        switch model.format {
        case .mlx:
            // Public paths throw this before building an engine; refusing here too keeps an internal caller from
            // loading weights that would ignore the tools.
            guard tools.isEmpty || runsToolLoop(for: model) else {
                return UnavailableBackend(failure: ModelConfigurationError.toolsUnsupported(.mlx))
            }
            #if targetEnvironment(simulator)
            // MLX needs a Metal GPU family the simulator does not provide; upstream mlx-swift
            // documents that MLX cannot run there. Fail clearly instead of crashing mid-load.
            return UnavailableBackend(
                "MLX models can't run on the iOS Simulator (no Metal GPU). Use a GGUF model here, "
                + "or run on a device.")
            #elseif canImport(MLXLLM)
            return MLXBackend(model: model, sampling: sampling, tools: tools)
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
                return LlamaCppBackend(model: model, sampling: sampling, tools: tools)

            case .streamingRequired:
                // Model too large for monolithic load — use layer streaming
                return LayerStreamingBackend(model: model, sampling: sampling, tools: tools)

            case .tooLarge:
                // Even streaming can't help (e.g. 70B on iPhone). ModelManager.load refuses this
                // case up front (throws .modelTooLarge before downloading), so this is only reached
                // if a caller drives the router directly — return llama.cpp as a defensive fallback
                // that fails at load rather than silently doing nothing.
                return LlamaCppBackend(model: model, sampling: sampling, tools: tools)
            }
        }
    }

    /// Throws ``ModelConfigurationError/toolsUnsupported(_:)`` when `model`'s backend has no tool-calling loop for
    /// it: an MLX vision or document model, or any MLX model where MLX cannot run.
    nonisolated static func requireToolSupport(_ tools: [any LLMTool], for model: Model) throws {
        guard !tools.isEmpty, !runsToolLoop(for: model) else { return }
        throw ModelConfigurationError.toolsUnsupported(.mlx)
    }

    /// Whether `model`'s backend runs a tool-calling loop, reported as ``ModelCapabilities/tools``: the llama.cpp
    /// backends for every GGUF model, ``MLXBackend`` for text models.
    nonisolated static func runsToolLoop(for model: Model) -> Bool {
        switch model.format {
        case .gguf:
            return true
        case .mlx:
            #if canImport(MLXLLM) && !targetEnvironment(simulator)
            return model.category == .text
            #else
            return false
            #endif
        }
    }

    /// Whether `model`'s backend passes an image to it, reported as ``ModelCapabilities/vision``. The llama.cpp
    /// backends are text-only and drop the image, whatever the model's category says.
    nonisolated static func acceptsImages(for model: Model) -> Bool {
        model.format == .mlx
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

/// A no-op ``InferenceBackend`` for a model this configuration cannot run: MLX on the simulator or without the
/// mlx-swift products, or an MLX vision or document model asked for tools. It never loads and every call throws
/// `failure`, so the model fails clearly instead of crashing the router or answering without its tools.
@MainActor
final class UnavailableBackend: InferenceBackend {
    let failure: any Error
    init(_ message: String) { self.failure = TonamoxError.invalidResponse(message) }
    init(failure: any Error) { self.failure = failure }

    var isLoaded: Bool { false }

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        throw failure
    }

    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        throw failure
    }

    func generate(prompt: String, systemPrompt: String?, maxTokens: Int, constraint: GenerationConstraint,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        throw failure
    }

    func generate(prompt: String, image: PlatformImage?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        throw failure
    }

    func generate(messages: [[String: String]], maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        throw failure
    }

    func unload() {}
}
