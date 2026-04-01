import Foundation

// MARK: - InferenceBackend

/// Abstraction over different inference engines (MLX, llama.cpp, layer-streaming).
///
/// Each backend handles model loading, token generation, and resource cleanup.
/// ``AuraEngine`` delegates to a backend chosen by ``BackendRouter`` based on
/// the model format and device capabilities.
@MainActor
public protocol InferenceBackend: AnyObject {

    /// Whether the backend has a model loaded and ready for inference.
    var isLoaded: Bool { get }

    /// Load the model into memory, reporting progress via `onProgress`.
    func load(onProgress: @escaping @MainActor (String) -> Void) async throws

    /// Generate text from a prompt with optional system prompt.
    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String

    /// Generate text from a vision input (image + prompt).
    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String

    /// Generate text from a multi-turn conversation history.
    func generate(
        messages: [[String: String]],
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String

    /// Release all model resources from memory.
    func unload()
}

// MARK: - BackendKind

/// Identifies which inference engine is in use.
public enum BackendKind: String, Sendable {
    /// Apple MLX framework — loads full model into unified memory.
    case mlx
    /// llama.cpp C library — GGUF format, Metal compute.
    case llamaCpp
    /// Layer-streaming via llama.cpp — loads one transformer layer at a time.
    case layerStreaming
}
