import Foundation

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - AuraEngine

/// Unified internal engine that delegates inference to an ``InferenceBackend``.
///
/// By default uses ``MLXBackend`` for models in MLX format.
/// ``BackendRouter`` can select alternative backends (llama.cpp, layer-streaming)
/// based on model format and device capabilities.
@MainActor
final class AuraEngine {

    // MARK: - State

    let backend: any InferenceBackend
    let model: Model

    // MARK: - Init

    init(model: Model, temperature: Float? = nil) {
        self.model = model
        self.backend = BackendRouter.selectBackend(for: model, temperature: temperature)
    }

    /// Create an engine with a specific backend (used by BackendRouter for GGUF models).
    init(model: Model, backend: any InferenceBackend) {
        self.model = model
        self.backend = backend
    }

    // MARK: - Load

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        try await backend.load(onProgress: onProgress)
    }

    // MARK: - Generate (text-only)

    func generate(
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        try await backend.generate(
            prompt: prompt,
            systemPrompt: systemPrompt,
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    // MARK: - Generate (vision)

    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int = 800,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        try await backend.generate(
            prompt: prompt,
            image: image,
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    // MARK: - Generate (multi-turn messages)

    func generate(
        messages: [[String: String]],
        maxTokens: Int = 1024,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        try await backend.generate(
            messages: messages,
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    // MARK: - Cleanup

    func unload() {
        backend.unload()
    }
}
