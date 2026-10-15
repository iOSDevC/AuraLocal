import Foundation
import LocalLLMClientCore

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
    /// Copied from the backend when a call returns and cleared when one starts, so a call that throws before
    /// decoding cannot leave the previous call's numbers behind.
    private(set) var lastGenerationStats: GenerationStats?
    /// Kept the same way as ``lastGenerationStats``.
    private(set) var lastContextOverflow: ContextWindowOverflow?
    let appLifecycle: BackgroundLifecycle
    /// Opt-in guard checked before and during every generation; `nil` is off.
    var deviceGuard: DeviceConditions?

    // MARK: - Init

    init(model: Model, temperature: Float? = nil, tools: [any LLMTool] = [],
         lifecycle: BackgroundLifecycle = .shared) {
        self.model = model
        self.backend = BackendRouter.selectBackend(for: model, temperature: temperature, tools: tools)
        self.appLifecycle = lifecycle
    }

    /// Create an engine with a specific backend (used by BackendRouter for GGUF models).
    init(model: Model, backend: any InferenceBackend, lifecycle: BackgroundLifecycle = .shared) {
        self.model = model
        self.backend = backend
        self.appLifecycle = lifecycle
    }

    // MARK: - Load

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        appLifecycle.startObserving()
        try await backend.load(onProgress: onProgress)
    }

    // MARK: - Generate (text-only)

    func generate(
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024,
        constraint: GenerationConstraint? = nil,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        try await recordingStats(onToken: onToken) { [self] onToken in
            if let constraint {
                return try await backend.generate(
                    prompt: prompt,
                    systemPrompt: systemPrompt,
                    maxTokens: maxTokens,
                    constraint: constraint,
                    onToken: onToken
                )
            }
            return try await backend.generate(
                prompt: prompt,
                systemPrompt: systemPrompt,
                maxTokens: maxTokens,
                onToken: onToken
            )
        }
    }

    // MARK: - Generate (vision)

    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int = 800,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        try await recordingStats(onToken: onToken) { [self] onToken in
            try await backend.generate(
                prompt: prompt,
                image: image,
                maxTokens: maxTokens,
                onToken: onToken
            )
        }
    }

    // MARK: - Generate (multi-turn messages)

    func generate(
        messages: [[String: String]],
        maxTokens: Int = 1024,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        try await recordingStats(onToken: onToken) { [self] onToken in
            try await backend.generate(
                messages: messages,
                maxTokens: maxTokens,
                onToken: onToken
            )
        }
    }

    private func recordingStats(
        onToken: @escaping @MainActor (String) -> Void,
        _ body: @escaping @MainActor (_ onToken: @escaping @MainActor (String) -> Void) async throws -> String
    ) async throws -> String {
        lastGenerationStats = nil
        lastContextOverflow = nil
        let reply = try await supervised(onToken: onToken, body)
        lastGenerationStats = backend.lastGenerationStats
        lastContextOverflow = backend.lastContextOverflow
        return reply
    }

    // MARK: - Cleanup

    func unload() {
        backend.unload()
    }
}
