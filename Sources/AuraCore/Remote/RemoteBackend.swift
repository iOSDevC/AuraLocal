import Foundation

/// Adapts a ``RemoteLLMProvider`` to the ``InferenceBackend`` protocol so remote
/// generation flows through the same `AuraEngine` / `AuraLocal.stream()` path as
/// local models. Honors the cumulative-`onToken` contract (accumulate SSE deltas
/// before each call) and cancellation.
@MainActor
final class RemoteBackend: InferenceBackend {
    let target: RemoteTarget
    private let temperature: Float

    /// Usage reported by the last successful generation (for the cost ledger).
    private(set) var lastUsage: TokenUsage?

    init(target: RemoteTarget, temperature: Float = 0.7) {
        self.target = target
        self.temperature = temperature
    }

    var isLoaded: Bool { true }

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        onProgress("Using \(target.provider.displayName)")
    }

    func unload() {}

    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        try await run(system: systemPrompt,
                      messages: [["role": "user", "content": prompt]],
                      maxTokens: maxTokens, onToken: onToken)
    }

    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        throw AuraError.invalidResponse("Remote vision escalation is not supported yet.")
    }

    func generate(
        messages: [[String: String]],
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        var system: String?
        var msgs = messages
        if let first = msgs.first, first["role"] == "system" {
            system = first["content"]
            msgs.removeFirst()
        }
        return try await run(system: system, messages: msgs, maxTokens: maxTokens, onToken: onToken)
    }

    // MARK: - Core

    private func run(
        system: String?,
        messages: [[String: String]],
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        lastUsage = nil
        let request = RemoteRequest(
            model: target.modelID, system: system, messages: messages,
            maxTokens: maxTokens, temperature: temperature)
        var full = ""
        for try await event in target.provider.stream(request) {
            try Task.checkCancellation()
            switch event {
            case .token(let delta):
                full += delta
                onToken(full)          // cumulative — AuraLocal.stream() re-diffs to deltas
            case .usage(let usage):
                lastUsage = usage
            }
        }
        return full
    }
}
