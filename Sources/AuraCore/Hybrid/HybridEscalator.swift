import Foundation

/// Manual escalation engine (Phase 1 MVP): sends a request to a more powerful
/// remote model — the user's own `llama-server`/Ollama box — after compressing
/// the context to save tokens. Local-first and fail-closed: callers keep the
/// local answer if this throws.
@MainActor
public final class HybridEscalator {
    private let compressor = ContextCompressor()
    private let ledger: CostLedger

    public init(ledger: CostLedger = .shared) { self.ledger = ledger }

    /// The result of an escalation: the remote answer plus the compression
    /// receipt and usage (for the cost ledger / UI).
    public struct Result: Sendable {
        public let answer: String
        public let compression: CompressionResult
        public let usage: TokenUsage?
        public let providerName: String
    }

    // MARK: - Target discovery

    /// Discover the best local-network escalation target: prefers `llama-server`,
    /// then Ollama, choosing the model with the largest context window. Returns
    /// `nil` if no local provider is running.
    public static func bestLocalTarget() async -> RemoteTarget? {
        let available = await LocalProviderDetector.detectAll().filter(\.isAvailable)
        let ordered = available.sorted { a, _ in a.kind == .llamaServer }
        guard let status = ordered.first, let model = bestModel(status) else { return nil }
        return RemoteTarget(
            provider: OpenAICompatibleProvider.from(status),
            modelID: model.name,
            contextLength: model.contextLength,
            origin: .localNetwork(status.kind))
    }

    private static func bestModel(_ status: LocalProviderStatus) -> LocalProviderModel? {
        status.models.max { ($0.contextLength ?? 0) < ($1.contextLength ?? 0) }
    }

    // MARK: - Escalation

    /// Escalate to `target`: compress `context` toward the remote's budget, send
    /// system + context + question, stream the answer, and record cost.
    public func escalate(
        to target: RemoteTarget,
        systemPrompt: String? = nil,
        context: String,
        question: String,
        maxTokens: Int = 1024,
        onToken: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> Result {
        // Reserve room for the answer; keep well under the remote context window.
        let contextWindow = target.contextLength ?? 8192
        let budget = max(256, Int(Double(contextWindow) * 0.5) - maxTokens)
        let compressed = compressor.compress(context: context, question: question, budgetTokens: budget)

        let userContent = compressed.keptText.isEmpty
            ? question
            : "Context:\n\(compressed.keptText)\n\nQuestion: \(question)"

        let backend = RemoteBackend(target: target)
        let answer = try await backend.generate(
            prompt: userContent, systemPrompt: systemPrompt,
            maxTokens: maxTokens, onToken: onToken)

        let didCompress = compressed.originalTokens > compressed.compressedTokens
        ledger.record(
            provider: target.provider.id,
            usage: backend.lastUsage,
            origin: target.origin,
            compressionRatio: didCompress ? compressed.ratio : nil)

        return Result(
            answer: answer,
            compression: compressed,
            usage: backend.lastUsage,
            providerName: target.provider.displayName)
    }
}
