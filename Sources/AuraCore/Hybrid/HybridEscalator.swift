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
        /// Number of secrets/PII items redacted before sending (0 if redaction off).
        public let redactedPIICount: Int
        /// `true` when the answer came from the response cache (no remote call, $0).
        public let fromCache: Bool
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
        redactPII: Bool = false,
        onToken: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> Result {
        // Reserve room for the answer; keep well under the remote context window.
        let contextWindow = target.contextLength ?? 8192
        let budget = max(256, Int(Double(contextWindow) * 0.5) - maxTokens)
        let compressed = compressor.compress(context: context, question: question, budgetTokens: budget)

        // Optional privacy backstop: strip obvious secrets/PII before sending.
        var contextText = compressed.keptText
        var redactedCount = 0
        if redactPII {
            let redaction = PIIRedactor().redact(compressed.keptText)
            contextText = redaction.redacted
            redactedCount = redaction.count
        }

        let userContent = contextText.isEmpty
            ? question
            : "Context:\n\(contextText)\n\nQuestion: \(question)"

        // Response cache: don't pay twice for an identical request.
        if let cached = ResponseCache.shared.lookup(
            provider: target.provider.id, model: target.modelID, prompt: userContent) {
            onToken(cached)
            return Result(
                answer: cached, compression: compressed, usage: nil,
                providerName: target.provider.displayName,
                redactedPIICount: redactedCount, fromCache: true)
        }

        let backend = RemoteBackend(target: target)
        let answer = try await backend.generate(
            prompt: userContent, systemPrompt: systemPrompt,
            maxTokens: maxTokens, onToken: onToken)

        ResponseCache.shared.insert(
            answer, provider: target.provider.id, model: target.modelID, prompt: userContent)

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
            providerName: target.provider.displayName,
            redactedPIICount: redactedCount,
            fromCache: false)
    }

    // MARK: - Cloud targets (Phase 2)

    /// Build cloud escalation targets from Keychain-stored API keys (BYOK).
    public static func cloudTargets(
        allowCloud: Bool,
        anthropicModel: String = "claude-sonnet-4-5",
        openAIModel: String = "gpt-4o"
    ) -> [RemoteTarget] {
        guard allowCloud else { return [] }
        var targets: [RemoteTarget] = []
        if let key = KeychainStore.read(for: "cloud.anthropic") {
            targets.append(RemoteTarget(
                provider: AnthropicProvider(apiKey: key),
                modelID: anthropicModel, contextLength: 200_000, origin: .cloud))
        }
        if let key = KeychainStore.read(for: "cloud.openai") {
            let provider = OpenAICompatibleProvider(
                id: "cloud.openai", displayName: "OpenAI",
                baseURL: URL(string: "https://api.openai.com/v1")!, apiKey: key,
                retentionNote: "Sent to OpenAI's API. See their data-retention policy.")
            targets.append(RemoteTarget(
                provider: provider, modelID: openAIModel, contextLength: 128_000, origin: .cloud))
        }
        return targets
    }

    /// Candidate targets in preference order: the user's own LAN box first, then
    /// cloud (only if the policy allows and keys are configured).
    public static func candidateTargets(policy: EscalationPolicy) async -> [RemoteTarget] {
        var targets: [RemoteTarget] = []
        if let lan = await bestLocalTarget() { targets.append(lan) }
        targets.append(contentsOf: cloudTargets(allowCloud: policy.allowCloud))
        return targets
    }

    // MARK: - Policy-driven escalation (Phase 2)

    /// Consult the router and (if needed) the consent gate, then escalate.
    /// Returns `nil` when the policy/router keeps the request local. Throws
    /// ``AuraError/escalationDeclined`` if the user declines an offer.
    public func routeAndEscalate(
        policy: EscalationPolicy,
        systemPrompt: String? = nil,
        context: String,
        question: String,
        domain: Model.Domain? = nil,
        localContextWindow: Int = 8192,
        consent: any ConsentGate = DenyingConsentGate(),
        maxTokens: Int = 1024,
        onToken: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> Result? {
        let candidates = await Self.candidateTargets(policy: policy)
        guard let target = candidates.first else { return nil }   // R2: no target

        let promptTokens = ContextCompressor.estimateTokens((systemPrompt ?? "") + context + question)
        let projected = CostLedger.projectedCost(target: target, inputTokens: promptTokens, maxOutput: maxTokens)

        let decision = EscalationRouter.decide(RoutingInput(
            policy: policy,
            hasCandidateTarget: true,
            candidateIsCloud: !target.isLocalNetwork,
            online: NetworkMonitor.shared.isOnline,
            promptTokens: promptTokens,
            localContextWindow: localContextWindow,
            localAnswer: nil,
            domain: domain,
            projectedCostUSD: projected))

        switch decision {
        case .stayLocal:
            return nil
        case .escalate:
            return try await escalate(to: target, systemPrompt: systemPrompt,
                                      context: context, question: question,
                                      maxTokens: maxTokens, onToken: onToken)
        case .offer:
            let budget = max(256, Int(Double(target.contextLength ?? 8192) * policy.keepRatio) - maxTokens)
            let preview = compressor.compress(context: context, question: question, budgetTokens: budget)
            guard await consent.requestConsent(target: target, preview: preview, projectedCostUSD: projected) else {
                throw AuraError.escalationDeclined
            }
            return try await escalate(to: target, systemPrompt: systemPrompt,
                                      context: context, question: question,
                                      maxTokens: maxTokens, onToken: onToken)
        }
    }
}
