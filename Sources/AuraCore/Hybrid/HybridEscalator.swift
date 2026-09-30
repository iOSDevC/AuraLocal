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
    /// then Ollama, choosing the model with the largest context window. Models the
    /// server forwards off the machine (Ollama cloud models) are skipped. Returns
    /// `nil` if no local provider is running.
    public static func bestLocalTarget() async -> RemoteTarget? {
        bestLocalTarget(from: await LocalProviderDetector.detectAll())
    }

    /// The pure half of ``bestLocalTarget()``: picks from already-probed statuses.
    nonisolated static func bestLocalTarget(from statuses: [LocalProviderStatus]) -> RemoteTarget? {
        let candidates = statuses.filter(\.isAvailable).compactMap { status in
            bestModel(status).map { (status: status, model: $0) }
        }
        guard let pick = candidates.first(where: { $0.status.kind == .llamaServer }) ?? candidates.first else {
            return nil
        }
        return RemoteTarget(
            provider: OpenAICompatibleProvider.from(pick.status),
            modelID: pick.model.name,
            contextLength: pick.model.contextLength,
            origin: .localNetwork(pick.status.kind))
    }

    private nonisolated static func bestModel(_ status: LocalProviderStatus) -> LocalProviderModel? {
        status.models.filter(\.runsLocally).max { ($0.contextLength ?? 0) < ($1.contextLength ?? 0) }
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
        var questionText = question
        var redactedCount = 0
        if redactPII {
            let redactor = PIIRedactor()
            let contextRedaction = redactor.redact(compressed.keptText)
            let questionRedaction = redactor.redact(question)
            contextText = contextRedaction.redacted
            questionText = questionRedaction.redacted
            redactedCount = contextRedaction.count + questionRedaction.count
        }

        let userContent = contextText.isEmpty
            ? questionText
            : "Context:\n\(contextText)\n\nQuestion: \(questionText)"

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

    /// Build cloud escalation targets from Keychain-stored API keys (BYOK):
    /// `cloud.anthropic`, then `cloud.openai`.
    public static func cloudTargets(
        allowCloud: Bool,
        anthropicModel: String = RemoteTarget.defaultAnthropicModel,
        openAIModel: String = RemoteTarget.defaultOpenAIModel
    ) -> [RemoteTarget] {
        cloudTargets(
            allowCloud: allowCloud, anthropicModel: anthropicModel, openAIModel: openAIModel,
            readKey: KeychainStore.read(for:))
    }

    /// ``cloudTargets(allowCloud:anthropicModel:openAIModel:)`` with an injected key reader.
    nonisolated static func cloudTargets(
        allowCloud: Bool,
        anthropicModel: String = RemoteTarget.defaultAnthropicModel,
        openAIModel: String = RemoteTarget.defaultOpenAIModel,
        readKey: (String) -> String?
    ) -> [RemoteTarget] {
        guard allowCloud else { return [] }
        var targets: [RemoteTarget] = []
        if let key = readKey(CloudAccount.anthropic) {
            targets.append(.anthropic(apiKey: key, model: anthropicModel))
        }
        if let key = readKey(CloudAccount.openAI) {
            targets.append(.openAI(apiKey: key, model: openAIModel))
        }
        return targets
    }

    /// Retired: GitHub shut down GitHub Models on 2026-07-30. A key still stored
    /// under `cloud.github-models` is no longer read.
    @available(*, unavailable, message: "GitHub retired GitHub Models on 2026-07-30. Drop gitHubModelsModel and call cloudTargets(allowCloud:anthropicModel:openAIModel:), or use a local llama-server/Ollama or OpenAICompatibleProvider(id:displayName:baseURL:apiKey:).")
    public static func cloudTargets(
        allowCloud: Bool,
        anthropicModel: String = RemoteTarget.defaultAnthropicModel,
        openAIModel: String = RemoteTarget.defaultOpenAIModel,
        gitHubModelsModel: String
    ) -> [RemoteTarget] {
        fatalError("GitHub retired GitHub Models on 2026-07-30.")
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
    ///
    /// Pass `localAnswer` (the local model's draft) to enable the router's
    /// low-confidence trigger (R5/R6) — without it only size-overflow can escalate.
    /// `targets` lets callers reuse already-discovered candidates (e.g. tests).
    public func routeAndEscalate(
        policy: EscalationPolicy,
        systemPrompt: String? = nil,
        context: String,
        question: String,
        domain: Model.Domain? = nil,
        localContextWindow: Int = 8192,
        localAnswer: String? = nil,
        consent: any ConsentGate = DenyingConsentGate(),
        maxTokens: Int = 1024,
        targets: [RemoteTarget]? = nil,
        onToken: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> Result? {
        let candidates: [RemoteTarget]
        if let targets {
            candidates = targets
        } else {
            candidates = await Self.candidateTargets(policy: policy)
        }
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
            localAnswer: localAnswer,
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
