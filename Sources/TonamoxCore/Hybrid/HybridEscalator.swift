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

    /// Escalate to `target`: optionally redact, compress `context` to half the remote's
    /// context window minus `maxTokens`, send system + context + question, stream the
    /// answer, and record cost. Checks no policy, consent or cost cap.
    public func escalate(
        to target: RemoteTarget,
        systemPrompt: String? = nil,
        context: String,
        question: String,
        maxTokens: Int = 1024,
        redactPII: Bool = false,
        onToken: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> Result {
        let payload = preparePayload(
            for: target, systemPrompt: systemPrompt, context: context, question: question,
            maxTokens: maxTokens, keepRatio: 0.5, redactPII: redactPII)
        return try await escalate(payload, to: target, onToken: onToken)
    }

    /// Build the exact request an escalation to `target` sends, without sending it:
    /// with `redactPII`, the context and `question` are redacted, then the context is
    /// compressed to `keepRatio` of the remote's context window minus `maxTokens` (at
    /// least 256 tokens). Show it for consent, then pass it to ``escalate(_:to:onToken:)``.
    public func preparePayload(
        for target: RemoteTarget,
        systemPrompt: String? = nil,
        context: String,
        question: String,
        maxTokens: Int = 1024,
        keepRatio: Double = 0.5,
        redactPII: Bool = false
    ) -> EscalationPayload {
        var contextText = context
        var questionText = question
        var redactedCount = 0
        if redactPII {
            // Before compression: its sentence split breaks emails and dotted tokens apart, hiding them from the patterns.
            let redactor = PIIRedactor()
            let contextRedaction = redactor.redact(context)
            let questionRedaction = redactor.redact(question)
            contextText = contextRedaction.redacted
            questionText = questionRedaction.redacted
            redactedCount = contextRedaction.count + questionRedaction.count
        }

        let contextWindow = target.contextLength ?? 8192
        let budget = max(256, Int(Double(contextWindow) * keepRatio) - maxTokens)
        let compression = compressor.compress(context: contextText, question: questionText, budgetTokens: budget)

        let userMessage = compression.keptText.isEmpty
            ? questionText
            : "Context:\n\(compression.keptText)\n\nQuestion: \(questionText)"
        return EscalationPayload(
            systemPrompt: systemPrompt, userMessage: userMessage, maxTokens: maxTokens,
            redactionEnabled: redactPII, redactedPIICount: redactedCount, compression: compression)
    }

    /// Send `payload` to `target` exactly as built (no recompression or redaction),
    /// stream the answer, and record cost. An earlier request with the same provider,
    /// model, messages and `maxTokens` is answered from ``ResponseCache`` without a remote call.
    public func escalate(
        _ payload: EscalationPayload,
        to target: RemoteTarget,
        onToken: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> Result {
        let compressed = payload.compression
        if let cached = ResponseCache.shared.lookup(
            provider: target.provider.id, model: target.modelID, payload: payload) {
            onToken(cached)
            return Result(
                answer: cached, compression: compressed, usage: nil,
                providerName: target.provider.displayName,
                redactedPIICount: payload.redactedPIICount, fromCache: true)
        }

        let backend = RemoteBackend(target: target)
        let answer: String
        do {
            answer = try await backend.generate(
                prompt: payload.userMessage, systemPrompt: payload.systemPrompt,
                maxTokens: payload.maxTokens, onToken: onToken)
        } catch {
            // Output means tokens were billed; without usage the record's cost is unknown, never $0.
            if backend.didReceiveOutput || backend.lastUsage != nil {
                record(target, usage: backend.lastUsage, compression: compressed)
            }
            throw error
        }

        ResponseCache.shared.insert(
            answer, provider: target.provider.id, model: target.modelID, payload: payload)
        record(target, usage: backend.lastUsage, compression: compressed)

        return Result(
            answer: answer,
            compression: compressed,
            usage: backend.lastUsage,
            providerName: target.provider.displayName,
            redactedPIICount: payload.redactedPIICount,
            fromCache: false)
    }

    private func record(_ target: RemoteTarget, usage: TokenUsage?, compression: CompressionResult) {
        let didCompress = compression.originalTokens > compression.compressedTokens
        ledger.record(
            provider: target.provider.id,
            model: target.modelID,
            usage: usage,
            origin: target.origin,
            compressionRatio: didCompress ? compression.ratio : nil)
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
    /// ``TonamoxError/escalationDeclined`` if the gate declines an offer, which the
    /// default ``ConsentGate/requestConsent(target:preview:offer:)`` does when it
    /// is over the cost cap.
    ///
    /// The payload is built once, compressed with `policy.keepRatio` and redacted per
    /// `policy.piiRedaction`; the gate is shown that payload and exactly it is sent. The projected
    /// cost and the cost cap are checked for that payload; size overflow (R4) reads the raw prompt.
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
        // Cost never keeps a request local, so this settles R1-R6 before paying for redaction and compression.
        let trigger = routingInput(
            policy: policy, target: target, promptTokens: promptTokens,
            maxTokens: maxTokens, localContextWindow: localContextWindow,
            localAnswer: localAnswer, domain: domain)
        if EscalationRouter.decide(trigger) == .stayLocal { return nil }

        let payload = preparePayload(
            for: target, systemPrompt: systemPrompt, context: context, question: question,
            maxTokens: maxTokens, keepRatio: policy.keepRatio, redactPII: policy.redactsPII(for: target))
        // R7 prices what is sent; the size rule keeps reading the raw prompt.
        let input = routingInput(
            policy: policy, target: target, promptTokens: promptTokens,
            pricedInputTokens: payload.estimatedInputTokens,
            maxTokens: maxTokens, localContextWindow: localContextWindow,
            localAnswer: localAnswer, domain: domain)
        let decision = EscalationRouter.decide(input)
        if decision == .stayLocal { return nil }

        if case .offer(let reason) = decision {
            let offer = EscalationOffer(
                reason: reason, projectedCostUSD: input.projectedCostUSD,
                sessionSpentUSD: input.sessionSpentUSD, unpricedRecordCount: input.unpricedRecordCount,
                costCapUSD: policy.costCapUSDPerSession, payload: payload)
            guard await consent.requestConsent(target: target, preview: payload.compression, offer: offer) else {
                throw TonamoxError.escalationDeclined
            }
        }
        return try await escalate(payload, to: target, onToken: onToken)
    }

    /// The router's view of one request, priced by this escalator's ledger and
    /// counting what the ledger's session has already spent. The cost is for
    /// `pricedInputTokens` (default `promptTokens`) plus `maxTokens`.
    func routingInput(
        policy: EscalationPolicy,
        target: RemoteTarget,
        promptTokens: Int,
        pricedInputTokens: Int? = nil,
        maxTokens: Int,
        localContextWindow: Int,
        localAnswer: String?,
        domain: Model.Domain?
    ) -> RoutingInput {
        RoutingInput(
            policy: policy,
            hasCandidateTarget: true,
            candidateIsCloud: !target.isLocalNetwork,
            online: NetworkMonitor.shared.isOnline,
            promptTokens: promptTokens,
            localContextWindow: localContextWindow,
            localAnswer: localAnswer,
            domain: domain,
            projectedCostUSD: ledger.projectedCost(
                target: target, inputTokens: pricedInputTokens ?? promptTokens, maxOutput: maxTokens),
            sessionSpentUSD: ledger.sessionCostUSD,
            unpricedRecordCount: ledger.unpricedRecordCount)
    }
}
