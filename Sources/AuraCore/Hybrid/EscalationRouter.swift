import Foundation

/// Why an escalation is being considered.
public enum EscalationReason: String, Sendable, Equatable {
    case sizeOverflow      // prompt won't fit the local context
    case lowConfidence     // local answer looks weak / refused
    case domainSensitive   // security/medicine — bias toward a stronger model
    case userRequested     // explicit manual trigger
    case costCapped        // a trigger fired but session spend + projected cost exceeds the cap
    case costUnknown       // a trigger fired but the cloud target is unpriced, or the session spend is unknown
}

/// The router's per-request verdict.
public enum RoutingDecision: Sendable, Equatable {
    /// Keep the local answer.
    case stayLocal
    /// Escalate now (allowed only for LAN targets under auto mode).
    case escalate(reason: EscalationReason)
    /// Surface an offer; the user confirms.
    case offer(reason: EscalationReason)
}

/// Everything the router needs, gathered by the caller (I/O stays outside so the
/// decision is a pure, unit-testable function).
public struct RoutingInput: Sendable {
    public var policy: EscalationPolicy
    public var hasCandidateTarget: Bool
    public var candidateIsCloud: Bool
    public var online: Bool
    public var promptTokens: Int
    public var localContextWindow: Int
    /// The local answer (nil = pre-attempt, only size-overflow can fire).
    public var localAnswer: String?
    public var domain: Model.Domain?
    /// This request's projected cost; nil when the target's price is unknown.
    public var projectedCostUSD: Decimal?
    /// What the session has already spent (priced records only).
    public var sessionSpentUSD: Decimal
    /// Session records whose cost is unknown; above zero, `sessionSpentUSD` is only a lower bound.
    public var unpricedRecordCount: Int

    public init(
        policy: EscalationPolicy,
        hasCandidateTarget: Bool,
        candidateIsCloud: Bool,
        online: Bool = true,
        promptTokens: Int = 0,
        localContextWindow: Int = 8192,
        localAnswer: String? = nil,
        domain: Model.Domain? = nil,
        projectedCostUSD: Decimal? = nil,
        sessionSpentUSD: Decimal = 0,
        unpricedRecordCount: Int = 0
    ) {
        self.policy = policy
        self.hasCandidateTarget = hasCandidateTarget
        self.candidateIsCloud = candidateIsCloud
        self.online = online
        self.promptTokens = promptTokens
        self.localContextWindow = localContextWindow
        self.localAnswer = localAnswer
        self.domain = domain
        self.projectedCostUSD = projectedCostUSD
        self.sessionSpentUSD = sessionSpentUSD
        self.unpricedRecordCount = unpricedRecordCount
    }
}

/// Local-first escalation policy engine (rules R1–R7). Pure and fail-closed.
public enum EscalationRouter {

    public static func decide(_ input: RoutingInput) -> RoutingDecision {
        // R1 — consent hard gate: OFF wins over everything.
        guard input.policy.mode != .off else { return .stayLocal }
        // R2 — a usable target must exist.
        guard input.hasCandidateTarget else { return .stayLocal }
        // R3 — cloud requires connectivity; LAN/loopback does not.
        if input.candidateIsCloud && !input.online { return .stayLocal }

        // Determine whether a trigger fires.
        let reason: EscalationReason?
        if input.localAnswer == nil {
            // R4 — size overflow (pre-attempt): skip the doomed local call.
            reason = input.promptTokens > Int(0.9 * Double(input.localContextWindow)) ? .sizeOverflow : nil
        } else {
            // R5/R6 — post-attempt uncertainty, biased by sensitive domains.
            reason = isLowConfidence(input.localAnswer!, domain: input.domain) ? .lowConfidence : nil
        }
        guard let reason else { return .stayLocal }

        if let costOffer = costOffer(input) { return costOffer }

        // Escalate vs offer by mode + origin.
        switch input.policy.mode {
        case .off:
            return .stayLocal   // unreachable (R1)
        case .askEachTime:
            return .offer(reason: reason)
        case .autoWithConsentMemory:
            // Auto only for the user's own LAN box; cloud always asks per conversation.
            return input.candidateIsCloud ? .offer(reason: reason) : .escalate(reason: reason)
        }
    }

    /// R7 — cost: over the session cap, or unknown for cloud ⇒ offer, never silent. `nil` when cost does not
    /// stand in the way.
    private static func costOffer(_ input: RoutingInput) -> RoutingDecision? {
        guard let projected = input.projectedCostUSD else {
            return input.candidateIsCloud ? .offer(reason: .costUnknown) : nil
        }
        guard projected > 0 else { return nil }        // adds no spend, so a free LAN call is never capped
        if input.sessionSpentUSD + projected > input.policy.costCapUSDPerSession {
            return .offer(reason: .costCapped)          // certain even when unpriced calls hide more spend
        }
        return input.unpricedRecordCount > 0 ? .offer(reason: .costUnknown) : nil
    }

    // MARK: - Heuristics

    static func isLowConfidence(_ answer: String, domain: Model.Domain?) -> Bool {
        let trimmed = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count < 40 { return true }
        if containsRefusal(trimmed) { return true }
        let sensitive = domain == .security || domain == .medicine
        return sensitive && trimmed.count < 120
    }

    private static func containsRefusal(_ answer: String) -> Bool {
        let lowered = answer.lowercased()
        return refusalPhrases.contains { lowered.contains($0) }
    }

    // Spanish keeps the accented forms only: unaccented "no se" also matches ordinary
    // impersonal phrasing ("no se puede", "no se ha indicado").
    private static let refusalPhrases = [
        "i'm not sure", "i am not sure", "i cannot", "i can't", "i don't know",
        "i do not know", "as an ai", "unable to",
        "no estoy segur", "no puedo", "no sé", "no lo sé", "desconozco",
        "como modelo de lenguaje", "como una ia", "soy una ia",
        "no soy capaz", "me es imposible", "no me es posible",
    ]
}
