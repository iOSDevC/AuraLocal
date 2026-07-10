import Foundation

/// Why an escalation is being considered.
public enum EscalationReason: String, Sendable, Equatable {
    case sizeOverflow      // prompt won't fit the local context
    case lowConfidence     // local answer looks weak / refused
    case domainSensitive   // security/medicine — bias toward a stronger model
    case userRequested     // explicit manual trigger
    case costCapped        // a trigger fired but projected cost exceeds the cap
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
    public var projectedCostUSD: Decimal

    public init(
        policy: EscalationPolicy,
        hasCandidateTarget: Bool,
        candidateIsCloud: Bool,
        online: Bool = true,
        promptTokens: Int = 0,
        localContextWindow: Int = 8192,
        localAnswer: String? = nil,
        domain: Model.Domain? = nil,
        projectedCostUSD: Decimal = 0
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

        // R7 — cost cap: a trigger fired but the price is too high ⇒ offer, never silent.
        if input.projectedCostUSD > input.policy.costCapUSDPerSession {
            return .offer(reason: .costCapped)
        }

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
        return ["i'm not sure", "i am not sure", "i cannot", "i can't", "i don't know",
                "i do not know", "as an ai", "unable to"].contains { lowered.contains($0) }
    }
}
