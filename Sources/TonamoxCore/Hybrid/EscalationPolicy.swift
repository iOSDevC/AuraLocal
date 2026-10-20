import Foundation

/// Per-profile policy for escalating to a remote model. Default is OFF — the
/// user opts in. Codable so it can persist with profile settings.
public struct EscalationPolicy: Sendable, Equatable, Codable {

    public enum Mode: String, Sendable, Codable {
        /// Never escalate (default).
        case off
        /// Offer escalation; the user confirms each time.
        case askEachTime
        /// Auto-escalate to the user's own LAN box; cloud still asks per conversation.
        case autoWithConsentMemory
    }

    /// Which targets get ``PIIRedactor`` applied to the context and the question before sending.
    public enum RedactionScope: String, Sendable, Codable {
        /// Cloud targets only; the user's own LAN box gets the text verbatim (default).
        case cloudOnly
        /// Every target, the LAN box included.
        case allTargets
        /// No target.
        case never
    }

    public var mode: Mode
    /// Allow cloud (BYOK) targets in addition to the user's own LAN box.
    public var allowCloud: Bool
    /// Ceiling on priced remote spend per ``CostLedger`` session. A priced request that would cross it is offered
    /// as `.costCapped`, which the default ``ConsentGate/requestConsent(target:preview:offer:)`` declines; only a
    /// gate that implements that method can let the user go over. A $0 LAN request never counts against it.
    public var costCapUSDPerSession: Decimal
    /// Share of the target's context window, minus `maxTokens`, that the compressed context may fill (0…1).
    public var keepRatio: Double
    /// Where `HybridEscalator.routeAndEscalate` redacts secrets/PII. Default `.cloudOnly`.
    public var piiRedaction: RedactionScope

    public init(
        mode: Mode = .off,
        allowCloud: Bool = false,
        costCapUSDPerSession: Decimal = 1.0,
        keepRatio: Double = 0.5,
        piiRedaction: RedactionScope = .cloudOnly
    ) {
        self.mode = mode
        self.allowCloud = allowCloud
        self.costCapUSDPerSession = costCapUSDPerSession
        self.keepRatio = keepRatio
        self.piiRedaction = piiRedaction
    }

    /// Escalation disabled — the safe default.
    public static let off = EscalationPolicy()

    /// Whether this policy redacts what it sends to `target`.
    public func redactsPII(for target: RemoteTarget) -> Bool {
        switch piiRedaction {
        case .cloudOnly: !target.isLocalNetwork
        case .allTargets: true
        case .never: false
        }
    }

    private enum CodingKeys: String, CodingKey {
        case mode, allowCloud, costCapUSDPerSession, keepRatio, piiRedaction
    }

    // Policies saved before `piiRedaction` existed decode with the default.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        mode = try container.decode(Mode.self, forKey: .mode)
        allowCloud = try container.decode(Bool.self, forKey: .allowCloud)
        costCapUSDPerSession = try container.decode(Decimal.self, forKey: .costCapUSDPerSession)
        keepRatio = try container.decode(Double.self, forKey: .keepRatio)
        piiRedaction = try container.decodeIfPresent(RedactionScope.self, forKey: .piiRedaction) ?? .cloudOnly
    }
}
