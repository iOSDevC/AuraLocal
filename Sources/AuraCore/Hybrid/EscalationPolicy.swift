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

    public var mode: Mode
    /// Allow cloud (BYOK) targets in addition to the user's own LAN box.
    public var allowCloud: Bool
    /// Hard ceiling on remote spend per session; the router never exceeds it silently.
    public var costCapUSDPerSession: Decimal
    /// Fraction of context to keep when compressing (0…1).
    public var keepRatio: Double

    public init(
        mode: Mode = .off,
        allowCloud: Bool = false,
        costCapUSDPerSession: Decimal = 1.0,
        keepRatio: Double = 0.5
    ) {
        self.mode = mode
        self.allowCloud = allowCloud
        self.costCapUSDPerSession = costCapUSDPerSession
        self.keepRatio = keepRatio
    }

    /// Escalation disabled — the safe default.
    public static let off = EscalationPolicy()
}
