import Foundation

/// What the router saw when it offered an escalation, so a ``ConsentGate`` can show the budget and decide on it.
public struct EscalationOffer: Sendable, Equatable {
    public let reason: EscalationReason
    /// This request's projected cost; nil when the target's price is unknown.
    public let projectedCostUSD: Decimal?
    /// Priced spend since the ledger's session started; a lower bound while `unpricedRecordCount > 0`.
    public let sessionSpentUSD: Decimal
    /// Calls this session whose cost is unknown.
    public let unpricedRecordCount: Int
    /// The policy's `costCapUSDPerSession`.
    public let costCapUSD: Decimal

    public init(reason: EscalationReason, projectedCostUSD: Decimal?, sessionSpentUSD: Decimal,
                unpricedRecordCount: Int, costCapUSD: Decimal) {
        self.reason = reason
        self.projectedCostUSD = projectedCostUSD
        self.sessionSpentUSD = sessionSpentUSD
        self.unpricedRecordCount = unpricedRecordCount
        self.costCapUSD = costCapUSD
    }
}

/// Asks the user to approve sending a (compressed) payload off-device to a remote
/// target. AuraUI implements the presenting conformer (showing the exact
/// compressed payload, cost projection, and the provider's retention note).
@MainActor
public protocol ConsentGate: Sendable {
    /// Return `true` to proceed with the remote call, `false` to keep the local answer.
    /// `projectedCostUSD` is nil when the target's price is unknown; show it as
    /// unknown, not as free.
    func requestConsent(
        target: RemoteTarget,
        preview: CompressionResult,
        projectedCostUSD: Decimal?
    ) async -> Bool

    /// What `HybridEscalator.routeAndEscalate` calls, with the router's reason and the session's spend.
    /// The default declines a `.costCapped` offer and passes any other to
    /// ``requestConsent(target:preview:projectedCostUSD:)``; implement it to let the user override the cap.
    func requestConsent(
        target: RemoteTarget,
        preview: CompressionResult,
        offer: EscalationOffer
    ) async -> Bool
}

extension ConsentGate {
    public func requestConsent(
        target: RemoteTarget,
        preview: CompressionResult,
        offer: EscalationOffer
    ) async -> Bool {
        guard offer.reason != .costCapped else { return false }
        return await requestConsent(target: target, preview: preview, projectedCostUSD: offer.projectedCostUSD)
    }
}

/// Fail-closed default: denies every request. Escalation stays off until a real
/// consent UI is wired in.
@MainActor
public struct DenyingConsentGate: ConsentGate {
    public init() {}
    public func requestConsent(
        target: RemoteTarget,
        preview: CompressionResult,
        projectedCostUSD: Decimal?
    ) async -> Bool { false }
}
