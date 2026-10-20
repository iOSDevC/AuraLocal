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
    /// The exact request sent on approval. `routeAndEscalate` always sets it; nil only when the caller built the
    /// offer without one.
    public let payload: EscalationPayload?

    public init(reason: EscalationReason, projectedCostUSD: Decimal?, sessionSpentUSD: Decimal,
                unpricedRecordCount: Int, costCapUSD: Decimal, payload: EscalationPayload? = nil) {
        self.reason = reason
        self.projectedCostUSD = projectedCostUSD
        self.sessionSpentUSD = sessionSpentUSD
        self.unpricedRecordCount = unpricedRecordCount
        self.costCapUSD = costCapUSD
        self.payload = payload
    }
}

/// Asks the user to approve sending a payload off-device to a remote target.
/// Tonamox ships no presenting conformer; the app shows the payload, cost and retention note.
@MainActor
public protocol ConsentGate: Sendable {
    /// Return `true` to proceed with the remote call, `false` to keep the local answer.
    /// `preview.keptText` is the context as sent; the question and system prompt reach a gate only
    /// through the offer method's ``EscalationOffer/payload``. `projectedCostUSD` is nil when the
    /// target's price is unknown; show it as unknown, not as free.
    func requestConsent(
        target: RemoteTarget,
        preview: CompressionResult,
        projectedCostUSD: Decimal?
    ) async -> Bool

    /// What `HybridEscalator.routeAndEscalate` calls, with the router's reason, the session's spend and
    /// the exact payload. The default declines a `.costCapped` offer and passes any other to
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
