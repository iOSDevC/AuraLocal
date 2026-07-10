import Foundation

/// Asks the user to approve sending a (compressed) payload off-device to a remote
/// target. AuraUI implements the presenting conformer (showing the exact
/// compressed payload, cost projection, and the provider's retention note).
@MainActor
public protocol ConsentGate: Sendable {
    /// Return `true` to proceed with the remote call, `false` to keep the local answer.
    func requestConsent(
        target: RemoteTarget,
        preview: CompressionResult,
        projectedCostUSD: Decimal
    ) async -> Bool
}

/// Fail-closed default: denies every request. Escalation stays off until a real
/// consent UI is wired in.
@MainActor
public struct DenyingConsentGate: ConsentGate {
    public init() {}
    public func requestConsent(
        target: RemoteTarget,
        preview: CompressionResult,
        projectedCostUSD: Decimal
    ) async -> Bool { false }
}
