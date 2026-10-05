import SwiftUI
import Combine
import AuraCore

/// App-level hybrid escalation settings + the consent gate the escalation flow awaits.
@MainActor
final class HybridSettings: ObservableObject {
    static let shared = HybridSettings()

    /// Escalation policy used by the manual/auto escalation flow.
    @Published var policy = EscalationPolicy(mode: .askEachTime, allowCloud: false)

    /// The consent gate presented before any cloud send.
    let consent = UIConsentGate()

    private init() {}
}

/// A ``ConsentGate`` that presents a SwiftUI sheet and suspends until the user
/// approves or declines. Fail-closed: if resolution never comes, the awaiting
/// call simply never proceeds (the caller keeps the local answer).
@MainActor
final class UIConsentGate: ObservableObject, ConsentGate {
    struct Request: Identifiable {
        let id = UUID()
        let target: RemoteTarget
        let preview: CompressionResult
        /// nil when the target's price is unknown.
        let cost: Decimal?
        /// Why it was offered, with the session's spend and cap; nil when the caller passed only a cost.
        let offer: EscalationOffer?
    }

    @Published var pending: Request?
    private var continuation: CheckedContinuation<Bool, Never>?

    func requestConsent(
        target: RemoteTarget,
        preview: CompressionResult,
        projectedCostUSD: Decimal?
    ) async -> Bool {
        await present(Request(target: target, preview: preview, cost: projectedCostUSD, offer: nil))
    }

    /// Shows the session budget, so the user can decide on an offer over the cost cap.
    func requestConsent(
        target: RemoteTarget,
        preview: CompressionResult,
        offer: EscalationOffer
    ) async -> Bool {
        await present(Request(target: target, preview: preview, cost: offer.projectedCostUSD, offer: offer))
    }

    private func present(_ request: Request) async -> Bool {
        await withCheckedContinuation { cont in
            self.continuation = cont
            self.pending = request
        }
    }

    /// Called by the consent sheet's buttons.
    func resolve(_ approved: Bool) {
        pending = nil
        continuation?.resume(returning: approved)
        continuation = nil
    }
}
