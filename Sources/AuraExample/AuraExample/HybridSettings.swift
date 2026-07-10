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
        let cost: Decimal
    }

    @Published var pending: Request?
    private var continuation: CheckedContinuation<Bool, Never>?

    func requestConsent(
        target: RemoteTarget,
        preview: CompressionResult,
        projectedCostUSD: Decimal
    ) async -> Bool {
        await withCheckedContinuation { cont in
            self.continuation = cont
            self.pending = Request(target: target, preview: preview, cost: projectedCostUSD)
        }
    }

    /// Called by the consent sheet's buttons.
    func resolve(_ approved: Bool) {
        pending = nil
        continuation?.resume(returning: approved)
        continuation = nil
    }
}
