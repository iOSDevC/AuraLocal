import Foundation
import Network

/// Observes network reachability so the router can require connectivity for
/// cloud targets. Loopback / local-network targets don't need this.
@MainActor
public final class NetworkMonitor {
    public static let shared = NetworkMonitor()

    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "dev.auralocal.network")

    /// `true` when a usable network path exists (assumed true until proven otherwise).
    public private(set) var isOnline = true

    private init() {
        monitor.pathUpdateHandler = { [weak self] path in
            let online = path.status == .satisfied
            Task { @MainActor in self?.isOnline = online }
        }
        monitor.start(queue: queue)
    }
}
