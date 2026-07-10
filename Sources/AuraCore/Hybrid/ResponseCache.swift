import Foundation

/// In-session cache of remote answers keyed by (provider, model, exact payload),
/// so an identical escalation isn't paid for twice. Bounded, FIFO-evicted.
/// In-memory only (cleared on relaunch) — a persistent cache is a later step.
@MainActor
public final class ResponseCache {
    public static let shared = ResponseCache()
    public init(capacity: Int = 64) { self.capacity = capacity }

    private let capacity: Int
    private var store: [String: String] = [:]
    private var order: [String] = []

    public var count: Int { store.count }

    /// Return a cached answer for this exact request, if present.
    public func lookup(provider: String, model: String, prompt: String) -> String? {
        store[Self.key(provider, model, prompt)]
    }

    /// Store an answer for this request (FIFO-evicts the oldest past capacity).
    public func insert(_ answer: String, provider: String, model: String, prompt: String) {
        let k = Self.key(provider, model, prompt)
        if store[k] == nil {
            order.append(k)
            while order.count > capacity, let oldest = order.first {
                order.removeFirst()
                store[oldest] = nil
            }
        }
        store[k] = answer
    }

    public func clear() {
        store.removeAll()
        order.removeAll()
    }

    private static func key(_ provider: String, _ model: String, _ prompt: String) -> String {
        "\(provider)\u{1}\(model)\u{1}\(prompt)"
    }
}
