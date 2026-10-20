import Foundation

// MARK: - DecodeGate

/// Runs async bodies one at a time, in arrival order. A waiter that is cancelled still takes its turn,
/// so bodies must check cancellation themselves.
@MainActor
final class DecodeGate {
    private var isHeld = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func run<T>(_ body: () async throws -> T) async rethrows -> T {
        if isHeld {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            isHeld = true
        }
        defer { handOff() }
        return try await body()
    }

    /// Ownership passes straight to the next waiter, so a newcomer can never slip in between.
    private func handOff() {
        if waiters.isEmpty {
            isHeld = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}
