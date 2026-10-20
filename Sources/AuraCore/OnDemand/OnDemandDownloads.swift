import Foundation
import Observation
import Synchronization

// MARK: - OnDemandDownloads

/// Downloads models only when the user asks, one transfer at a time, and publishes their progress.
///
/// ``ModelDownloadPlaceholder`` drives it. Transfers go through ``ModelDownloads/download(_:auth:progress:)``, so
/// files land where the backends load them from, the free-space check applies and a cancelled download resumes
/// from its `.partial` file. Models are tracked by file (``DownloadRecords/key(for:)``), so two entries for one file
/// share one transfer. Observe ``revision`` to refresh anything that lists downloaded models.
@MainActor
@Observable
public final class OnDemandDownloads {

    /// Where one model's download stands.
    public enum Phase: Equatable, Sendable {
        /// Not queued: never asked for, finished, or cancelled.
        case idle
        /// Queued behind another model's download.
        case waiting
        /// Bytes on disk so far and the total; the total is the catalog estimate until the transfer reports one.
        case downloading(done: Int64, total: Int64)
        /// The last attempt failed with this message. Downloading again resumes from the partial file.
        case failed(String)

        /// Queued or transferring.
        public var isActive: Bool {
            switch self {
            case .waiting, .downloading: true
            case .idle, .failed: false
            }
        }
    }

    /// What the queue reads from disk and fetches over the network; tests swap it.
    struct DiskAndNetwork: Sendable {
        var isOnDisk: @MainActor @Sendable (Model) -> Bool
        var bytesOnDisk: @MainActor @Sendable (Model) -> Int64
        var fetch: @Sendable (Model, @escaping @Sendable (Int64, Int64) -> Void) async throws -> Void

        static let live = DiskAndNetwork(
            isOnDisk: { $0.isDownloaded },
            bytesOnDisk: { ModelDownloads.bytesOnDisk(for: $0) },
            fetch: { model, progress in _ = try await ModelDownloads.download(model, progress: progress) })
    }

    public static let shared = OnDemandDownloads(io: .live)

    /// Bumped whenever a download finishes, fails or is cancelled.
    public private(set) var revision = 0

    private let io: DiskAndNetwork
    private var stateByKey: [String: Phase] = [:]
    private var queue: [Model] = []
    private var activeKey: String?
    private var transfer: Task<Phase, Never>?
    private var worker: Task<Void, Never>?

    init(io: DiskAndNetwork) {
        self.io = io
    }

    // MARK: State

    /// `.idle` for a model that was never queued or has finished; check ``Model/isDownloaded`` for the latter.
    public func phase(of model: Model) -> Phase {
        stateByKey[DownloadRecords.key(for: model)] ?? .idle
    }

    /// Whether `model`'s file is queued or transferring here.
    public func isDownloading(_ model: Model) -> Bool {
        phase(of: model).isActive
    }

    // MARK: Actions

    /// Queues every model in `models` that is not on disk and not already queued, in order.
    public func download(_ models: [Model]) {
        for model in models where !io.isOnDisk(model) && !isDownloading(model) {
            stateByKey[DownloadRecords.key(for: model)] = .waiting
            queue.append(model)
        }
        guard worker == nil, !queue.isEmpty else { return }
        worker = Task { await drain() }
    }

    /// Removes `models` from the queue and stops the one transferring, keeping its partial file for next time.
    public func cancel(_ models: [Model]) {
        let keys = Set(models.map(DownloadRecords.key(for:)))
        queue.removeAll { keys.contains(DownloadRecords.key(for: $0)) }
        for key in keys where stateByKey[key] == .waiting {
            stateByKey[key] = nil
        }
        if let activeKey, keys.contains(activeKey) {
            transfer?.cancel()
        }
        revision += 1
    }

    // MARK: Transfer

    private func drain() async {
        while !queue.isEmpty {
            await fetch(queue.removeFirst())
        }
        worker = nil
    }

    private func fetch(_ model: Model) async {
        let key = DownloadRecords.key(for: model)
        activeKey = key
        stateByKey[key] = .downloading(done: io.bytesOnDisk(model), total: model.approximateSizeBytes)
        let sampler = ProgressSampler()
        let fetch = io.fetch
        let job = Task { () -> Phase in
            do {
                try await fetch(model) { done, total in
                    guard sampler.admits(done: done, total: total) else { return }
                    Task { @MainActor in self.report(key, done: done, total: total) }
                }
                return .idle
            } catch {
                // A cancel can also surface as URLError(.cancelled).
                return Task.isCancelled ? .idle : .failed(error.localizedDescription)
            }
        }
        transfer = job
        let outcome = await job.value
        transfer = nil
        activeKey = nil
        stateByKey[key] = outcome == .idle ? nil : outcome
        revision += 1
    }

    private func report(_ key: String, done: Int64, total: Int64) {
        guard activeKey == key else { return }
        stateByKey[key] = .downloading(done: done, total: total)
    }
}

// MARK: - Helpers

public extension Model {
    /// ``approximateSizeMB`` in bytes, counting a megabyte as 1,000,000 bytes as size labels do.
    var approximateSizeBytes: Int64 {
        Int64(approximateSizeMB) * 1_000_000
    }
}

/// Admits a progress report at most every 100 ms, plus the final one, so SwiftUI is not flooded.
final class ProgressSampler: Sendable {
    private let lastAdmitted = Mutex<ContinuousClock.Instant?>(nil)

    func admits(done: Int64, total: Int64) -> Bool {
        let now = ContinuousClock.now
        return lastAdmitted.withLock { last in
            guard done >= total || last.map({ now - $0 >= .milliseconds(100) }) ?? true else { return false }
            last = now
            return true
        }
    }
}
