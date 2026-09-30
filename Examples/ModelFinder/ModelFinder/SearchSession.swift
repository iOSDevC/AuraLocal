import SwiftUI
import AuraCore

/// Which weights to search for.
enum FormatFilter: String, CaseIterable, Identifiable {
    case any, mlx, gguf

    var id: String { rawValue }

    var title: String {
        switch self {
        case .any: "MLX + GGUF"
        case .mlx: "MLX"
        case .gguf: "GGUF"
        }
    }

    /// Hugging Face search tag; `nil` searches everything.
    var tag: String? { self == .any ? nil : rawValue }
}

/// Search state plus a per-repo cache of fetched snapshots. A snapshot is device-independent, so changing the
/// device re-evaluates the cache without touching the network.
@MainActor
@Observable
final class SearchSession {
    var query = "qwen 4bit"
    var formatChoice = FormatFilter.any
    var ordering = HuggingFaceSearch.Sort.downloads
    var onlyRunnable = false
    var deviceID = DevicePreset.thisDeviceID {
        didSet { reevaluate() }
    }

    private(set) var matches: [HFModelHit] = []
    private(set) var searchError: String?
    private(set) var reports: [String: CompatibilityReport] = [:]

    let devices = DevicePreset.all()
    private var snapshots: [String: RepoSnapshot] = [:]
    private var fetches: [String: SnapshotFetch] = [:]
    private var searchTask: Task<Void, Never>?
    private let inspector = ModelCompatibilityChecker()

    var isSearching: Bool { searchTask != nil }

    var currentTarget: DevicePreset {
        devices.first { $0.id == deviceID } ?? devices[0]
    }

    /// Search results, minus the ones already known not to run when the filter is on.
    var visibleMatches: [HFModelHit] {
        guard onlyRunnable else { return matches }
        return matches.filter { reports[$0.id].map { $0.status != .notRunnable } ?? true }
    }

    /// Starts a search and cancels the one in flight, so an older query can never overwrite a newer one.
    func search() {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        searchTask?.cancel()
        searchError = nil
        let sort = ordering
        let tag = formatChoice.tag
        searchTask = Task {
            let found: [HFModelHit]
            let failure: String?
            do {
                found = try await HuggingFaceSearch.search(text, limit: 40, sort: sort, tag: tag)
                failure = nil
            } catch {
                found = []
                failure = error.localizedDescription
            }
            // Runs on the main actor, so a newer search() either cancelled this one already or has not started.
            guard !Task.isCancelled else { return }
            matches = found
            searchError = failure
            searchTask = nil
        }
    }

    /// Fetches one repo's snapshot the first time a row or the detail pane asks. Every caller waits on the same
    /// fetch; it is cancelled only when all of them have gone away, and a finished fetch is always cached.
    func loadReport(for repoID: String) async {
        guard snapshots[repoID] == nil else { return }
        let fetch = join(repoID)
        let token = fetch.token
        let snapshot = await withTaskCancellationHandler {
            await fetch.work.value
        } onCancel: {
            Task { @MainActor in self.leave(repoID, token: token) }
        }
        if fetches[repoID]?.token == token { fetches[repoID] = nil }
        guard !fetch.work.isCancelled, snapshots[repoID] == nil else { return }
        snapshots[repoID] = snapshot
        reports[repoID] = CompatibilityEvaluator.evaluate(snapshot, on: currentTarget)
    }

    private func join(_ repoID: String) -> SnapshotFetch {
        if var running = fetches[repoID] {
            running.waiters += 1
            fetches[repoID] = running
            return running
        }
        let inspector = inspector
        let started = SnapshotFetch(work: Task { await inspector.snapshot(of: repoID) }, token: UUID(), waiters: 1)
        fetches[repoID] = started
        return started
    }

    private func leave(_ repoID: String, token: UUID) {
        guard var running = fetches[repoID], running.token == token else { return }
        running.waiters -= 1
        if running.waiters > 0 {
            fetches[repoID] = running
        } else {
            running.work.cancel()
            fetches[repoID] = nil
        }
    }

    private func reevaluate() {
        let target = currentTarget
        reports = snapshots.mapValues { CompatibilityEvaluator.evaluate($0, on: target) }
    }
}

/// One in-flight snapshot fetch and how many views wait on it.
private struct SnapshotFetch {
    let work: Task<RepoSnapshot, Never>
    let token: UUID
    var waiters: Int
}
