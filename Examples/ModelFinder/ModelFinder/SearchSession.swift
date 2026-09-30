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
    private(set) var isSearching = false
    private(set) var searchError: String?
    private(set) var reports: [String: CompatibilityReport] = [:]

    let devices = DevicePreset.all()
    private var snapshots: [String: RepoSnapshot] = [:]
    private var loading: Set<String> = []
    private let inspector = ModelCompatibilityChecker()

    var currentTarget: DevicePreset {
        devices.first { $0.id == deviceID } ?? devices[0]
    }

    /// Search results, minus the ones already known not to run when the filter is on.
    var visibleMatches: [HFModelHit] {
        guard onlyRunnable else { return matches }
        return matches.filter { reports[$0.id].map { $0.status != .notRunnable } ?? true }
    }

    func search() async {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        isSearching = true
        searchError = nil
        defer { isSearching = false }
        do {
            matches = try await HuggingFaceSearch.search(text, limit: 40, sort: ordering, tag: formatChoice.tag)
        } catch {
            matches = []
            searchError = error.localizedDescription
        }
    }

    /// Fetch one repo's snapshot the first time its row appears. The row's `.task` cancels this when it
    /// scrolls away; a cancelled fetch is dropped so it runs again next time.
    func loadReport(for repoID: String) async {
        guard snapshots[repoID] == nil, !loading.contains(repoID) else { return }
        loading.insert(repoID)
        defer { loading.remove(repoID) }
        let snapshot = await inspector.snapshot(of: repoID)
        guard !Task.isCancelled else { return }
        snapshots[repoID] = snapshot
        reports[repoID] = CompatibilityEvaluator.evaluate(snapshot, on: currentTarget)
    }

    private func reevaluate() {
        let target = currentTarget
        reports = snapshots.mapValues { CompatibilityEvaluator.evaluate($0, on: target) }
    }
}
