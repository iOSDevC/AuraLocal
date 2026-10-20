import Foundation
import Observation
import AuraCore
#if canImport(UIKit)
import UIKit
#endif

/// Where one model's download stands in this run of the app.
enum DownloadPhase: Equatable {
    case idle
    case downloading(done: Int64, total: Int64)
    case failed(String)
    case finished(URL)
}

/// Thrown when a delete targets a file that is still being written.
struct DownloadInProgressError: LocalizedError {
    let name: String
    var errorDescription: String? { "\(name) is downloading. Cancel the download before deleting it." }
}

/// Owns the single running download from the Hugging Face browser, records it, registers it when it finishes and
/// tells the pickers when files appear or disappear.
///
/// One download at a time keeps memory, disk checks and GGUF single-flight simple; a second start is refused while
/// one runs. Observe ``revision`` to refresh anything that lists models or scans the disk.
@MainActor
@Observable
public final class ModelDownloadCenter {
    public static let shared = ModelDownloadCenter()

    /// Bumped whenever models are downloaded, deleted or registered, here or through ``OnDemandDownloads``.
    public var revision: Int { changes + OnDemandDownloads.shared.revision }
    private var changes = 0
    /// Keeps the screen on while a download runs (iOS).
    public var keepsScreenAwake = false
    /// Whether a download runs.
    public var isBusy: Bool { job != nil }

    private(set) var phases: [String: DownloadPhase] = [:]
    private(set) var activeModelID: String?
    private var activeKey: String?
    /// File keys that a catalog row (``ModelSection``) is downloading.
    private var claimedKeys: Set<String> = []
    private var job: Task<Void, Never>?
    #if canImport(UIKit)
    private var backgroundTask = UIBackgroundTaskIdentifier.invalid
    #endif

    private init() {}

    func phase(of candidate: Model) -> DownloadPhase {
        phases[candidate.id] ?? .idle
    }

    /// Whether another download writes `candidate`'s file: this center's, a catalog row's or a download placeholder's.
    func isWritingElsewhere(_ candidate: Model) -> Bool {
        let key = DownloadRecords.key(for: candidate)
        return claimedKeys.contains(key) || (activeKey == key && activeModelID != candidate.id)
            || OnDemandDownloads.shared.isDownloading(candidate)
    }

    /// - Parameter records: record `candidate` now and register it when the download finishes. Pass `true` only
    ///   for a sized model built from a search result or read from a record; a catalog model is already
    ///   registered, and an unsized one would pass the memory guard (size 0 always fits).
    func start(_ candidate: Model, records: Bool) {
        let key = DownloadRecords.key(for: candidate)
        guard job == nil, !claimedKeys.contains(key), !OnDemandDownloads.shared.isDownloading(candidate) else { return }
        let records = records && candidate.approximateSizeMB > 0
        if records { DownloadRecords.standard.remember(candidate) }
        activeModelID = candidate.id
        activeKey = key
        phases[candidate.id] = .downloading(done: ModelDownloads.bytesOnDisk(for: candidate), total: 0)
        beginBackgroundWindow()
        let throttle = ProgressThrottle()
        let id = candidate.id
        job = Task {
            let outcome: DownloadPhase
            do {
                let url = try await ModelDownloads.download(candidate) { done, total in
                    guard throttle.admits(done: done, total: total) else { return }
                    Task { @MainActor in self.report(id: id, done: done, total: total) }
                }
                if records { DownloadRecords.standard.registerIfDownloaded(candidate) }
                outcome = .finished(url)
            } catch is CancellationError {
                outcome = .idle
            } catch {
                // A cancel can also surface as URLError(.cancelled).
                outcome = Task.isCancelled ? .idle : .failed(DownloadMessages.text(for: error))
            }
            finish(id: id, outcome: outcome)
        }
    }

    /// Stops the transfer; the `.partial` file stays for Resume.
    func cancel() {
        job?.cancel()
    }

    /// Whether a download writes `candidate`'s file now: this center's, a ``ModelSection`` row's or a
    /// ``ModelDownloadPlaceholder``'s.
    public func isDownloading(_ candidate: Model) -> Bool {
        let key = DownloadRecords.key(for: candidate)
        return activeKey == key || claimedKeys.contains(key) || OnDemandDownloads.shared.isDownloading(candidate)
    }

    /// Unloads every model that reads this file, removes its finished and partial files, forgets its record and
    /// unregisters it. Catalog models stay registered. Throws while a download writes the file.
    public func delete(_ candidate: Model) throws {
        guard activeModelID != candidate.id, !isWritingElsewhere(candidate) else {
            throw DownloadInProgressError(name: candidate.displayName)
        }
        evictAll(sharingFileWith: candidate)
        try ModelDownloads.delete(candidate)
        DownloadRecords.standard.remove(candidate)
        phases[candidate.id] = nil
        changes += 1
    }

    /// Removes only `partial`, a paused file whose finished copy is already on disk; the finished file, its record
    /// and its registration stay.
    func deleteLeftover(_ partial: URL, of candidate: Model) throws {
        guard activeModelID != candidate.id, !isWritingElsewhere(candidate) else {
            throw DownloadInProgressError(name: candidate.displayName)
        }
        try FileManager.default.removeItem(at: partial)
        changes += 1
    }

    /// The cache is keyed by id: a search record and the catalog entry for the same file are two cache entries,
    /// and the picker loads the catalog one.
    private func evictAll(sharingFileWith candidate: Model) {
        let key = DownloadRecords.key(for: candidate)
        let manager = ModelManager.shared
        manager.evict(candidate)
        for model in ModelRegistry.shared.models where DownloadRecords.key(for: model) == key {
            manager.evict(model)
        }
    }

    // MARK: Catalog rows

    /// Reserves a file for a catalog row's own download, so two transfers never write one `.partial`.
    func claim(_ key: String) -> Bool {
        guard activeKey != key, !claimedKeys.contains(key) else { return false }
        claimedKeys.insert(key)
        return true
    }

    func release(_ key: String) {
        claimedKeys.remove(key)
    }

    /// Files changed outside this center (a catalog row finished or stopped).
    func noteDiskChange() {
        changes += 1
    }

    // MARK: Progress

    private func report(id: String, done: Int64, total: Int64) {
        guard activeModelID == id else { return }
        phases[id] = .downloading(done: done, total: total)
    }

    private func finish(id: String, outcome: DownloadPhase) {
        phases[id] = outcome
        activeModelID = nil
        activeKey = nil
        job = nil
        changes += 1
        endBackgroundWindow()
    }

    // MARK: App lifecycle

    /// Not a background download: it only buys ~30 s so a quick app switch does not fail the transfer.
    private func beginBackgroundWindow() {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = keepsScreenAwake
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "Model download") { [weak self] in
            MainActor.assumeIsolated {
                self?.job?.cancel()
                self?.endBackgroundWindow()
            }
        }
        #endif
    }

    private func endBackgroundWindow() {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = false
        guard backgroundTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
        #endif
    }
}
