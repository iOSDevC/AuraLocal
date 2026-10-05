import Foundation
import Observation
import Synchronization
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

/// Owns the single running download. One at a time keeps memory, disk checks and GGUF single-flight simple;
/// a second start is refused while one runs.
@MainActor
@Observable
final class DownloadCenter {
    private(set) var phases: [String: DownloadPhase] = [:]
    private(set) var activeModelID: String?
    /// Bumped whenever files appear or disappear, so disk-scanning views refresh.
    private(set) var diskRevision = 0
    var keepsScreenAwake = false

    private var job: Task<Void, Never>?
    #if canImport(UIKit)
    private var backgroundTask = UIBackgroundTaskIdentifier.invalid
    #endif

    var isBusy: Bool { job != nil }

    func phase(of candidate: Model) -> DownloadPhase {
        phases[candidate.id] ?? .idle
    }

    func start(_ candidate: Model) {
        guard job == nil else { return }
        DownloadRecords.remember(candidate)
        activeModelID = candidate.id
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
                outcome = .finished(url)
            } catch is CancellationError {
                outcome = .idle
            } catch {
                outcome = .failed(DownloadMessages.text(for: error))
            }
            finish(id: id, outcome: outcome)
        }
    }

    /// Stops the transfer; the `.partial` file stays for Resume.
    func cancel() {
        job?.cancel()
    }

    /// Unloads, then removes finished and partial files.
    func delete(_ candidate: Model) throws {
        guard activeModelID != candidate.id else { return }
        ModelManager.shared.evict(candidate)
        try ModelDownloads.delete(candidate)
        DownloadRecords.forget(candidate)
        phases[candidate.id] = nil
        diskRevision += 1
    }

    private func report(id: String, done: Int64, total: Int64) {
        guard activeModelID == id else { return }
        phases[id] = .downloading(done: done, total: total)
    }

    private func finish(id: String, outcome: DownloadPhase) {
        phases[id] = outcome
        activeModelID = nil
        job = nil
        diskRevision += 1
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

/// Admits a progress report at most every 100 ms, plus the final one, so SwiftUI is not flooded.
final class ProgressThrottle: Sendable {
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

/// User-facing text for a failed download.
enum DownloadMessages {
    static let gated = "This repo is gated. Accept its license on huggingface.co and add a read token in Settings."

    @MainActor
    static func text(for error: any Error) -> String {
        guard let failure = error as? ModelDownloadError else {
            return "\(error.localizedDescription) The partial file is kept; Resume continues from it."
        }
        switch failure {
        case .unauthorized:
            return gated
        case .notFound:
            return "File not found at main (renamed or removed)."
        case .notFoundOrPrivate:
            return "Repo not found, renamed or private. A private repo needs a read token with access in Settings."
        case .insufficientSpace(let needed, let available):
            return "Needs \(ByteText.text(needed)), \(ByteText.text(available)) free."
        case .sizeMismatch:
            return (failure.errorDescription ?? "") + " Delete the partial file and retry."
        case .http, .incompleteRepo, .unsupported:
            return failure.errorDescription ?? "Download failed."
        }
    }

    static func isGated(_ error: any Error) -> Bool {
        if case .unauthorized = error as? ModelDownloadError { return true }
        return false
    }
}

/// Byte counts for display.
@MainActor
enum ByteText {
    private static let fileStyle: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter
    }()

    static func text(_ bytes: Int64) -> String {
        fileStyle.string(fromByteCount: bytes)
    }
}

/// The sized `Model` behind each download, so the Downloads tab can offer Try it and Delete after a relaunch.
/// Files on disk stay the source of truth; this only describes them.
enum DownloadRecords {
    private static let defaultsKey = "downloadRecords"

    static func remember(_ candidate: Model) {
        guard let data = try? JSONEncoder().encode(candidate) else { return }
        var stored = all()
        stored[key(for: candidate)] = data
        UserDefaults.standard.set(stored, forKey: defaultsKey)
    }

    static func forget(_ candidate: Model) {
        var stored = all()
        stored[key(for: candidate)] = nil
        UserDefaults.standard.set(stored, forKey: defaultsKey)
    }

    static func gguf(repoID: String, filename: String) -> Model? {
        decode(all()["gguf:\(repoID)/\(filename)"])
    }

    static func mlx(repoID: String) -> Model? {
        decode(all()["mlx:\(repoID)"])
    }

    private static func key(for candidate: Model) -> String {
        switch candidate.format {
        case .gguf: "gguf:\(candidate.repoID)/\(candidate.ggufFilename ?? "")"
        case .mlx: "mlx:\(candidate.repoID)"
        }
    }

    private static func all() -> [String: Data] {
        UserDefaults.standard.dictionary(forKey: defaultsKey) as? [String: Data] ?? [:]
    }

    private static func decode(_ data: Data?) -> Model? {
        data.flatMap { try? JSONDecoder().decode(Model.self, from: $0) }
    }
}
