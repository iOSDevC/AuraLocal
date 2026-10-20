import Darwin
import Foundation
import os

// MARK: - DownloadGate

/// One writer per destination: a snapshot download holds `snapshots/main`, a file download its file; overlapping paths
/// wait in this process, identical ones across processes (``DownloadFileLock``). Not reentrant for overlapping paths.
actor DownloadGate {
    static let shared = DownloadGate()

    private struct Turn {
        let id: UInt64
        let path: [String]
        let request: String?
        var isHolding = false
        /// An earlier turn for the same path and request ended successfully while this one waited.
        var peerFinished = false
        var continuation: CheckedContinuation<Bool, any Error>?
    }

    /// Arrival order. A turn may hold once no earlier turn overlaps it, holding or still waiting.
    private var turns: [Turn] = []
    private var nextID: UInt64 = 0
    /// How often a caller retries a lock file another process holds.
    let lockRetryInterval: Duration

    init(lockRetryInterval: Duration = .milliseconds(250)) {
        self.lockRetryInterval = lockRetryInterval
    }

    /// Runs `body` once every earlier caller for an overlapping path is done. `body` learns whether one with the same
    /// `request` (a string naming what is fetched) succeeded while this one waited.
    func exclusive<T>(
        _ destination: URL,
        request: String? = nil,
        isolation: isolated (any Actor)? = #isolation,
        _ body: (_ peerFinished: Bool) async throws -> T
    ) async throws -> T {
        let turn = try await enter(Self.components(of: destination), request: request)
        let lock: DownloadFileLock
        do {
            lock = try await DownloadFileLock.acquire(for: destination, retryInterval: lockRetryInterval)
        } catch {
            await leave(turn.id, succeeded: false)
            throw error
        }
        do {
            let value = try await body(turn.peerFinished)
            lock.release()
            await leave(turn.id, succeeded: true)
            return value
        } catch {
            lock.release()
            await leave(turn.id, succeeded: false)
            throw error
        }
    }

    /// Callers queued behind another for exactly `destination`.
    func waitingCount(at destination: URL) -> Int {
        let path = Self.components(of: destination)
        return turns.filter { !$0.isHolding && $0.path == path }.count
    }

    static func components(of destination: URL) -> [String] {
        destination.standardizedFileURL.pathComponents
    }

    // MARK: Queue

    private func enter(_ path: [String], request: String?) async throws -> (id: UInt64, peerFinished: Bool) {
        let id = nextID
        nextID += 1
        turns.append(Turn(id: id, path: path, request: request))
        if isClear(turns.count - 1) {
            turns[turns.count - 1].isHolding = true
            return (id, false)
        }
        let peerFinished = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Bool, any Error>) in
                guard let index = index(of: id) else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                if turns[index].isHolding {
                    continuation.resume(returning: turns[index].peerFinished)
                } else if Task.isCancelled {
                    turns.remove(at: index)
                    continuation.resume(throwing: CancellationError())
                    admitWaiting()
                } else {
                    turns[index].continuation = continuation
                }
            }
        } onCancel: {
            Task { await self.abandon(id) }
        }
        return (id, peerFinished)
    }

    /// A cancelled caller still queued leaves at once; one already holding ends through `leave`.
    private func abandon(_ id: UInt64) {
        guard let index = index(of: id), !turns[index].isHolding else { return }
        let continuation = turns.remove(at: index).continuation
        continuation?.resume(throwing: CancellationError())
        admitWaiting()
    }

    private func leave(_ id: UInt64, succeeded: Bool) {
        guard let index = index(of: id) else { return }
        let ended = turns.remove(at: index)
        if succeeded, let request = ended.request {
            for waiting in turns.indices where turns[waiting].path == ended.path && turns[waiting].request == request {
                turns[waiting].peerFinished = true
            }
        }
        admitWaiting()
    }

    private func admitWaiting() {
        for index in turns.indices where !turns[index].isHolding && isClear(index) {
            turns[index].isHolding = true
            let continuation = turns[index].continuation
            turns[index].continuation = nil
            continuation?.resume(returning: turns[index].peerFinished)
        }
    }

    private func isClear(_ index: Int) -> Bool {
        let path = turns[index].path
        return !turns[..<index].contains { Self.overlaps($0.path, path) }
    }

    private func index(of id: UInt64) -> Int? {
        turns.firstIndex { $0.id == id }
    }

    /// Equal paths, or one inside the other.
    static func overlaps(_ lhs: [String], _ rhs: [String]) -> Bool {
        lhs.count <= rhs.count ? rhs.starts(with: lhs) : lhs.starts(with: rhs)
    }
}

// MARK: - DownloadFileLock

/// An advisory `flock(2)` on `<destination>.lock` for processes sharing the storage root. The kernel drops it with
/// the holder, so a crash leaves nothing stale; waiting polls a non-blocking `flock` so no thread blocks.
struct DownloadFileLock: Sendable {
    let lockFile: URL
    private let descriptor: Int32
    /// Folders created to hold the lock file; removed on release if still empty.
    private let createdFolders: [URL]

    static let log = Logger(subsystem: "dev.auralocal", category: "DownloadGate")

    private init(lockFile: URL, descriptor: Int32, createdFolders: [URL]) {
        self.lockFile = lockFile
        self.descriptor = descriptor
        self.createdFolders = createdFolders
    }

    static func lockFile(for destination: URL) -> URL {
        let target = destination.standardizedFileURL
        return target.deletingLastPathComponent()
            .appendingPathComponent(target.lastPathComponent + ".lock", isDirectory: false)
    }

    /// Waits until no other open file holds the lock on `destination`'s lock file, then holds it.
    static func acquire(for destination: URL, retryInterval: Duration) async throws -> DownloadFileLock {
        let lockFile = lockFile(for: destination)
        var created: [URL] = []
        var reportedWait = false
        do {
            while true {
                try Task.checkCancellation()
                created += try createMissingFolders(above: lockFile)
                if let descriptor = try lockIfFree(lockFile) {
                    return held(lockFile, descriptor: descriptor, createdFolders: created)
                }
                if !reportedWait {
                    reportedWait = true
                    log.notice("waiting for another process writing \(destination.lastPathComponent, privacy: .public)")
                }
                try await Task.sleep(for: retryInterval)
            }
        } catch {
            removeEmpty(created)
            throw error
        }
    }

    /// Unlinks the lock file while still holding it, so a process that opened it starts over. A file recreated after a
    /// delete removed this one belongs to another holder and stays.
    func release() {
        let path = lockFile.path(percentEncoded: false)
        if Self.isLinked(descriptor, at: path) {
            unlink(path)
        }
        flock(descriptor, LOCK_UN)
        close(descriptor)
        Self.removeEmpty(createdFolders)
    }

    private static func held(_ lockFile: URL, descriptor: Int32, createdFolders: [URL]) -> DownloadFileLock {
        DownloadFileLock(lockFile: lockFile, descriptor: descriptor, createdFolders: createdFolders)
    }

    /// The locked descriptor, or `nil` to retry: another open file holds the lock, or the file or its folder vanished.
    private static func lockIfFree(_ lockFile: URL) throws -> Int32? {
        let path = lockFile.path(percentEncoded: false)
        let descriptor = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else {
            let failure = errno
            // A releasing holder removed the folder it had created; it is created again on the next try.
            if failure == ENOENT { return nil }
            throw posixError(failure)
        }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let failure = errno
            close(descriptor)
            if failure == EWOULDBLOCK { return nil }
            throw posixError(failure)
        }
        // A holder unlinks the file before unlocking it; a lock on that old file guards nothing.
        guard isLinked(descriptor, at: path) else {
            close(descriptor)
            return nil
        }
        return descriptor
    }

    private static func isLinked(_ descriptor: Int32, at path: String) -> Bool {
        var opened = stat()
        var named = stat()
        guard fstat(descriptor, &opened) == 0, stat(path, &named) == 0 else { return false }
        return opened.st_dev == named.st_dev && opened.st_ino == named.st_ino
    }

    /// Creates the missing folders above `lockFile` and returns them, deepest first.
    private static func createMissingFolders(above lockFile: URL) throws -> [URL] {
        var missing: [URL] = []
        var folder = lockFile.deletingLastPathComponent()
        while folder.pathComponents.count > 1,
              !FileManager.default.fileExists(atPath: folder.path(percentEncoded: false)) {
            missing.append(folder)
            folder.deleteLastPathComponent()
        }
        if let deepest = missing.first {
            try FileManager.default.createDirectory(at: deepest, withIntermediateDirectories: true)
        }
        return missing
    }

    /// Deepest first; `rmdir` only removes an empty folder, so one another download writes into stays.
    private static func removeEmpty(_ folders: [URL]) {
        let deepestFirst = Set(folders).sorted { $0.pathComponents.count > $1.pathComponents.count }
        for folder in deepestFirst {
            rmdir(folder.path(percentEncoded: false))
        }
    }

    private static func posixError(_ code: Int32) -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
    }
}
