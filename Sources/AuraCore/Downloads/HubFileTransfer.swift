import Darwin
import Foundation
import os
import Synchronization

/// A file of a Hugging Face repo as the tree API lists it.
struct RemoteFile: Sendable, Equatable {
    let path: String
    /// LFS size for LFS files, plain size otherwise; `nil` when the listing had none.
    let size: Int64?
}

/// Resumable Hugging Face file transfer shared by the GGUF and MLX download paths.
///
/// Each file is fetched in 32 MB `Range` chunks appended to `<file>.partial`, which survives network cuts,
/// app suspension and relaunch; the next call continues from its size. The partial records the ETag of the bytes
/// it holds, and a chunk with another ETag restarts it from zero, so a file re-uploaded under the same name between
/// cancel and resume is never spliced. The file is moved into place only when its size matches, so a failed or
/// refused request never becomes a model file.
struct HubFileTransfer: Sendable {
    static let chunkSize: Int64 = 32 * 1024 * 1024
    static let maxChunkAttempts = 6
    static let spaceMargin = 1.2
    static let partialExtension = "partial"
    static let validatorAttribute = "dev.auralocal.etag"

    let session: URLSession
    let auth: any DownloadAuthorizing
    let freeSpace: @Sendable (URL) -> Int64?
    let retryDelay: @Sendable (Int) -> Duration
    /// How often the in-flight chunk reports received bytes.
    let progressInterval: Duration

    init(session: URLSession = HubFileTransfer.sharedSession,
         auth: any DownloadAuthorizing,
         freeSpace: @escaping @Sendable (URL) -> Int64? = HubFileTransfer.volumeFreeSpace,
         retryDelay: @escaping @Sendable (Int) -> Duration = HubFileTransfer.backoff,
         progressInterval: Duration = .milliseconds(250)) {
        self.session = session
        self.auth = auth
        self.freeSpace = freeSpace
        self.retryDelay = retryDelay
        self.progressInterval = progressInterval
    }

    static let log = Logger(subsystem: "dev.auralocal", category: "HFDownloader")

    static let sharedSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 7200
        config.waitsForConnectivity = true
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.httpMaximumConnectionsPerHost = 2
        return URLSession(configuration: config)
    }()

    @Sendable static func backoff(_ attempt: Int) -> Duration {
        .seconds(min(30.0, pow(2.0, Double(attempt - 1))))
    }

    /// Free space for important data on the volume holding `url` (or its nearest existing parent).
    @Sendable static func volumeFreeSpace(_ url: URL) -> Int64? {
        var probe = url
        while !FileManager.default.fileExists(atPath: probe.path), probe.pathComponents.count > 1 {
            probe.deleteLastPathComponent()
        }
        let values = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    static func partialURL(for destination: URL) -> URL {
        destination.appendingPathExtension(partialExtension)
    }

    // MARK: Listing

    /// Top-level files of `repoID` at `revision`. Any non-200 status or an unreadable body throws, so a gated
    /// repo never looks like an empty one.
    func listTopLevelFiles(repoID: String, revision: String) async throws -> [RemoteFile] {
        guard let url = Self.treeURL(repoID: repoID, revision: revision) else {
            throw ModelDownloadError.unsupported("Not a Hugging Face repository id: \(repoID)")
        }
        let request = auth.request(for: url)
        let (data, response) = try await load(request, onBytes: nil)
        guard let http = response as? HTTPURLResponse else { throw ModelDownloadError.http(status: 0) }
        if let failure = ModelDownloadError(response: http, request: request, repoID: repoID, path: "") {
            throw failure
        }
        guard http.statusCode == 200, let entries = try? JSONDecoder().decode([TreeEntry].self, from: data) else {
            throw ModelDownloadError.unsupported("Hugging Face returned an unreadable file list for \(repoID).")
        }
        return entries.filter { $0.type == "file" && !$0.path.contains("/") }
            .map { RemoteFile(path: $0.path, size: $0.largeFile?.size ?? $0.size) }
    }

    /// Size of a remote file from a HEAD request: `x-linked-size` (LFS) first, then `Content-Length`.
    func size(of url: URL, repoID: String, path: String) async throws -> Int64 {
        var request = auth.request(for: url)
        request.httpMethod = "HEAD"
        let (_, response) = try await load(request, onBytes: nil)
        guard let http = response as? HTTPURLResponse else { throw ModelDownloadError.http(status: 0) }
        if let failure = ModelDownloadError(response: http, request: request, repoID: repoID, path: path) {
            throw failure
        }
        let linked = http.value(forHTTPHeaderField: "x-linked-size").flatMap { Int64($0) }
        let size = linked ?? http.expectedContentLength
        guard size > 0 else {
            throw ModelDownloadError.unsupported("Hugging Face did not report a size for \(path).")
        }
        return size
    }

    /// Throws ``ModelDownloadError/insufficientSpace(neededBytes:availableBytes:)`` when `remaining` bytes plus
    /// the margin do not fit on the volume of `directory`. An unknown free space does not block.
    func ensureSpace(remaining: Int64, at directory: URL) throws {
        guard remaining > 0, let free = freeSpace(directory) else { return }
        let needed = Int64((Double(remaining) * Self.spaceMargin).rounded(.up))
        if free < needed {
            throw ModelDownloadError.insufficientSpace(neededBytes: needed, availableBytes: free)
        }
    }

    // MARK: Transfer

    /// Downloads `url` into `destination`, resuming from `<destination>.partial`. `progress` receives the
    /// bytes of this file on disk so far. On cancellation the partial file is kept.
    func fetch(_ url: URL, to destination: URL, expectedSize: Int64, repoID: String, path: String,
               progress: @escaping @Sendable (Int64) -> Void) async throws {
        let fileManager = FileManager.default
        let partial = Self.partialURL(for: destination)
        var offset = Self.fileSize(at: partial) ?? 0
        if expectedSize > 0, offset > expectedSize {
            // Longer than the file: left over from another revision.
            Self.log.notice("\(path, privacy: .public): partial of \(offset) bytes exceeds \(expectedSize), restarting")
            try fileManager.removeItem(at: partial)
            offset = 0
        }
        if offset > 0 {
            Self.log.info("resume \(path, privacy: .public) from byte \(offset)")
        }
        progress(offset)

        if expectedSize == 0 || offset < expectedSize {
            if !fileManager.fileExists(atPath: partial.path) {
                fileManager.createFile(atPath: partial.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: partial)
            defer { try? handle.close() }
            try handle.seekToEnd()
            offset = try await appendChunks(from: url, to: handle, partial: partial, offset: offset,
                                            expectedSize: expectedSize, repoID: repoID, path: path,
                                            progress: progress)
            try handle.close()
        }

        let finalSize = Self.fileSize(at: partial) ?? 0
        if expectedSize > 0, finalSize != expectedSize {
            Self.log.error("\(path, privacy: .public): \(finalSize) bytes, expected \(expectedSize)")
            throw ModelDownloadError.sizeMismatch(path: path, expected: expectedSize, actual: finalSize)
        }
        try? fileManager.removeItem(at: destination)
        try fileManager.moveItem(at: partial, to: destination)
        Self.log.info("\(path, privacy: .public) complete · \(finalSize) bytes")
    }

    private func appendChunks(from url: URL, to handle: FileHandle, partial: URL, offset start: Int64,
                              expectedSize: Int64, repoID: String, path: String,
                              progress: @escaping @Sendable (Int64) -> Void) async throws -> Int64 {
        var offset = start
        var validator = Self.validator(of: partial)
        while expectedSize == 0 || offset < expectedSize {
            try Task.checkCancellation()
            let rangeEnd = expectedSize > 0
                ? min(offset + Self.chunkSize - 1, expectedSize - 1)
                : offset + Self.chunkSize - 1
            let request = makeRangeRequest(url: url, from: offset, to: rangeEnd)
            let base = offset
            let reply = try await chunkWithRetries(request, offset: offset, repoID: repoID, path: path) { received in
                progress(base + received)
            }
            guard let reply, !reply.data.isEmpty else { break }
            let chunk = reply.data
            let etag = reply.etag
            if let etag, etag != validator {
                if let validator, offset > 0 {
                    Self.log.notice("""
                        \(path, privacy: .public) changed on the server \
                        (\(validator, privacy: .public) → \(etag, privacy: .public)), restarting
                        """)
                    try handle.truncate(atOffset: 0)
                    offset = 0
                    progress(0)
                    Self.setValidator(etag, of: partial)
                    continue
                }
                validator = etag
                Self.setValidator(etag, of: partial)
            }
            try handle.write(contentsOf: chunk)
            offset += Self.length(of: chunk)
            progress(offset)
            // Unknown size: a short chunk is the last one.
            if expectedSize == 0, Self.length(of: chunk) < rangeEnd - base + 1 { break }
        }
        return offset
    }

    func makeRangeRequest(url: URL, from start: Int64, to end: Int64) -> URLRequest {
        var request = auth.request(for: url)
        request.setValue("bytes=\(start)-\(end)", forHTTPHeaderField: "Range")
        return request
    }

    /// One chunk and its ETag, retried with backoff on network errors and 5xx. `nil` means the server has
    /// nothing past `offset` (416).
    private func chunkWithRetries(_ request: URLRequest, offset: Int64, repoID: String, path: String,
                                  onBytes: @escaping @Sendable (Int64) -> Void) async throws -> ChunkReply? {
        var lastError: any Error = URLError(.cannotConnectToHost)
        for attempt in 1...Self.maxChunkAttempts {
            if attempt > 1 {
                Self.log.warning("\(path, privacy: .public) chunk @\(offset) retry \(attempt)")
                try await Task.sleep(for: retryDelay(attempt))
            }
            do {
                return try await chunk(request, offset: offset, repoID: repoID, path: path, onBytes: onBytes)
            } catch let error where !Self.isRetryable(error) {
                throw error
            } catch {
                lastError = error
                let reason = error.localizedDescription
                Self.log.error("\(path, privacy: .public) chunk @\(offset) #\(attempt): \(reason, privacy: .public)")
            }
        }
        throw lastError
    }

    private func chunk(_ request: URLRequest, offset: Int64, repoID: String, path: String,
                       onBytes: @escaping @Sendable (Int64) -> Void) async throws -> ChunkReply? {
        let (data, response) = try await load(request, onBytes: onBytes)
        guard let http = response as? HTTPURLResponse else { throw ModelDownloadError.http(status: 0) }
        let etag = http.value(forHTTPHeaderField: "ETag")
        switch http.statusCode {
        case 416:
            return nil
        case 206:
            return (data, etag)
        case 200 where offset == 0:
            // The server ignored Range and sent the whole file; fine only from the start.
            return (data, etag)
        default:
            throw ModelDownloadError(response: http, request: request, repoID: repoID, path: path)
                ?? ModelDownloadError.http(status: http.statusCode)
        }
    }

    typealias ChunkReply = (data: Data, etag: String?)

    static func isRetryable(_ error: any Error) -> Bool {
        if error is CancellationError { return false }
        if let failure = error as? ModelDownloadError { return failure.isTransient }
        return true
    }

    // MARK: Request plumbing

    /// A data task whose redirects drop credentials for other hosts and whose received bytes are polled for
    /// progress; `URLSession.data(for:)` exposes neither.
    private func load(_ request: URLRequest,
                      onBytes: (@Sendable (Int64) -> Void)?) async throws -> (Data, URLResponse) {
        let handle = DataTaskHandle()
        let redirects = AuthRedirectDelegate(auth: auth)
        let poller = onBytes.map { report in
            Task { [progressInterval] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: progressInterval)
                    if let received = handle.bytesReceived, received > 0 { report(received) }
                }
            }
        }
        defer { poller?.cancel() }
        do {
            return try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    let task = session.dataTask(with: request) { data, response, error in
                        if let error {
                            continuation.resume(throwing: error)
                        } else if let response {
                            continuation.resume(returning: (data ?? Data(), response))
                        } else {
                            continuation.resume(throwing: URLError(.badServerResponse))
                        }
                    }
                    task.delegate = redirects
                    handle.start(task)
                }
            } onCancel: {
                handle.cancel()
            }
        } catch let error as URLError where error.code == .cancelled {
            throw CancellationError()
        }
    }

    static func treeURL(repoID: String, revision: String) -> URL? {
        let parts = repoID.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else { return nil }
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/api/models/\(repoID)/tree/\(revision)"
        return components.url
    }

    static func length(of data: Data) -> Int64 { Int64(data.count) }

    static func fileSize(at url: URL) -> Int64? {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes?[.size] as? NSNumber)?.int64Value
    }
}

// MARK: - Partial validator

extension HubFileTransfer {
    // The ETag rides on the partial as an extended attribute, so deleting or moving the partial takes it along.

    static func validator(of partial: URL) -> String? {
        partial.withUnsafeFileSystemRepresentation { path -> String? in
            guard let path else { return nil }
            let length = getxattr(path, validatorAttribute, nil, 0, 0, 0)
            guard length > 0 else { return nil }
            var bytes = [UInt8](repeating: 0, count: length)
            guard getxattr(path, validatorAttribute, &bytes, length, 0, 0) == length else { return nil }
            return String(bytes: bytes, encoding: .utf8)
        }
    }

    static func setValidator(_ etag: String, of partial: URL) {
        let bytes = Array(etag.utf8)
        partial.withUnsafeFileSystemRepresentation { path in
            guard let path else { return }
            if setxattr(path, validatorAttribute, bytes, bytes.count, 0, 0) != 0 {
                log.error("could not record the ETag of \(partial.lastPathComponent, privacy: .public): errno \(errno)")
            }
        }
    }
}

/// One entry of the Hugging Face tree API.
private struct TreeEntry: Decodable {
    struct LargeFileInfo: Decodable { let size: Int64? }
    let type: String
    let path: String
    let size: Int64?
    let largeFile: LargeFileInfo?

    enum CodingKeys: String, CodingKey {
        case type, path, size
        case largeFile = "lfs"
    }
}

// MARK: - Redirects

/// Rebuilds a redirected request's credentials from the authorizer, so the Hugging Face token reaches only
/// `huggingface.co` / `hf.co` and never the CDN host a file redirects to. Whether URLSession strips
/// `Authorization` across hosts by itself is not something to rely on.
final class AuthRedirectDelegate: NSObject, URLSessionTaskDelegate, Sendable {
    let auth: any DownloadAuthorizing

    init(auth: any DownloadAuthorizing) {
        self.auth = auth
    }

    func redirected(_ request: URLRequest) -> URLRequest {
        var rebuilt = request
        rebuilt.setValue(nil, forHTTPHeaderField: "Authorization")
        auth.authorize(&rebuilt)
        return rebuilt
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest) async -> URLRequest? {
        redirected(request)
    }
}

// MARK: - Task handle

/// The running data task, shared with the cancellation handler and the progress poller.
private final class DataTaskHandle: Sendable {
    private struct State {
        var task: URLSessionDataTask?
        var isCancelled = false
    }

    private let guarded = Mutex(State())

    func start(_ task: URLSessionDataTask) {
        let cancelled = guarded.withLock { current -> Bool in
            current.task = task
            return current.isCancelled
        }
        task.resume()
        if cancelled { task.cancel() }
    }

    func cancel() {
        let task = guarded.withLock { current -> URLSessionDataTask? in
            current.isCancelled = true
            return current.task
        }
        task?.cancel()
    }

    var bytesReceived: Int64? {
        guarded.withLock { $0.task?.countOfBytesReceived }
    }
}
