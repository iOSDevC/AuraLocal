import Foundation
import Synchronization

// MARK: - GGUFModelDownloader

/// Downloads GGUF model files from Hugging Face with progress tracking and **pause/resume** support.
///
/// GGUF files are large (4-8 GB for 7B models), so this downloader:
/// - Supports **pausing** an in-flight download and **resuming** it later via `URLSessionDownloadTask`'s
///   resume-data mechanism (HTTP `Range` under the hood) — a pause suspends the awaiting `download(...)`
///   call, it does not fail it; only ``cancel()`` fails it.
/// - Reports progress for UI display.
/// - Stores files in the model's ``Model/cacheDirectory`` (under ``ModelStorage/rootDirectory``), and uses a file an
///   earlier version left in Caches instead of downloading it again.
/// - Hashes a Hugging Face download against the SHA-256 published for it before moving it into place; a mismatch
///   deletes it and fails with ``ModelIntegrityError/checksumMismatch(path:expected:actual:)``.
/// - Waits while another download writes the same file (any downloader, ``ModelDownloads``, another process), then
///   returns the finished file without a transfer. A paused download keeps the file reserved until it ends.
@MainActor
public final class GGUFModelDownloader: ObservableObject {

    // MARK: - State

    @Published public private(set) var progress: Double = 0
    @Published public private(set) var downloadedBytes: Int64 = 0
    @Published public private(set) var totalBytes: Int64 = 0
    @Published public private(set) var isDownloading = false
    @Published public private(set) var isPaused = false
    @Published public private(set) var error: String?

    private var downloadTask: URLSessionDownloadTask?
    private var delegateSession: URLSession?
    private var resumeData: Data?
    /// The awaiting `download(...)` call — resumed exactly once on genuine finish/error/cancel, and kept
    /// pending across a pause/resume cycle.
    private var continuation: CheckedContinuation<URL, Error>?
    /// Retained so `resume()` can rebuild the download task against the same delegate session + continuation.
    private var context: DownloadContext?
    /// Bumped on every `pause()`. The resume-data callback captures its generation and drops if a newer
    /// pause/resume has since happened — so a late callback from an old pause can't clobber current state.
    private var pauseGeneration = 0
    /// Bumped on every `cancel()`, so it also ends calls still queued for the file and drops late delegate results.
    private var cancelGeneration = 0
    /// Adds the Keychain Hugging Face token to requests for `huggingface.co` only.
    private let auth: any DownloadAuthorizing
    private let sessionConfiguration: URLSessionConfiguration
    /// Fetches the published SHA-256 before each download.
    private let checksums: HubFileTransfer

    init(auth: any DownloadAuthorizing = KeychainDownloadAuth(),
         sessionConfiguration: URLSessionConfiguration = .default,
         checksumSession: URLSession = HubFileTransfer.sharedSession) {
        self.auth = auth
        self.sessionConfiguration = sessionConfiguration
        self.checksums = HubFileTransfer(session: checksumSession, auth: auth)
    }

    private struct DownloadContext {
        let url: URL
        let destination: URL
        let modelName: String
        let onProgress: @MainActor (String) -> Void
    }

    // MARK: - Download

    /// Download the GGUF file for a model. Returns the local file URL on success. Stays suspended across
    /// `pause()`/`resume()`; throws only on a genuine error or ``cancel()``.
    public func download(
        model: Model,
        onProgress: @escaping @MainActor (String) -> Void
    ) async throws -> URL {
        // A user-supplied local file is used in place — nothing to download.
        if let local = model.localFileURL {
            guard FileManager.default.fileExists(atPath: local.path) else {
                throw TonamoxError.invalidResponse("File not found: \(local.path)")
            }
            onProgress("\(model.displayName) (local file)")
            return local
        }

        guard let filename = model.ggufFilename else {
            throw TonamoxError.invalidResponse("Model \(model.displayName) has no GGUF filename")
        }

        // Already downloaded, in the storage root or in Caches.
        if let existing = model.resolvedFileURL, FileManager.default.fileExists(atPath: existing.path) {
            onProgress("\(model.displayName) already downloaded")
            return existing
        }

        let cacheDir = model.cacheDirectory
        let destinationURL = cacheDir.appendingPathComponent(filename)

        // Prefer the model's explicit download URL — user-supplied models carry the exact HF
        // `resolve/<rev>/…` URL the user pasted; catalog models derive it from repo + filename at `main`.
        let hfURL: URL
        if let explicit = model.downloadURL {
            hfURL = explicit
        } else if let derived = huggingFaceURL(repo: model.rawValue, filename: filename) {
            hfURL = derived
        } else {
            throw TonamoxError.invalidResponse("Couldn't build a download URL for \(model.displayName)")
        }

        // Another downloader, ModelDownloads or another process may be writing this file's .partial.
        let generation = cancelGeneration
        return try await DownloadGate.shared.exclusive(destinationURL) { _ in
            try await downloadHolding(model, from: hfURL, to: destinationURL, generation: generation,
                                      onProgress: onProgress)
        }
    }

    /// The rest of ``download(model:onProgress:)``, run while this file's ``DownloadGate`` turn is held.
    private func downloadHolding(
        _ model: Model,
        from hfURL: URL,
        to destinationURL: URL,
        generation: Int,
        onProgress: @escaping @MainActor (String) -> Void
    ) async throws -> URL {
        try checkNotCancelled(since: generation)
        // The download this call waited for may have finished the file.
        if let existing = model.resolvedFileURL, FileManager.default.fileExists(atPath: existing.path) {
            onProgress("\(model.displayName) already downloaded")
            return existing
        }
        try FileManager.default.createDirectory(at: destinationURL.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)

        // Looked up before the single-flight check: no suspension may separate that check from claiming the slot.
        let expectedSHA256 = try await checksums.publishedSHA256(of: hfURL)
        try checkNotCancelled(since: generation)

        // Single-flight: this downloader owns one continuation. A second download() while one is in flight
        // (running OR paused) must not orphan the first awaiter — reject it instead of overwriting the slot.
        guard continuation == nil else {
            throw TonamoxError.invalidResponse("A model download is already in progress")
        }

        isDownloading = true
        isPaused = false
        error = nil
        onProgress("Downloading \(model.displayName)...")

        do {
            let localURL = try await performDownload(
                from: hfURL, to: destinationURL, modelName: model.displayName, repoID: model.repoID,
                expectedSHA256: expectedSHA256, onProgress: onProgress)
            isDownloading = false
            onProgress("\(model.displayName) download complete")
            return localURL
        } catch {
            isDownloading = false
            isPaused = false
            self.error = error.localizedDescription
            throw error
        }
    }

    /// Pause the current download, retaining progress. The awaiting `download(...)` stays suspended; call
    /// ``resume()`` to continue or ``cancel()`` to abandon it.
    public func pause() {
        guard isDownloading, !isPaused, let task = downloadTask else { return }
        isDownloading = false
        isPaused = true
        pauseGeneration &+= 1
        let gen = pauseGeneration
        // Produces resume data for this pause. Keep the delegate session alive so resume() can start a new task
        // on it. The gen guard drops a late callback whose pause was already superseded by a resume/re-pause.
        task.cancel(byProducingResumeData: { [weak self] data in
            Task { @MainActor in
                guard let self, self.isPaused, gen == self.pauseGeneration else { return }
                self.resumeData = data
            }
        })
        downloadTask = nil
    }

    /// Resume a paused download from where it stopped (or restart cleanly if resume data was unavailable).
    public func resume() {
        guard isPaused, let session = delegateSession, let context else { return }
        isPaused = false
        isDownloading = true
        error = nil
        let task: URLSessionDownloadTask
        if let data = resumeData {
            task = session.downloadTask(withResumeData: data)
            resumeData = nil
        } else {
            task = session.downloadTask(with: auth.request(for: context.url))   // fresh start (no resume data)
        }
        downloadTask = task
        context.onProgress("Resuming \(context.modelName)...")
        task.resume()
    }

    /// Cancel and abandon the current download — fails the awaiting `download(...)` with a cancellation, and any
    /// call of this downloader still waiting for its file.
    public func cancel() {
        cancelGeneration &+= 1
        isPaused = false
        isDownloading = false
        resumeData = nil
        downloadTask?.cancel()
        downloadTask = nil
        let session = delegateSession
        delegateSession = nil
        if let continuation {
            self.continuation = nil
            self.context = nil
            // Failed once the session's last callback returns: until then a file being verified may still be
            // promoted, and the next writer must not take the file before that.
            if let delegate = session?.delegate as? DownloadDelegate {
                delegate.cancel { continuation.resume(throwing: CancellationError()) }
            } else {
                continuation.resume(throwing: CancellationError())
            }
        }
        session?.invalidateAndCancel()
    }

    // MARK: - Private

    private func performDownload(
        from url: URL,
        to destination: URL,
        modelName: String,
        repoID: String,
        expectedSHA256: String?,
        onProgress: @escaping @MainActor (String) -> Void
    ) async throws -> URL {
        context = DownloadContext(url: url, destination: destination, modelName: modelName, onProgress: onProgress)
        let generation = cancelGeneration
        return try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation
            let delegate = DownloadDelegate(
                destination: destination,
                repoID: repoID,
                expectedSHA256: expectedSHA256,
                auth: auth,
                onVerifying: {
                    Task { @MainActor in onProgress("Verifying \(modelName)…") }
                },
                onProgress: { [weak self] _, totalWritten, totalExpected in
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.downloadedBytes = totalWritten
                        self.totalBytes = totalExpected
                        if totalExpected > 0 {
                            self.progress = Double(totalWritten) / Double(totalExpected)
                            let pct = Int(self.progress * 100)
                            let mbDownloaded = totalWritten / (1024 * 1024)
                            let mbTotal = totalExpected / (1024 * 1024)
                            onProgress("Downloading \(modelName): \(pct)% (\(mbDownloaded)/\(mbTotal) MB)")
                        }
                    }
                },
                onFinished: { [weak self] url in
                    Task { @MainActor [weak self] in self?.finish(.success(url), generation: generation) }
                },
                onError: { [weak self] error in
                    Task { @MainActor [weak self] in self?.finish(.failure(error), generation: generation) }
                },
                onCancelled: { _ in
                    // A cancellation reaches the delegate from pause() — resume data is captured by pause()'s
                    // own (generation-guarded) resume-data closure — or from cancel(), which hands the
                    // continuation to the delegate. Either way there is nothing to resolve here.
                }
            )
            let session = URLSession(configuration: sessionConfiguration, delegate: delegate, delegateQueue: nil)
            self.delegateSession = session
            let task = session.downloadTask(with: auth.request(for: url))
            self.downloadTask = task
            task.resume()
        }
    }

    /// Resolve the awaiting `download(...)` exactly once and tear down the session.
    private func finish(_ result: Result<URL, Error>, generation: Int) {
        // A cancelled transfer's late result must not resolve a download started after the cancel.
        guard generation == cancelGeneration, let continuation else { return }
        self.continuation = nil
        self.context = nil
        isPaused = false   // a pause that raced this finish must not strand isPaused=true on an idle downloader
        delegateSession?.finishTasksAndInvalidate()
        delegateSession = nil
        continuation.resume(with: result)
    }

    /// A queued call holds no continuation or task, so `cancel()` reaches it only through the generation.
    private func checkNotCancelled(since generation: Int) throws {
        try Task.checkCancellation()
        guard generation == cancelGeneration else { throw CancellationError() }
    }

    private func huggingFaceURL(repo: String, filename: String) -> URL? {
        URL(string: "https://huggingface.co/\(repo)/resolve/main/\(filename)")
    }

    /// Throws for a non-2xx download: URLSession saves an error body like any file, and moving it into place
    /// would leave a `.gguf` that is really an HTML or JSON error page.
    nonisolated static func validateDownloadResponse(_ response: URLResponse?, request: URLRequest? = nil,
                                                     repoID: String, path: String) throws {
        guard let http = response as? HTTPURLResponse else { throw ModelDownloadError.http(status: 0) }
        let sentToken = request?.value(forHTTPHeaderField: "Authorization") != nil
        if let failure = ModelDownloadError(response: http, repoID: repoID, path: path, sentToken: sentToken) {
            throw failure
        }
    }
}

// MARK: - DownloadDelegate

private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate {
    let destination: URL
    let repoID: String
    let expectedSHA256: String?
    let redirects: AuthRedirectDelegate
    let onVerifying: @Sendable () -> Void
    let onProgress: @Sendable (Int64, Int64, Int64) -> Void
    let onFinished: @Sendable (URL) -> Void
    let onError: @Sendable (Error) -> Void
    let onCancelled: @Sendable (Data?) -> Void
    /// Set by `cancel(then:)`: what to run once the session is invalidated.
    private let cancellation = Mutex<(@Sendable () -> Void)?>(nil)

    init(
        destination: URL,
        repoID: String,
        expectedSHA256: String?,
        auth: any DownloadAuthorizing,
        onVerifying: @escaping @Sendable () -> Void,
        onProgress: @escaping @Sendable (Int64, Int64, Int64) -> Void,
        onFinished: @escaping @Sendable (URL) -> Void,
        onError: @escaping @Sendable (Error) -> Void,
        onCancelled: @escaping @Sendable (Data?) -> Void
    ) {
        self.destination = destination
        self.repoID = repoID
        self.expectedSHA256 = expectedSHA256
        self.redirects = AuthRedirectDelegate(auth: auth)
        self.onVerifying = onVerifying
        self.onProgress = onProgress
        self.onFinished = onFinished
        self.onError = onError
        self.onCancelled = onCancelled
    }

    /// Stops a finished file from being promoted; `then` runs after the session's last callback has returned.
    func cancel(then: @escaping @Sendable () -> Void) {
        cancellation.withLock { $0 = then }
    }

    private var isCancelled: Bool {
        cancellation.withLock { $0 != nil }
    }

    func urlSession(_ session: URLSession, didBecomeInvalidWithError error: (any Error)?) {
        cancellation.withLock { $0 }?()
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress(bytesWritten, totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        do {
            try GGUFModelDownloader.validateDownloadResponse(
                downloadTask.response, request: downloadTask.originalRequest, repoID: repoID,
                path: destination.lastPathComponent)
            // URLSession deletes `location` when this returns, so the file is staged before hashing. As a finished
            // `.partial` it is verified again by ModelDownloads if the app dies mid-hash.
            let staged = HubFileTransfer.partialURL(for: destination)
            try? FileManager.default.removeItem(at: staged)
            try FileManager.default.moveItem(at: location, to: staged)
            let stagedSize = HubFileTransfer.fileSize(at: staged) ?? 0
            // A cancelled download leaves the staged file to the next caller, which verifies it again.
            guard !isCancelled else { return }
            if let expectedSHA256 {
                onVerifying()
                try FileIntegrity.verify(staged, expectedSHA256: expectedSHA256, path: destination.lastPathComponent)
            }
            guard !isCancelled else { return }
            try HubFileTransfer.promote(staged, to: destination, expectedSize: stagedSize,
                                        path: destination.lastPathComponent)
            onFinished(destination)
        } catch {
            try? FileManager.default.removeItem(at: location)
            onError(error)
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        completionHandler(redirects.redirected(request))
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        guard let error else { return }   // success is delivered via didFinishDownloadingTo
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled {
            onCancelled(nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data)
        } else {
            onError(error)
        }
    }
}
