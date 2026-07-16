import Foundation

// MARK: - GGUFModelDownloader

/// Downloads GGUF model files from Hugging Face with progress tracking and **pause/resume** support.
///
/// GGUF files are large (4-8 GB for 7B models), so this downloader:
/// - Supports **pausing** an in-flight download and **resuming** it later via `URLSessionDownloadTask`'s
///   resume-data mechanism (HTTP `Range` under the hood) — a pause suspends the awaiting `download(...)`
///   call, it does not fail it; only ``cancel()`` fails it.
/// - Reports progress for UI display.
/// - Stores files in the model's ``Model/cacheDirectory``.
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
                throw AuraError.invalidResponse("File not found: \(local.path)")
            }
            onProgress("\(model.displayName) (local file)")
            return local
        }

        guard let filename = model.ggufFilename else {
            throw AuraError.invalidResponse("Model \(model.displayName) has no GGUF filename")
        }

        let cacheDir = model.cacheDirectory
        let destinationURL = cacheDir.appendingPathComponent(filename)

        // Check if already downloaded
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            onProgress("\(model.displayName) already downloaded")
            return destinationURL
        }

        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        // Prefer the model's explicit download URL — user-supplied models carry the exact HF
        // `resolve/<rev>/…` URL the user pasted; catalog models derive it from repo + filename at `main`.
        let hfURL: URL
        if let explicit = model.downloadURL {
            hfURL = explicit
        } else if let derived = huggingFaceURL(repo: model.rawValue, filename: filename) {
            hfURL = derived
        } else {
            throw AuraError.invalidResponse("Couldn't build a download URL for \(model.displayName)")
        }

        // Single-flight: this downloader owns one continuation. A second download() while one is in flight
        // (running OR paused) must not orphan the first awaiter — reject it instead of overwriting the slot.
        guard continuation == nil else {
            throw AuraError.invalidResponse("A model download is already in progress")
        }

        isDownloading = true
        isPaused = false
        error = nil
        onProgress("Downloading \(model.displayName)...")

        do {
            let localURL = try await performDownload(
                from: hfURL, to: destinationURL, modelName: model.displayName, onProgress: onProgress)
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
            task = session.downloadTask(with: context.url)   // fresh start (resume data not yet/ever available)
        }
        downloadTask = task
        context.onProgress("Resuming \(context.modelName)...")
        task.resume()
    }

    /// Cancel and abandon the current download — fails the awaiting `download(...)` with a cancellation.
    public func cancel() {
        isPaused = false
        isDownloading = false
        resumeData = nil
        downloadTask?.cancel()
        downloadTask = nil
        // Fail the awaiting call ourselves; the delegate's later cancellation callback is then a no-op.
        if let continuation {
            self.continuation = nil
            self.context = nil
            continuation.resume(throwing: CancellationError())
        }
        delegateSession?.invalidateAndCancel()
        delegateSession = nil
    }

    // MARK: - Private

    private func performDownload(
        from url: URL,
        to destination: URL,
        modelName: String,
        onProgress: @escaping @MainActor (String) -> Void
    ) async throws -> URL {
        context = DownloadContext(url: url, destination: destination, modelName: modelName, onProgress: onProgress)
        return try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation
            let delegate = DownloadDelegate(
                destination: destination,
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
                onFinished: { [weak self] url in Task { @MainActor [weak self] in self?.finish(.success(url)) } },
                onError: { [weak self] error in Task { @MainActor [weak self] in self?.finish(.failure(error)) } },
                onCancelled: { _ in
                    // A cancellation reaches the delegate from pause() — resume data is captured by pause()'s
                    // own (generation-guarded) resume-data closure — or from cancel(), which already failed the
                    // continuation. Either way there is nothing to resolve here.
                }
            )
            let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
            self.delegateSession = session
            let task = session.downloadTask(with: url)
            self.downloadTask = task
            task.resume()
        }
    }

    /// Resolve the awaiting `download(...)` exactly once and tear down the session.
    private func finish(_ result: Result<URL, Error>) {
        guard let continuation else { return }
        self.continuation = nil
        self.context = nil
        isPaused = false   // a pause that raced this finish must not strand isPaused=true on an idle downloader
        delegateSession?.finishTasksAndInvalidate()
        delegateSession = nil
        continuation.resume(with: result)
    }

    private func huggingFaceURL(repo: String, filename: String) -> URL? {
        URL(string: "https://huggingface.co/\(repo)/resolve/main/\(filename)")
    }
}

// MARK: - DownloadDelegate

private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate {
    let destination: URL
    let onProgress: @Sendable (Int64, Int64, Int64) -> Void
    let onFinished: @Sendable (URL) -> Void
    let onError: @Sendable (Error) -> Void
    let onCancelled: @Sendable (Data?) -> Void

    init(
        destination: URL,
        onProgress: @escaping @Sendable (Int64, Int64, Int64) -> Void,
        onFinished: @escaping @Sendable (URL) -> Void,
        onError: @escaping @Sendable (Error) -> Void,
        onCancelled: @escaping @Sendable (Data?) -> Void
    ) {
        self.destination = destination
        self.onProgress = onProgress
        self.onFinished = onFinished
        self.onError = onError
        self.onCancelled = onCancelled
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
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: location, to: destination)
            onFinished(destination)
        } catch {
            onError(error)
        }
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
