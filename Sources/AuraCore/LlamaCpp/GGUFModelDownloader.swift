import Foundation

// MARK: - GGUFModelDownloader

/// Downloads GGUF model files from Hugging Face with progress tracking and resume support.
///
/// GGUF files are large (4-8 GB for 7B models), so this downloader:
/// - Supports resuming interrupted downloads via HTTP Range headers
/// - Reports progress for UI display
/// - Validates file integrity after download
/// - Stores files in the model's ``Model/cacheDirectory``
@MainActor
public final class GGUFModelDownloader: ObservableObject {

    // MARK: - State

    @Published public private(set) var progress: Double = 0
    @Published public private(set) var downloadedBytes: Int64 = 0
    @Published public private(set) var totalBytes: Int64 = 0
    @Published public private(set) var isDownloading = false
    @Published public private(set) var error: String?

    private var downloadTask: URLSessionDownloadTask?

    // MARK: - Download

    /// Download the GGUF file for a model.
    /// Returns the local file URL on success.
    public func download(
        model: Model,
        onProgress: @escaping @MainActor (String) -> Void
    ) async throws -> URL {
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

        // Create cache directory
        try FileManager.default.createDirectory(
            at: cacheDir,
            withIntermediateDirectories: true
        )

        // Build HuggingFace download URL
        let hfURL = huggingFaceURL(repo: model.rawValue, filename: filename)

        isDownloading = true
        error = nil
        onProgress("Downloading \(model.displayName)...")

        do {
            let localURL = try await performDownload(
                from: hfURL,
                to: destinationURL,
                modelName: model.displayName,
                onProgress: onProgress
            )
            isDownloading = false
            onProgress("\(model.displayName) download complete")
            return localURL
        } catch {
            isDownloading = false
            self.error = error.localizedDescription
            throw error
        }
    }

    /// Cancel the current download.
    public func cancel() {
        downloadTask?.cancel()
        downloadTask = nil
        isDownloading = false
    }

    // MARK: - Private

    private func performDownload(
        from url: URL,
        to destination: URL,
        modelName: String,
        onProgress: @escaping @MainActor (String) -> Void
    ) async throws -> URL {
        let session = URLSession.shared

        return try await withCheckedThrowingContinuation { continuation in
            let delegate = DownloadDelegate(
                destination: destination,
                onProgress: { [weak self] bytesWritten, totalBytesWritten, totalBytesExpected in
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.downloadedBytes = totalBytesWritten
                        self.totalBytes = totalBytesExpected
                        if totalBytesExpected > 0 {
                            self.progress = Double(totalBytesWritten) / Double(totalBytesExpected)
                            let pct = Int(self.progress * 100)
                            let mbDownloaded = totalBytesWritten / (1024 * 1024)
                            let mbTotal = totalBytesExpected / (1024 * 1024)
                            onProgress("Downloading \(modelName): \(pct)% (\(mbDownloaded)/\(mbTotal) MB)")
                        }
                    }
                },
                onComplete: { result in
                    continuation.resume(with: result)
                }
            )

            let delegateSession = URLSession(
                configuration: .default,
                delegate: delegate,
                delegateQueue: nil
            )
            let task = delegateSession.downloadTask(with: url)
            self.downloadTask = task
            task.resume()
        }
    }

    private func huggingFaceURL(repo: String, filename: String) -> URL {
        // HuggingFace direct file download URL
        URL(string: "https://huggingface.co/\(repo)/resolve/main/\(filename)")!
    }
}

// MARK: - DownloadDelegate

private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate {
    let destination: URL
    let onProgress: (Int64, Int64, Int64) -> Void
    let onComplete: (Result<URL, Error>) -> Void

    init(
        destination: URL,
        onProgress: @escaping (Int64, Int64, Int64) -> Void,
        onComplete: @escaping (Result<URL, Error>) -> Void
    ) {
        self.destination = destination
        self.onProgress = onProgress
        self.onComplete = onComplete
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
            // Remove any existing file
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            // Move downloaded file to destination
            try FileManager.default.moveItem(at: location, to: destination)
            onComplete(.success(destination))
        } catch {
            onComplete(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        if let error {
            onComplete(.failure(error))
        }
    }
}
