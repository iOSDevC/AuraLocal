//
//  ChunkedDownloader.swift
//  AuraCore
//
//  Downloads large files in parallel chunks using HTTP Range requests,
//  then assembles them into the final file. Falls back to single-stream
//  download when the server doesn't support ranges.
//
//  Designed for downloading ML model weights from HuggingFace Hub
//  on memory-constrained iOS devices.

import Foundation
import os.log

private let logger = Logger(subsystem: "dev.auralocal", category: "ChunkedDownloader")

/// Downloads files using parallel HTTP Range requests for large files.
///
/// Usage:
/// ```swift
/// let downloader = ChunkedDownloader()
/// try await downloader.download(
///     from: url,
///     to: localFile,
///     fileName: "model.safetensors"
/// ) { progress in
///     print("\(progress.percent)% — \(progress.bytesDownloaded)/\(progress.totalBytes)")
/// }
/// ```
public actor ChunkedDownloader {

    /// Number of parallel chunks for large files.
    private let maxConcurrentChunks: Int

    /// Minimum file size (bytes) to activate chunked download.
    private let chunkThreshold: Int

    private let session: URLSession

    /// Create a chunked downloader.
    /// - Parameters:
    ///   - concurrentChunks: Max parallel connections per file (default 4).
    ///   - chunkThresholdMB: Files smaller than this (MB) download in a single stream (default 10).
    ///   - timeoutSeconds: Per-request timeout in seconds (default 60).
    ///   - resourceTimeoutSeconds: Total per-file timeout in seconds (default 1800 = 30min).
    public init(
        concurrentChunks: Int = 4,
        chunkThresholdMB: Int = 10,
        timeoutSeconds: TimeInterval = 60,
        resourceTimeoutSeconds: TimeInterval = 1800
    ) {
        self.maxConcurrentChunks = max(1, concurrentChunks)
        self.chunkThreshold = chunkThresholdMB * 1024 * 1024

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = timeoutSeconds
        config.timeoutIntervalForResource = resourceTimeoutSeconds
        config.waitsForConnectivity = true
        config.httpMaximumConnectionsPerHost = max(1, concurrentChunks) + 2
        self.session = URLSession(configuration: config)
    }

    /// Progress report for an in-flight download.
    public struct DownloadProgress: Sendable {
        public let bytesDownloaded: Int64
        public let totalBytes: Int64
        public let fileName: String

        /// Percentage complete (0–100).
        public var percent: Int {
            totalBytes > 0 ? Int(bytesDownloaded * 100 / totalBytes) : 0
        }
    }

    /// Download a file, using chunked parallel download for large files.
    ///
    /// - If the server supports `Accept-Ranges: bytes` and the file exceeds
    ///   `chunkThresholdMB`, the file is split into `concurrentChunks` byte ranges
    ///   downloaded in parallel, then assembled on disk.
    /// - Otherwise falls back to a single-stream download.
    ///
    /// - Parameters:
    ///   - url: Remote file URL.
    ///   - destination: Local file URL to write to.
    ///   - fileName: Display name for progress reporting.
    ///   - onProgress: Called periodically with download progress.
    public func download(
        from url: URL,
        to destination: URL,
        fileName: String,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws {
        // HEAD to check size and Range support
        var headRequest = URLRequest(url: url)
        headRequest.httpMethod = "HEAD"
        let (_, headResponse) = try await session.data(for: headRequest)

        guard let http = headResponse as? HTTPURLResponse, http.statusCode == 200 else {
            try await singleDownload(from: url, to: destination, fileName: fileName, onProgress: onProgress)
            return
        }

        let contentLength = http.expectedContentLength
        let acceptsRanges = http.value(forHTTPHeaderField: "Accept-Ranges")?.lowercased() == "bytes"

        if !acceptsRanges || contentLength <= 0 || contentLength < Int64(chunkThreshold) {
            try await singleDownload(from: url, to: destination, fileName: fileName, onProgress: onProgress)
            return
        }

        // Chunked parallel download
        logger.info("Chunked: \(fileName) (\(contentLength / 1024 / 1024)MB, \(self.maxConcurrentChunks) chunks)")

        let chunkSize = contentLength / Int64(maxConcurrentChunks)
        var ranges: [(start: Int64, end: Int64)] = []
        for i in 0..<maxConcurrentChunks {
            let start = Int64(i) * chunkSize
            let end = (i == maxConcurrentChunks - 1) ? contentLength - 1 : start + chunkSize - 1
            ranges.append((start, end))
        }

        let tracker = ProgressTracker(totalBytes: contentLength, fileName: fileName, onProgress: onProgress)

        let tempDir = FileManager.default.temporaryDirectory
            .appending(path: "chunked_\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        defer { try? FileManager.default.removeItem(at: tempDir) }

        try await withThrowingTaskGroup(of: (Int, URL).self) { group in
            for (index, range) in ranges.enumerated() {
                group.addTask {
                    let chunkFile = tempDir.appending(path: "chunk_\(index)")
                    try await self.downloadChunk(from: url, range: range, to: chunkFile, tracker: tracker)
                    return (index, chunkFile)
                }
            }

            var chunkFiles = [(Int, URL)]()
            for try await result in group {
                chunkFiles.append(result)
            }

            // Assemble in order
            chunkFiles.sort { $0.0 < $1.0 }
            FileManager.default.createFile(atPath: destination.path(), contents: nil)
            let handle = try FileHandle(forWritingTo: destination)
            defer { try? handle.close() }

            for (_, chunkFile) in chunkFiles {
                let data = try Data(contentsOf: chunkFile)
                handle.write(data)
            }
        }

        logger.info("Assembled: \(fileName) (\(contentLength / 1024 / 1024)MB)")
    }

    // MARK: - Single download (fallback)

    private func singleDownload(
        from url: URL,
        to destination: URL,
        fileName: String,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws {
        let (tempURL, response) = try await session.download(from: url)

        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            try? FileManager.default.removeItem(at: tempURL)
            throw DownloadError.httpError(http.statusCode, fileName)
        }

        try FileManager.default.moveItem(at: tempURL, to: destination)

        let size = (try? FileManager.default.attributesOfItem(atPath: destination.path())[.size] as? Int64) ?? 0
        onProgress(DownloadProgress(bytesDownloaded: size, totalBytes: size, fileName: fileName))
    }

    // MARK: - Chunk download

    private func downloadChunk(
        from url: URL,
        range: (start: Int64, end: Int64),
        to destination: URL,
        tracker: ProgressTracker
    ) async throws {
        var request = URLRequest(url: url)
        request.setValue("bytes=\(range.start)-\(range.end)", forHTTPHeaderField: "Range")

        let (bytes, response) = try await session.bytes(for: request)

        guard let http = response as? HTTPURLResponse,
              http.statusCode == 206 || http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw DownloadError.httpError(code, url.lastPathComponent)
        }

        FileManager.default.createFile(atPath: destination.path(), contents: nil)
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }

        var buffer = Data()
        let flushSize = 256 * 1024  // 256KB

        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= flushSize {
                handle.write(buffer)
                await tracker.add(Int64(buffer.count))
                buffer.removeAll(keepingCapacity: true)
            }
        }

        if !buffer.isEmpty {
            handle.write(buffer)
            await tracker.add(Int64(buffer.count))
        }
    }

    // MARK: - Error

    public enum DownloadError: LocalizedError {
        case httpError(Int, String)

        public var errorDescription: String? {
            switch self {
            case .httpError(let code, let file):
                return "\(file): HTTP \(code)"
            }
        }
    }
}

// MARK: - Progress Tracker

/// Thread-safe progress aggregator across parallel chunk downloads.
actor ProgressTracker {
    private var bytesDownloaded: Int64 = 0
    private let totalBytes: Int64
    private let fileName: String
    private let onProgress: @Sendable (ChunkedDownloader.DownloadProgress) -> Void
    private var lastReportedPercent: Int = -1

    init(
        totalBytes: Int64,
        fileName: String,
        onProgress: @Sendable @escaping (ChunkedDownloader.DownloadProgress) -> Void
    ) {
        self.totalBytes = totalBytes
        self.fileName = fileName
        self.onProgress = onProgress
    }

    func add(_ bytes: Int64) {
        bytesDownloaded += bytes
        let percent = totalBytes > 0 ? Int(bytesDownloaded * 100 / totalBytes) : 0
        if percent != lastReportedPercent && percent % 2 == 0 {
            lastReportedPercent = percent
            onProgress(ChunkedDownloader.DownloadProgress(
                bytesDownloaded: bytesDownloaded,
                totalBytes: totalBytes,
                fileName: fileName
            ))
        }
    }
}
