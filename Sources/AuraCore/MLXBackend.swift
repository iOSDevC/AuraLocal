// MLX backend — compiled in only when the MLX products (mlx-swift-lm) are linked.
#if canImport(MLXLLM)
import Foundation
import MLX
import MLXLLM
import MLXVLM
import MLXLMCommon
import os
import Tokenizers

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - MLXBackend

/// Inference backend powered by Apple's MLX framework.
///
/// Loads the full model into unified memory via `LLMModelFactory` or
/// `VLMModelFactory` and delegates generation to `MLXLMCommon.generate()`.
/// Best for models that fit entirely in device RAM (typically up to 4B on iOS).
@MainActor
final class MLXBackend: InferenceBackend {

    // MARK: - State

    private var modelContainer: ModelContainer?
    private let model: Model
    let parameters: GenerateParameters
    private(set) var lastGenerationStats: GenerationStats?
    private(set) var lastContextOverflow: ContextWindowOverflow?

    var isLoaded: Bool { modelContainer != nil }

    // MARK: - Init

    /// `nil` sampling takes ``SamplingSettings/defaults(for:)``. Temperature 0 selects MLX's argmax sampler.
    init(model: Model, sampling: SamplingSettings? = nil) {
        self.model = model
        let settings = sampling ?? .defaults(for: model)
        // Unbounded, the KV cache grows with every token until jetsam; the window makes MLX rotate it instead.
        self.parameters = GenerateParameters(
            maxKVSize: HardwareAnalyzer.recommendedContextWindow(for: model),
            temperature: settings.temperature,
            topP: settings.topP)
    }

    // MARK: - InferenceBackend

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        guard modelContainer == nil else { return }

        // Scale GPU cache limit proportionally to model size.
        let modelWeightsBytes = model.approximateSizeMB * 1024 * 1024
        let proportionalCache = modelWeightsBytes / 12
        let cacheLimitBytes   = max(128 * 1024 * 1024, min(proportionalCache, 512 * 1024 * 1024))
        MLX.GPU.set(cacheLimit: cacheLimitBytes)

        let config = Self.configuration(for: model)
        let downloader = AuraHFDownloader()
        let tokenizer = AuraTokenizerLoader()

        switch model.purpose {
        case .text:
            modelContainer = try await LLMModelFactory.shared.loadContainer(
                from: downloader,
                using: tokenizer,
                configuration: config
            ) { [model] progress in
                let pct = Int(progress.fractionCompleted * 100)
                Task { @MainActor in
                    onProgress("Downloading \(model.displayName): \(pct)%")
                }
            }

        case .vision, .visionSpecialized:
            modelContainer = try await VLMModelFactory.shared.loadContainer(
                from: downloader,
                using: tokenizer,
                configuration: config
            ) { [model] progress in
                let pct = Int(progress.fractionCompleted * 100)
                Task { @MainActor in
                    onProgress("Downloading \(model.displayName): \(pct)%")
                }
            }
        }

        onProgress("\(model.displayName) ready")
    }

    /// Some converted repositories omit a turn terminator from
    /// `generation_config.json`. Keep the exception next to the runtime instead
    /// of pretending that catalog metadata changes tokenizer behavior.
    nonisolated static func configuration(for model: Model) -> ModelConfiguration {
        let extraEOSTokens: Set<String>
        switch model.id {
        case "mediphi_3_8b_mlx":
            extraEOSTokens = ["<|end|>"]
        case "medgemma_4b", "medgemma_1_5_4b":
            extraEOSTokens = ["<end_of_turn>"]
        default:
            extraEOSTokens = []
        }
        return ModelConfiguration(id: model.rawValue, extraEOSTokens: extraEOSTokens)
    }

    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw AuraError.modelNotLoaded
        }

        var msgs: [[String: String]] = []
        if let sys = systemPrompt {
            msgs.append(["role": "system", "content": sys])
        }
        msgs.append(["role": "user", "content": prompt])
        let capturedMessages = msgs

        return try await performGeneration(
            container: container,
            prepareInput: { context in
                try await context.processor.prepare(input: .init(messages: capturedMessages))
            },
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    /// Nothing in the MLX path applies a grammar to the logits.
    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        constraint: GenerationConstraint,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        throw GenerationConstraintError.unsupportedBackend(.mlx)
    }

    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw AuraError.modelNotLoaded
        }

        var tempURL: URL?
        let capturedInput: UserInput
        if let img = image, let url = saveImageToTemp(img) {
            tempURL = url
            capturedInput = UserInput(prompt: prompt, images: [.url(url)])
        } else {
            capturedInput = UserInput(prompt: prompt)
        }

        defer {
            if let url = tempURL {
                try? FileManager.default.removeItem(at: url)
            }
        }

        nonisolated(unsafe) let unsafeInput = capturedInput
        return try await performGeneration(
            container: container,
            prepareInput: { context in
                try await context.processor.prepare(input: unsafeInput)
            },
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    func generate(
        messages: [[String: String]],
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw AuraError.modelNotLoaded
        }

        return try await performGeneration(
            container: container,
            prepareInput: { context in
                try await context.processor.prepare(input: .init(messages: messages))
            },
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    func unload() {
        modelContainer = nil
    }

    // MARK: - Core generation

    private struct Output: Sendable {
        let text: String
        let stats: GenerationStats
        let overflow: ContextWindowOverflow?
    }

    private func performGeneration(
        container: ModelContainer,
        prepareInput: @escaping @Sendable (ModelContext) async throws -> LMInput,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        lastGenerationStats = nil
        let params = self.parameters
        lastContextOverflow = nil
        let output = try await container.perform { (context: ModelContext) async throws -> Output in
            let start = ContinuousClock.now
            var firstToken: ContinuousClock.Instant?
            var lastToken: ContinuousClock.Instant?
            let input = try await prepareInput(context)
            let result = try MLXLMCommon.generate(
                input: input,
                parameters: params,
                context: context
            ) { tokens in
                let now = ContinuousClock.now
                if firstToken == nil { firstToken = now }
                lastToken = now
                guard !Task.isCancelled else { return .stop }

                let partial = context.tokenizer.decode(tokenIds: tokens)
                Task { @MainActor in
                    guard !Task.isCancelled else { return }
                    onToken(partial)
                }
                return tokens.count >= maxTokens ? .stop : .more
            }
            // `result.generateTime` also covers the step that drew the stop token, which `generationTokenCount`
            // leaves out; timing the callbacks matches llama.cpp's first-to-last-piece span.
            let decodeTime = firstToken.flatMap { first in lastToken.map { GenerationStats.seconds($0 - first) } } ?? 0
            let stats = GenerationStats(
                backend: .mlx,
                source: .runtime,
                promptTokens: result.promptTokenCount,
                generatedTokens: result.generationTokenCount,
                promptTime: result.promptTime,
                timeToFirstToken: GenerationStats.seconds((firstToken ?? ContinuousClock.now) - start),
                decodeTime: decodeTime
            )
            let overflow = params.maxKVSize.flatMap {
                ContextWindowOverflow.detect(promptTokens: result.promptTokenCount,
                                             generatedTokens: result.generationTokenCount, windowSize: $0)
            }.flatMap { candidate in
                Self.cacheHonorsWindow(context.model, params) ? candidate : nil
            }
            return Output(text: context.tokenizer.decode(tokenIds: result.tokenIds), stats: stats, overflow: overflow)
        }
        lastGenerationStats = output.stats
        lastContextOverflow = output.overflow
        return output.text
    }

    /// Several architectures (Qwen3.5 text, Gemma 3/4, LFM2…) build their own caches and ignore `maxKVSize`, so
    /// nothing was dropped there. Comparing the caches built with and without it tells them apart.
    nonisolated private static func cacheHonorsWindow(_ model: any LanguageModel, _ params: GenerateParameters) -> Bool {
        var unbounded = params
        unbounded.maxKVSize = nil
        return model.newCache(parameters: params).map(\.maxSize)
            != model.newCache(parameters: unbounded).map(\.maxSize)
    }

    // MARK: - Helpers

    private func saveImageToTemp(_ image: PlatformImage) -> URL? {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "mlxedge_\(UUID().uuidString).jpg")
#if canImport(UIKit)
        guard let data = image.jpegData(compressionQuality: 0.9) else { return nil }
#else
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let data = NSBitmapImageRep(cgImage: cgImage)
            .representation(using: .jpeg, properties: [:]) else { return nil }
#endif
        try? data.write(to: url)
        return url
    }
}

// MARK: - HuggingFace Downloader (MLXLMCommon.Downloader)

/// Resumable HuggingFace Hub snapshot downloader.
///
/// Chunked + Range instead of `URLSession.download(from:)`: the latter
/// restarts from byte 0 on every network failure, so large LFS files stalled
/// forever (~40%) when the CDN paused mid-stream. Each 32 MB chunk is written
/// to a `.partial` on disk and persists across cuts; progress is monotonic.
/// A finished LFS file is hashed against the tree's `lfs.oid` before it is promoted.
struct AuraHFDownloader: Downloader {
    private static let base = "https://huggingface.co"
    private static let chunkSize: Int64 = 32 * 1024 * 1024
    private static let maxChunkAttempts = 6

    private static let log = Logger(subsystem: "dev.auralocal", category: "HFDownloader")

    private struct TransientHTTPError: LocalizedError {
        let statusCode: Int
        var errorDescription: String? { "Download temporarily failed (HTTP \(statusCode))" }
    }

    /// Dedicated session with generous timeouts for large downloads.
    private static let defaultSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 7200
        config.waitsForConnectivity = true
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.httpMaximumConnectionsPerHost = 2
        return URLSession(configuration: config)
    }()

    private let session: URLSession
    private let cacheDirectory: URL
    /// Read only: a snapshot already completed there is loaded in place instead of downloaded again.
    private let fallbackDirectories: [URL]
    private let waitBeforeRetry: @Sendable (Double) async throws -> Void
    private let availableDiskCapacity: @Sendable (URL) throws -> Int64?

    /// A `nil` `cacheDirectory` downloads into ``ModelStorage/rootDirectory`` and falls back to Caches;
    /// otherwise only `fallbackDirectories` are read besides it.
    init(
        session: URLSession = Self.defaultSession,
        cacheDirectory: URL? = nil,
        fallbackDirectories: [URL] = [],
        waitBeforeRetry: @escaping @Sendable (Double) async throws -> Void = { seconds in
            try await Task.sleep(for: .seconds(seconds))
        },
        availableDiskCapacity: @escaping @Sendable (URL) throws -> Int64? = { url in
            let capacity = try url.resourceValues(
                forKeys: [.volumeAvailableCapacityForImportantUsageKey]
            ).volumeAvailableCapacityForImportantUsage
            return capacity
        }
    ) {
        self.session = session
        self.cacheDirectory = cacheDirectory ?? ModelStorage.rootDirectory
        self.fallbackDirectories = cacheDirectory == nil ? ModelStorage.fallbackRoots : fallbackDirectories
        self.waitBeforeRetry = waitBeforeRetry
        self.availableDiskCapacity = availableDiskCapacity
    }

    func download(
        id: String, revision: String?, matching patterns: [String],
        useLatest: Bool, progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        try Task.checkCancellation()
        let sanitized = id.replacingOccurrences(of: "/", with: "--")
        let snapshotPath = "huggingface/hub/models--\(sanitized)/snapshots/main"
        let dir = cacheDirectory.appending(path: snapshotPath)
        // Two loads of one repo (e.g. two sampling settings) would write the same .partial files and race.
        return try await SnapshotDownloadGate.shared.serialized(dir.path(percentEncoded: false)) {
            try await self.downloadSnapshot(id: id, revision: revision, patterns: patterns, useLatest: useLatest,
                                            snapshotPath: snapshotPath, dir: dir, progressHandler: progressHandler)
        }
    }

    private func downloadSnapshot(
        id: String, revision: String?, patterns: [String], useLatest: Bool,
        snapshotPath: String, dir: URL, progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        try Task.checkCancellation()

        let completionMarker = dir.appending(path: ".complete")

        if !useLatest, FileManager.default.fileExists(atPath: completionMarker.path(percentEncoded: false)) {
            Self.log.info("snapshot already complete — skip · \(dir.path(percentEncoded: false), privacy: .public)")
            return dir
        }
        let previous = fallbackDirectories.lazy.map { $0.appending(path: snapshotPath) }
            .first { FileManager.default.fileExists(atPath: $0.appending(path: ".complete").path(percentEncoded: false)) }
        if !useLatest, let previous {
            Self.log.info("snapshot complete in Caches — load in place · \(previous.path(percentEncoded: false), privacy: .public)")
            return previous
        }
        // Refreshing may replace previously complete files. Invalidate their old marker before
        // touching the snapshot, so a failed or cancelled refresh cannot take the offline fast path.
        if FileManager.default.fileExists(atPath: completionMarker.path(percentEncoded: false)) {
            try FileManager.default.removeItem(at: completionMarker)
        }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let rev = revision ?? "main"
        Self.log.info("download start: \(id, privacy: .public) rev=\(rev, privacy: .public)")

        // 1. Tree API — repo file list
        let (data, treeResp) = try await session.data(
            from: URL(string: "\(Self.base)/api/models/\(id)/tree/\(rev)")!)
        guard let treeHTTP = treeResp as? HTTPURLResponse,
              (200..<300).contains(treeHTTP.statusCode) else {
            throw AuraError.invalidResponse("Hugging Face tree request failed")
        }
        Self.log.info("tree API status=\(treeHTTP.statusCode)")

        struct LFSPointer: Codable { let oid: String? }
        struct F: Codable {
            let path: String?
            let type: String?
            let size: Int64?
            let largeFile: LFSPointer?
            enum CodingKeys: String, CodingKey {
                case path, type, size
                case largeFile = "lfs"
            }
        }
        let entries = try JSONDecoder().decode([F].self, from: data).filter { entry in
            guard entry.type == "file", let path = entry.path else { return false }
            return ModelDownloads.isSnapshotFile(path, patterns: patterns)
        }
        let files = entries.compactMap(\.path)
        var checksums: [String: String] = [:]
        for entry in entries {
            if let path = entry.path, let sha256 = FileIntegrity.normalizedSHA256(entry.largeFile?.oid) {
                checksums[path] = sha256
            }
        }
        try ModelDownloads.requireLoadable(files, patterns: patterns, repoID: id)
        Self.log.info("fetching \(files.count) files: \(files.joined(separator: ", "), privacy: .public)")

        try Task.checkCancellation()

        // 2. Sizes come from the tree (an LFS entry's size is the real file's); HEAD only for an unsized entry.
        var sizes: [String: Int64] = [:]
        for entry in entries {
            if let path = entry.path, let size = entry.size, size >= 0 { sizes[path] = size }
        }
        var totalBytes: Int64 = 0
        for name in files {
            try Task.checkCancellation()
            let size: Int64
            if let listed = sizes[name] {
                size = listed
            } else {
                size = try await remoteSize(of: name, id: id, rev: rev)
                sizes[name] = size
            }
            totalBytes += size
            Self.log.info("\(name, privacy: .public): \(size) bytes")
        }
        Self.log.info("total to download: \(totalBytes / 1_000_000) MB")

        // 3. Disk space pre-check (20% margin)
        let needed = Int64(ceil(Double(totalBytes) * 1.2))
        if let free = try? availableDiskCapacity(dir) {
            Self.log.info("free space: \(free / 1_000_000) MB · required: \(needed / 1_000_000) MB")
            if Int64(free) < needed {
                Self.log.error("insufficient disk space")
                throw AuraError.invalidResponse(
                    "Espacio insuficiente: se necesitan \(needed / 1_000_000) MB, disponibles \(free / 1_000_000) MB"
                )
            }
        }

        let progress = Progress(totalUnitCount: totalBytes)
        var aggregated: Int64 = 0

        // 4. Download each file, chunked + resumable
        for name in files {
            try Task.checkCancellation()
            let dest = dir.appending(path: name)
            let expectedSize = sizes[name] ?? 0

            if FileManager.default.fileExists(atPath: dest.path(percentEncoded: false)) {
                let attributes = try FileManager.default.attributesOfItem(atPath: dest.path(percentEncoded: false))
                if attributes[.type] as? FileAttributeType == .typeRegular,
                   attributes[.size] as? Int64 == expectedSize,
                   try FileIntegrity.reusable(dest, expectedSHA256: checksums[name]) {
                    aggregated += expectedSize
                    progress.completedUnitCount = aggregated
                    progressHandler(progress)
                    Self.log.info("\(name, privacy: .public) already on disk — size verified")
                    continue
                }
                // `reusable` has already deleted a file that hashed wrong.
                if FileManager.default.fileExists(atPath: dest.path(percentEncoded: false)) {
                    try FileManager.default.removeItem(at: dest)
                }
            }
            try FileManager.default.createDirectory(
                at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)

            let url = URL(string: "\(Self.base)/\(id)/resolve/\(rev)/\(name)")!
            try await downloadFileChunked(
                from: url,
                to: dest,
                expectedSize: expectedSize,
                expectedSHA256: checksums[name],
                fileBaseBytes: aggregated,
                progress: progress,
                progressHandler: progressHandler
            )
            aggregated += expectedSize
            progress.completedUnitCount = aggregated
            progressHandler(progress)
        }

        try Task.checkCancellation()
        // A tokenizer-only call (mlx-swift-lm's `tokenizerDownloadPatterns`) fetches no weights,
        // so it must not mark the snapshot loadable.
        guard files.contains(where: { $0.lowercased().hasSuffix(".safetensors") }) else {
            Self.log.info("no weights requested — \(aggregated / 1_000_000) MB · no marker")
            return dir
        }
        try Data().write(to: completionMarker, options: .atomic)
        Self.log.info("snapshot complete — \(aggregated / 1_000_000) MB · marker written")
        return dir
    }

    /// HEAD size of one file. `identity` matters: Hugging Face compresses text files and then omits Content-Length.
    private func remoteSize(of name: String, id: String, rev: String) async throws -> Int64 {
        var head = URLRequest(url: URL(string: "\(Self.base)/\(id)/resolve/\(rev)/\(name)")!)
        head.httpMethod = "HEAD"
        head.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        let (_, resp) = try await session.data(for: head)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw AuraError.invalidResponse("Hugging Face size request failed for \(name)")
        }
        // LFS files redirect to a CDN; x-linked-size beats the pointer's ~134-byte Content-Length.
        let header = http.value(forHTTPHeaderField: "x-linked-size") ?? http.value(forHTTPHeaderField: "Content-Length")
        guard let header, let size = Int64(header), size >= 0 else {
            throw AuraError.invalidResponse("Missing download size for \(name)")
        }
        return size
    }

    /// Downloads one file in 32 MB `Range` chunks, appending to a `.partial`
    /// on disk. Resumes from the last completed chunk after any network cut.
    /// With `expectedSHA256` the finished partial must hash to it before it is promoted.
    private func downloadFileChunked(
        from url: URL,
        to destination: URL,
        expectedSize: Int64,
        expectedSHA256: String?,
        fileBaseBytes: Int64,
        progress: Progress,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws {
        guard expectedSize >= 0 else {
            throw AuraError.invalidResponse("Missing download size for \(destination.lastPathComponent)")
        }
        let partial = destination.appendingPathExtension("partial")

        // Bytes already in the .partial from a prior attempt → resume.
        var offset: Int64 = 0
        if let attrs = try? FileManager.default.attributesOfItem(atPath: partial.path(percentEncoded: false)),
           let existing = attrs[.size] as? Int64 {
            offset = existing
        }
        if offset > 0 {
            Self.log.info("resume \(destination.lastPathComponent, privacy: .public) from \(offset / 1_000_000) MB")
        }

        guard offset <= expectedSize else {
            throw AuraError.invalidResponse("Oversized partial download of \(destination.lastPathComponent)")
        }

        if expectedSize == 0 {
            try Task.checkCancellation()
            try Data().write(to: destination, options: .atomic)
            return
        }

        // Partial already complete (only the move failed last time) → promote.
        if offset == expectedSize {
            try Task.checkCancellation()
            if let expectedSHA256 {
                try FileIntegrity.verify(partial, expectedSHA256: expectedSHA256, path: destination.lastPathComponent)
            }
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: partial, to: destination)
            Self.log.info("\(destination.lastPathComponent, privacy: .public): partial already complete — promoted")
            return
        }

        if !FileManager.default.fileExists(atPath: partial.path(percentEncoded: false)) {
            FileManager.default.createFile(atPath: partial.path(percentEncoded: false), contents: nil)
        }
        let handle = try FileHandle(forWritingTo: partial)
        try handle.seekToEnd()
        defer { try? handle.close() }

        let startTime = Date()
        var lastLoggedPct = -1

        while offset < expectedSize {
            try Task.checkCancellation()

            let rangeEnd = min(offset + Self.chunkSize - 1, expectedSize - 1)

            var chunkOK = false
            var lastError: Error?
            for attempt in 1...Self.maxChunkAttempts {
                do {
                    try Task.checkCancellation()
                    if attempt > 1 {
                        let waitSec = min(30.0, pow(2.0, Double(attempt - 1)))
                        Self.log.warning("\(destination.lastPathComponent, privacy: .public) chunk @\(offset) retry \(attempt) after \(waitSec)s")
                        try await waitBeforeRetry(waitSec)
                    }
                    var request = URLRequest(url: url)
                    request.setValue("bytes=\(offset)-\(rangeEnd)", forHTTPHeaderField: "Range")
                    request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")   // byte ranges of the stored file, not of a compressed reply
                    let (chunk, resp) = try await session.data(for: request)
                    guard let http = resp as? HTTPURLResponse else {
                        throw AuraError.invalidResponse("No HTTP response")
                    }
                    if http.statusCode == 408 || http.statusCode == 429 || (500..<600).contains(http.statusCode) {
                        throw TransientHTTPError(statusCode: http.statusCode)
                    }
                    // A completed partial was handled before the loop. Here a 416 means the
                    // advertised size and the available bytes disagree; retrying cannot advance.
                    if http.statusCode == 416 {
                        throw AuraError.invalidResponse("Premature end of \(destination.lastPathComponent)")
                    }
                    // 206 = Range honored. 200 = server ignored Range; only OK at offset 0.
                    guard http.statusCode == 206 || (http.statusCode == 200 && offset == 0) else {
                        throw AuraError.invalidResponse("Unexpected status \(http.statusCode) for Range @\(offset)")
                    }
                    guard !chunk.isEmpty else {
                        throw AuraError.invalidResponse("Empty download chunk for \(destination.lastPathComponent)")
                    }
                    guard Int64(chunk.count) <= expectedSize - offset else {
                        throw AuraError.invalidResponse("Oversized download chunk for \(destination.lastPathComponent)")
                    }
                    try Task.checkCancellation()
                    try handle.write(contentsOf: chunk)
                    offset += Int64(chunk.count)
                    progress.completedUnitCount = fileBaseBytes + offset
                    progressHandler(progress)
                    chunkOK = true

                    // Progress + speed log every 5%
                    if expectedSize > 0 {
                        let pct = Int(Double(offset) / Double(expectedSize) * 100)
                        if pct / 5 > lastLoggedPct / 5 {
                            let elapsed = max(0.1, Date().timeIntervalSince(startTime))
                            let mbps = Double(offset) / elapsed / 1_000_000
                            Self.log.info("\(destination.lastPathComponent, privacy: .public): \(pct)% · \(offset / 1_000_000)/\(expectedSize / 1_000_000) MB · \(String(format: "%.1f", mbps)) MB/s")
                            lastLoggedPct = pct
                        }
                    }
                    break
                } catch is CancellationError {
                    throw CancellationError()
                } catch let urlError as URLError where urlError.code == .cancelled {
                    throw CancellationError()
                } catch let integrityError as AuraError {
                    // Invalid HTTP responses and size mismatches are not transient network cuts.
                    throw integrityError
                } catch {
                    lastError = error
                    let code = (error as? URLError)?.code.rawValue ?? -1
                    Self.log.error("\(destination.lastPathComponent, privacy: .public) chunk @\(offset) attempt \(attempt) failed: \(error.localizedDescription, privacy: .public) (URLError=\(code)) — partial keeps \(offset / 1_000_000) MB")
                }
            }

            guard chunkOK else {
                // Chunk retries exhausted. The .partial keeps what was downloaded;
                // a re-launch continues from here.
                try? handle.close()
                Self.log.fault("\(destination.lastPathComponent, privacy: .public): chunk @\(offset) exhausted \(Self.maxChunkAttempts) retries")
                throw lastError ?? URLError(.cannotConnectToHost)
            }

            if offset >= expectedSize { break }
        }

        try Task.checkCancellation()
        try handle.close()

        // Size integrity check before promoting.
        if expectedSize > 0 {
            let attrs = try? FileManager.default.attributesOfItem(atPath: partial.path(percentEncoded: false))
            let finalSize = (attrs?[.size] as? Int64) ?? 0
            guard finalSize == expectedSize else {
                Self.log.error("\(destination.lastPathComponent, privacy: .public): final size \(finalSize) != expected \(expectedSize)")
                throw AuraError.invalidResponse("Incomplete download of \(destination.lastPathComponent): \(finalSize)/\(expectedSize) bytes")
            }
        }
        if let expectedSHA256 {
            try FileIntegrity.verify(partial, expectedSHA256: expectedSHA256, path: destination.lastPathComponent)
        }

        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: partial, to: destination)
        Self.log.info("\(destination.lastPathComponent, privacy: .public) complete")
    }
}

/// Runs downloads into the same snapshot folder one at a time; a later call then finds the finished files.
private actor SnapshotDownloadGate {
    static let shared = SnapshotDownloadGate()
    private var tails: [String: (id: UUID, done: Task<Void, Never>)] = [:]

    func serialized<T: Sendable>(_ key: String, _ body: @escaping @Sendable () async throws -> T) async throws -> T {
        let previous = tails[key]?.done
        let work = Task<T, Error> {
            await previous?.value
            try Task.checkCancellation()
            return try await body()
        }
        let ticket = UUID()
        tails[key] = (ticket, Task { _ = await work.result })
        defer { if tails[key]?.id == ticket { tails[key] = nil } }
        return try await withTaskCancellationHandler { try await work.value } onCancel: { work.cancel() }
    }
}

// MARK: - Tokenizer Loader (MLXLMCommon.TokenizerLoader)

/// Loads tokenizer via swift-transformers and bridges to MLXLMCommon.Tokenizer.
struct AuraTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let t = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return Bridge(t)
    }

    private struct Bridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
        let t: any Tokenizers.Tokenizer
        init(_ t: any Tokenizers.Tokenizer) { self.t = t }
        var bosToken: String? { t.bosToken }
        var eosToken: String? { t.eosToken }
        var unknownToken: String? { t.unknownToken }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            t.encode(text: text, addSpecialTokens: addSpecialTokens)
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            t.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }
        func convertTokenToId(_ token: String) -> Int? { t.convertTokenToId(token) }
        func convertIdToToken(_ id: Int) -> String? { t.convertIdToToken(id) }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try t.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
        }
    }
}

#endif
