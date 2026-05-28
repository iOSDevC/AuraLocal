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
    private let generateParameters: GenerateParameters

    var isLoaded: Bool { modelContainer != nil }

    // MARK: - Init

    init(model: Model, temperature: Float? = nil) {
        self.model = model
        let defaultTemp: Float
        switch model.purpose {
        case .text:              defaultTemp = 0.7
        case .vision:            defaultTemp = 0.1
        case .visionSpecialized: defaultTemp = 0.0
        }
        self.generateParameters = GenerateParameters(temperature: temperature ?? defaultTemp)
    }

    // MARK: - InferenceBackend

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        guard modelContainer == nil else { return }

        // Scale GPU cache limit proportionally to model size.
        let modelWeightsBytes = model.approximateSizeMB * 1024 * 1024
        let proportionalCache = modelWeightsBytes / 12
        let cacheLimitBytes   = max(128 * 1024 * 1024, min(proportionalCache, 512 * 1024 * 1024))
        MLX.GPU.set(cacheLimit: cacheLimitBytes)

        let config = ModelConfiguration(id: model.rawValue)
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

    private func performGeneration(
        container: ModelContainer,
        prepareInput: @escaping @Sendable (ModelContext) async throws -> LMInput,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        let params = self.generateParameters
        return try await container.perform { (context: ModelContext) async throws -> sending String in
            let input = try await prepareInput(context)
            let result = try MLXLMCommon.generate(
                input: input,
                parameters: params,
                context: context
            ) { tokens in
                guard !Task.isCancelled else { return .stop }

                let partial = context.tokenizer.decode(tokenIds: tokens)
                Task { @MainActor in
                    guard !Task.isCancelled else { return }
                    onToken(partial)
                }
                return tokens.count >= maxTokens ? .stop : .more
            }
            return context.tokenizer.decode(tokenIds: result.tokens)
        }
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
struct AuraHFDownloader: Downloader {
    private static let base = "https://huggingface.co"
    private static let chunkSize: Int64 = 32 * 1024 * 1024
    private static let maxChunkAttempts = 6

    private static let log = Logger(subsystem: "dev.auralocal", category: "HFDownloader")

    /// Dedicated session with generous timeouts for large downloads.
    private static let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 7200
        config.waitsForConnectivity = true
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.httpMaximumConnectionsPerHost = 2
        return URLSession(configuration: config)
    }()

    func download(
        id: String, revision: String?, matching patterns: [String],
        useLatest: Bool, progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let sanitized = id.replacingOccurrences(of: "/", with: "--")
        let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appending(path: "huggingface/hub/models--\(sanitized)/snapshots/main")

        let completionMarker = dir.appending(path: ".complete")

        if !useLatest, FileManager.default.fileExists(atPath: completionMarker.path()) {
            Self.log.info("snapshot already complete — skip · \(dir.path(), privacy: .public)")
            return dir
        }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let rev = revision ?? "main"
        Self.log.info("download start: \(id, privacy: .public) rev=\(rev, privacy: .public)")

        // 1. Tree API — repo file list
        let (data, treeResp) = try await Self.session.data(
            from: URL(string: "\(Self.base)/api/models/\(id)/tree/\(rev)")!)
        if let http = treeResp as? HTTPURLResponse {
            Self.log.info("tree API status=\(http.statusCode)")
        }

        struct F: Codable { let path: String?; let type: String? }
        let files = ((try? JSONDecoder().decode([F].self, from: data)) ?? [])
            .filter { $0.type == "file" }
            .compactMap(\.path)
        Self.log.info("repo has \(files.count) files: \(files.joined(separator: ", "), privacy: .public)")

        try Task.checkCancellation()

        // 2. HEAD per file for the real size. LFS files redirect to a CDN;
        //    prefer the `x-linked-size` header over Content-Length (which for an
        //    LFS pointer would be ~134 bytes and break the total).
        var sizes: [String: Int64] = [:]
        var totalBytes: Int64 = 0
        for name in files {
            try Task.checkCancellation()
            var head = URLRequest(url: URL(string: "\(Self.base)/\(id)/resolve/\(rev)/\(name)")!)
            head.httpMethod = "HEAD"
            let (_, resp) = try await Self.session.data(for: head)
            let http = resp as? HTTPURLResponse
            let linkedSize = (http?.value(forHTTPHeaderField: "x-linked-size")).flatMap { Int64($0) }
            let contentLength = http?.expectedContentLength ?? 0
            let sz = linkedSize ?? (contentLength > 0 ? contentLength : 0)
            sizes[name] = sz
            totalBytes += sz
            Self.log.info("HEAD \(name, privacy: .public): \(sz) bytes (x-linked-size=\(linkedSize.map(String.init) ?? "nil", privacy: .public), content-length=\(contentLength))")
        }
        Self.log.info("total to download: \(totalBytes / 1_000_000) MB")

        // 3. Disk space pre-check (20% margin)
        let needed = Int64(Double(totalBytes) * 1.2)
        if let free = try? dir.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            .volumeAvailableCapacityForImportantUsage {
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

            if FileManager.default.fileExists(atPath: dest.path()) {
                aggregated += expectedSize
                progress.completedUnitCount = aggregated
                progressHandler(progress)
                Self.log.info("\(name, privacy: .public) already on disk — skip")
                continue
            }
            try FileManager.default.createDirectory(
                at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)

            let url = URL(string: "\(Self.base)/\(id)/resolve/\(rev)/\(name)")!
            try await Self.downloadFileChunked(
                from: url,
                to: dest,
                expectedSize: expectedSize,
                fileBaseBytes: aggregated,
                progress: progress,
                progressHandler: progressHandler
            )
            aggregated += expectedSize
            progress.completedUnitCount = aggregated
            progressHandler(progress)
        }

        try? Data().write(to: completionMarker)
        Self.log.info("snapshot complete — \(aggregated / 1_000_000) MB · marker written")
        return dir
    }

    /// Downloads one file in 32 MB `Range` chunks, appending to a `.partial`
    /// on disk. Resumes from the last completed chunk after any network cut.
    private static func downloadFileChunked(
        from url: URL,
        to destination: URL,
        expectedSize: Int64,
        fileBaseBytes: Int64,
        progress: Progress,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws {
        let partial = destination.appendingPathExtension("partial")

        // Bytes already in the .partial from a prior attempt → resume.
        var offset: Int64 = 0
        if let attrs = try? FileManager.default.attributesOfItem(atPath: partial.path()),
           let existing = attrs[.size] as? Int64 {
            offset = existing
        }
        if offset > 0 {
            log.info("resume \(destination.lastPathComponent, privacy: .public) from \(offset / 1_000_000) MB")
        }

        // Partial already complete (only the move failed last time) → promote.
        if expectedSize > 0, offset >= expectedSize {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: partial, to: destination)
            log.info("\(destination.lastPathComponent, privacy: .public): partial already complete — promoted")
            return
        }

        if !FileManager.default.fileExists(atPath: partial.path()) {
            FileManager.default.createFile(atPath: partial.path(), contents: nil)
        }
        let handle = try FileHandle(forWritingTo: partial)
        try handle.seekToEnd()
        defer { try? handle.close() }

        let startTime = Date()
        var lastLoggedPct = -1

        // Chunk loop. With expectedSize == 0 (HEAD gave no size) we read until
        // the server returns an empty chunk / 416.
        while expectedSize == 0 || offset < expectedSize {
            try Task.checkCancellation()

            let rangeEnd: Int64 = expectedSize > 0
                ? min(offset + chunkSize - 1, expectedSize - 1)
                : offset + chunkSize - 1

            var chunkOK = false
            var lastError: Error?
            for attempt in 1...maxChunkAttempts {
                do {
                    try Task.checkCancellation()
                    if attempt > 1 {
                        let waitSec = min(30.0, pow(2.0, Double(attempt - 1)))
                        log.warning("\(destination.lastPathComponent, privacy: .public) chunk @\(offset) retry \(attempt) after \(waitSec)s")
                        try await Task.sleep(for: .seconds(waitSec))
                    }
                    var request = URLRequest(url: url)
                    request.setValue("bytes=\(offset)-\(rangeEnd)", forHTTPHeaderField: "Range")
                    let (chunk, resp) = try await session.data(for: request)
                    guard let http = resp as? HTTPURLResponse else {
                        throw AuraError.invalidResponse("No HTTP response")
                    }
                    // 416 Range Not Satisfiable → already have everything.
                    if http.statusCode == 416 {
                        log.info("\(destination.lastPathComponent, privacy: .public): 416 — file already complete")
                        chunkOK = true
                        break
                    }
                    // 206 = Range honored. 200 = server ignored Range; only OK at offset 0.
                    guard http.statusCode == 206 || (http.statusCode == 200 && offset == 0) else {
                        throw AuraError.invalidResponse("Unexpected status \(http.statusCode) for Range @\(offset)")
                    }
                    if chunk.isEmpty {
                        chunkOK = true
                        break
                    }
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
                            log.info("\(destination.lastPathComponent, privacy: .public): \(pct)% · \(offset / 1_000_000)/\(expectedSize / 1_000_000) MB · \(String(format: "%.1f", mbps)) MB/s")
                            lastLoggedPct = pct
                        }
                    }
                    break
                } catch is CancellationError {
                    throw CancellationError()
                } catch let urlError as URLError where urlError.code == .cancelled {
                    throw CancellationError()
                } catch {
                    lastError = error
                    let code = (error as? URLError)?.code.rawValue ?? -1
                    log.error("\(destination.lastPathComponent, privacy: .public) chunk @\(offset) attempt \(attempt) failed: \(error.localizedDescription, privacy: .public) (URLError=\(code)) — partial keeps \(offset / 1_000_000) MB")
                }
            }

            guard chunkOK else {
                // Chunk retries exhausted. The .partial keeps what was downloaded;
                // a re-launch continues from here.
                try? handle.close()
                log.fault("\(destination.lastPathComponent, privacy: .public): chunk @\(offset) exhausted \(maxChunkAttempts) retries")
                throw lastError ?? URLError(.cannotConnectToHost)
            }

            if expectedSize == 0 { break }
            if offset >= expectedSize { break }
        }

        try handle.close()

        // Size integrity check before promoting.
        if expectedSize > 0 {
            let attrs = try? FileManager.default.attributesOfItem(atPath: partial.path())
            let finalSize = (attrs?[.size] as? Int64) ?? 0
            guard finalSize == expectedSize else {
                log.error("\(destination.lastPathComponent, privacy: .public): final size \(finalSize) != expected \(expectedSize)")
                throw AuraError.invalidResponse("Incomplete download of \(destination.lastPathComponent): \(finalSize)/\(expectedSize) bytes")
            }
        }

        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: partial, to: destination)
        log.info("\(destination.lastPathComponent, privacy: .public) complete")
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
