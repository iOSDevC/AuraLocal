import Foundation

// MARK: - Plan

/// One file a download will fetch.
public struct PlannedFile: Sendable, Equatable {
    /// Repo-relative path.
    public let path: String
    public let sizeBytes: Int64
    /// The SHA-256 Hugging Face publishes for an LFS file; `nil` for a file kept in git.
    let sha256: String?

    public init(path: String, sizeBytes: Int64) {
        self.init(path: path, sizeBytes: sizeBytes, sha256: nil)
    }

    init(path: String, sizeBytes: Int64, sha256: String?) {
        self.path = path
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }

    /// Path and size, as before the checksum existed: it is download bookkeeping, not part of what a plan shows.
    public static func == (lhs: PlannedFile, rhs: PlannedFile) -> Bool {
        lhs.path == rhs.path && lhs.sizeBytes == rhs.sizeBytes
    }
}

/// What downloading a model involves, computed without downloading it.
public struct DownloadPlan: Sendable, Equatable {
    public let repoID: String
    public let files: [PlannedFile]
    public let totalBytes: Int64
    /// Finished files plus `.partial` bytes already on disk.
    public let bytesOnDisk: Int64

    public init(repoID: String, files: [PlannedFile], bytesOnDisk: Int64) {
        self.repoID = repoID
        self.files = files
        self.totalBytes = files.map(\.sizeBytes).saturatingSum()
        self.bytesOnDisk = bytesOnDisk
    }

    public var remainingBytes: Int64 { max(0, totalBytes - bytesOnDisk) }
}

// MARK: - ModelDownloads

/// Download-only API: fetches a model to the exact path its backend loads from, without loading it. Paths are under
/// ``ModelStorage/rootDirectory`` (`<root>` below):
///
/// - GGUF: `<root>/models/<org>/<repo>/<ggufFilename>` (``Model/resolvedFileURL``).
/// - MLX: `<root>/huggingface/hub/models--<org>--<repo>/snapshots/main/` plus a `.complete` marker
///   (``Model/isDownloaded``), with only the files mlx-swift-lm reads.
///
/// A model an earlier version downloaded into Caches is used from there until it is migrated, and is not fetched
/// again. Transfers resume from `<file>.partial` after a cancel, a network cut or a relaunch. Each finished LFS file
/// is hashed and compared with the SHA-256 Hugging Face publishes for it before it is moved into place; a mismatch
/// deletes it and throws ``ModelIntegrityError/checksumMismatch(path:expected:actual:)``. Only single-file GGUF
/// quants at the top level of a repo are supported. The Hugging Face token comes from ``KeychainDownloadAuth``.
/// ``downloadFolder(repoID:revision:folder:include:auth:progress:)`` fetches one folder of a repo instead, such as
/// a Core ML model, and sends no token by default. Two downloads that would write the same `.partial` files never run
/// at once; see ``download(_:auth:progress:)``.
public enum ModelDownloads {

    /// Mirrors mlx-swift-lm's `modelDownloadPatterns` (`ModelFactory.swift`, package-internal there).
    static let mlxFilePatterns = ["*.safetensors", "*.json", "*.jinja"]
    static let completeMarker = ".complete"
    /// The MLX snapshot folder is always `main`, whatever revision was fetched; ``Model/isDownloaded`` reads it.
    static let snapshotFolder = "main"

    static var storageRoot: URL { ModelStorage.rootDirectory }

    // MARK: Public

    /// Files and sizes for `model` without downloading. GGUF: one file (HEAD). MLX: the filtered top-level tree.
    public static func plan(for model: Model,
                            auth: any DownloadAuthorizing = KeychainDownloadAuth()) async throws -> DownloadPlan {
        try await plan(for: model, transfer: HubFileTransfer(auth: auth), root: storageRoot)
    }

    /// Downloads to the exact path the backends load from, resuming any `.partial` (a GGUF one left in Caches is
    /// moved into the storage root first), and verifies each LFS file's SHA-256. Returns the GGUF file or the MLX
    /// snapshot folder; one still in Caches from an earlier version is returned as is. `progress` receives (bytes
    /// on disk, total bytes). Cancel the calling task to stop; the partial file is kept for the next call.
    ///
    /// Downloads of one file or snapshot run one at a time, from any API or process sharing the storage root: a later
    /// call waits, then returns the finished model without a transfer. Cancelling a waiting call stops only that call.
    public static func download(_ model: Model,
                                auth: any DownloadAuthorizing = KeychainDownloadAuth(),
                                progress: @escaping @Sendable (Int64, Int64) -> Void) async throws -> URL {
        try await download(model, transfer: HubFileTransfer(auth: auth), root: storageRoot,
                           fallbackRoots: ModelStorage.fallbackRoots, progress: progress)
    }

    /// Removes the model's files, finished and partial, from the storage root and from Caches. GGUF: the file, then
    /// its empty `<org>/<repo>` folders; MLX: the whole `models--<org>--<repo>` folder (in Caches only its
    /// `snapshots/main`, since a Hugging Face cache there may hold the same repo). Unload the model first
    /// (``ModelManager/evict(_:)``).
    public static func delete(_ model: Model) throws {
        try delete(model, root: storageRoot)
        for legacyRoot in ModelStorage.fallbackRoots {
            try deleteLegacy(model, root: legacyRoot)
        }
    }

    /// Bytes on disk for `model`, finished plus partial, in the storage root and in Caches; 0 when nothing is there.
    public static func bytesOnDisk(for model: Model) -> Int64 {
        ModelStorage.readRoots.map { bytesOnDisk(for: model, root: $0) }.saturatingSum()
    }

    /// Free space for important data on the volume of ``ModelStorage/rootDirectory``, where every download lands;
    /// `nil` when unknown. ``download(_:auth:progress:)`` requires the remaining bytes plus a 20 % margin.
    public static func availableDiskBytes() -> Int64? {
        HubFileTransfer.volumeFreeSpace(storageRoot)
    }

    /// The SHA-256 a download verified `file` against, or `nil` when it was not verified: downloaded before
    /// verification existed, from a source that publishes no SHA-256, or changed in size since.
    public static func verifiedSHA256(of file: URL) -> String? {
        FileIntegrity.verifiedSHA256(of: file)
    }

    // MARK: Paths

    static func modelsRoot(_ root: URL) -> URL {
        root.appendingPathComponent("models")
    }

    static func repoFolder(repoID: String, root: URL) -> URL {
        root.appending(path: "huggingface/hub/models--\(repoID.replacingOccurrences(of: "/", with: "--"))")
    }

    static func snapshotDirectory(repoID: String, root: URL) -> URL {
        repoFolder(repoID: repoID, root: root).appending(path: "snapshots/\(snapshotFolder)")
    }

    /// The GGUF destination, or why this model cannot be downloaded here.
    static func ggufTarget(for model: Model, root: URL) throws -> (file: URL, source: URL) {
        guard model.localFileURL == nil else {
            throw ModelDownloadError.unsupported("\(model.displayName) is a local file; there is nothing to download.")
        }
        guard let filename = model.ggufFilename, !filename.isEmpty else {
            throw ModelDownloadError.unsupported("\(model.displayName) has no GGUF file name.")
        }
        guard !filename.contains("/") else {
            throw ModelDownloadError.unsupported("\(filename) sits in a subfolder of the repo; not supported yet.")
        }
        guard !HuggingFaceRepo.isShard(filename) else {
            throw ModelDownloadError.unsupported("\(filename) is one part of a split quant; Tonamox loads single files.")
        }
        let parts = model.repoID.split(separator: "/").map(String.init)
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0 != ".." }),
              let source = model.downloadURL
                ?? HuggingFaceRepo.resolveURL(owner: parts[0], repo: parts[1], revision: "main", path: filename)
        else {
            throw ModelDownloadError.unsupported("Not a Hugging Face repository id: \(model.repoID)")
        }
        let file = modelsRoot(root).appendingPathComponent(model.repoID).appendingPathComponent(filename)
        return (file, source)
    }

    // MARK: Plan

    static func plan(for model: Model, transfer: HubFileTransfer, root: URL) async throws -> DownloadPlan {
        switch model.format {
        case .gguf:
            let target = try ggufTarget(for: model, root: root)
            let path = target.file.lastPathComponent
            let size = try await transfer.size(of: target.source, repoID: model.repoID, path: path)
            return DownloadPlan(repoID: model.repoID, files: [PlannedFile(path: path, sizeBytes: size)],
                                bytesOnDisk: bytesOnDisk(for: model, root: root))
        case .mlx:
            let files = try await snapshotFiles(repoID: model.repoID, patterns: mlxFilePatterns, transfer: transfer)
            let directory = snapshotDirectory(repoID: model.repoID, root: root)
            let present = files.map { onDiskBytes(directory.appendingPathComponent($0.path)) }.saturatingSum()
            return DownloadPlan(repoID: model.repoID, files: files, bytesOnDisk: present)
        }
    }

    /// Top-level files matching `patterns`, sized from the tree (HEAD when the tree has no size). Throws
    /// ``ModelDownloadError/incompleteRepo(repoID:)`` when a model download would miss weights or config.
    static func snapshotFiles(repoID: String, patterns: [String], revision: String = snapshotFolder,
                              transfer: HubFileTransfer) async throws -> [PlannedFile] {
        let listed = try await transfer.listTopLevelFiles(repoID: repoID, revision: revision)
        let matching = listed.filter { isSnapshotFile($0.path, patterns: patterns) }
        try requireLoadable(matching.map(\.path), patterns: patterns, repoID: repoID)
        var planned: [PlannedFile] = []
        for file in matching {
            planned.append(try await sized(file, repoID: repoID, revision: revision, transfer: transfer))
        }
        return planned
    }

    static func sized(_ file: RemoteFile, repoID: String, revision: String,
                      transfer: HubFileTransfer) async throws -> PlannedFile {
        if let size = file.size, size > 0 {
            return PlannedFile(path: file.path, sizeBytes: size, sha256: file.sha256)
        }
        guard let url = fileURL(repoID: repoID, revision: revision, path: file.path) else {
            throw ModelDownloadError.unsupported("Not a Hugging Face repository id: \(repoID)")
        }
        let size = try await transfer.size(of: url, repoID: repoID, path: file.path)
        return PlannedFile(path: file.path, sizeBytes: size, sha256: file.sha256)
    }

    static func hasWeightsAndConfig(_ paths: [String]) -> Bool {
        paths.contains { $0.lowercased().hasSuffix(".safetensors") } && paths.contains("config.json")
    }

    /// A top-level entry a snapshot fetches: not hidden (`.gitattributes`) and matching one of `patterns`.
    static func isSnapshotFile(_ path: String, patterns: [String]) -> Bool {
        !path.hasPrefix(".") && patterns.contains { matches(path, pattern: $0) }
    }

    /// Throws ``ModelDownloadError/incompleteRepo(repoID:)`` when nothing matched, or when a weights download
    /// (`patterns` include `*.safetensors`) would miss the weights or `config.json`.
    static func requireLoadable(_ paths: [String], patterns: [String], repoID: String) throws {
        if paths.isEmpty || (patterns.contains("*.safetensors") && !hasWeightsAndConfig(paths)) {
            throw ModelDownloadError.incompleteRepo(repoID: repoID)
        }
    }

    /// `*.ext` globs as mlx-swift-lm passes them; anything else must match the name exactly.
    static func matches(_ path: String, pattern: String) -> Bool {
        if pattern.hasPrefix("*") { return path.hasSuffix(pattern.dropFirst()) }
        return path == pattern
    }

    static func fileURL(repoID: String, revision: String, path: String) -> URL? {
        let parts = repoID.split(separator: "/").map(String.init)
        guard parts.count == 2 else { return nil }
        return HuggingFaceRepo.resolveURL(owner: parts[0], repo: parts[1], revision: revision, path: path)
    }

    // MARK: Download

    /// `fallbackRoots` are only read: a finished model found there is returned instead of downloaded again.
    static func download(_ model: Model, transfer: HubFileTransfer, root: URL, fallbackRoots: [URL] = [],
                         progress: @escaping @Sendable (Int64, Int64) -> Void) async throws -> URL {
        switch model.format {
        case .gguf:
            return try await downloadGGUF(model, transfer: transfer, root: root, fallbackRoots: fallbackRoots,
                                          progress: progress)
        case .mlx:
            let finished = ([root] + fallbackRoots).lazy
                .map { snapshotDirectory(repoID: model.repoID, root: $0) }
                .first { FileManager.default.fileExists(atPath: $0.appendingPathComponent(completeMarker).path) }
            if let finished { return finished }
            return try await downloadSnapshot(repoID: model.repoID, patterns: mlxFilePatterns,
                                              revision: snapshotFolder, transfer: transfer, root: root,
                                              progress: progress)
        }
    }

    private static func downloadGGUF(_ model: Model, transfer: HubFileTransfer, root: URL, fallbackRoots: [URL],
                                     progress: @escaping @Sendable (Int64, Int64) -> Void) async throws -> URL {
        let target = try ggufTarget(for: model, root: root)
        if let finished = try finishedGGUF(model, roots: [root] + fallbackRoots, progress: progress) {
            return finished
        }
        return try await DownloadGate.shared.exclusive(target.file) { _ in
            // The call this one waited for may have finished the file.
            if let finished = try finishedGGUF(model, roots: [root], progress: progress) { return finished }
            adoptFallbackPartial(of: model, to: target.file, from: fallbackRoots)
            let path = target.file.lastPathComponent
            let total = try await transfer.size(of: target.source, repoID: model.repoID, path: path)
            let directory = target.file.deletingLastPathComponent()
            let present = onDiskBytes(target.file)
            try transfer.ensureSpace(remaining: max(0, total - present), at: directory)
            let sha256 = try await transfer.publishedSHA256(of: target.source)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try await transfer.fetch(target.source, to: target.file, expectedSize: total, expectedSHA256: sha256,
                                     repoID: model.repoID, path: path) { bytes in
                progress(bytes, total)
            }
            return target.file
        }
    }

    /// The finished GGUF file in the first of `roots` that has one.
    private static func finishedGGUF(_ model: Model, roots: [URL],
                                     progress: @Sendable (Int64, Int64) -> Void) throws -> URL? {
        for candidate in roots {
            let file = try ggufTarget(for: model, root: candidate).file
            if let size = HubFileTransfer.fileSize(at: file) {
                progress(size, size)
                return file
            }
        }
        return nil
    }

    /// Moves a `.partial` an earlier version left in a fallback root next to `file`, so the transfer resumes from
    /// it instead of starting over and leaving it behind. Nothing moves when `file` already has a partial.
    static func adoptFallbackPartial(of model: Model, to file: URL, from fallbackRoots: [URL]) {
        let partial = HubFileTransfer.partialURL(for: file)
        guard HubFileTransfer.fileSize(at: partial) == nil else { return }
        for fallback in fallbackRoots {
            guard let old = try? HubFileTransfer.partialURL(for: ggufTarget(for: model, root: fallback).file) else {
                continue
            }
            if ModelStorage.moveExclusively(old, to: partial, fileManager: .default) == .moved {
                ModelStorage.removeEmptyFolders(from: old.deletingLastPathComponent(), upTo: fallback)
                return
            }
        }
    }

    /// Fetches the matching top-level files into the `snapshots/main` folder and writes `.complete` only after
    /// every file is in place, and only for a weights download (a tokenizer-only call must not mark it loadable).
    static func downloadSnapshot(repoID: String, patterns: [String], revision: String, transfer: HubFileTransfer,
                                 root: URL,
                                 progress: @escaping @Sendable (Int64, Int64) -> Void) async throws -> URL {
        let directory = snapshotDirectory(repoID: repoID, root: root)
        let marker = directory.appendingPathComponent(completeMarker)
        if FileManager.default.fileExists(atPath: marker.path) { return directory }

        return try await DownloadGate.shared.exclusive(directory) { _ in
            // The call this one waited for may have finished the snapshot.
            if FileManager.default.fileExists(atPath: marker.path) { return directory }
            let files = try await snapshotFiles(repoID: repoID, patterns: patterns, revision: revision,
                                                transfer: transfer)
            let total = files.map(\.sizeBytes).saturatingSum()
            let present = files.map { onDiskBytes(directory.appendingPathComponent($0.path)) }.saturatingSum()
            try transfer.ensureSpace(remaining: max(0, total - present), at: directory)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

            var finished: Int64 = 0
            for file in files {
                try Task.checkCancellation()
                try await fetchSnapshotFile(file, repoID: repoID, revision: revision, into: directory,
                                            transfer: transfer, finished: finished, total: total, progress: progress)
                finished += file.sizeBytes
            }
            progress(total, total)
            if patterns.contains("*.safetensors") {
                FileManager.default.createFile(atPath: marker.path, contents: Data())
            }
            return directory
        }
    }

    private static func fetchSnapshotFile(_ file: PlannedFile, repoID: String, revision: String, into directory: URL,
                                          transfer: HubFileTransfer, finished: Int64, total: Int64,
                                          progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        let destination = directory.appendingPathComponent(file.path)
        if HubFileTransfer.fileSize(at: destination) == file.sizeBytes,
           try FileIntegrity.reusable(destination, expectedSHA256: file.sha256) { return }
        guard let url = fileURL(repoID: repoID, revision: revision, path: file.path) else {
            throw ModelDownloadError.unsupported("Not a Hugging Face repository id: \(repoID)")
        }
        try await transfer.fetch(url, to: destination, expectedSize: file.sizeBytes, expectedSHA256: file.sha256,
                                 repoID: repoID, path: file.path) { bytes in
            progress(finished + bytes, total)
        }
    }

    // MARK: Disk

    /// Size of a finished file, else of its `.partial`, else 0.
    static func onDiskBytes(_ file: URL) -> Int64 {
        HubFileTransfer.fileSize(at: file) ?? HubFileTransfer.fileSize(at: HubFileTransfer.partialURL(for: file)) ?? 0
    }

    static func bytesOnDisk(for model: Model, root: URL) -> Int64 {
        switch model.format {
        case .gguf:
            guard let target = try? ggufTarget(for: model, root: root) else { return 0 }
            return onDiskBytes(target.file)
        case .mlx:
            return DownloadedModels.directorySize(snapshotDirectory(repoID: model.repoID, root: root),
                                                  fileManager: .default)
        }
    }

    static func delete(_ model: Model, root: URL, fileManager: FileManager = .default) throws {
        switch model.format {
        case .gguf:
            let target = try ggufTarget(for: model, root: root)
            try removeIfPresent(target.file, fileManager: fileManager)
            try removeIfPresent(HubFileTransfer.partialURL(for: target.file), fileManager: fileManager)
            removeEmptyFolders(from: target.file.deletingLastPathComponent(), upTo: modelsRoot(root),
                               fileManager: fileManager)
        case .mlx:
            try removeIfPresent(repoFolder(repoID: model.repoID, root: root), fileManager: fileManager)
        }
    }

    /// ``delete(_:root:fileManager:)`` for a root Tonamox shares with other tools: an MLX model loses only its
    /// `snapshots/main` and the folders that leaves empty.
    static func deleteLegacy(_ model: Model, root: URL, fileManager: FileManager = .default) throws {
        switch model.format {
        case .gguf:
            try delete(model, root: root, fileManager: fileManager)
        case .mlx:
            try deleteLegacySnapshot(repoID: model.repoID, root: root, fileManager: fileManager)
        }
    }

    static func deleteLegacySnapshot(repoID: String, root: URL, fileManager: FileManager = .default) throws {
        let snapshot = snapshotDirectory(repoID: repoID, root: root)
        try removeIfPresent(snapshot, fileManager: fileManager)
        removeEmptyFolders(from: snapshot.deletingLastPathComponent(), upTo: DownloadedModels.hubDirectory(root: root),
                           fileManager: fileManager)
    }

    static func removeIfPresent(_ url: URL, fileManager: FileManager) throws {
        guard fileManager.fileExists(atPath: url.path) else { return }
        try fileManager.removeItem(at: url)
    }

    /// Walks up from `folder`, removing each empty folder, and stops at `root` or the first non-empty one.
    static func removeEmptyFolders(from folder: URL, upTo root: URL, fileManager: FileManager) {
        let rootPath = root.standardizedFileURL.path
        var current = folder.standardizedFileURL
        while current.path.hasPrefix(rootPath + "/"),
              let contents = try? fileManager.contentsOfDirectory(atPath: current.path), contents.isEmpty {
            try? fileManager.removeItem(at: current)
            current.deleteLastPathComponent()
        }
    }
}
