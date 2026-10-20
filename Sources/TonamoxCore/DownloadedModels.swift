import Foundation

// MARK: - DownloadedGGUF

/// A `.gguf` model already present on disk under a models root (`<root>/models/<org>/<repo>/<file>.gguf`, see
/// ``DownloadedModels/modelsDirectories``).
///
/// Produced by ``DownloadedModels/scan(fileManager:)`` so a host app can offer an "already-downloaded" picker
/// instead of forcing the user to re-paste a Hugging Face URL. A downloaded file loads **in place** via
/// ``Model/fromURL(_:displayName:)`` with its ``fileURL`` — never re-downloaded.
public struct DownloadedGGUF: Sendable, Identifiable, Equatable {

    /// `<org>/<repo>` derived from the file's path relative to the models root, or `""` for a flat file that
    /// sits directly under the models root (no repo folder).
    public let repoID: String

    /// The bare GGUF file name, e.g. `gemma-4-E2B-it-Q4_K_M.gguf`.
    public let filename: String

    /// Absolute on-disk location of the file — feed straight into ``Model/fromURL(_:displayName:)``.
    public let fileURL: URL

    /// File size in bytes.
    public let sizeBytes: Int64

    public init(repoID: String, filename: String, fileURL: URL, sizeBytes: Int64) {
        self.repoID = repoID
        self.filename = filename
        self.fileURL = fileURL
        self.sizeBytes = sizeBytes
    }

    /// Stable identity — the file path is unique on disk.
    public var id: String { fileURL.path }

    /// A friendly label for a list row, e.g. `"gemma-4-E2B-it-GGUF · Q4_K_M"` — the repo tail plus the parsed
    /// quant (falling back to the full filename when no quant token is recognizable). A flat file with no repo
    /// folder shows just the filename.
    public var displayName: String {
        let repoTail = repoID.split(separator: "/").last.map(String.init) ?? repoID
        let detail = HuggingFaceRepo.quantLabel(for: filename) ?? filename
        return repoTail.isEmpty ? filename : "\(repoTail) · \(detail)"
    }

    /// Human-readable size (e.g. `1.6 GB`). Formatting lives here, not in the view.
    public var sizeText: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }
}

// MARK: - DownloadedModels

/// Enumerates the GGUF models a user has **already downloaded** (or dropped in place) under the model storage
/// root, and under Caches where earlier versions downloaded. Pure and deterministic: the scan takes an injectable
/// `FileManager`/root so it can be unit-tested against a temp directory without touching the real folders.
public enum DownloadedModels {

    /// The root directory GGUF models download into: `<root>/models` under ``ModelStorage/rootDirectory``.
    /// Mirrors the base of ``Model/cacheDirectory`` (which appends `<repoID>` to this).
    public static var modelsDirectory: URL {
        modelsDirectory(root: ModelStorage.rootDirectory)
    }

    /// Every GGUF models root that is read: ``modelsDirectory`` first, then `<Caches>/models`, where earlier
    /// versions downloaded and where files stay until ``ModelStorage/migrateLegacyDownloads()`` moves them.
    public static var modelsDirectories: [URL] {
        ModelStorage.readRoots.map(modelsDirectory(root:))
    }

    static func modelsDirectory(root: URL) -> URL {
        ModelDownloads.modelsRoot(root)
    }

    /// Recursively enumerate every `.gguf` file under ``modelsDirectories``, sorted by
    /// ``DownloadedGGUF/displayName``. A file in both roots is listed once, from the storage root. Partial and
    /// URLSession temp files (no `.gguf` extension) are skipped.
    public static func scan(fileManager: FileManager = .default) -> [DownloadedGGUF] {
        scan(roots: modelsDirectories, fileManager: fileManager)
    }

    /// ``scan(root:fileManager:)`` over each root, keeping the first file for each `<repo>/<file>`.
    static func scan(roots: [URL], fileManager: FileManager) -> [DownloadedGGUF] {
        var seen = Set<String>()
        return roots.flatMap { scan(root: $0, fileManager: fileManager) }
            .filter { seen.insert($0.repoID + "/" + $0.filename).inserted }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
    }

    /// Testable core — enumerate `*.gguf` files under `root`. A missing/empty root yields `[]`.
    static func scan(root: URL, fileManager: FileManager) -> [DownloadedGGUF] {
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles],
            errorHandler: nil
        ) else { return [] }

        var results: [DownloadedGGUF] = []
        for case let url as URL in enumerator {
            guard url.pathExtension.lowercased() == "gguf" else { continue }
            let values = try? url.resourceValues(forKeys: Set(keys))
            guard values?.isRegularFile == true else { continue }
            let size = Int64(values?.fileSize ?? 0)
            results.append(DownloadedGGUF(
                repoID: repoID(of: url, root: root),
                filename: url.lastPathComponent,
                fileURL: url,
                sizeBytes: size
            ))
        }
        return results.sorted {
            $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
        }
    }

    /// Derive `<org>/<repo>` from `url`'s path relative to `root`, minus the filename. A file sitting directly
    /// under `root` (no intervening folder) → `""`.
    static func repoID(of url: URL, root: URL) -> String {
        let rootComponents = root.standardizedFileURL.pathComponents
        let fileComponents = url.standardizedFileURL.pathComponents
        guard fileComponents.count > rootComponents.count,
              Array(fileComponents.prefix(rootComponents.count)) == rootComponents else {
            return ""
        }
        // Components strictly between the root and the filename form the repo path.
        return fileComponents[rootComponents.count..<(fileComponents.count - 1)].joined(separator: "/")
    }
}

// MARK: - MLX snapshots and partial files

/// An MLX snapshot folder under `<root>/huggingface/hub/models--<org>--<repo>/snapshots/main`, in the storage root
/// or in Caches.
public struct DownloadedSnapshot: Sendable, Identifiable, Equatable {
    public let repoID: String
    public let directory: URL
    /// Every file in the folder, `.partial` files included.
    public let sizeBytes: Int64
    /// The `.complete` marker is present, so ``Model/isDownloaded`` is `true` and MLX loads it without fetching.
    public let isComplete: Bool

    public init(repoID: String, directory: URL, sizeBytes: Int64, isComplete: Bool) {
        self.repoID = repoID
        self.directory = directory
        self.sizeBytes = sizeBytes
        self.isComplete = isComplete
    }

    public var id: String { directory.path }
}

public extension DownloadedModels {

    /// MLX snapshots on disk, complete or not, sorted by repo id. A repo in both the storage root and Caches is
    /// listed once: the complete copy, else the storage root's.
    static func scanSnapshots(fileManager: FileManager = .default) -> [DownloadedSnapshot] {
        scanSnapshots(hubRoots: ModelStorage.readRoots.map(hubDirectory(root:)), fileManager: fileManager)
    }

    /// Unfinished `*.partial` downloads under the GGUF models roots and the MLX hub folders.
    static func partialFiles(fileManager: FileManager = .default) -> [URL] {
        partialFiles(roots: ModelStorage.readRoots.flatMap { [modelsDirectory(root: $0), hubDirectory(root: $0)] },
                     fileManager: fileManager)
    }
}

extension DownloadedModels {

    static func hubDirectory(root: URL) -> URL {
        root.appending(path: "huggingface/hub")
    }

    static func scanSnapshots(hubRoots: [URL], fileManager: FileManager) -> [DownloadedSnapshot] {
        var chosen: [String: DownloadedSnapshot] = [:]
        for snapshot in hubRoots.flatMap({ scanSnapshots(hubRoot: $0, fileManager: fileManager) }) {
            if let kept = chosen[snapshot.repoID], kept.isComplete || !snapshot.isComplete { continue }
            chosen[snapshot.repoID] = snapshot
        }
        return chosen.values
            .sorted { $0.repoID.localizedCaseInsensitiveCompare($1.repoID) == .orderedAscending }
    }

    static func scanSnapshots(hubRoot: URL, fileManager: FileManager) -> [DownloadedSnapshot] {
        let folders = (try? fileManager.contentsOfDirectory(atPath: hubRoot.path)) ?? []
        return folders.filter { $0.hasPrefix("models--") }
            .compactMap { snapshot(inRepoFolder: hubRoot.appendingPathComponent($0), fileManager: fileManager) }
            .sorted { $0.repoID.localizedCaseInsensitiveCompare($1.repoID) == .orderedAscending }
    }

    private static func snapshot(inRepoFolder folder: URL, fileManager: FileManager) -> DownloadedSnapshot? {
        let directory = folder.appending(path: "snapshots/\(ModelDownloads.snapshotFolder)")
        var isFolder: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isFolder), isFolder.boolValue else {
            return nil
        }
        let name = folder.lastPathComponent.dropFirst("models--".count)
        guard let separator = name.range(of: "--") else { return nil }
        let repoID = name.replacingCharacters(in: separator, with: "/")
        let marker = directory.appendingPathComponent(ModelDownloads.completeMarker)
        return DownloadedSnapshot(repoID: repoID, directory: directory,
                                  sizeBytes: directorySize(directory, fileManager: fileManager),
                                  isComplete: fileManager.fileExists(atPath: marker.path))
    }

    static func partialFiles(roots: [URL], fileManager: FileManager) -> [URL] {
        roots.flatMap { regularFiles(under: $0, fileManager: fileManager) }
            .filter { $0.pathExtension == HubFileTransfer.partialExtension }
            .sorted { $0.path < $1.path }
    }

    /// Total size of the regular files under `directory`; 0 when it does not exist.
    static func directorySize(_ directory: URL, fileManager: FileManager) -> Int64 {
        regularFiles(under: directory, fileManager: fileManager)
            .map { Int64((try? $0.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0) }
            .saturatingSum()
    }

    private static func regularFiles(under root: URL, fileManager: FileManager) -> [URL] {
        guard let enumerator = fileManager.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey], options: [], errorHandler: nil
        ) else { return [] }
        return enumerator.compactMap { item in
            guard let url = item as? URL,
                  (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else { return nil }
            return url
        }
    }
}
