import Foundation

// MARK: - DownloadedGGUF

/// A `.gguf` model already present on disk under the download cache (`<Caches>/models/<org>/<repo>/<file>.gguf`).
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

/// Enumerates the GGUF models a user has **already downloaded** (or dropped in place) under the shared model
/// cache. Pure and deterministic: the scan takes an injectable `FileManager`/root so it can be unit-tested
/// against a temp directory without touching the real caches.
public enum DownloadedModels {

    /// The root directory GGUF models download into: `<Caches>/models`. Mirrors the base of
    /// ``Model/cacheDirectory`` (which appends `<repoID>` to this).
    public static var modelsDirectory: URL {
        modelsDirectory(fileManager: .default)
    }

    /// The models root for a specific `FileManager` (injectable for tests).
    static func modelsDirectory(fileManager: FileManager) -> URL {
        fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models")
    }

    /// Recursively enumerate every `.gguf` file under the real models root (`<Caches>/models`), sorted by
    /// ``DownloadedGGUF/displayName``. Partial/URLSession temp files (no `.gguf` extension) are skipped.
    public static func scan(fileManager: FileManager = .default) -> [DownloadedGGUF] {
        scan(root: modelsDirectory(fileManager: fileManager), fileManager: fileManager)
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
