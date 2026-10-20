import Darwin
import Foundation
import os

// MARK: - ModelStorage

/// Where Tonamox keeps downloaded model weights: `<Application Support>/AuraLocal/Models`, excluded from iCloud
/// and device backups (and Time Machine on macOS) so multi-gigabyte weights are neither uploaded nor restored.
///
/// The layout under the root is the one Caches used: GGUF files at `models/<org>/<repo>/<file>`, MLX snapshots and
/// Core ML folders at `huggingface/hub/models--<org>--<repo>/snapshots/main/`.
///
/// Earlier versions downloaded into Caches, which iOS empties when storage runs low. Those files keep working: the
/// download and load paths read Caches as a fallback while the storage root has no finished copy, and in a
/// sandboxed process (every iOS app, a sandboxed Mac app) the first access to the root moves them over with
/// ``migrateLegacyDownloads()``. A non-sandboxed macOS process does not move anything by itself, because its Caches
/// (`~/Library/Caches`) is shared with every other process of the user, older Tonamox builds included. Resuming a
/// GGUF download is the exception: its `.partial` in Caches moves into the root so the transfer continues from it.
public enum ModelStorage {

    /// `<Application Support>/AuraLocal/Models`. Inside the app's container on iOS and in a sandboxed Mac app. A
    /// non-sandboxed macOS process — the `tonamox` CLI, an app without App Sandbox, `swift test` — gets
    /// `~/Library/Application Support/AuraLocal/Models`, shared by all of them so one download serves each.
    ///
    /// The first access creates the folder, excludes it from backup and, when sandboxed, migrates Caches.
    public static var rootDirectory: URL { prepared.root }

    /// Moves downloads an earlier version left in Caches into ``rootDirectory``: GGUF files and their `.partial`
    /// files under `models/`, and Tonamox's `snapshots/main` folders under `huggingface/hub/`. Each item is
    /// renamed, never copied, and only when nothing sits at its new path; anything else stays where it is and is
    /// still read from there. Other tools' files in Caches (a Hugging Face hub cache with `blobs/` and `refs/`) are
    /// left alone. Idempotent: a second call finds nothing to move.
    ///
    /// Runs on its own in a sandboxed process; a non-sandboxed Mac app may call it once it knows no older build is
    /// using `~/Library/Caches`.
    @discardableResult
    public static func migrateLegacyDownloads() -> ModelStorageMigration {
        migrate(from: prepared.legacyRoot, to: prepared.root, fileManager: .default)
    }

    // MARK: Internal

    static let log = Logger(subsystem: "dev.auralocal", category: "ModelStorage")
    static let relativeRootPath = "AuraLocal/Models"

    /// Caches, where earlier versions downloaded.
    static var legacyRoot: URL { prepared.legacyRoot }

    /// Roots read after ``rootDirectory`` when it has no finished copy of a model.
    static var fallbackRoots: [URL] { [prepared.legacyRoot] }

    /// ``rootDirectory`` first, then ``fallbackRoots``.
    static var readRoots: [URL] { [prepared.root, prepared.legacyRoot] }

    private static let prepared: StorageLocations = {
        let fileManager = FileManager.default
        let locations = StorageLocations(
            root: fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appending(path: relativeRootPath, directoryHint: .isDirectory),
            legacyRoot: fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0])
        prepare(locations.root, fileManager: fileManager)
        if isSandboxed {
            let report = migrate(from: locations.legacyRoot, to: locations.root, fileManager: fileManager)
            if !report.isEmpty {
                log.notice("""
                    moved \(report.moved.count) downloads out of Caches; \(report.conflicts.count) kept there \
                    (already in the storage root), \(report.failures.count) failed
                    """)
            }
        }
        return locations
    }()

    static var isSandboxed: Bool {
        #if os(macOS)
        ProcessInfo.processInfo.environment["APP_SANDBOX_CONTAINER_ID"] != nil
        #else
        true
        #endif
    }

    /// Creates `root` and excludes it, and so everything under it, from backup.
    static func prepare(_ root: URL, fileManager: FileManager) {
        do {
            try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var excluded = root
            try excluded.setResourceValues(values)
        } catch {
            let reason = error.localizedDescription
            log.error("could not prepare \(root.path, privacy: .public): \(reason, privacy: .public)")
        }
    }

    /// The first of `roots` where `relativePath` names an existing regular file, or a symlink to one (a GGUF linked
    /// in to avoid a second multi-gigabyte copy). The returned URL is the path in that root, not the link's target.
    static func firstFile(_ relativePath: String, in roots: [URL]) -> URL? {
        roots.lazy.map { $0.appending(path: relativePath) }.first { url in
            (try? url.resolvingSymlinksInPath().resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
        }
    }

    // MARK: Migration

    static func migrate(from legacyRoot: URL, to root: URL, fileManager: FileManager) -> ModelStorageMigration {
        var moved: [String] = []
        var conflicts: [String] = []
        var failures: [String] = []
        let items = legacyGGUFFiles(in: legacyRoot, fileManager: fileManager)
            + legacySnapshots(in: legacyRoot, fileManager: fileManager)
        for relative in items {
            let source = legacyRoot.appending(path: relative)
            switch moveExclusively(source, to: root.appending(path: relative), fileManager: fileManager) {
            case .moved:
                moved.append(relative)
                removeEmptyFolders(from: source.deletingLastPathComponent(), upTo: legacyRoot)
            case .conflict:
                conflicts.append(relative)
            case .vanished:
                break
            case .failed(let code):
                failures.append(relative)
                log.error("could not move \(relative, privacy: .public) out of Caches: errno \(code)")
            }
        }
        return ModelStorageMigration(moved: moved, conflicts: conflicts, failures: failures)
    }

    /// `models/…/<file>.gguf` and `.gguf.partial` files, relative to `legacyRoot`.
    static func legacyGGUFFiles(in legacyRoot: URL, fileManager: FileManager) -> [String] {
        let models = legacyRoot.appending(path: "models")
        guard let enumerator = fileManager.enumerator(at: models, includingPropertiesForKeys: [.isRegularFileKey],
                                                      options: [.skipsHiddenFiles], errorHandler: nil) else {
            return []
        }
        return enumerator.compactMap { item -> String? in
            guard let url = item as? URL, isGGUFDownload(url.lastPathComponent),
                  (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else { return nil }
            return relativePath(of: url, under: legacyRoot)
        }.sorted()
    }

    /// `huggingface/hub/models--<org>--<repo>/snapshots/main` folders, relative to `legacyRoot`. Only that folder
    /// moves: a Hugging Face hub cache keeps `blobs/`, `refs/` and commit-named snapshots beside it.
    static func legacySnapshots(in legacyRoot: URL, fileManager: FileManager) -> [String] {
        let hub = legacyRoot.appending(path: "huggingface/hub")
        let repos = (try? fileManager.contentsOfDirectory(atPath: hub.path)) ?? []
        return repos.filter { $0.hasPrefix("models--") }.sorted().compactMap { repo -> String? in
            let relative = "huggingface/hub/\(repo)/snapshots/\(ModelDownloads.snapshotFolder)"
            var isFolder: ObjCBool = false
            guard fileManager.fileExists(atPath: legacyRoot.appending(path: relative).path, isDirectory: &isFolder),
                  isFolder.boolValue else { return nil }
            return relative
        }
    }

    static func isGGUFDownload(_ name: String) -> Bool {
        let lowered = name.lowercased()
        return lowered.hasSuffix(".gguf") || lowered.hasSuffix(".gguf.\(HubFileTransfer.partialExtension)")
    }

    static func relativePath(of url: URL, under root: URL) -> String? {
        let rootComponents = root.standardizedFileURL.pathComponents
        let components = url.standardizedFileURL.pathComponents
        guard components.count > rootComponents.count,
              Array(components.prefix(rootComponents.count)) == rootComponents else { return nil }
        return components.dropFirst(rootComponents.count).joined(separator: "/")
    }

    enum MoveOutcome: Equatable {
        case moved, conflict, vanished
        case failed(Int32)
    }

    /// Renames `source` to `destination` unless something already sits there. `rename(2)` with `RENAME_EXCL` is
    /// atomic and never copies: a destination created concurrently is not overwritten, and an item on another
    /// volume fails (`EXDEV`) instead of being copied at launch.
    static func moveExclusively(_ source: URL, to destination: URL, fileManager: FileManager) -> MoveOutcome {
        guard fileManager.fileExists(atPath: source.path) else { return .vanished }
        do {
            try fileManager.createDirectory(at: destination.deletingLastPathComponent(),
                                            withIntermediateDirectories: true)
        } catch {
            return .failed(EIO)
        }
        let code = source.withUnsafeFileSystemRepresentation { from -> Int32 in
            destination.withUnsafeFileSystemRepresentation { to -> Int32 in
                guard let from, let to else { return EINVAL }
                return renamex_np(from, to, UInt32(RENAME_EXCL)) == 0 ? 0 : errno
            }
        }
        switch code {
        case 0: return .moved
        case EEXIST, ENOTEMPTY: return .conflict
        case ENOENT where !fileManager.fileExists(atPath: source.path): return .vanished
        default: return .failed(code)
        }
    }

    /// Removes `folder` and its parents while they are empty, stopping before `root`.
    static func removeEmptyFolders(from folder: URL, upTo root: URL) {
        let rootPath = root.standardizedFileURL.path
        var current = folder.standardizedFileURL
        while current.path.hasPrefix(rootPath + "/"), rmdir(current.path) == 0 {
            current.deleteLastPathComponent()
        }
    }
}

// MARK: - StorageLocations

/// The storage root and the Caches directory earlier versions downloaded into.
struct StorageLocations: Sendable {
    let root: URL
    let legacyRoot: URL
}

// MARK: - ModelStorageMigration

/// What ``ModelStorage/migrateLegacyDownloads()`` did, as paths relative to the roots (for example
/// `models/org/repo/model.gguf` or `huggingface/hub/models--org--repo/snapshots/main`).
public struct ModelStorageMigration: Sendable, Equatable {
    /// Renamed into the storage root.
    public let moved: [String]
    /// Left in Caches because the storage root already has an item at that path. The storage root's copy is used
    /// once it is complete; until then the one in Caches is read.
    public let conflicts: [String]
    /// Left in Caches because the rename failed (another volume, permissions). Still read from there.
    public let failures: [String]

    public init(moved: [String], conflicts: [String], failures: [String]) {
        self.moved = moved
        self.conflicts = conflicts
        self.failures = failures
    }

    /// Nothing was found in Caches.
    public var isEmpty: Bool { moved.isEmpty && conflicts.isEmpty && failures.isEmpty }
}
