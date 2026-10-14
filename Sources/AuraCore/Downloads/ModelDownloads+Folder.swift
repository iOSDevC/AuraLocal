import Foundation

// MARK: - Folder downloads

extension ModelDownloads {

    // MARK: Public

    /// Downloads every file under `folder` of `repoID` at `revision` whose path relative to `folder` equals or
    /// starts with one of `include` (a trailing "/" means a directory), keeping the repo layout. Resumes `.partial`
    /// files, checks free space, and writes `.complete` (holding `revision`) inside the folder after the last file.
    /// Returns the local folder (``localFolder(repoID:folder:)``). `progress` receives (bytes on disk, total bytes).
    ///
    /// The folder records the revision its files come from. Files from another revision, finished or partial, are
    /// deleted before fetching, so a file of the same size is never kept from the old revision.
    ///
    /// The listing is untrusted: an absolute path, a `.` or `..` segment, a backslash, a NUL or a path outside
    /// `folder` throws ``ModelDownloadError/unsupported(_:)`` before anything is written. An `include` entry that
    /// matches no file throws ``ModelDownloadError/notFound(repoID:path:)``.
    public static func downloadFolder(repoID: String, revision: String, folder: String, include: [String],
                                      auth: any DownloadAuthorizing = NoDownloadAuth(),
                                      progress: @escaping @Sendable (Int64, Int64) -> Void) async throws -> URL {
        try await downloadFolder(repoID: repoID, revision: revision, folder: folder, include: include,
                                 transfer: HubFileTransfer(auth: auth), root: cachesRoot, progress: progress)
    }

    /// Files and sizes ``downloadFolder(repoID:revision:folder:include:auth:progress:)`` would fetch, without
    /// downloading them.
    public static func planFolder(repoID: String, revision: String, folder: String, include: [String],
                                  auth: any DownloadAuthorizing = NoDownloadAuth()) async throws -> DownloadPlan {
        try await planFolder(repoID: repoID, revision: revision, folder: folder, include: include,
                             transfer: HubFileTransfer(auth: auth), root: cachesRoot)
    }

    /// Where `downloadFolder` puts `folder`: `<Caches>/huggingface/hub/models--org--repo/snapshots/main/<folder>`.
    public static func localFolder(repoID: String, folder: String) -> URL {
        localFolder(repoID: repoID, folder: folder, root: cachesRoot)
    }

    /// Whether `downloadFolder` finished `folder` of `repoID` at `revision`.
    public static func isFolderComplete(repoID: String, folder: String, revision: String) -> Bool {
        isFolderComplete(at: localFolder(repoID: repoID, folder: folder), revision: revision)
    }

    /// Whether `directory` holds the `.complete` marker `downloadFolder` writes after its last file, for
    /// `revision`. A folder finished at another revision is not complete.
    public static func isFolderComplete(at directory: URL, revision: String) -> Bool {
        recordedRevision(in: directory, marker: completeMarker) == revision
    }

    /// Removes `models--org--repo` (finished and partial files).
    public static func deleteRepo(_ repoID: String) throws {
        try deleteRepo(repoID, root: cachesRoot)
    }

    // MARK: Revision

    /// Written before the first file is fetched: the revision the files in the folder come from.
    static let revisionMarker = ".revision"

    static func recordedRevision(in directory: URL, marker: String) -> String? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(marker)) else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    /// Deletes `directory` when its files were fetched for another revision, or for an unrecorded one.
    static func discardOtherRevision(in directory: URL, keeping revision: String) throws {
        guard FileManager.default.fileExists(atPath: directory.path),
              recordedRevision(in: directory, marker: revisionMarker) != revision else { return }
        try FileManager.default.removeItem(at: directory)
    }

    // MARK: Paths

    static func localFolder(repoID: String, folder: String, root: URL) -> URL {
        snapshotDirectory(repoID: repoID, root: root).appending(path: folder, directoryHint: .isDirectory)
    }

    /// Relative, non-empty, and free of `.`/`..`/empty segments, backslashes and NULs.
    static func isSafeRelativePath(_ path: String) -> Bool {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"), !path.contains("\0") else { return false }
        return path.split(separator: "/", omittingEmptySubsequences: false)
            .allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }

    static func isSafeListedPath(_ path: String, under folder: String) -> Bool {
        isSafeRelativePath(path) && path.hasPrefix(folder + "/")
    }

    /// `path` inside `snapshot`, refusing anything that would land outside it.
    static func destination(for path: String, in snapshot: URL) throws -> URL {
        let destination = snapshot.appending(path: path).standardizedFileURL
        guard isSafeRelativePath(path),
              destination.path.hasPrefix(snapshot.standardizedFileURL.path + "/") else {
            throw ModelDownloadError.unsupported("Unsafe path in the Hugging Face listing: \(path)")
        }
        return destination
    }

    static func validated(repoID: String) throws -> String {
        let parts = repoID.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy({ isSafeRelativePath(String($0)) }) else {
            throw ModelDownloadError.unsupported("Not a Hugging Face repository id: \(repoID)")
        }
        return repoID
    }

    static func validated(folder: String) throws -> String {
        guard isSafeRelativePath(folder) else {
            throw ModelDownloadError.unsupported("Not a safe folder path: \(folder)")
        }
        return folder
    }

    static func validated(include: [String]) throws -> [String] {
        let unsafe = include.first { !isSafeRelativePath($0.hasSuffix("/") ? String($0.dropLast()) : $0) }
        if let unsafe { throw ModelDownloadError.unsupported("Not a safe path to include: \(unsafe)") }
        return include
    }

    // MARK: Plan

    static func planFolder(repoID: String, revision: String, folder: String, include: [String],
                           transfer: HubFileTransfer, root: URL) async throws -> DownloadPlan {
        let files = try await folderFiles(repoID: repoID, revision: revision, folder: folder, include: include,
                                          transfer: transfer)
        let snapshot = snapshotDirectory(repoID: repoID, root: root)
        let present = try files.map { onDiskBytes(try destination(for: $0.path, in: snapshot)) }.saturatingSum()
        // Files from another revision are deleted before the download, so they do not count.
        let directory = localFolder(repoID: repoID, folder: folder, root: root)
        let current = recordedRevision(in: directory, marker: revisionMarker) == revision
        return DownloadPlan(repoID: repoID, files: files, bytesOnDisk: current ? present : 0)
    }

    /// The listed files under `folder` selected by `include`, sized. Rejects the whole listing if any path is unsafe.
    static func folderFiles(repoID: String, revision: String, folder: String, include: [String],
                            transfer: HubFileTransfer) async throws -> [PlannedFile] {
        _ = try validated(repoID: repoID)
        _ = try validated(include: include)
        let prefix = try validated(folder: folder) + "/"
        let listed = try await transfer.listFiles(repoID: repoID, revision: revision, under: folder)
        if let unsafe = listed.first(where: { !isSafeListedPath($0.path, under: folder) }) {
            throw ModelDownloadError.unsupported("Unsafe path in the Hugging Face listing: \(unsafe.path)")
        }
        let relative: (RemoteFile) -> String = { String($0.path.dropFirst(prefix.count)) }
        let matching = listed.filter { file in include.contains { matchesInclude(relative(file), entry: $0) } }
        guard !matching.isEmpty else { throw ModelDownloadError.incompleteRepo(repoID: repoID) }
        if let missing = include.first(where: { entry in
            !matching.contains { matchesInclude(relative($0), entry: entry) }
        }) {
            throw ModelDownloadError.notFound(repoID: repoID, path: prefix + missing)
        }
        var planned: [PlannedFile] = []
        for file in matching {
            planned.append(try await sized(file, repoID: repoID, revision: revision, transfer: transfer))
        }
        return planned
    }

    /// An entry ending in "/" selects everything under that directory; any other entry selects one file.
    static func matchesInclude(_ relativePath: String, entry: String) -> Bool {
        entry.hasSuffix("/") ? relativePath.hasPrefix(entry) : relativePath == entry
    }

    // MARK: Download

    static func downloadFolder(repoID: String, revision: String, folder: String, include: [String],
                               transfer: HubFileTransfer, root: URL,
                               progress: @escaping @Sendable (Int64, Int64) -> Void) async throws -> URL {
        _ = try validated(repoID: repoID)
        _ = try validated(include: include)
        let directory = localFolder(repoID: repoID, folder: try validated(folder: folder), root: root)
        let marker = directory.appendingPathComponent(completeMarker)
        if isFolderComplete(at: directory, revision: revision) {
            let intact = include.allSatisfy {
                FileManager.default.fileExists(atPath: directory.appending(path: $0).path)
            }
            if intact { return directory }
            // Something was removed after completion (iOS purges Caches): refetch what is missing.
            try FileManager.default.removeItem(at: marker)
        }

        let snapshot = snapshotDirectory(repoID: repoID, root: root)
        let files = try await folderFiles(repoID: repoID, revision: revision, folder: folder, include: include,
                                          transfer: transfer)
        try discardOtherRevision(in: directory, keeping: revision)
        let targets = try files.map { (file: $0, destination: try destination(for: $0.path, in: snapshot)) }
        let total = files.map(\.sizeBytes).saturatingSum()
        let present = targets.map { onDiskBytes($0.destination) }.saturatingSum()
        try transfer.ensureSpace(remaining: max(0, total - present), at: snapshot)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(revision.utf8).write(to: directory.appendingPathComponent(revisionMarker))

        var finished: Int64 = 0
        for target in targets {
            try Task.checkCancellation()
            try await fetchFolderFile(target.file, to: target.destination, repoID: repoID, revision: revision,
                                      transfer: transfer, finished: finished, total: total, progress: progress)
            finished += target.file.sizeBytes
        }
        progress(total, total)
        try Data(revision.utf8).write(to: marker)
        return directory
    }

    private static func fetchFolderFile(_ file: PlannedFile, to destination: URL, repoID: String, revision: String,
                                        transfer: HubFileTransfer, finished: Int64, total: Int64,
                                        progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        if HubFileTransfer.fileSize(at: destination) == file.sizeBytes { return }
        guard let url = fileURL(repoID: repoID, revision: revision, path: file.path) else {
            throw ModelDownloadError.unsupported("Not a Hugging Face repository id: \(repoID)")
        }
        try await transfer.fetch(url, to: destination, expectedSize: file.sizeBytes,
                                 repoID: repoID, path: file.path) { bytes in
            progress(finished + bytes, total)
        }
    }

    // MARK: Delete

    static func deleteRepo(_ repoID: String, root: URL, fileManager: FileManager = .default) throws {
        try removeIfPresent(repoFolder(repoID: try validated(repoID: repoID), root: root), fileManager: fileManager)
    }
}
