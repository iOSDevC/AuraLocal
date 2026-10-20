import Foundation
import Testing
@testable import TonamoxCore

/// The storage root, its backup exclusion, and the move out of Caches, on temporary folders.
struct ModelStorageTests {
    private let base: URL
    private let legacy: URL
    private let root: URL

    init() throws {
        base = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelStorageTests-\(UUID().uuidString)", isDirectory: true)
        legacy = base.appending(path: "Caches", directoryHint: .isDirectory)
        root = base.appending(path: "Application Support/Tonamox/Models", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: legacy, withIntermediateDirectories: true)
    }

    private func cleanUp() {
        try? FileManager.default.removeItem(at: base)
    }

    private func write(_ bytes: Int, at relative: String, under folder: URL) throws -> URL {
        let file = folder.appending(path: relative)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 7, count: bytes).write(to: file)
        return file
    }

    private func exists(_ relative: String, under folder: URL) -> Bool {
        FileManager.default.fileExists(atPath: folder.appending(path: relative).path)
    }

    private func migrate() -> ModelStorageMigration {
        ModelStorage.migrate(from: legacy, to: root, fileManager: .default)
    }

    // MARK: Root

    @Test func rootKeepsAuraLocalModelsPathInApplicationSupportAndExcludesItFromBackup() throws {
        let support = try #require(FileManager.default.urls(for: .applicationSupportDirectory,
                                                            in: .userDomainMask).first)
        let rootDirectory = ModelStorage.rootDirectory

        #expect(rootDirectory.standardizedFileURL.path
                == support.appending(path: "AuraLocal/Models").standardizedFileURL.path)
        let values = try URL(fileURLWithPath: rootDirectory.path).resourceValues(forKeys: [.isExcludedFromBackupKey])
        #expect(values.isExcludedFromBackup == true)
        #expect(ModelStorage.readRoots.first == rootDirectory)
        #expect(ModelStorage.fallbackRoots.first?.lastPathComponent == "Caches")
    }

    @Test func prepareCreatesTheRootAndExcludesItFromBackup() throws {
        defer { cleanUp() }
        ModelStorage.prepare(root, fileManager: .default)
        ModelStorage.prepare(root, fileManager: .default)

        var isFolder: ObjCBool = false
        #expect(FileManager.default.fileExists(atPath: root.path, isDirectory: &isFolder) && isFolder.boolValue)
        let values = try URL(fileURLWithPath: root.path).resourceValues(forKeys: [.isExcludedFromBackupKey])
        #expect(values.isExcludedFromBackup == true)
    }

    #if os(macOS)
    @Test func aNonSandboxedTestProcessDoesNotMigrateOnItsOwn() {
        #expect(!ModelStorage.isSandboxed)
    }
    #endif

    // MARK: Migration

    @Test func movesGGUFFilesAndPartialsKeepingLayoutAndAttributes() throws {
        defer { cleanUp() }
        _ = try write(10, at: "models/org/repo/a.gguf", under: legacy)
        let partial = try write(4, at: "models/org/repo/b.gguf.partial", under: legacy)
        HubFileTransfer.setValidator("\"etag-1\"", of: partial)
        _ = try write(3, at: "models/org/repo/notes.txt", under: legacy)

        let report = migrate()

        #expect(report.moved == ["models/org/repo/a.gguf", "models/org/repo/b.gguf.partial"])
        #expect(report.conflicts.isEmpty && report.failures.isEmpty)
        #expect(exists("models/org/repo/a.gguf", under: root))
        #expect(HubFileTransfer.validator(of: root.appending(path: "models/org/repo/b.gguf.partial")) == "\"etag-1\"")
        #expect(!exists("models/org/repo/a.gguf", under: legacy))
        #expect(exists("models/org/repo/notes.txt", under: legacy))
    }

    @Test func emptyLegacyFoldersAreRemovedButCachesStays() throws {
        defer { cleanUp() }
        _ = try write(10, at: "models/org/repo/a.gguf", under: legacy)

        _ = migrate()

        #expect(!exists("models", under: legacy))
        #expect(FileManager.default.fileExists(atPath: legacy.path))
    }

    @Test func movesOnlyTonamoxSnapshotFolders() throws {
        defer { cleanUp() }
        let snapshot = "huggingface/hub/models--org--mlx/snapshots/main"
        _ = try write(30, at: "\(snapshot)/model.safetensors", under: legacy)
        _ = try write(0, at: "\(snapshot)/\(ModelDownloads.completeMarker)", under: legacy)
        // A Hugging Face hub cache of another tool: blobs, refs and a commit-named snapshot.
        _ = try write(5, at: "huggingface/hub/models--org--py/blobs/abc", under: legacy)
        _ = try write(1, at: "huggingface/hub/models--org--py/refs/main", under: legacy)
        _ = try write(5, at: "huggingface/hub/models--org--py/snapshots/0123abcd/config.json", under: legacy)

        let report = migrate()

        #expect(report.moved == [snapshot])
        #expect(exists("\(snapshot)/model.safetensors", under: root))
        #expect(exists("\(snapshot)/\(ModelDownloads.completeMarker)", under: root))
        #expect(!exists("huggingface/hub/models--org--mlx", under: legacy))
        #expect(exists("huggingface/hub/models--org--py/blobs/abc", under: legacy))
        #expect(exists("huggingface/hub/models--org--py/snapshots/0123abcd/config.json", under: legacy))
        #expect(!exists("huggingface/hub/models--org--py", under: root))
    }

    @Test func aSecondMigrationFindsNothing() throws {
        defer { cleanUp() }
        _ = try write(10, at: "models/org/repo/a.gguf", under: legacy)
        _ = try write(30, at: "huggingface/hub/models--org--mlx/snapshots/main/model.safetensors", under: legacy)

        #expect(migrate().moved.count == 2)
        #expect(migrate().isEmpty)
        #expect(exists("models/org/repo/a.gguf", under: root))
    }

    @Test func aConflictKeepsBothCopiesAndOverwritesNothing() throws {
        defer { cleanUp() }
        _ = try write(10, at: "models/org/repo/a.gguf", under: legacy)
        _ = try write(20, at: "models/org/repo/a.gguf", under: root)
        let snapshot = "huggingface/hub/models--org--mlx/snapshots/main"
        _ = try write(30, at: "\(snapshot)/model.safetensors", under: legacy)
        _ = try write(0, at: "\(snapshot)/\(ModelDownloads.completeMarker)", under: legacy)
        _ = try write(7, at: "\(snapshot)/model.safetensors.partial", under: root)

        let report = migrate()

        #expect(report.moved.isEmpty)
        #expect(report.conflicts == ["models/org/repo/a.gguf", snapshot])
        #expect(HubFileTransfer.fileSize(at: root.appending(path: "models/org/repo/a.gguf")) == 20)
        #expect(HubFileTransfer.fileSize(at: legacy.appending(path: "models/org/repo/a.gguf")) == 10)
        #expect(exists("\(snapshot)/\(ModelDownloads.completeMarker)", under: legacy))
        #expect(!exists("\(snapshot)/model.safetensors", under: root))
    }

    @Test func aFailedMoveLeavesTheFileInCaches() throws {
        defer { cleanUp() }
        _ = try write(10, at: "models/org/repo/a.gguf", under: legacy)
        // A file where the destination folder has to be: the move cannot happen.
        _ = try write(1, at: "models", under: root)

        let report = migrate()

        #expect(report.failures == ["models/org/repo/a.gguf"])
        #expect(HubFileTransfer.fileSize(at: legacy.appending(path: "models/org/repo/a.gguf")) == 10)
    }

    @Test func aPartialStillBeingWrittenKeepsEveryByteAfterTheMove() throws {
        defer { cleanUp() }
        let partial = try write(4, at: "models/org/repo/a.gguf.partial", under: legacy)
        let writer = try FileHandle(forWritingTo: partial)
        try writer.seekToEnd()

        #expect(migrate().moved == ["models/org/repo/a.gguf.partial"])
        try writer.write(contentsOf: Data(repeating: 9, count: 6))
        try writer.close()

        #expect(HubFileTransfer.fileSize(at: root.appending(path: "models/org/repo/a.gguf.partial")) == 10)
        #expect(!FileManager.default.fileExists(atPath: partial.path))
    }

    @Test func moveExclusivelyNeverReplacesAnExistingFile() throws {
        defer { cleanUp() }
        let source = try write(10, at: "a", under: legacy)
        let destination = try write(3, at: "a", under: root)

        #expect(ModelStorage.moveExclusively(source, to: destination, fileManager: .default) == .conflict)
        #expect(HubFileTransfer.fileSize(at: destination) == 3)
        #expect(ModelStorage.moveExclusively(legacy.appending(path: "gone"), to: destination,
                                             fileManager: .default) == .vanished)
    }

    @Test func deletingAnMLXModelInCachesLeavesAHubCacheOfTheSameRepo() throws {
        defer { cleanUp() }
        let entry = CatalogEntry(id: "storage_mlx", repoID: "org/mlx", displayName: "MLX", modelCategory: .text,
                                 weightFormat: .mlx, approximateSizeMB: 1, isUncensored: false, ggufFilename: nil,
                                 numLayers: 24, kvHeads: 0, headDim: 0, maxContextLength: nil)
        let model = try #require(entry.decodedModel())
        _ = try write(30, at: "huggingface/hub/models--org--mlx/snapshots/main/model.safetensors", under: legacy)
        _ = try write(5, at: "huggingface/hub/models--org--mlx/blobs/abc", under: legacy)

        try ModelDownloads.deleteLegacy(model, root: legacy)

        #expect(!exists("huggingface/hub/models--org--mlx/snapshots", under: legacy))
        #expect(exists("huggingface/hub/models--org--mlx/blobs/abc", under: legacy))

        try FileManager.default.removeItem(at: legacy.appending(path: "huggingface/hub/models--org--mlx/blobs"))
        _ = try write(30, at: "huggingface/hub/models--org--mlx/snapshots/main/model.safetensors", under: legacy)
        try ModelDownloads.deleteLegacy(model, root: legacy)
        #expect(!exists("huggingface/hub/models--org--mlx", under: legacy))
    }

    // MARK: Reading both roots

    @Test func firstFilePrefersTheStorageRoot() throws {
        defer { cleanUp() }
        let old = try write(10, at: "models/org/repo/a.gguf", under: legacy)
        #expect(ModelStorage.firstFile("models/org/repo/a.gguf", in: [root, legacy]) == old)

        let new = try write(10, at: "models/org/repo/a.gguf", under: root)
        #expect(ModelStorage.firstFile("models/org/repo/a.gguf", in: [root, legacy]) == new)
        #expect(ModelStorage.firstFile("models/org/repo/b.gguf", in: [root, legacy]) == nil)
    }

    @Test func firstFileFollowsASymlinkToARegularFile() throws {
        defer { cleanUp() }
        let target = try write(10, at: "elsewhere/a.gguf", under: base)
        let link = legacy.appending(path: "models/org/repo/a.gguf")
        try FileManager.default.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        let folderLink = legacy.appending(path: "models/org/repo/folder.gguf")
        try FileManager.default.createSymbolicLink(at: folderLink, withDestinationURL: base)

        #expect(ModelStorage.firstFile("models/org/repo/a.gguf", in: [root, legacy]) == link)
        #expect(ModelStorage.firstFile("models/org/repo/folder.gguf", in: [root, legacy]) == nil)
    }

    @Test func scanListsAFileInBothRootsOnce() throws {
        defer { cleanUp() }
        _ = try write(10, at: "models/org/repo/a.gguf", under: legacy)
        _ = try write(10, at: "models/org/repo/a.gguf", under: root)
        _ = try write(10, at: "models/org/repo/b.gguf", under: legacy)

        let found = DownloadedModels.scan(roots: [root, legacy].map(DownloadedModels.modelsDirectory(root:)),
                                          fileManager: .default)

        #expect(found.map(\.filename) == ["a.gguf", "b.gguf"])
        #expect(found.map(\.repoID) == ["org/repo", "org/repo"])
        // The enumerator reports /private/var for a temporary /var path; standardizing maps both to /var.
        #expect(found.map(\.fileURL.standardizedFileURL.path)
                == [root.appending(path: "models/org/repo/a.gguf").standardizedFileURL.path,
                    legacy.appending(path: "models/org/repo/b.gguf").standardizedFileURL.path])
    }

    @Test func scanSnapshotsPrefersTheCompleteCopy() throws {
        defer { cleanUp() }
        let snapshot = "huggingface/hub/models--org--mlx/snapshots/main"
        _ = try write(7, at: "\(snapshot)/model.safetensors.partial", under: root)
        _ = try write(30, at: "\(snapshot)/model.safetensors", under: legacy)
        _ = try write(0, at: "\(snapshot)/\(ModelDownloads.completeMarker)", under: legacy)

        let found = DownloadedModels.scanSnapshots(hubRoots: [root, legacy].map(DownloadedModels.hubDirectory(root:)),
                                                   fileManager: .default)

        #expect(found.count == 1)
        #expect(found.first?.isComplete == true)
        #expect(found.first?.directory.path.hasPrefix(legacy.path) == true)
    }

    @Test func completedFolderFallsBackToCachesUntilTheRootIsComplete() throws {
        defer { cleanUp() }
        let old = ModelDownloads.localFolder(repoID: "org/sd", folder: "compiled", root: legacy)
        try FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
        try Data("rev1".utf8).write(to: old.appendingPathComponent(ModelDownloads.completeMarker))

        #expect(ModelDownloads.completedFolder(repoID: "org/sd", folder: "compiled", revision: "rev1",
                                               roots: [root, legacy]) == old)
        #expect(ModelDownloads.completedFolder(repoID: "org/sd", folder: "compiled", revision: "rev2",
                                               roots: [root, legacy]) == nil)

        let new = ModelDownloads.localFolder(repoID: "org/sd", folder: "compiled", root: root)
        try FileManager.default.createDirectory(at: new, withIntermediateDirectories: true)
        try Data("rev1".utf8).write(to: new.appendingPathComponent(ModelDownloads.completeMarker))
        #expect(ModelDownloads.completedFolder(repoID: "org/sd", folder: "compiled", revision: "rev1",
                                               roots: [root, legacy]) == new)
    }
}
