import Foundation
import Synchronization
import Testing
@testable import AuraCore

/// `ModelDownloads.downloadFolder` against a stubbed Hugging Face: no network, a temp Caches root per test, and
/// its own stub routes so it never races `ModelDownloadsTests`, which resets the shared ones.
@Suite(.serialized)
final class ModelDownloadsFolderTests {
    private let root: URL
    private let hub = FolderStubProtocol.hub

    private static let folder = "split/compiled"
    private static let revision = "0123abc"

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelDownloadsFolderTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        hub.reset()
    }

    deinit {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: Fixtures

    private static let stubSession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FolderStubProtocol.self]
        return URLSession(configuration: configuration)
    }()

    private func transfer(auth: any DownloadAuthorizing = NoDownloadAuth(),
                          freeBytes: Int64? = nil) -> HubFileTransfer {
        HubFileTransfer(session: Self.stubSession, auth: auth, freeSpace: { _ in freeBytes },
                        retryDelay: { _ in .zero }, progressInterval: .milliseconds(10))
    }

    private static func payload(_ count: Int, salt: Int = 0) -> Data {
        Data((0..<count).map { UInt8(($0 + salt) % 251) })
    }

    /// Lists `files` (paths relative to the repo) under the folder, plus a directory entry the filter must skip.
    private func serveTree(_ repoID: String, _ files: [(path: String, size: Int)],
                           revision: String = ModelDownloadsFolderTests.revision) {
        let directory: [String: Any] = ["type": "directory", "path": "\(Self.folder)/Unet.mlmodelc", "size": 0]
        let entries = [directory] + files.map { ["type": "file", "path": $0.path, "size": $0.size] as [String: Any] }
        let body = (try? JSONSerialization.data(withJSONObject: entries)) ?? Data()
        hub.serve("api/models/\(repoID)/tree/\(revision)/\(Self.folder)?recursive=true", body: body)
    }

    /// Each revision serves different bytes for the same path and size.
    private func serveFile(_ repoID: String, _ path: String, size: Int,
                           revision: String = ModelDownloadsFolderTests.revision) {
        hub.serve("\(repoID)/resolve/\(revision)/\(path)", body: Self.payload(size, salt: Self.salt(path, revision)))
    }

    private static func salt(_ path: String, _ revision: String) -> Int {
        path.count + (revision == Self.revision ? 0 : 7)
    }

    private static let modelFiles: [(path: String, size: Int)] = [
        ("\(folder)/TextEncoder.mlmodelc/model.mil", 300),
        ("\(folder)/TextEncoder.mlmodelc/weights/weight.bin", 900),
        ("\(folder)/VAEEncoder.mlmodelc/weights/weight.bin", 700),
        ("\(folder)/vocab.json", 120),
        ("\(folder)/merges.txt", 80),
    ]

    /// A Core ML-shaped folder: two model directories, the encoder that must be skipped, and two loose files.
    private func serveModelFolder(_ repoID: String, revision: String = ModelDownloadsFolderTests.revision) {
        serveTree(repoID, Self.modelFiles, revision: revision)
        Self.modelFiles.forEach { serveFile(repoID, $0.path, size: $0.size, revision: revision) }
    }

    private static let include = ["TextEncoder.mlmodelc/", "vocab.json", "merges.txt"]

    private func download(_ repoID: String, include: [String] = ModelDownloadsFolderTests.include,
                          revision: String = ModelDownloadsFolderTests.revision,
                          auth: any DownloadAuthorizing = NoDownloadAuth(),
                          progress: @escaping @Sendable (Int64, Int64) -> Void = { _, _ in }) async throws -> URL {
        try await ModelDownloads.downloadFolder(repoID: repoID, revision: revision, folder: Self.folder,
                                                include: include, transfer: transfer(auth: auth), root: root,
                                                progress: progress)
    }

    // MARK: Listing

    @Test func treeURLCarriesThePathAndRecursiveFlag() throws {
        let url = try #require(HubFileTransfer.treeURL(repoID: "o/r", revision: "rev", path: "a/b", recursive: true))
        #expect(url.absoluteString == "https://huggingface.co/api/models/o/r/tree/rev/a/b?recursive=true")
        #expect(HubFileTransfer.treeURL(repoID: "o/r", revision: "rev", path: "a", recursive: false)?.query == nil)
        for bad in ["", "/abs", "a/../b", "./a", "a//b"] {
            #expect(HubFileTransfer.treeURL(repoID: "o/r", revision: "rev", path: bad, recursive: true) == nil)
        }
    }

    @Test func localFolderSitsInTheRepoSnapshot() {
        let folder = ModelDownloads.localFolder(repoID: "org/repo", folder: "a/b", root: root)
        let expected = root.appending(path: "huggingface/hub/models--org--repo/snapshots/main/a/b")
        #expect(folder.standardizedFileURL.path == expected.standardizedFileURL.path)
    }

    // MARK: Include filter

    @Test func planKeepsIncludedDirectoriesAndExactFilesOnly() async throws {
        let repo = "stub/plan-folder"
        serveModelFolder(repo)

        let plan = try await ModelDownloads.planFolder(repoID: repo, revision: Self.revision, folder: Self.folder,
                                                       include: Self.include, transfer: transfer(), root: root)

        #expect(plan.files.map(\.path).sorted() == [
            "\(Self.folder)/TextEncoder.mlmodelc/model.mil",
            "\(Self.folder)/TextEncoder.mlmodelc/weights/weight.bin",
            "\(Self.folder)/merges.txt",
            "\(Self.folder)/vocab.json",
        ])
        #expect(plan.totalBytes == 300 + 900 + 120 + 80)
        #expect(plan.bytesOnDisk == 0)
        #expect(hub.requests(containing: "VAEEncoder").isEmpty)
    }

    @Test func includeEntriesMatchWholeNamesNotPrefixes() {
        #expect(ModelDownloads.matchesInclude("vocab.json", entry: "vocab.json"))
        #expect(!ModelDownloads.matchesInclude("vocab.json.bak", entry: "vocab.json"))
        #expect(ModelDownloads.matchesInclude("Unet.mlmodelc/model.mil", entry: "Unet.mlmodelc/"))
        #expect(!ModelDownloads.matchesInclude("UnetChunk1.mlmodelc/model.mil", entry: "Unet.mlmodelc/"))
    }

    @Test func anIncludeThatMatchesNothingNamesTheMissingPath() async throws {
        let repo = "stub/missing-folder"
        serveModelFolder(repo)

        await #expect(throws: ModelDownloadError.notFound(repoID: repo, path: "\(Self.folder)/Unet.mlmodelc/")) {
            try await self.download(repo, include: ["vocab.json", "Unet.mlmodelc/"])
        }
        await #expect(throws: ModelDownloadError.incompleteRepo(repoID: repo)) {
            try await self.download(repo, include: ["Unet.mlmodelc/"])
        }
        #expect(!FileManager.default.fileExists(atPath: ModelDownloads.repoFolder(repoID: repo, root: root).path))
    }

    // MARK: Download

    @Test func downloadKeepsTheLayoutAndMarksCompleteLast() async throws {
        let repo = "stub/dl-folder"
        serveModelFolder(repo)
        let reports = FolderProgressLog()

        let directory = try await download(repo) { done, total in reports.append(done, total) }

        #expect(directory == ModelDownloads.localFolder(repoID: repo, folder: Self.folder, root: root))
        let weightPath = "\(Self.folder)/TextEncoder.mlmodelc/weights/weight.bin"
        let weight = directory.appending(path: "TextEncoder.mlmodelc/weights/weight.bin")
        #expect(try Data(contentsOf: weight) == Self.payload(900, salt: weightPath.count))
        #expect(ModelDownloads.recordedRevision(in: directory, marker: ModelDownloads.completeMarker) == Self.revision)
        #expect(FileManager.default.fileExists(atPath: directory.appending(path: "vocab.json").path))
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: "VAEEncoder.mlmodelc").path))
        #expect(ModelDownloads.isFolderComplete(at: directory, revision: Self.revision))
        let last = try #require(reports.last)
        #expect(last.0 == 1400 && last.1 == 1400)
    }

    @Test func aFailedFileLeavesNoCompletionMarker() async throws {
        let repo = "stub/fail-folder"
        serveTree(repo, [("\(Self.folder)/vocab.json", 120), ("\(Self.folder)/merges.txt", 80)])
        serveFile(repo, "\(Self.folder)/vocab.json", size: 120)
        hub.serve("\(repo)/resolve/\(Self.revision)/\(Self.folder)/merges.txt", body: Data(), status: 404)

        await #expect(throws: ModelDownloadError.self) {
            try await self.download(repo, include: ["vocab.json", "merges.txt"])
        }
        let directory = ModelDownloads.localFolder(repoID: repo, folder: Self.folder, root: root)
        #expect(FileManager.default.fileExists(atPath: directory.appending(path: "vocab.json").path))
        #expect(!ModelDownloads.isFolderComplete(at: directory, revision: Self.revision))
    }

    @Test func completeFolderMissingAComponentIsFetchedAgain() async throws {
        let repo = "stub/purged-folder"
        serveModelFolder(repo)
        let directory = try await download(repo)
        try FileManager.default.removeItem(at: directory.appending(path: "merges.txt"))
        let before = hub.requests(containing: "\(repo)/resolve").count

        _ = try await download(repo)

        #expect(FileManager.default.fileExists(atPath: directory.appending(path: "merges.txt").path))
        #expect(ModelDownloads.isFolderComplete(at: directory, revision: Self.revision))
        let refetched = hub.requests(containing: "\(repo)/resolve").dropFirst(before)
            .compactMap(\.url?.lastPathComponent)
        #expect(refetched == ["merges.txt"])
    }

    @Test func resumesFromThePartialFile() async throws {
        let repo = "stub/resume-folder"
        let path = "\(Self.folder)/vocab.json"
        serveTree(repo, [(path, 1000)])
        serveFile(repo, path, size: 1000)
        let destination = ModelDownloads.snapshotDirectory(repoID: repo, root: root).appending(path: path)
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Self.payload(1000, salt: path.count).prefix(400).write(to: HubFileTransfer.partialURL(for: destination))
        try Data(Self.revision.utf8).write(to: destination.deletingLastPathComponent()
            .appendingPathComponent(ModelDownloads.revisionMarker))

        let plan = try await ModelDownloads.planFolder(repoID: repo, revision: Self.revision, folder: Self.folder,
                                                       include: ["vocab.json"], transfer: transfer(), root: root)
        #expect(plan.bytesOnDisk == 400)
        _ = try await download(repo, include: ["vocab.json"])

        let ranges = hub.requests(containing: "\(repo)/resolve").compactMap { $0.value(forHTTPHeaderField: "Range") }
        #expect(ranges == ["bytes=400-999"])
        #expect(try Data(contentsOf: destination) == Self.payload(1000, salt: path.count))
    }

    // MARK: Revisions

    @Test func aFolderFinishedAtAnotherRevisionIsNotComplete() async throws {
        let repo = "stub/revision-marker"
        serveModelFolder(repo)

        let directory = try await download(repo)

        #expect(ModelDownloads.isFolderComplete(at: directory, revision: Self.revision))
        #expect(!ModelDownloads.isFolderComplete(at: directory, revision: "fedcba9"))
    }

    @Test func anotherRevisionIsFetchedAfreshNotMixedWithTheOld() async throws {
        let repo = "stub/revision-bump"
        let newRevision = "fedcba9"
        serveModelFolder(repo)
        serveModelFolder(repo, revision: newRevision)
        let directory = try await download(repo)
        try Data(count: 4).write(to: directory.appending(path: "stale.bin"))

        let plan = try await ModelDownloads.planFolder(repoID: repo, revision: newRevision, folder: Self.folder,
                                                       include: Self.include, transfer: transfer(), root: root)
        _ = try await download(repo, revision: newRevision)

        #expect(plan.bytesOnDisk == 0)
        let fetched = hub.requests(containing: "\(repo)/resolve/\(newRevision)").compactMap(\.url?.lastPathComponent)
        #expect(fetched.sorted() == ["merges.txt", "model.mil", "vocab.json", "weight.bin"])
        let vocabPath = "\(Self.folder)/vocab.json"
        #expect(try Data(contentsOf: directory.appending(path: "vocab.json"))
                == Self.payload(120, salt: Self.salt(vocabPath, newRevision)))
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: "stale.bin").path))
        #expect(ModelDownloads.isFolderComplete(at: directory, revision: newRevision))
        #expect(!ModelDownloads.isFolderComplete(at: directory, revision: Self.revision))
    }

    @Test func aPartialFromAnotherRevisionIsDiscarded() async throws {
        let repo = "stub/revision-partial"
        let path = "\(Self.folder)/vocab.json"
        serveTree(repo, [(path, 120)])
        serveFile(repo, path, size: 120)
        let directory = ModelDownloads.localFolder(repoID: repo, folder: Self.folder, root: root)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("0ld0ld0".utf8).write(to: directory.appendingPathComponent(ModelDownloads.revisionMarker))
        try Data(count: 120).write(to: directory.appending(path: "vocab.json"))

        _ = try await download(repo, include: ["vocab.json"])

        #expect(try Data(contentsOf: directory.appending(path: "vocab.json"))
                == Self.payload(120, salt: Self.salt(path, Self.revision)))
        #expect(hub.requests(containing: "\(repo)/resolve").count == 1)
    }

    @Test func insufficientSpaceThrowsBeforeWriting() async throws {
        let repo = "stub/space-folder"
        serveModelFolder(repo)

        await #expect(throws: ModelDownloadError.insufficientSpace(neededBytes: 1680, availableBytes: 100)) {
            try await ModelDownloads.downloadFolder(repoID: repo, revision: Self.revision, folder: Self.folder,
                                                    include: Self.include, transfer: self.transfer(freeBytes: 100),
                                                    root: self.root) { _, _ in }
        }
        #expect(hub.requests(containing: "\(repo)/resolve").isEmpty)
    }

    // MARK: Path safety

    @Test(arguments: [
        "split/compiled/../escape.bin", "../escape.bin", "/abs/escape.bin", "split/compiled/a\\b.bin",
        "other/escape.bin", "split/compiled/./x.bin", "split/compiled//x.bin", "split/compiled/x\0.bin",
    ])
    func unsafeListedPathsThrowBeforeAnyWrite(_ unsafePath: String) async throws {
        let repo = "stub/unsafe-\(abs(unsafePath.hashValue))"
        serveTree(repo, [("\(Self.folder)/vocab.json", 10), (unsafePath, 10)])
        serveFile(repo, "\(Self.folder)/vocab.json", size: 10)

        await #expect(throws: ModelDownloadError.self) {
            try await self.download(repo, include: ["vocab.json"])
        }
        #expect(hub.requests(containing: "\(repo)/resolve").isEmpty)
        #expect(!FileManager.default.fileExists(atPath: ModelDownloads.repoFolder(repoID: repo, root: root).path))
        #expect(!ModelDownloads.isSafeListedPath(unsafePath, under: Self.folder))
    }

    @Test func unsafeArgumentsAreRejected() async throws {
        await #expect(throws: ModelDownloadError.self) {
            try await ModelDownloads.downloadFolder(repoID: "stub/x", revision: Self.revision, folder: "../up",
                                                    include: ["a"], transfer: self.transfer(),
                                                    root: self.root) { _, _ in }
        }
        await #expect(throws: ModelDownloadError.self) {
            try await self.download("stub/x", include: ["../../a"])
        }
        #expect(throws: ModelDownloadError.self) { try ModelDownloads.deleteRepo("../..", root: self.root) }
        #expect(throws: ModelDownloadError.self) { try ModelDownloads.destination(for: "a/../../b", in: self.root) }
        #expect(hub.requests(containing: "stub/x").isEmpty)
    }

    // MARK: Auth and delete

    @Test func noDownloadAuthSendsNoAuthorizationHeader() async throws {
        let repo = "stub/anon-folder"
        serveModelFolder(repo)

        _ = try await download(repo)

        let sent = hub.requests(containing: repo)
        #expect(!sent.isEmpty)
        #expect(sent.allSatisfy { $0.value(forHTTPHeaderField: "Authorization") == nil })

        _ = try await ModelDownloads.planFolder(repoID: repo, revision: Self.revision, folder: Self.folder,
                                                include: Self.include,
                                                transfer: transfer(auth: FolderTokenAuth(token: "hf_x")), root: root)
        #expect(hub.requests(containing: "api/models/\(repo)").last?.value(forHTTPHeaderField: "Authorization")
                == "Bearer hf_x")
    }

    @Test func deleteRepoRemovesFinishedAndPartialFiles() async throws {
        let repo = "stub/delete-folder"
        serveModelFolder(repo)
        let directory = try await download(repo)
        try Data(count: 4).write(to: HubFileTransfer.partialURL(for: directory.appending(path: "extra.bin")))

        try ModelDownloads.deleteRepo(repo, root: root)

        #expect(!FileManager.default.fileExists(atPath: ModelDownloads.repoFolder(repoID: repo, root: root).path))
        #expect(!ModelDownloads.isFolderComplete(at: directory, revision: Self.revision))
    }

    // MARK: Delete one folder

    /// A finished folder written straight to disk: a component, both markers and a partial file.
    private func makeLocalFolder(_ repoID: String, _ folder: String) throws -> URL {
        let directory = ModelDownloads.localFolder(repoID: repoID, folder: folder, root: root)
        try FileManager.default.createDirectory(at: directory.appending(path: "Unet.mlmodelc"),
                                                withIntermediateDirectories: true)
        try Data(count: 8).write(to: directory.appending(path: "Unet.mlmodelc/model.mil"))
        try Data(count: 4).write(to: HubFileTransfer.partialURL(for: directory.appending(path: "vocab.json")))
        try Data(Self.revision.utf8).write(to: directory.appendingPathComponent(ModelDownloads.revisionMarker))
        try Data(Self.revision.utf8).write(to: directory.appendingPathComponent(ModelDownloads.completeMarker))
        return directory
    }

    @Test func deleteFolderKeepsSiblingFolders() throws {
        let repo = "stub/two-folders"
        let removed = try makeLocalFolder(repo, "original/compiled")
        let kept = try makeLocalFolder(repo, "split_einsum_v2/compiled")

        try ModelDownloads.deleteFolder(repoID: repo, folder: "original/compiled", root: root)

        #expect(!FileManager.default.fileExists(atPath: removed.path))
        #expect(!FileManager.default.fileExists(atPath: removed.deletingLastPathComponent().path),
                "the emptied parent goes too")
        #expect(ModelDownloads.isFolderComplete(at: kept, revision: Self.revision))
        #expect(FileManager.default.fileExists(atPath: kept.appending(path: "Unet.mlmodelc/model.mil").path))
    }

    @Test func deleteFolderRemovesTheRepoWithItsLastFolderAndKeepsTheHub() throws {
        let repo = "stub/last-folder"
        let directory = try makeLocalFolder(repo, "compiled")
        let other = try makeLocalFolder("stub/neighbour", "compiled")

        try ModelDownloads.deleteFolder(repoID: repo, folder: "compiled", root: root)

        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(!FileManager.default.fileExists(atPath: ModelDownloads.repoFolder(repoID: repo, root: root).path))
        #expect(FileManager.default.fileExists(atPath: other.path))
        let hub = ModelDownloads.repoFolder(repoID: repo, root: root).deletingLastPathComponent()
        #expect(FileManager.default.fileExists(atPath: hub.path))
    }

    @Test func deleteFolderRejectsUnsafeFoldersAndIgnoresAbsentOnes() throws {
        for bad in ["", "..", "a/../..", "/abs/compiled"] {
            #expect(throws: ModelDownloadError.self) {
                try ModelDownloads.deleteFolder(repoID: "stub/unsafe", folder: bad, root: self.root)
            }
        }
        #expect(throws: ModelDownloadError.self) {
            try ModelDownloads.deleteFolder(repoID: "../..", folder: "compiled", root: self.root)
        }
        try ModelDownloads.deleteFolder(repoID: "stub/absent", folder: "compiled", root: root)
        let absent = ModelDownloads.repoFolder(repoID: "stub/absent", root: root)
        #expect(!FileManager.default.fileExists(atPath: absent.path))
    }
}

// MARK: - Stubs

/// Adds a bearer token for `huggingface.co` only.
private struct FolderTokenAuth: DownloadAuthorizing {
    let token: String

    func headers(for url: URL) -> [String: String] {
        url.host == "huggingface.co" ? ["Authorization": "Bearer \(token)"] : [:]
    }
}

private final class FolderProgressLog: Sendable {
    private let entries = Mutex<[(Int64, Int64)]>([])

    func append(_ done: Int64, _ total: Int64) {
        entries.withLock { $0.append((done, total)) }
    }

    var last: (Int64, Int64)? { entries.withLock { $0.last } }
}

/// Serves ``FolderStubProtocol/hub``, a route table separate from `DownloadStubs.shared`.
private final class FolderStubProtocol: URLProtocol, @unchecked Sendable {
    static let hub = DownloadStubs()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        guard let url = request.url else { return }
        let reply = Self.hub.reply(to: request)
        if let response = HTTPURLResponse(url: url, statusCode: reply.status, httpVersion: "HTTP/1.1",
                                          headerFields: reply.headers) {
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        }
        client?.urlProtocol(self, didLoad: reply.body)
        client?.urlProtocolDidFinishLoading(self)
    }
}
