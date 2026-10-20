import CryptoKit
import Foundation
import Testing
@testable import TonamoxCore

/// SHA-256 verification of downloads, and reading a model an earlier version left in Caches, against a stubbed
/// Hugging Face with its own route table (no network).
@Suite(.serialized)
final class ModelChecksumTests {
    private let base: URL
    private let root: URL
    private let legacy: URL
    private let hub = ChecksumStubProtocol.hub

    init() throws {
        base = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelChecksumTests-\(UUID().uuidString)", isDirectory: true)
        root = base.appending(path: "Models", directoryHint: .isDirectory)
        legacy = base.appending(path: "Caches", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: legacy, withIntermediateDirectories: true)
        hub.reset()
    }

    deinit {
        try? FileManager.default.removeItem(at: base)
    }

    // MARK: Fixtures

    private static func stubConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ChecksumStubProtocol.self]
        return configuration
    }

    private static let stubSession = URLSession(configuration: stubConfiguration())

    private func transfer() -> HubFileTransfer {
        HubFileTransfer(session: Self.stubSession, auth: NoDownloadAuth(), freeSpace: { _ in nil },
                        retryDelay: { _ in .zero }, progressInterval: .milliseconds(10))
    }

    private static func payload(_ count: Int, salt: Int = 0) -> Data {
        Data((0..<count).map { UInt8(($0 + salt) % 251) })
    }

    private static func sha256(_ data: Data) -> String {
        FileIntegrity.hex(SHA256.hash(data: data))
    }

    private static func model(repoID: String, gguf file: String?) throws -> Model {
        let entry = CatalogEntry(id: "checksum_\(file ?? "mlx")", repoID: repoID, displayName: "Checksum test",
                                 modelCategory: .text, weightFormat: file == nil ? .mlx : .gguf,
                                 approximateSizeMB: 1, isUncensored: false, ggufFilename: file, numLayers: 24,
                                 kvHeads: 2, headDim: 64, maxContextLength: nil)
        return try #require(entry.decodedModel())
    }

    private static func entries(_ files: [(path: String, body: Data, sha256: String?)]) -> Data {
        let list = files.map { file -> [String: Any] in
            var entry: [String: Any] = ["type": "file", "path": file.path, "size": file.body.count]
            if let sha256 = file.sha256 { entry["lfs"] = ["oid": sha256, "size": file.body.count] }
            return entry
        }
        return (try? JSONSerialization.data(withJSONObject: list)) ?? Data()
    }

    private func serveGGUF(_ repoID: String, _ file: String, body: Data, published sha256: String?) {
        hub.serve("\(repoID)/resolve/main/\(file)", body: body)
        hub.serve("api/models/\(repoID)/paths-info/main", body: Self.entries([(file, body, sha256)]))
    }

    private func serveSnapshot(_ repoID: String, _ files: [(path: String, body: Data, sha256: String?)]) {
        hub.serve("api/models/\(repoID)/tree/main", body: Self.entries(files))
        files.forEach { hub.serve("\(repoID)/resolve/main/\($0.path)", body: $0.body) }
    }

    private func write(_ data: Data, to file: URL) throws {
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: file)
    }

    // MARK: Hashing

    @Test func sha256MatchesKnownVectorsAndCryptoKitAcrossReads() throws {
        let abc = base.appending(path: "abc")
        let empty = base.appending(path: "empty")
        let large = base.appending(path: "large")
        try write(Data("abc".utf8), to: abc)
        try write(Data(), to: empty)
        let body = Self.payload(FileIntegrity.readSize * 2 + 3)
        try write(body, to: large)

        #expect(try FileIntegrity.sha256(of: abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        #expect(try FileIntegrity.sha256(of: empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        #expect(try FileIntegrity.sha256(of: large) == Self.sha256(body))
    }

    @Test func verifyRecordsAMatchUntilTheSizeChanges() throws {
        let file = base.appending(path: "m.bin")
        let body = Self.payload(100)
        try write(body, to: file)

        try FileIntegrity.verify(file, expectedSHA256: Self.sha256(body), path: "m.bin")
        #expect(ModelDownloads.verifiedSHA256(of: file) == Self.sha256(body))

        try write(Self.payload(101), to: file)
        #expect(ModelDownloads.verifiedSHA256(of: file) == nil)
    }

    @Test func verifyDeletesAMismatchAndThrows() throws {
        let file = base.appending(path: "m.bin")
        let body = Self.payload(100)
        try write(body, to: file)
        let wrong = Self.sha256(Self.payload(100, salt: 1))

        #expect(throws: ModelIntegrityError.checksumMismatch(path: "m.bin", expected: wrong,
                                                             actual: Self.sha256(body))) {
            try FileIntegrity.verify(file, expectedSHA256: wrong, path: "m.bin")
        }
        #expect(!FileManager.default.fileExists(atPath: file.path))
    }

    @Test func aRecordedFileIsReusedWithoutHashingAgain() throws {
        let file = base.appending(path: "m.bin")
        try write(Self.payload(100), to: file)
        // Not the content's hash: reuse can only succeed by trusting the record.
        let recorded = String(repeating: "a", count: 64)
        FileIntegrity.recordVerified(file, sha256: recorded)

        #expect(try FileIntegrity.reusable(file, expectedSHA256: recorded))
        #expect(FileManager.default.fileExists(atPath: file.path))
    }

    @Test func anUnrecordedFileThatHashesWrongIsNotReusedAndIsDeleted() throws {
        let file = base.appending(path: "m.bin")
        try write(Self.payload(100), to: file)

        #expect(try !FileIntegrity.reusable(file, expectedSHA256: Self.sha256(Self.payload(100, salt: 3))))
        #expect(!FileManager.default.fileExists(atPath: file.path))
        #expect(try FileIntegrity.reusable(file, expectedSHA256: nil))
    }

    @Test func plannedFilesCompareByPathAndSizeOnly() {
        let listed = PlannedFile(path: "model.safetensors", sizeBytes: 10, sha256: String(repeating: "a", count: 64))
        #expect(listed == PlannedFile(path: "model.safetensors", sizeBytes: 10))
        #expect(listed != PlannedFile(path: "model.safetensors", sizeBytes: 11))
    }

    @Test func onlySixtyFourHexDigitsCountAsASHA256() {
        let upper = String(repeating: "AB", count: 32)
        #expect(FileIntegrity.normalizedSHA256(upper) == upper.lowercased())
        #expect(FileIntegrity.normalizedSHA256(String(repeating: "a", count: 63)) == nil)
        #expect(FileIntegrity.normalizedSHA256(String(repeating: "g", count: 64)) == nil)
        #expect(FileIntegrity.normalizedSHA256("a6344aac8c09253b3b630fb776ae94478aa0275b") == nil)
        #expect(FileIntegrity.normalizedSHA256(nil) == nil)
    }

    @Test func resolveURLsNameRepoRevisionAndPath() throws {
        let nested = try #require(URL(string: "https://huggingface.co/org/repo/resolve/abc123/sub/m.gguf?download=1"))
        #expect(HubFileTransfer.hubFile(at: nested)
                == HubFileReference(repoID: "org/repo", revision: "abc123", path: "sub/m.gguf"))
        let short = try #require(URL(string: "https://hf.co/org/repo/resolve/main/m.gguf"))
        #expect(HubFileTransfer.hubFile(at: short)?.path == "m.gguf")
        #expect(HubFileTransfer.hubFile(at: try #require(URL(string: "https://example.com/org/repo/resolve/main/m.gguf")))
                == nil)
        #expect(HubFileTransfer.hubFile(at: try #require(URL(string: "https://huggingface.co/org/repo/blob/main/m.gguf")))
                == nil)
    }

    @Test func aBranchWithSlashesStaysOneRevisionSegment() throws {
        let url = try #require(URL(string: "https://huggingface.co/org/repo/resolve/refs%2Fpr%2F1/m.gguf"))
        let file = try #require(HubFileTransfer.hubFile(at: url))
        #expect(file == HubFileReference(repoID: "org/repo", revision: "refs/pr/1", path: "m.gguf"))
        #expect(HubFileTransfer.pathsInfoURL(repoID: file.repoID, revision: file.revision)?.absoluteString
                == "https://huggingface.co/api/models/org/repo/paths-info/refs%2Fpr%2F1")
        #expect(HubFileTransfer.treeURL(repoID: "org/repo", revision: "main", path: "a b/c", recursive: true)?
            .absoluteString == "https://huggingface.co/api/models/org/repo/tree/main/a%20b/c?recursive=true")
    }

    // MARK: GGUF

    @Test func aGGUFDownloadIsVerifiedAgainstPathsInfo() async throws {
        let repo = "stub/verified-gguf"
        let body = Self.payload(5000)
        serveGGUF(repo, "m.gguf", body: body, published: Self.sha256(body))
        let model = try Self.model(repoID: repo, gguf: "m.gguf")

        let file = try await ModelDownloads.download(model, transfer: transfer(), root: root) { _, _ in }

        #expect(try Data(contentsOf: file) == body)
        #expect(ModelDownloads.verifiedSHA256(of: file) == Self.sha256(body))
        let lookup = try #require(hub.requests(containing: "paths-info").first)
        #expect(lookup.httpMethod == "POST")
        #expect(lookup.url?.absoluteString == "https://huggingface.co/api/models/\(repo)/paths-info/main")
    }

    @Test func aGGUFThatDoesNotMatchIsDeletedAndThrows() async throws {
        let repo = "stub/tampered-gguf"
        let body = Self.payload(5000)
        let published = Self.sha256(Self.payload(5000, salt: 9))
        serveGGUF(repo, "m.gguf", body: body, published: published)
        let model = try Self.model(repoID: repo, gguf: "m.gguf")

        await #expect(throws: ModelIntegrityError.checksumMismatch(path: "m.gguf", expected: published,
                                                                   actual: Self.sha256(body))) {
            try await ModelDownloads.download(model, transfer: self.transfer(), root: self.root) { _, _ in }
        }
        let file = root.appending(path: "models/\(repo)/m.gguf")
        #expect(!FileManager.default.fileExists(atPath: file.path))
        #expect(!FileManager.default.fileExists(atPath: HubFileTransfer.partialURL(for: file).path))
    }

    @Test func aResumedGGUFIsHashedWhole() async throws {
        let repo = "stub/resumed-gguf"
        let body = Self.payload(5000)
        serveGGUF(repo, "m.gguf", body: body, published: Self.sha256(body))
        let model = try Self.model(repoID: repo, gguf: "m.gguf")
        let file = root.appending(path: "models/\(repo)/m.gguf")
        try write(body.prefix(1200), to: HubFileTransfer.partialURL(for: file))

        _ = try await ModelDownloads.download(model, transfer: transfer(), root: root) { _, _ in }

        let ranges = hub.requests(containing: "resumed-gguf/resolve").compactMap { $0.value(forHTTPHeaderField: "Range") }
        #expect(ranges == ["bytes=1200-4999"])
        #expect(ModelDownloads.verifiedSHA256(of: file) == Self.sha256(body))
    }

    @Test func aGGUFPartialLeftInCachesIsMovedIntoTheRootAndResumed() async throws {
        let repo = "stub/legacy-partial-gguf"
        let body = Self.payload(5000)
        serveGGUF(repo, "m.gguf", body: body, published: Self.sha256(body))
        let model = try Self.model(repoID: repo, gguf: "m.gguf")
        let oldPartial = HubFileTransfer.partialURL(for: legacy.appending(path: "models/\(repo)/m.gguf"))
        try write(body.prefix(1200), to: oldPartial)

        let file = try await ModelDownloads.download(model, transfer: transfer(), root: root,
                                                     fallbackRoots: [legacy]) { _, _ in }

        let ranges = hub.requests(containing: "\(repo)/resolve").compactMap { $0.value(forHTTPHeaderField: "Range") }
        #expect(ranges == ["bytes=1200-4999"])
        #expect(file == root.appending(path: "models/\(repo)/m.gguf"))
        #expect(ModelDownloads.verifiedSHA256(of: file) == Self.sha256(body))
        #expect(!FileManager.default.fileExists(atPath: oldPartial.path))
        #expect(!FileManager.default.fileExists(atPath: legacy.appending(path: "models").path))
        #expect(FileManager.default.fileExists(atPath: legacy.path))
    }

    @Test func aPartialInTheRootIsResumedAndTheOneInCachesLeftAlone() async throws {
        let repo = "stub/both-partials-gguf"
        let body = Self.payload(5000)
        serveGGUF(repo, "m.gguf", body: body, published: Self.sha256(body))
        let model = try Self.model(repoID: repo, gguf: "m.gguf")
        let oldPartial = HubFileTransfer.partialURL(for: legacy.appending(path: "models/\(repo)/m.gguf"))
        try write(body.prefix(2000), to: oldPartial)
        try write(body.prefix(1200), to: HubFileTransfer.partialURL(for: root.appending(path: "models/\(repo)/m.gguf")))

        _ = try await ModelDownloads.download(model, transfer: transfer(), root: root,
                                              fallbackRoots: [legacy]) { _, _ in }

        let ranges = hub.requests(containing: "\(repo)/resolve").compactMap { $0.value(forHTTPHeaderField: "Range") }
        #expect(ranges == ["bytes=1200-4999"])
        #expect(HubFileTransfer.fileSize(at: oldPartial) == 2000)
    }

    @Test func aGGUFWithoutAPublishedSHA256IsKeptUnverified() async throws {
        let repo = "stub/unpublished-gguf"
        let body = Self.payload(300)
        serveGGUF(repo, "m.gguf", body: body, published: nil)
        let model = try Self.model(repoID: repo, gguf: "m.gguf")

        let file = try await ModelDownloads.download(model, transfer: transfer(), root: root) { _, _ in }

        #expect(try Data(contentsOf: file) == body)
        #expect(ModelDownloads.verifiedSHA256(of: file) == nil)
    }

    @Test func aGGUFLeftInCachesIsReturnedWithoutTheNetwork() async throws {
        let repo = "stub/legacy-gguf"
        let model = try Self.model(repoID: repo, gguf: "m.gguf")
        let old = legacy.appending(path: "models/\(repo)/m.gguf")
        try write(Self.payload(40), to: old)

        let file = try await ModelDownloads.download(model, transfer: transfer(), root: root,
                                                     fallbackRoots: [legacy]) { _, _ in }

        #expect(file == old)
        #expect(hub.requests(containing: repo).isEmpty)
    }

    // MARK: MLX snapshots

    @Test func snapshotFilesAreVerifiedFromTheTree() async throws {
        let repo = "stub/verified-mlx"
        let weights = Self.payload(3000)
        let config = Self.payload(20, salt: 4)
        serveSnapshot(repo, [("model.safetensors", weights, Self.sha256(weights)), ("config.json", config, nil)])
        let model = try Self.model(repoID: repo, gguf: nil)

        let directory = try await ModelDownloads.download(model, transfer: transfer(), root: root) { _, _ in }

        #expect(FileManager.default.fileExists(atPath: directory.appending(path: ModelDownloads.completeMarker).path))
        #expect(ModelDownloads.verifiedSHA256(of: directory.appending(path: "model.safetensors")) == Self.sha256(weights))
        #expect(ModelDownloads.verifiedSHA256(of: directory.appending(path: "config.json")) == nil)
    }

    @Test func aSnapshotMismatchWritesNoMarker() async throws {
        let repo = "stub/tampered-mlx"
        let weights = Self.payload(3000)
        let published = Self.sha256(Self.payload(3000, salt: 2))
        serveSnapshot(repo, [("model.safetensors", weights, published), ("config.json", Self.payload(20), nil)])
        let model = try Self.model(repoID: repo, gguf: nil)

        await #expect(throws: ModelIntegrityError.self) {
            try await ModelDownloads.download(model, transfer: self.transfer(), root: self.root) { _, _ in }
        }
        let directory = ModelDownloads.snapshotDirectory(repoID: repo, root: root)
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: ModelDownloads.completeMarker).path))
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: "model.safetensors").path))
    }

    @Test func aWrongFinishedFileOfTheRightSizeIsFetchedAgain() async throws {
        let repo = "stub/stale-mlx"
        let weights = Self.payload(3000)
        serveSnapshot(repo, [("model.safetensors", weights, Self.sha256(weights)), ("config.json", Self.payload(20), nil)])
        let directory = ModelDownloads.snapshotDirectory(repoID: repo, root: root)
        try write(Self.payload(3000, salt: 5), to: directory.appending(path: "model.safetensors"))
        let model = try Self.model(repoID: repo, gguf: nil)

        _ = try await ModelDownloads.download(model, transfer: transfer(), root: root) { _, _ in }

        #expect(try Data(contentsOf: directory.appending(path: "model.safetensors")) == weights)
        #expect(ModelDownloads.verifiedSHA256(of: directory.appending(path: "model.safetensors")) == Self.sha256(weights))
    }

    @Test func aSnapshotCompleteInCachesIsReturnedWithoutTheNetwork() async throws {
        let repo = "stub/legacy-mlx"
        let old = ModelDownloads.snapshotDirectory(repoID: repo, root: legacy)
        try write(Self.payload(30), to: old.appending(path: "model.safetensors"))
        try write(Data(), to: old.appending(path: ModelDownloads.completeMarker))
        let model = try Self.model(repoID: repo, gguf: nil)

        let directory = try await ModelDownloads.download(model, transfer: transfer(), root: root,
                                                          fallbackRoots: [legacy]) { _, _ in }

        #expect(directory == old)
        #expect(hub.requests(containing: repo).isEmpty)
    }

    // MARK: Folders

    @Test func aFolderMismatchWritesNoMarker() async throws {
        let repo = "stub/tampered-folder"
        let weights = Self.payload(900)
        let published = Self.sha256(Self.payload(900, salt: 1))
        let files: [(path: String, body: Data, sha256: String?)] = [("compiled/weight.bin", weights, published)]
        hub.serve("api/models/\(repo)/tree/rev1/compiled?recursive=true", body: Self.entries(files))
        hub.serve("\(repo)/resolve/rev1/compiled/weight.bin", body: weights)

        await #expect(throws: ModelIntegrityError.self) {
            try await ModelDownloads.downloadFolder(repoID: repo, revision: "rev1", folder: "compiled",
                                                    include: ["weight.bin"], transfer: self.transfer(),
                                                    root: self.root) { _, _ in }
        }
        let directory = ModelDownloads.localFolder(repoID: repo, folder: "compiled", root: root)
        #expect(!ModelDownloads.isFolderComplete(at: directory, revision: "rev1"))
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: "weight.bin").path))
    }

    @Test func aFolderVerifiesAndCompletes() async throws {
        let repo = "stub/verified-folder"
        let weights = Self.payload(900)
        let files: [(path: String, body: Data, sha256: String?)] = [("compiled/weight.bin", weights,
                                                                      Self.sha256(weights))]
        hub.serve("api/models/\(repo)/tree/rev1/compiled?recursive=true", body: Self.entries(files))
        hub.serve("\(repo)/resolve/rev1/compiled/weight.bin", body: weights)

        let directory = try await ModelDownloads.downloadFolder(repoID: repo, revision: "rev1", folder: "compiled",
                                                                include: ["weight.bin"], transfer: transfer(),
                                                                root: root) { _, _ in }

        #expect(ModelDownloads.isFolderComplete(at: directory, revision: "rev1"))
        #expect(ModelDownloads.verifiedSHA256(of: directory.appending(path: "weight.bin")) == Self.sha256(weights))
    }

    @Test func aFolderCompleteInCachesIsReturnedWithoutTheNetwork() async throws {
        let repo = "stub/legacy-folder"
        let old = ModelDownloads.localFolder(repoID: repo, folder: "compiled", root: legacy)
        try write(Self.payload(10), to: old.appending(path: "weight.bin"))
        try write(Data("rev1".utf8), to: old.appending(path: ModelDownloads.completeMarker))

        let directory = try await ModelDownloads.downloadFolder(repoID: repo, revision: "rev1", folder: "compiled",
                                                                include: ["weight.bin"], transfer: transfer(),
                                                                root: root, fallbackRoots: [legacy]) { _, _ in }

        #expect(directory == old)
        #expect(hub.requests(containing: repo).isEmpty)
    }

    @Test func aFolderCompleteInBothRootsIsReturnedFromTheStorageRoot() async throws {
        let repo = "stub/both-folder"
        let current = ModelDownloads.localFolder(repoID: repo, folder: "compiled", root: root)
        let old = ModelDownloads.localFolder(repoID: repo, folder: "compiled", root: legacy)
        let weights = Self.payload(10)
        let completed = Data("rev1".utf8)
        for folder in [current, old] {
            try write(weights, to: folder.appending(path: "weight.bin"))
            try write(completed, to: folder.appending(path: ModelDownloads.completeMarker))
        }

        let directory = try await ModelDownloads.downloadFolder(repoID: repo, revision: "rev1", folder: "compiled",
                                                                include: ["weight.bin"], transfer: transfer(),
                                                                root: root, fallbackRoots: [legacy]) { _, _ in }

        #expect(directory == current)
        #expect(hub.requests(containing: repo).isEmpty)
    }

    // MARK: GGUFModelDownloader

    /// Downloads into the real storage root, so the repo id is unique and its folder removed afterwards.
    @MainActor
    @Test(arguments: [true, false])
    func theLoadPathDownloaderVerifiesToo(matches: Bool) async throws {
        let repo = "auralocal-tests/checksum-\(UUID().uuidString.lowercased())"
        let folder = DownloadedModels.modelsDirectory.appending(path: repo)
        defer {
            try? FileManager.default.removeItem(at: folder)
            ModelStorage.removeEmptyFolders(from: folder.deletingLastPathComponent(),
                                            upTo: DownloadedModels.modelsDirectory)
        }
        let body = Self.payload(4000)
        let published = matches ? Self.sha256(body) : Self.sha256(Self.payload(4000, salt: 6))
        serveGGUF(repo, "m.gguf", body: body, published: published)
        let model = try Self.model(repoID: repo, gguf: "m.gguf")
        let downloader = GGUFModelDownloader(auth: NoDownloadAuth(), sessionConfiguration: Self.stubConfiguration(),
                                             checksumSession: Self.stubSession)
        let file = folder.appending(path: "m.gguf")

        if matches {
            let downloaded = try await downloader.download(model: model) { _ in }
            #expect(downloaded == file)
            #expect(ModelDownloads.verifiedSHA256(of: file) == published)
        } else {
            await #expect(throws: ModelIntegrityError.self) {
                _ = try await downloader.download(model: model) { _ in }
            }
            #expect(!FileManager.default.fileExists(atPath: file.path))
            #expect(!FileManager.default.fileExists(atPath: HubFileTransfer.partialURL(for: file).path))
        }
    }
}

// MARK: - Stub

/// Serves ``ChecksumStubProtocol/hub``, a route table separate from `DownloadStubs.shared`.
private final class ChecksumStubProtocol: URLProtocol, @unchecked Sendable {
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
