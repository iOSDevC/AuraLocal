import Foundation
import Synchronization
import Testing
@testable import AuraCore

/// `ModelDownloads` against a stubbed Hugging Face: no network, a temp Caches root per test.
@Suite(.serialized)
final class ModelDownloadsTests {
    private let root: URL
    private let hub = DownloadStubs.shared

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelDownloadsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        hub.reset()
    }

    deinit {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: Fixtures

    private static let stubSession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [DownloadStubProtocol.self]
        return URLSession(configuration: configuration)
    }()

    private func transfer(auth: any DownloadAuthorizing = NoDownloadAuth(),
                          freeBytes: Int64? = nil) -> HubFileTransfer {
        HubFileTransfer(session: Self.stubSession, auth: auth, freeSpace: { _ in freeBytes },
                        retryDelay: { _ in .zero }, progressInterval: .milliseconds(10))
    }

    private static func ggufModel(repoID: String, file: String) throws -> Model {
        let entry = CatalogEntry(id: "test_gguf", repoID: repoID, displayName: "Test GGUF", modelCategory: .text,
                                 weightFormat: .gguf, approximateSizeMB: 1, isUncensored: false,
                                 ggufFilename: file, numLayers: 24, kvHeads: 2, headDim: 64,
                                 maxContextLength: nil)
        return try #require(entry.decodedModel())
    }

    private static func mlxModel(repoID: String) throws -> Model {
        let entry = CatalogEntry(id: "test_mlx", repoID: repoID, displayName: "Test MLX", modelCategory: .text,
                                 weightFormat: .mlx, approximateSizeMB: 1, isUncensored: false,
                                 ggufFilename: nil, numLayers: 24, kvHeads: 0, headDim: 0,
                                 maxContextLength: nil)
        return try #require(entry.decodedModel())
    }

    private static func payload(_ count: Int) -> Data {
        Data((0..<count).map { UInt8($0 % 251) })
    }

    private func ggufFile(_ repoID: String, _ file: String) -> URL {
        root.appending(path: "models/\(repoID)/\(file)")
    }

    private func serveTree(_ repoID: String, _ files: [(path: String, size: Int)], status: Int = 200,
                           errorCode: String? = nil) {
        let entries = files.map { ["type": "file", "path": $0.path, "size": $0.size] as [String: Any] }
        let body = (try? JSONSerialization.data(withJSONObject: entries)) ?? Data()
        hub.serve("api/models/\(repoID)/tree/main", body: body, status: status, errorCode: errorCode)
    }

    // MARK: GGUF plan

    @Test func ggufPlanUsesLinkedSizeFromHEAD() async throws {
        let model = try Self.ggufModel(repoID: "stub/plan-gguf", file: "m-Q4_K_M.gguf")
        hub.serve("stub/plan-gguf/resolve/main/m-Q4_K_M.gguf", body: Self.payload(10), linkedSize: 4_000_000)

        let plan = try await ModelDownloads.plan(for: model, transfer: transfer(), root: root)

        #expect(plan.files == [PlannedFile(path: "m-Q4_K_M.gguf", sizeBytes: 4_000_000)])
        #expect(plan.totalBytes == 4_000_000)
        #expect(plan.bytesOnDisk == 0)
        #expect(hub.requests(containing: "plan-gguf").first?.httpMethod == "HEAD")
    }

    @Test func ggufPlanMapsGatedAndMissingStatuses() async throws {
        let gated = try Self.ggufModel(repoID: "stub/gated-gguf", file: "g.gguf")
        let missing = try Self.ggufModel(repoID: "stub/missing-gguf", file: "x.gguf")
        hub.serve("stub/gated-gguf/resolve/main/g.gguf", body: Data(), status: 401, errorCode: "GatedRepo")

        await #expect(throws: ModelDownloadError.unauthorized(repoID: "stub/gated-gguf")) {
            try await ModelDownloads.plan(for: gated, transfer: self.transfer(), root: self.root)
        }
        await #expect(throws: ModelDownloadError.notFound(repoID: "stub/missing-gguf", path: "x.gguf")) {
            try await ModelDownloads.plan(for: missing, transfer: self.transfer(), root: self.root)
        }
    }

    @Test func anonymous401WithoutCodeIsMissingOrPrivateNotGated() async throws {
        let model = try Self.ggufModel(repoID: "stub/gone-gguf", file: "m.gguf")
        hub.serve("stub/gone-gguf/resolve/main/m.gguf", body: Data(), status: 401)

        await #expect(throws: ModelDownloadError.notFoundOrPrivate(repoID: "stub/gone-gguf")) {
            try await ModelDownloads.plan(for: model, transfer: self.transfer(), root: self.root)
        }
        await #expect(throws: ModelDownloadError.unauthorized(repoID: "stub/gone-gguf")) {
            try await ModelDownloads.plan(for: model, transfer: self.transfer(auth: FixedTokenAuth(token: "hf_x")),
                                          root: self.root)
        }
    }

    @Test func errorCodesOverrideTheStatus() throws {
        let url = try #require(URL(string: "https://huggingface.co/o/r/resolve/main/m.gguf"))
        func response(_ status: Int, _ code: String) throws -> HTTPURLResponse {
            try #require(HTTPURLResponse(url: url, statusCode: status, httpVersion: nil,
                                         headerFields: ["X-Error-Code": code]))
        }

        #expect(ModelDownloadError(response: try response(401, "RepoNotFound"), repoID: "o/r", path: "m.gguf",
                                   sentToken: false) == .notFound(repoID: "o/r", path: ""))
        #expect(ModelDownloadError(response: try response(404, "EntryNotFound"), repoID: "o/r", path: "m.gguf",
                                   sentToken: true) == .notFound(repoID: "o/r", path: "m.gguf"))
        #expect(ModelDownloadError(response: try response(403, "GatedRepo"), repoID: "o/r", path: "m.gguf",
                                   sentToken: true) == .unauthorized(repoID: "o/r"))
    }

    @Test func ggufTargetRejectsShardsAndSubfolders() throws {
        let shard = try Self.ggufModel(repoID: "stub/split", file: "m-Q8_0-00001-of-00002.gguf")
        let nested = try Self.ggufModel(repoID: "stub/nested", file: "Q4/m.gguf")

        #expect(throws: ModelDownloadError.self) { try ModelDownloads.ggufTarget(for: shard, root: self.root) }
        #expect(throws: ModelDownloadError.self) { try ModelDownloads.ggufTarget(for: nested, root: self.root) }
    }

    // MARK: GGUF download

    @Test func ggufDownloadLandsWhereTheBackendLoads() async throws {
        let model = try Self.ggufModel(repoID: "stub/dl-gguf", file: "m.gguf")
        let body = Self.payload(5000)
        hub.serve("stub/dl-gguf/resolve/main/m.gguf", body: body)
        let reports = ProgressLog()

        let file = try await ModelDownloads.download(model, transfer: transfer(), root: root) { done, total in
            reports.append(done, total)
        }

        #expect(file == ggufFile("stub/dl-gguf", "m.gguf"))
        let resolved = try #require(model.resolvedFileURL)
        #expect(resolved.path.replacingOccurrences(of: ModelDownloads.storageRoot.path, with: root.path) == file.path)
        #expect(try Data(contentsOf: file) == body)
        #expect(!FileManager.default.fileExists(atPath: HubFileTransfer.partialURL(for: file).path))
        let last = try #require(reports.last)
        #expect(last.0 == 5000 && last.1 == 5000)
    }

    @Test func ggufResumeSendsRangeFromThePartialSize() async throws {
        let model = try Self.ggufModel(repoID: "stub/resume-gguf", file: "m.gguf")
        let body = Self.payload(1000)
        hub.serve("stub/resume-gguf/resolve/main/m.gguf", body: body)
        let file = ggufFile("stub/resume-gguf", "m.gguf")
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try body.prefix(400).write(to: HubFileTransfer.partialURL(for: file))

        let plan = try await ModelDownloads.plan(for: model, transfer: transfer(), root: root)
        #expect(plan.bytesOnDisk == 400)
        #expect(plan.remainingBytes == 600)

        _ = try await ModelDownloads.download(model, transfer: transfer(), root: root) { _, _ in }

        let ranged = hub.requests(containing: "resume-gguf").filter { $0.httpMethod == "GET" }
        #expect(ranged.first?.value(forHTTPHeaderField: "Range") == "bytes=400-999")
        #expect(try Data(contentsOf: file) == body)
    }

    @Test func partialOfAnotherRevisionRestartsInsteadOfSplicing() async throws {
        let model = try Self.ggufModel(repoID: "stub/reupload-gguf", file: "m.gguf")
        let current = Data(Self.payload(5000).reversed())
        hub.serve("stub/reupload-gguf/resolve/main/m.gguf", body: current, etag: "\"new\"")
        let file = ggufFile("stub/reupload-gguf", "m.gguf")
        let partial = HubFileTransfer.partialURL(for: file)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.payload(3000).write(to: partial)
        HubFileTransfer.setValidator("\"old\"", of: partial)

        _ = try await ModelDownloads.download(model, transfer: transfer(), root: root) { _, _ in }

        #expect(try Data(contentsOf: file) == current)
        let ranges = hub.requests(containing: "reupload-gguf").filter { $0.httpMethod == "GET" }
            .compactMap { $0.value(forHTTPHeaderField: "Range") }
        #expect(ranges == ["bytes=3000-4999", "bytes=0-4999"])
    }

    @Test func partialOfTheSameRevisionResumes() async throws {
        let model = try Self.ggufModel(repoID: "stub/same-gguf", file: "m.gguf")
        let body = Self.payload(5000)
        hub.serve("stub/same-gguf/resolve/main/m.gguf", body: body, etag: "\"v1\"")
        let file = ggufFile("stub/same-gguf", "m.gguf")
        let partial = HubFileTransfer.partialURL(for: file)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try body.prefix(3000).write(to: partial)
        HubFileTransfer.setValidator("\"v1\"", of: partial)

        _ = try await ModelDownloads.download(model, transfer: transfer(), root: root) { _, _ in }

        #expect(try Data(contentsOf: file) == body)
        let ranges = hub.requests(containing: "same-gguf").filter { $0.httpMethod == "GET" }
            .compactMap { $0.value(forHTTPHeaderField: "Range") }
        #expect(ranges == ["bytes=3000-4999"])
    }

    @Test func partialLongerThanTheFileRestarts() async throws {
        let model = try Self.ggufModel(repoID: "stub/shrunk-gguf", file: "m.gguf")
        let body = Self.payload(1000)
        hub.serve("stub/shrunk-gguf/resolve/main/m.gguf", body: body)
        let file = ggufFile("stub/shrunk-gguf", "m.gguf")
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.payload(1500).write(to: HubFileTransfer.partialURL(for: file))

        _ = try await ModelDownloads.download(model, transfer: transfer(), root: root) { _, _ in }

        #expect(try Data(contentsOf: file) == body)
    }

    @Test func fullBodyForAResumedRangeIsRejectedNotAppended() async throws {
        let model = try Self.ggufModel(repoID: "stub/norange-gguf", file: "m.gguf")
        let body = Self.payload(1000)
        hub.serve("stub/norange-gguf/resolve/main/m.gguf", body: body, honoursRange: false)
        let file = ggufFile("stub/norange-gguf", "m.gguf")
        let partial = HubFileTransfer.partialURL(for: file)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try body.prefix(400).write(to: partial)

        await #expect(throws: ModelDownloadError.http(status: 200)) {
            try await ModelDownloads.download(model, transfer: self.transfer(), root: self.root) { _, _ in }
        }
        #expect(HubFileTransfer.fileSize(at: partial) == 400)
        #expect(!FileManager.default.fileExists(atPath: file.path))
    }

    @Test func shortBodyIsASizeMismatchAndNeverBecomesTheModel() async throws {
        let model = try Self.ggufModel(repoID: "stub/short-gguf", file: "m.gguf")
        hub.serve("stub/short-gguf/resolve/main/m.gguf", body: Self.payload(80), linkedSize: 100)

        await #expect(throws: ModelDownloadError.sizeMismatch(path: "m.gguf", expected: 100, actual: 80)) {
            try await ModelDownloads.download(model, transfer: self.transfer(), root: self.root) { _, _ in }
        }
        #expect(!FileManager.default.fileExists(atPath: ggufFile("stub/short-gguf", "m.gguf").path))
    }

    @Test func insufficientSpaceThrowsBeforeWritingAnything() async throws {
        let model = try Self.ggufModel(repoID: "stub/space-gguf", file: "m.gguf")
        hub.serve("stub/space-gguf/resolve/main/m.gguf", body: Self.payload(1000))

        await #expect(throws: ModelDownloadError.insufficientSpace(neededBytes: 1200, availableBytes: 500)) {
            try await ModelDownloads.download(model, transfer: self.transfer(freeBytes: 500), root: self.root) { _, _ in }
        }
        #expect(!FileManager.default.fileExists(atPath: root.appending(path: "models/stub/space-gguf").path))
        #expect(hub.requests(containing: "space-gguf").allSatisfy { $0.httpMethod == "HEAD" })
    }

    // MARK: MLX

    @Test func mlxFetchesOnlyLoadableTopLevelFilesThenMarksComplete() async throws {
        let repo = "stub/dl-mlx"
        let model = try Self.mlxModel(repoID: repo)
        serveTree(repo, [("config.json", 20), ("model.safetensors", 300), ("tokenizer.json", 40),
                         ("README.md", 10), ("img/a.png", 5)])
        hub.serve("\(repo)/resolve/main/config.json", body: Self.payload(20))
        hub.serve("\(repo)/resolve/main/model.safetensors", body: Self.payload(300))
        hub.serve("\(repo)/resolve/main/tokenizer.json", body: Self.payload(40))

        let directory = try await ModelDownloads.download(model, transfer: transfer(), root: root) { _, _ in }

        #expect(directory == ModelDownloads.snapshotDirectory(repoID: repo, root: root))
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
        #expect(names == [".complete", "config.json", "model.safetensors", "tokenizer.json"])
        #expect(hub.requests(containing: "README.md").isEmpty)
        #expect(hub.requests(containing: "a.png").isEmpty)
    }

    @Test func mlxGatedTreeThrowsAndLeavesNoMarker() async throws {
        let repo = "stub/gated-mlx"
        let model = try Self.mlxModel(repoID: repo)
        serveTree(repo, [], status: 401, errorCode: "GatedRepo")

        await #expect(throws: ModelDownloadError.unauthorized(repoID: repo)) {
            try await ModelDownloads.download(model, transfer: self.transfer(), root: self.root) { _, _ in }
        }
        let marker = ModelDownloads.snapshotDirectory(repoID: repo, root: root)
            .appendingPathComponent(ModelDownloads.completeMarker)
        #expect(!FileManager.default.fileExists(atPath: marker.path))
    }

    @Test func mlxRepoWithoutWeightsIsIncomplete() async throws {
        let repo = "stub/noweights-mlx"
        let model = try Self.mlxModel(repoID: repo)
        serveTree(repo, [("config.json", 20), ("README.md", 10)])

        await #expect(throws: ModelDownloadError.incompleteRepo(repoID: repo)) {
            try await ModelDownloads.plan(for: model, transfer: self.transfer(), root: self.root)
        }
    }

    @Test func tokenizerOnlySnapshotIsNotMarkedComplete() async throws {
        let repo = "stub/tokenizer-mlx"
        serveTree(repo, [("tokenizer.json", 40), ("model.safetensors", 300)])
        hub.serve("\(repo)/resolve/main/tokenizer.json", body: Self.payload(40))

        let directory = try await ModelDownloads.downloadSnapshot(
            repoID: repo, patterns: ["*.json"], revision: "main", transfer: transfer(), root: root) { _, _ in }

        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        #expect(names == ["tokenizer.json"])
    }

    // MARK: Delete

    @Test func ggufDeleteRemovesFilePartialAndEmptyFoldersOnly() throws {
        let model = try Self.ggufModel(repoID: "org/repo", file: "a.gguf")
        let file = ggufFile("org/repo", "a.gguf")
        let sibling = ggufFile("org/other", "b.gguf")
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: sibling.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(count: 10).write(to: file)
        try Data(count: 4).write(to: HubFileTransfer.partialURL(for: file))
        try Data(count: 10).write(to: sibling)

        try ModelDownloads.delete(model, root: root)

        #expect(!FileManager.default.fileExists(atPath: file.deletingLastPathComponent().path))
        #expect(FileManager.default.fileExists(atPath: sibling.path))
        #expect(ModelDownloads.bytesOnDisk(for: model, root: root) == 0)
    }

    @Test func ggufDeleteStopsAtTheModelsRoot() throws {
        let model = try Self.ggufModel(repoID: "solo/repo", file: "a.gguf")
        let file = ggufFile("solo/repo", "a.gguf")
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(count: 10).write(to: file)

        try ModelDownloads.delete(model, root: root)

        #expect(!FileManager.default.fileExists(atPath: root.appending(path: "models/solo").path))
        #expect(FileManager.default.fileExists(atPath: root.appending(path: "models").path))
    }

    @Test func mlxDeleteRemovesTheRepoFolder() throws {
        let model = try Self.mlxModel(repoID: "org/mlx-repo")
        let directory = ModelDownloads.snapshotDirectory(repoID: "org/mlx-repo", root: root)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(count: 10).write(to: directory.appendingPathComponent("model.safetensors"))

        try ModelDownloads.delete(model, root: root)

        #expect(!FileManager.default.fileExists(atPath: ModelDownloads.repoFolder(repoID: "org/mlx-repo", root: root).path))
    }

    // MARK: Auth

    @Test func tokenGoesToHuggingFaceButNotToARedirectedHost() async throws {
        let auth = FixedTokenAuth(token: "hf_test")
        let model = try Self.ggufModel(repoID: "stub/auth-gguf", file: "m.gguf")
        hub.serve("stub/auth-gguf/resolve/main/m.gguf", body: Self.payload(10))

        _ = try await ModelDownloads.plan(for: model, transfer: transfer(auth: auth), root: root)
        let head = try #require(hub.requests(containing: "auth-gguf").first)
        #expect(head.value(forHTTPHeaderField: "Authorization") == "Bearer hf_test")

        var leaving = URLRequest(url: try #require(URL(string: "https://cas-bridge.xethub.hf.co/blob")))
        leaving.setValue("Bearer hf_test", forHTTPHeaderField: "Authorization")
        let redirects = AuthRedirectDelegate(auth: auth)
        #expect(redirects.redirected(leaving).value(forHTTPHeaderField: "Authorization") == nil)

        let staying = URLRequest(url: try #require(URL(string: "https://huggingface.co/org/repo/resolve/main/a")))
        #expect(redirects.redirected(staying).value(forHTTPHeaderField: "Authorization") == "Bearer hf_test")
    }

    @Test func ggufDownloaderRejectsErrorResponses() throws {
        let url = try #require(URL(string: "https://huggingface.co/o/r/resolve/main/m.gguf"))
        let forbidden = HTTPURLResponse(url: url, statusCode: 403, httpVersion: nil, headerFields: nil)
        let fine = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)

        #expect(throws: ModelDownloadError.unauthorized(repoID: "o/r")) {
            try GGUFModelDownloader.validateDownloadResponse(forbidden, repoID: "o/r", path: "m.gguf")
        }
        #expect(throws: Never.self) {
            try GGUFModelDownloader.validateDownloadResponse(fine, repoID: "o/r", path: "m.gguf")
        }
    }

    // MARK: Catalog and scan

    @Test func replacingQuantKeepsOnlySingleTopLevelGGUFFiles() throws {
        let base = CatalogEntry(id: "x", repoID: "org/m-GGUF", displayName: "m", modelCategory: .text,
                                weightFormat: .gguf, approximateSizeMB: 100, isUncensored: false,
                                ggufFilename: "m-Q4_K_M.gguf", numLayers: 24, kvHeads: 2, headDim: 64,
                                maxContextLength: nil)
        let mlx = CatalogEntry(id: "y", repoID: "org/m-4bit", displayName: "m", modelCategory: .text,
                               weightFormat: .mlx, approximateSizeMB: 100, isUncensored: false,
                               ggufFilename: nil, numLayers: 24, kvHeads: 0, headDim: 0, maxContextLength: nil)
        let single = GGUFQuantGroup(paths: ["m-Q2_K.gguf"], label: "Q2_K", totalBytes: 209_715_200)
        let split = GGUFQuantGroup(paths: ["m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf"],
                                   label: "Q8_0", totalBytes: 1)
        let nested = GGUFQuantGroup(paths: ["Q4/m.gguf"], label: "Q4_0", totalBytes: 1)

        let swapped = try #require(base.replacingQuant(single))
        #expect(swapped.ggufFilename == "m-Q2_K.gguf")
        #expect(swapped.approximateSizeMB == 200)
        #expect(swapped.numLayers == 24)
        #expect(swapped.id != base.id)
        #expect(base.replacingQuant(split) == nil)
        #expect(base.replacingQuant(nested) == nil)
        #expect(mlx.replacingQuant(single) == nil)
    }

    @Test func scanSnapshotsReportsCompletenessAndRepoID() throws {
        let hub = root.appending(path: "huggingface/hub")
        let done = ModelDownloads.snapshotDirectory(repoID: "org/done-4bit", root: root)
        let pending = ModelDownloads.snapshotDirectory(repoID: "org/pending-4bit", root: root)
        try FileManager.default.createDirectory(at: done, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pending, withIntermediateDirectories: true)
        try Data(count: 30).write(to: done.appendingPathComponent("model.safetensors"))
        FileManager.default.createFile(atPath: done.appendingPathComponent(".complete").path, contents: Data())
        try Data(count: 7).write(to: pending.appendingPathComponent("model.safetensors.partial"))
        try FileManager.default.createDirectory(at: hub.appending(path: "datasets--x--y"), withIntermediateDirectories: true)

        let found = DownloadedModels.scanSnapshots(hubRoot: hub, fileManager: .default)

        #expect(found.map(\.repoID) == ["org/done-4bit", "org/pending-4bit"])
        #expect(found.map(\.isComplete) == [true, false])
        #expect(found.map(\.sizeBytes) == [30, 7])
        let partials = DownloadedModels.partialFiles(roots: [hub], fileManager: .default)
        #expect(partials.map(\.lastPathComponent) == ["model.safetensors.partial"])
    }
}

// MARK: - Stubs

/// Adds a fixed bearer token for `huggingface.co` only, like ``KeychainDownloadAuth`` with a stored token.
private struct FixedTokenAuth: DownloadAuthorizing {
    let token: String

    func headers(for url: URL) -> [String: String] {
        url.host == "huggingface.co" ? ["Authorization": "Bearer \(token)"] : [:]
    }
}

/// (bytes on disk, total) pairs a download reported, in order.
private final class ProgressLog: Sendable {
    private let entries = Mutex<[(Int64, Int64)]>([])

    func append(_ done: Int64, _ total: Int64) {
        entries.withLock { $0.append((done, total)) }
    }

    var last: (Int64, Int64)? { entries.withLock { $0.last } }
}

/// Canned responses keyed by URL. Honours `Range: bytes=a-b` (206, or 416 past the end) unless a route opts out,
/// and answers HEAD with the size headers only. An unrouted paths-info request gets `[]`, Hugging Face's answer for a
/// path it has no LFS entry for, so a download without a published SHA-256 is not verified.
final class DownloadStubs: Sendable {
    static let shared = DownloadStubs()

    struct Route: Sendable {
        let body: Data
        let status: Int
        let linkedSize: Int64?
        let honoursRange: Bool
        let etag: String?
        let errorCode: String?
    }

    private struct State {
        var byURL: [String: Route] = [:]
        var received: [URLRequest] = []
    }

    private let guarded = Mutex(State())

    func serve(_ path: String, body: Data, status: Int = 200, linkedSize: Int64? = nil, honoursRange: Bool = true,
               etag: String? = nil, errorCode: String? = nil) {
        let route = Route(body: body, status: status, linkedSize: linkedSize, honoursRange: honoursRange,
                          etag: etag, errorCode: errorCode)
        guarded.withLock { $0.byURL["https://huggingface.co/" + path] = route }
    }

    func reset() {
        guarded.withLock { $0 = State() }
    }

    func requests(containing fragment: String) -> [URLRequest] {
        guarded.withLock { $0.received.filter { $0.url?.absoluteString.contains(fragment) == true } }
    }

    func reply(to request: URLRequest) -> (status: Int, headers: [String: String], body: Data) {
        let route = guarded.withLock { current -> Route? in
            current.received.append(request)
            return current.byURL[request.url?.absoluteString ?? ""]
        }
        guard let route else {
            let isPathsInfo = request.url?.path.contains("/paths-info/") == true
            return isPathsInfo ? (200, [:], Data("[]".utf8)) : (404, [:], Data())
        }
        var headers = ["Content-Length": "\(route.body.count)"]
        if let linkedSize = route.linkedSize { headers["x-linked-size"] = "\(linkedSize)" }
        if let etag = route.etag { headers["ETag"] = etag }
        if let errorCode = route.errorCode { headers["X-Error-Code"] = errorCode }
        guard route.status == 200 else { return (route.status, headers, Data()) }
        if request.httpMethod == "HEAD" { return (200, headers, Data()) }
        guard route.honoursRange, let range = Self.range(request.value(forHTTPHeaderField: "Range")) else {
            return (200, headers, route.body)
        }
        guard range.lowerBound < route.body.count else { return (416, [:], Data()) }
        let slice = route.body[range.lowerBound...min(range.upperBound, route.body.count - 1)]
        var partialHeaders = ["Content-Length": "\(slice.count)"]
        if let etag = route.etag { partialHeaders["ETag"] = etag }
        return (206, partialHeaders, Data(slice))
    }

    private static func range(_ header: String?) -> ClosedRange<Int>? {
        guard let header, header.hasPrefix("bytes=") else { return nil }
        let bounds = header.dropFirst("bytes=".count).split(separator: "-").compactMap { Int($0) }
        guard bounds.count == 2, bounds[0] <= bounds[1] else { return nil }
        return bounds[0]...bounds[1]
    }
}

final class DownloadStubProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        guard let url = request.url else { return }
        let reply = DownloadStubs.shared.reply(to: request)
        if let response = HTTPURLResponse(url: url, statusCode: reply.status, httpVersion: "HTTP/1.1",
                                          headerFields: reply.headers) {
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        }
        client?.urlProtocol(self, didLoad: reply.body)
        client?.urlProtocolDidFinishLoading(self)
    }
}
