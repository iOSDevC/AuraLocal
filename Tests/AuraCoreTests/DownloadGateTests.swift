import Foundation
import Synchronization
import Testing
@testable import AuraCore

// MARK: - Queue and lock file

/// `DownloadGate` and `DownloadFileLock` on their own, each test with its own gate and a folder whose name has a space.
final class DownloadGateTests {
    private let base: URL

    init() throws {
        base = FileManager.default.temporaryDirectory
            .appending(path: "DownloadGateTests-\(UUID().uuidString)/Application Support", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
    }

    deinit {
        try? FileManager.default.removeItem(at: base.deletingLastPathComponent())
    }

    private static func gate() -> DownloadGate {
        DownloadGate(lockRetryInterval: .milliseconds(5))
    }

    private static func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path(percentEncoded: false))
    }

    /// Holds `destination` for 20 ms, logging when it starts and ends.
    private static func record(_ name: String, on gate: DownloadGate, at destination: URL,
                               in log: EventLog) async throws {
        try await gate.exclusive(destination) { _ in
            log.append("start \(name)")
            try await Task.sleep(for: .milliseconds(20))
            log.append("end \(name)")
        }
    }

    /// Every start is followed directly by the same caller's end.
    private static func isSequential(_ events: [String]) -> Bool {
        events.count.isMultiple(of: 2) && stride(from: 0, to: events.count, by: 2).allSatisfy { index in
            events[index].hasPrefix("start ") && events[index + 1] == "end " + events[index].dropFirst("start ".count)
        }
    }

    // MARK: Queue

    @Test func callersOfOneDestinationRunOneAtATime() async throws {
        let gate = Self.gate()
        let destination = base.appending(path: "models/org/repo/m.gguf")
        let log = EventLog()

        try await withThrowingTaskGroup(of: Void.self) { group in
            for name in ["a", "b", "c"] {
                group.addTask { try await Self.record(name, on: gate, at: destination, in: log) }
            }
            try await group.waitForAll()
        }

        #expect(log.all.count == 6)
        #expect(Self.isSequential(log.all), "\(log.all)")
    }

    @Test func aPathInsideAHeldFolderWaitsForIt() async throws {
        let gate = Self.gate()
        let snapshot = base.appending(path: "huggingface/hub/models--o--r/snapshots/main", directoryHint: .isDirectory)
        let log = EventLog()

        async let folder: Void = Self.record("folder", on: gate, at: snapshot, in: log)
        async let file: Void = Self.record("file", on: gate, at: snapshot.appending(path: "config.json"), in: log)
        _ = try await (folder, file)

        #expect(Self.isSequential(log.all), "\(log.all)")
    }

    @Test func differentDestinationsRunAtTheSameTime() async throws {
        let gate = Self.gate()
        let first = Flag()
        let second = Flag()
        let one = base.appending(path: "a.gguf")
        let other = base.appending(path: "b.gguf")

        async let sawSecond: Bool = gate.exclusive(one) { _ in
            first.set()
            return await eventually { second.isSet }
        }
        async let sawFirst: Bool = gate.exclusive(other) { _ in
            second.set()
            return await eventually { first.isSet }
        }

        let (a, b) = try await (sawSecond, sawFirst)
        #expect(a && b)
    }

    @Test func cancellingAWaitingCallerLeavesTheRunningOneAlone() async throws {
        let gate = Self.gate()
        let destination = base.appending(path: "m.gguf")
        let holding = Flag()
        let mayFinish = Flag()
        let holderDone = Flag()
        let holder = Task {
            try await gate.exclusive(destination) { _ in
                holding.set()
                _ = await eventually(within: .seconds(10)) { mayFinish.isSet }
                holderDone.set()
                return "holder"
            }
        }
        #expect(await eventually { holding.isSet })
        let waiter = Task { try await gate.exclusive(destination) { _ in "waiter" } }
        #expect(await eventually { await gate.waitingCount(at: destination) == 1 })

        waiter.cancel()

        await #expect(throws: CancellationError.self) { try await waiter.value }
        #expect(!holderDone.isSet, "the cancelled caller returned before the running one finished")
        #expect(await gate.waitingCount(at: destination) == 0)
        mayFinish.set()
        #expect(try await holder.value == "holder")
        #expect(try await gate.exclusive(destination) { _ in "next" } == "next")
    }

    @Test func cancellingTheRunningCallerHandsTheTurnOn() async throws {
        let gate = Self.gate()
        let destination = base.appending(path: "m.gguf")
        let holding = Flag()
        let holder = Task {
            try await gate.exclusive(destination) { _ in
                holding.set()
                try await Task.sleep(for: .seconds(30))
            }
        }
        #expect(await eventually { holding.isSet })
        let waiter = Task { try await gate.exclusive(destination) { _ in "waiter" } }
        #expect(await eventually { await gate.waitingCount(at: destination) == 1 })

        holder.cancel()

        await #expect(throws: CancellationError.self) { try await holder.value }
        #expect(try await waiter.value == "waiter")
    }

    @Test(arguments: [true, false])
    func aWaitingCallerLearnsWhetherTheSameRequestFinished(succeeds: Bool) async throws {
        let gate = Self.gate()
        let destination = base.appending(path: "blob.bin")
        let holding = Flag()
        let mayFinish = Flag()
        let holder = Task {
            try await gate.exclusive(destination, request: "x") { _ in
                holding.set()
                _ = await eventually(within: .seconds(10)) { mayFinish.isSet }
                if !succeeds { throw CancellationError() }
            }
        }
        #expect(await eventually { holding.isSet })
        let same = Task { try await gate.exclusive(destination, request: "x") { $0 } }
        #expect(await eventually { await gate.waitingCount(at: destination) == 1 })
        let other = Task { try await gate.exclusive(destination, request: "y") { $0 } }
        #expect(await eventually { await gate.waitingCount(at: destination) == 2 })

        mayFinish.set()

        _ = await holder.result
        #expect(try await same.value == succeeds)
        #expect(try await other.value == false)
    }

    // MARK: Lock file

    @Test func aSecondOpenFileWaitsForTheLock() async throws {
        let destination = base.appending(path: "m.gguf")
        let first = try await DownloadFileLock.acquire(for: destination, retryInterval: .milliseconds(5))
        #expect(Self.exists(first.lockFile))
        let acquired = Flag()
        let second = Task {
            let lock = try await DownloadFileLock.acquire(for: destination, retryInterval: .milliseconds(5))
            acquired.set()
            return lock
        }

        try await Task.sleep(for: .milliseconds(200))
        #expect(!acquired.isSet)
        first.release()
        let lock = try await second.value
        #expect(Self.exists(lock.lockFile))
        lock.release()

        #expect(!Self.exists(lock.lockFile))
    }

    @Test func releasingRemovesOnlyTheEmptyFoldersTheLockCreated() async throws {
        let destination = base.appending(path: "models/org/repo/m.gguf")

        let lock = try await DownloadFileLock.acquire(for: destination, retryInterval: .milliseconds(5))
        #expect(Self.exists(destination.deletingLastPathComponent()))
        lock.release()

        #expect(!Self.exists(base.appending(path: "models")))
        #expect(Self.exists(base))
    }

    @Test func aLockFileLeftBehindIsTakenOverAtOnce() async throws {
        let destination = base.appending(path: "m.gguf")
        let leftover = DownloadFileLock.lockFile(for: destination)
        FileManager.default.createFile(atPath: leftover.path(percentEncoded: false), contents: Data("stale".utf8))
        let clock = ContinuousClock()
        let start = clock.now

        let lock = try await DownloadFileLock.acquire(for: destination, retryInterval: .seconds(5))

        #expect(clock.now - start < .seconds(2))
        lock.release()
        #expect(!Self.exists(leftover))
        #expect(Self.exists(base))
    }

    #if os(macOS)
    /// A real second process: `flock` held by Perl, then dropped by the kernel when it exits.
    @Test(.enabled(if: FileManager.default.isExecutableFile(atPath: "/usr/bin/perl")))
    func aLockHeldByAnotherProcessIsWaitedFor() async throws {
        let destination = base.appending(path: "m.gguf")
        let locked = base.appending(path: "perl locked")
        let done = base.appending(path: "perl done")
        let script = """
            use Fcntl qw(:flock);
            open(my $lock, ">>", $ARGV[0]) or die; flock($lock, LOCK_EX) or die;
            open(my $m, ">", $ARGV[1]) or die; close($m);
            select(undef, undef, undef, 0.3);
            open(my $d, ">", $ARGV[2]) or die; close($d);
            """
        let perl = Process()
        perl.executableURL = URL(fileURLWithPath: "/usr/bin/perl")
        perl.arguments = ["-e", script] + [DownloadFileLock.lockFile(for: destination), locked, done]
            .map { $0.path(percentEncoded: false) }
        try perl.run()
        defer { perl.waitUntilExit() }
        #expect(await eventually { Self.exists(locked) })

        let lock = try await DownloadFileLock.acquire(for: destination, retryInterval: .milliseconds(10))

        #expect(Self.exists(done), "the lock was taken while the other process still held it")
        lock.release()
    }
    #endif

    // MARK: Promotion

    @Test func aPartialThatVanishedIsAConflictNotAFileError() throws {
        let destination = base.appending(path: "m.gguf")

        #expect(throws: DownloadConflictError.partialChanged(path: "m.gguf")) {
            try HubFileTransfer.promote(HubFileTransfer.partialURL(for: destination), to: destination,
                                        expectedSize: 10, path: "m.gguf")
        }
        #expect(!Self.exists(destination))
    }

    @Test func aPartialOfAnotherSizeIsAConflictAndStaysWhereItIs() throws {
        let destination = base.appending(path: "m.gguf")
        let partial = HubFileTransfer.partialURL(for: destination)
        try Data(count: 12).write(to: partial)

        #expect(throws: DownloadConflictError.partialChanged(path: "m.gguf")) {
            try HubFileTransfer.promote(partial, to: destination, expectedSize: 10, path: "m.gguf")
        }
        #expect(HubFileTransfer.fileSize(at: partial) == 12)
        #expect(!Self.exists(destination))
    }

    @Test func anUnchangedPartialReplacesTheDestination() throws {
        let destination = base.appending(path: "m.gguf")
        let partial = HubFileTransfer.partialURL(for: destination)
        try Data(count: 3).write(to: destination)
        try Data(count: 10).write(to: partial)

        try HubFileTransfer.promote(partial, to: destination, expectedSize: 10, path: "m.gguf")

        #expect(HubFileTransfer.fileSize(at: destination) == 10)
        #expect(!Self.exists(partial))
    }

    @Test func conflictErrorsSayWhatHappened() {
        let message = DownloadConflictError.partialChanged(path: "m.gguf").errorDescription ?? ""
        #expect(message.contains("m.gguf") && message.contains("another download"))
    }
}

// MARK: - Entry points

/// Two calls for the same destination through each download entry point, against stubbed Hugging Face routes of
/// their own: both succeed and the file is transferred once.
@Suite(.serialized)
final class DownloadGuardEntryPointTests {
    private let base: URL
    private let root: URL
    private let hub = GuardStubProtocol.hub

    init() throws {
        base = FileManager.default.temporaryDirectory
            .appending(path: "DownloadGuardTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        root = base.appending(path: "Application Support/Models", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        hub.reset()
    }

    deinit {
        try? FileManager.default.removeItem(at: base)
    }

    // MARK: Fixtures

    private static func stubConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [GuardStubProtocol.self]
        return configuration
    }

    private static let stubSession = URLSession(configuration: stubConfiguration())

    private func transfer() -> HubFileTransfer {
        HubFileTransfer(session: Self.stubSession, auth: NoDownloadAuth(), freeSpace: { _ in nil },
                        retryDelay: { _ in .zero }, progressInterval: .milliseconds(10))
    }

    private static func payload(_ count: Int) -> Data {
        Data((0..<count).map { UInt8($0 % 251) })
    }

    private static func model(repoID: String, gguf file: String?) throws -> Model {
        let entry = CatalogEntry(id: "guard_\(file ?? "mlx")", repoID: repoID, displayName: "Guard test",
                                 modelCategory: .text, weightFormat: file == nil ? .mlx : .gguf,
                                 approximateSizeMB: 1, isUncensored: false, ggufFilename: file, numLayers: 24,
                                 kvHeads: 2, headDim: 64, maxContextLength: nil)
        return try #require(entry.decodedModel())
    }

    private func serveTree(_ path: String, _ files: [(path: String, size: Int)]) {
        let entries = files.map { ["type": "file", "path": $0.path, "size": $0.size] as [String: Any] }
        hub.serve(path, body: (try? JSONSerialization.data(withJSONObject: entries)) ?? Data())
    }

    /// GET requests for `resolvePath` (`<repo>/resolve/<revision>/<file>`), Range or whole-file.
    private func transfers(of resolvePath: String) -> Int {
        hub.requests(containing: resolvePath).filter { $0.httpMethod == "GET" }.count
    }

    private static func samePath(_ lhs: URL, _ rhs: URL) -> Bool {
        lhs.standardizedFileURL.path(percentEncoded: false) == rhs.standardizedFileURL.path(percentEncoded: false)
    }

    /// The real storage root, which the GGUF load path always writes to; the repo id is unique and removed after.
    @MainActor
    private static func withStorageRootRepo(_ body: (String) async throws -> Void) async throws {
        let repo = "auralocal-tests/guard-\(UUID().uuidString.lowercased())"
        let folder = DownloadedModels.modelsDirectory.appending(path: repo)
        defer {
            try? FileManager.default.removeItem(at: folder)
            ModelStorage.removeEmptyFolders(from: folder.deletingLastPathComponent(),
                                            upTo: DownloadedModels.modelsDirectory)
        }
        try await body(repo)
    }

    // MARK: ModelDownloads

    @Test func twoGGUFDownloadsOfOneFileTransferItOnce() async throws {
        let model = try Self.model(repoID: "stub/twice-gguf", gguf: "m.gguf")
        let body = Self.payload(5000)
        hub.serve("stub/twice-gguf/resolve/main/m.gguf", body: body)
        let transfer = self.transfer()
        let root = self.root

        async let first = ModelDownloads.download(model, transfer: transfer, root: root) { _, _ in }
        async let second = ModelDownloads.download(model, transfer: transfer, root: root) { _, _ in }
        let (a, b) = try await (first, second)

        #expect(a == b)
        #expect(try Data(contentsOf: a) == body)
        #expect(transfers(of: "stub/twice-gguf/resolve/main/m.gguf") == 1)
        #expect(!FileManager.default.fileExists(atPath: DownloadFileLock.lockFile(for: a).path(percentEncoded: false)))
    }

    @Test func twoMLXDownloadsOfOneSnapshotTransferEachFileOnce() async throws {
        let repo = "stub/twice-mlx"
        let model = try Self.model(repoID: repo, gguf: nil)
        serveTree("api/models/\(repo)/tree/main", [("config.json", 20), ("model.safetensors", 300)])
        hub.serve("\(repo)/resolve/main/config.json", body: Self.payload(20))
        hub.serve("\(repo)/resolve/main/model.safetensors", body: Self.payload(300))
        let transfer = self.transfer()
        let root = self.root

        async let first = ModelDownloads.download(model, transfer: transfer, root: root) { _, _ in }
        async let second = ModelDownloads.download(model, transfer: transfer, root: root) { _, _ in }
        let (a, b) = try await (first, second)

        #expect(a == b)
        #expect(FileManager.default.fileExists(atPath: a.appending(path: ModelDownloads.completeMarker).path))
        #expect(transfers(of: "\(repo)/resolve/main/config.json") == 1)
        #expect(transfers(of: "\(repo)/resolve/main/model.safetensors") == 1)
    }

    @Test func twoFolderDownloadsTransferEachFileOnce() async throws {
        let repo = "stub/twice-folder"
        serveTree("api/models/\(repo)/tree/rev1/compiled?recursive=true", [("compiled/weight.bin", 900)])
        hub.serve("\(repo)/resolve/rev1/compiled/weight.bin", body: Self.payload(900))
        let transfer = self.transfer()
        let root = self.root

        async let first = ModelDownloads.downloadFolder(repoID: repo, revision: "rev1", folder: "compiled",
                                                        include: ["weight.bin"], transfer: transfer,
                                                        root: root) { _, _ in }
        async let second = ModelDownloads.downloadFolder(repoID: repo, revision: "rev1", folder: "compiled",
                                                         include: ["weight.bin"], transfer: transfer,
                                                         root: root) { _, _ in }
        let (a, b) = try await (first, second)

        #expect(a == b)
        #expect(ModelDownloads.isFolderComplete(at: a, revision: "rev1"))
        #expect(transfers(of: "\(repo)/resolve/rev1/compiled/weight.bin") == 1)
    }

    @Test func aPartialRemovedBeforePromotionFailsWithAConflict() async throws {
        let model = try Self.model(repoID: "stub/vanish-gguf", gguf: "m.gguf")
        hub.serve("stub/vanish-gguf/resolve/main/m.gguf", body: Self.payload(5000))
        let file = root.appending(path: "models/stub/vanish-gguf/m.gguf")
        let partial = HubFileTransfer.partialURL(for: file)

        await #expect(throws: DownloadConflictError.partialChanged(path: "m.gguf")) {
            // Stands in for another writer that took the partial once its last byte arrived.
            try await ModelDownloads.download(model, transfer: self.transfer(), root: self.root) { done, total in
                if total > 0, done == total { try? FileManager.default.removeItem(at: partial) }
            }
        }
        #expect(!FileManager.default.fileExists(atPath: file.path(percentEncoded: false)))
    }

    #if canImport(MLXLLM)
    @Test func theMLXLoadPathAndModelDownloadsShareOneSnapshot() async throws {
        let repo = "stub/shared-mlx"
        let model = try Self.model(repoID: repo, gguf: nil)
        serveTree("api/models/\(repo)/tree/main", [("config.json", 20), ("model.safetensors", 300)])
        hub.serve("\(repo)/resolve/main/config.json", body: Self.payload(20))
        hub.serve("\(repo)/resolve/main/model.safetensors", body: Self.payload(300))
        let loader = AuraHFDownloader(session: Self.stubSession, cacheDirectory: root, waitBeforeRetry: { _ in },
                                      availableDiskCapacity: { _ in nil })
        let transfer = self.transfer()
        let root = self.root

        async let loaded = loader.download(id: repo, revision: nil, matching: ModelDownloads.mlxFilePatterns,
                                           useLatest: false) { _ in }
        async let fetched = ModelDownloads.download(model, transfer: transfer, root: root) { _, _ in }
        let (a, b) = try await (loaded, fetched)

        #expect(Self.samePath(a, b))
        #expect(transfers(of: "\(repo)/resolve/main/config.json") == 1)
        #expect(transfers(of: "\(repo)/resolve/main/model.safetensors") == 1)
    }
    #endif

    // MARK: GGUFModelDownloader

    @MainActor
    @Test func twoLoadPathDownloadersOfOneFileTransferItOnce() async throws {
        try await Self.withStorageRootRepo { repo in
            let body = Self.payload(4000)
            hub.serve("\(repo)/resolve/main/m.gguf", body: body)
            let model = try Self.model(repoID: repo, gguf: "m.gguf")
            let first = GGUFModelDownloader(auth: NoDownloadAuth(), sessionConfiguration: Self.stubConfiguration(),
                                            checksumSession: Self.stubSession)
            let second = GGUFModelDownloader(auth: NoDownloadAuth(), sessionConfiguration: Self.stubConfiguration(),
                                             checksumSession: Self.stubSession)

            async let a = first.download(model: model) { _ in }
            async let b = second.download(model: model) { _ in }
            let (x, y) = try await (a, b)

            #expect(x == y)
            #expect(try Data(contentsOf: x) == body)
            #expect(transfers(of: "\(repo)/resolve/main/m.gguf") == 1)
        }
    }

    @MainActor
    @Test func theGGUFLoadPathAndModelDownloadsShareOneTransfer() async throws {
        try await Self.withStorageRootRepo { repo in
            let body = Self.payload(4000)
            hub.serve("\(repo)/resolve/main/m.gguf", body: body)
            let model = try Self.model(repoID: repo, gguf: "m.gguf")
            let downloader = GGUFModelDownloader(auth: NoDownloadAuth(),
                                                 sessionConfiguration: Self.stubConfiguration(),
                                                 checksumSession: Self.stubSession)
            let transfer = self.transfer()
            let storageRoot = ModelStorage.rootDirectory

            async let loaded = downloader.download(model: model) { _ in }
            async let fetched = ModelDownloads.download(model, transfer: transfer, root: storageRoot) { _, _ in }
            let (a, b) = try await (loaded, fetched)

            #expect(Self.samePath(a, b))
            #expect(try Data(contentsOf: a) == body)
            #expect(transfers(of: "\(repo)/resolve/main/m.gguf") == 1)
        }
    }

    // MARK: ChunkedDownloader

    @Test func twoChunkedDownloadsToOneFileTransferItOnce() async throws {
        let url = try #require(URL(string: "https://huggingface.co/stub/chunked/resolve/main/blob.bin"))
        let body = Self.payload(3000)
        hub.serve("stub/chunked/resolve/main/blob.bin", body: body)
        let destination = root.appending(path: "blob.bin")
        let first = ChunkedDownloader(session: Self.stubSession, concurrentChunks: 2, chunkThresholdMB: 10,
                                      auth: NoDownloadAuth())
        let second = ChunkedDownloader(session: Self.stubSession, concurrentChunks: 2, chunkThresholdMB: 10,
                                       auth: NoDownloadAuth())

        async let a: Void = first.download(from: url, to: destination, fileName: "blob.bin") { _ in }
        async let b: Void = second.download(from: url, to: destination, fileName: "blob.bin") { _ in }
        _ = try await (a, b)

        #expect(try Data(contentsOf: destination) == body)
        #expect(transfers(of: "stub/chunked/resolve/main/blob.bin") == 1)
    }
}

// MARK: - Helpers

/// Polls `condition` every 5 ms; `false` once `limit` passes without it holding.
private func eventually(within limit: Duration = .seconds(5),
                        _ condition: @Sendable () async -> Bool) async -> Bool {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: limit)
    while clock.now < deadline {
        if await condition() { return true }
        try? await Task.sleep(for: .milliseconds(5))
    }
    return await condition()
}

private final class Flag: Sendable {
    private let value = Mutex(false)

    func set() { value.withLock { $0 = true } }
    var isSet: Bool { value.withLock { $0 } }
}

private final class EventLog: Sendable {
    private let entries = Mutex<[String]>([])

    func append(_ entry: String) { entries.withLock { $0.append(entry) } }
    var all: [String] { entries.withLock { $0 } }
}

/// Serves ``GuardStubProtocol/hub``, a route table separate from `DownloadStubs.shared`.
private final class GuardStubProtocol: URLProtocol, @unchecked Sendable {
    static let hub = DownloadStubs()
    /// File transfers take this long, so two downloads that are not kept apart always overlap.
    static let transferDelay: TimeInterval = 0.1

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        guard let url = request.url else { return }
        if request.httpMethod == "GET", url.path.contains("/resolve/") {
            Thread.sleep(forTimeInterval: Self.transferDelay)
        }
        let reply = Self.hub.reply(to: request)
        if let response = HTTPURLResponse(url: url, statusCode: reply.status, httpVersion: "HTTP/1.1",
                                          headerFields: reply.headers) {
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        }
        client?.urlProtocol(self, didLoad: reply.body)
        client?.urlProtocolDidFinishLoading(self)
    }
}
