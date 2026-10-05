import Foundation
import Testing
@testable import AuraCore

#if canImport(MLXLLM)

@Suite("Hugging Face snapshot download integrity")
struct AuraHFDownloaderTests {
    @Test("An HTTP error, malformed JSON or empty tree never creates a completion marker", arguments: [
        TreeFailure(status: 401, body: "[]"),
        TreeFailure(status: 500, body: "[]"),
        TreeFailure(status: 200, body: "not JSON"),
        TreeFailure(status: 200, body: "[]")
    ])
    private func rejectsInvalidTree(failure: TreeFailure) async throws {
        let fixture = try SnapshotDownloadFixture(handler: { request in
            if request.url?.path.contains("/tree/") == true {
                return .init(status: failure.status, data: Data(failure.body.utf8))
            }
            return .file()
        })
        await #expect(throws: (any Error).self) {
            _ = try await fixture.download()
        }
        #expect(!fixture.hasCompletionMarker)
    }

    @Test("HEAD errors never download an error body or mark the snapshot complete")
    func rejectsHeadError() async throws {
        let fixture = try SnapshotDownloadFixture(handler: { request in
            if request.httpMethod == "HEAD" {
                return .init(status: 404, data: Data(), headers: ["Content-Length": "4"])
            }
            return .standard(request)
        })
        await #expect(throws: (any Error).self) {
            _ = try await fixture.download()
        }
        #expect(!fixture.hasCompletionMarker)
        #expect(fixture.rangeRequests == 0)
    }

    @Test("An explicitly empty repository file is created without a Range transfer")
    func supportsZeroByteFile() async throws {
        let fixture = try SnapshotDownloadFixture(handler: { request in
            if request.httpMethod == "HEAD" {
                return .init(status: 200, data: Data(), headers: ["Content-Length": "0"])
            }
            return .standard(request)
        })
        _ = try await fixture.download()
        #expect(try Data(contentsOf: fixture.fileURL).isEmpty)
        #expect(fixture.rangeRequests == 0)
        #expect(fixture.hasCompletionMarker)
    }

    @Test("An absent or negative HEAD size fails instead of treating the file as empty", arguments: [nil, "-1"] as [String?])
    func rejectsMissingHeadSize(size: String?) async throws {
        let fixture = try SnapshotDownloadFixture(handler: { request in
            if request.httpMethod == "HEAD" {
                return .init(status: 200, data: Data(), headers: size.map { ["Content-Length": $0] } ?? [:])
            }
            return .standard(request)
        })
        await #expect(throws: (any Error).self) {
            _ = try await fixture.download()
        }
        #expect(fixture.rangeRequests == 0)
        #expect(!fixture.hasCompletionMarker)
    }

    @Test("A failed refresh invalidates the previous marker before replacing any files")
    func invalidatesMarkerBeforeRefresh() async throws {
        let fixture = try SnapshotDownloadFixture(handler: { request in
            if request.value(forHTTPHeaderField: "Range") != nil {
                return .init(status: 416, data: Data())
            }
            return .standard(request)
        })
        try Data([9]).write(to: fixture.fileURL)
        try Data().write(to: fixture.markerURL)
        await #expect(throws: (any Error).self) {
            _ = try await fixture.download(useLatest: true)
        }
        #expect(!fixture.hasCompletionMarker)
        // A normal call must attempt recovery, rather than returning the invalid old snapshot.
        await #expect(throws: (any Error).self) {
            _ = try await fixture.download()
        }
        #expect(fixture.requests.filter { $0.url?.path.contains("/tree/") == true }.count == 2)
    }

    @Test("Cancelling a refresh never restores the previous completion marker")
    func cancellationInvalidatesPreviousMarker() async throws {
        let fixture = try SnapshotDownloadFixture()
        try Data([9]).write(to: fixture.fileURL)
        try Data().write(to: fixture.markerURL)
        let task = Task {
            try await fixture.download(useLatest: true) { progress in
                if progress.completedUnitCount == 4 {
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            }
        }
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
        #expect(!fixture.hasCompletionMarker)
    }

    @Test("Transient HTTP failures retry the same Range", arguments: [408, 429, 503])
    func retriesTransientHTTPStatus(status: Int) async throws {
        let counter = DownloadAttemptCounter()
        let fixture = try SnapshotDownloadFixture(maximumRangeRequests: 2, handler: { request in
            if request.value(forHTTPHeaderField: "Range") != nil, counter.next() == 1 {
                return .init(status: status, data: Data())
            }
            return .standard(request)
        })
        _ = try await fixture.download()
        let ranges = fixture.requests.compactMap { $0.value(forHTTPHeaderField: "Range") }
        #expect(ranges == ["bytes=0-3", "bytes=0-3"])
        #expect(try Data(contentsOf: fixture.fileURL) == DownloadResponse.fileData)
        #expect(fixture.hasCompletionMarker)
    }

    @Test("Transient HTTP retries remain bounded and never claim completion")
    func boundsTransientHTTPRetries() async throws {
        let fixture = try SnapshotDownloadFixture(maximumRangeRequests: 6, handler: { request in
            if request.value(forHTTPHeaderField: "Range") != nil {
                return .init(status: 503, data: Data())
            }
            return .standard(request)
        })
        await #expect(throws: (any Error).self) {
            _ = try await fixture.download()
        }
        #expect(fixture.rangeRequests == 6)
        #expect(!fixture.hasCompletionMarker)
    }

    @Test("Only a correctly sized existing file may be reused", arguments: [0, 1, 4, 7])
    func validatesExistingFile(size: Int) async throws {
        let fixture = try SnapshotDownloadFixture()
        let existing = Data(repeating: 9, count: size)
        try existing.write(to: fixture.fileURL)
        _ = try await fixture.download()
        let result = try Data(contentsOf: fixture.fileURL)
        #expect(result == (size == 4 ? existing : DownloadResponse.fileData))
        #expect(fixture.rangeRequests == (size == 4 ? 0 : 1))
        #expect(fixture.hasCompletionMarker)
    }

    @Test("An exact partial is promoted without another transfer")
    func promotesExactPartial() async throws {
        let fixture = try SnapshotDownloadFixture()
        let existing = Data(repeating: 9, count: 4)
        try existing.write(to: fixture.partialURL)
        _ = try await fixture.download()
        #expect(try Data(contentsOf: fixture.fileURL) == existing)
        #expect(fixture.rangeRequests == 0)
        #expect(fixture.hasCompletionMarker)
    }

    @Test("An oversized partial is rejected instead of being promoted")
    func rejectsOversizedPartial() async throws {
        let fixture = try SnapshotDownloadFixture()
        try Data(repeating: 9, count: 5).write(to: fixture.partialURL)
        await #expect(throws: (any Error).self) {
            _ = try await fixture.download()
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.fileURL.path))
        #expect(!fixture.hasCompletionMarker)
    }

    @Test("Premature 416, empty and oversized chunks fail without being persisted", arguments: [
        ChunkFailure.rangeNotSatisfiable,
        .empty,
        .oversized
    ])
    private func rejectsInvalidChunk(failure: ChunkFailure) async throws {
        let fixture = try SnapshotDownloadFixture(handler: { request in
            guard request.value(forHTTPHeaderField: "Range") != nil else {
                return .standard(request)
            }
            switch failure {
            case .rangeNotSatisfiable: return .init(status: 416, data: Data())
            case .empty: return .init(status: 206, data: Data())
            case .oversized: return .init(status: 206, data: Data(repeating: 1, count: 5))
            }
        })
        await #expect {
            _ = try await fixture.download()
        } throws: { error in
            guard case .invalidResponse = error as? AuraError else { return false }
            return true
        }
        #expect(fixture.rangeRequests == 1)
        #expect(!fixture.hasCompletionMarker)
        let partialSize = (try? Data(contentsOf: fixture.partialURL).count) ?? 0
        #expect(partialSize == 0)
    }

    @Test("Cancelling after bytes arrive preserves the partial without a completion marker")
    func cancellationDoesNotCompleteSnapshot() async throws {
        let fixture = try SnapshotDownloadFixture()
        let task = Task {
            try await fixture.download { progress in
                if progress.completedUnitCount == 4 {
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            }
        }
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
        #expect(!fixture.hasCompletionMarker)
        #expect(FileManager.default.fileExists(atPath: fixture.partialURL.path))
    }

    @Test("A failed marker write is reported instead of claiming completion")
    func reportsMarkerWriteFailure() async throws {
        let fixture = try SnapshotDownloadFixture()
        await #expect(throws: (any Error).self) {
            _ = try await fixture.download { progress in
                if progress.completedUnitCount == 4 {
                    try? FileManager.default.createDirectory(
                        at: fixture.markerURL, withIntermediateDirectories: true
                    )
                }
            }
        }
    }

    @Test("Legacy completion markers retain their offline fast path")
    func reusesCompletedSnapshotWithoutNetwork() async throws {
        let fixture = try SnapshotDownloadFixture()
        try Data().write(to: fixture.markerURL)
        #expect(try await fixture.download() == fixture.snapshotURL)
        #expect(fixture.requests.isEmpty)
    }

    @Test("Insufficient free space fails before downloading model bytes")
    func rejectsInsufficientDiskSpace() async throws {
        let fixture = try SnapshotDownloadFixture(availableDiskCapacity: { _ in 4 })
        await #expect(throws: (any Error).self) {
            _ = try await fixture.download()
        }
        #expect(fixture.rangeRequests == 0)
        #expect(!fixture.hasCompletionMarker)
    }
}

private struct TreeFailure: Sendable {
    let status: Int
    let body: String
}

private enum ChunkFailure: Sendable {
    case rangeNotSatisfiable, empty, oversized
}

private final class DownloadAttemptCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    func next() -> Int {
        lock.withLock {
            value += 1
            return value
        }
    }
}

private struct DownloadResponse: Sendable {
    let status: Int
    let data: Data
    var headers: [String: String] = [:]

    static let fileData = Data([1, 2, 3, 4])

    static func file() -> Self {
        .init(status: 206, data: fileData, headers: ["Content-Length": "4"])
    }

    static func standard(_ request: URLRequest) -> Self {
        if request.url?.path.contains("/tree/") == true {
            return .init(status: 200, data: Data("[{\"path\":\"model.safetensors\",\"type\":\"file\"}]".utf8))
        }
        if request.httpMethod == "HEAD" {
            return .init(status: 200, data: Data(), headers: ["Content-Length": "4"])
        }
        return .file()
    }
}

private final class SnapshotDownloadFixture: @unchecked Sendable {
    let rootURL: URL
    let repoID: String
    let snapshotURL: URL
    let session: URLSession
    private let lock = NSLock()
    private var recordedRequests: [URLRequest] = []
    private let identifier: String
    private let availableDiskCapacity: @Sendable (URL) throws -> Int64?

    init(
        maximumRangeRequests: Int = 1,
        availableDiskCapacity: @escaping @Sendable (URL) throws -> Int64? = { _ in nil },
        handler: @escaping @Sendable (URLRequest) -> DownloadResponse = DownloadResponse.standard
    ) throws {
        identifier = UUID().uuidString
        repoID = "tests/\(identifier)"
        rootURL = FileManager.default.temporaryDirectory.appending(path: "aura-hf-tests-\(identifier)")
        snapshotURL = rootURL.appending(path: "huggingface/hub/models--tests--\(identifier)/snapshots/main")
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [SnapshotDownloadURLProtocol.self]
        session = URLSession(configuration: configuration)
        self.availableDiskCapacity = availableDiskCapacity
        try FileManager.default.createDirectory(at: snapshotURL, withIntermediateDirectories: true)
        SnapshotDownloadURLProtocol.register(identifier: identifier) { [weak self] request in
            self?.record(request)
            // A second defective request cancels the old looping implementation, so a regression
            // fails promptly instead of hanging the test runner indefinitely.
            if request.value(forHTTPHeaderField: "Range") != nil,
               (self?.rangeRequests ?? 0) > maximumRangeRequests {
                throw URLError(.cancelled)
            }
            return handler(request)
        }
    }

    deinit {
        SnapshotDownloadURLProtocol.unregister(identifier: identifier)
        session.invalidateAndCancel()
        try? FileManager.default.removeItem(at: rootURL)
    }

    var fileURL: URL { snapshotURL.appending(path: "model.safetensors") }
    var partialURL: URL { fileURL.appendingPathExtension("partial") }
    var markerURL: URL { snapshotURL.appending(path: ".complete") }
    var hasCompletionMarker: Bool { FileManager.default.fileExists(atPath: markerURL.path) }
    var requests: [URLRequest] { lock.withLock { recordedRequests } }
    var rangeRequests: Int { requests.filter { $0.value(forHTTPHeaderField: "Range") != nil }.count }

    func download(
        useLatest: Bool = false,
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> URL {
        try await AuraHFDownloader(
            session: session,
            cacheDirectory: rootURL,
            waitBeforeRetry: { _ in },
            availableDiskCapacity: availableDiskCapacity
        ).download(
            id: repoID, revision: nil, matching: ["*"], useLatest: useLatest,
            progressHandler: progressHandler
        )
    }

    private func record(_ request: URLRequest) {
        lock.withLock { recordedRequests.append(request) }
    }
}

private final class SnapshotDownloadURLProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    private static nonisolated(unsafe) var handlers: [String: @Sendable (URLRequest) throws -> DownloadResponse] = [:]

    static func register(identifier: String, handler: @escaping @Sendable (URLRequest) throws -> DownloadResponse) {
        lock.withLock { handlers[identifier] = handler }
    }

    static func unregister(identifier: String) {
        lock.withLock { _ = handlers.removeValue(forKey: identifier) }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        do {
            let url = try #require(request.url)
            let handler = try #require(Self.lock.withLock {
                Self.handlers.first { url.pathComponents.contains($0.key) }?.value
            })
            let stub = try handler(request)
            let response = try #require(HTTPURLResponse(
                url: url, statusCode: stub.status, httpVersion: "HTTP/1.1", headerFields: stub.headers
            ))
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: stub.data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

#endif
