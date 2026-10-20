import Foundation
import Testing
@testable import TonamoxCore

@MainActor
@Suite("On-demand downloads")
struct OnDemandDownloadsTests {

    @Test("Models download in order, one at a time, and ones on disk are skipped")
    func queuesInOrder() async throws {
        let fake = FakeTransfers(onDisk: [Model.qwen3_0_6b.id])
        let downloads = OnDemandDownloads(io: fake.io)

        downloads.download([.qwen3_0_6b, .qwen3_1_7b, .fastVLM_0_5b_fp16])

        #expect(downloads.phase(of: .qwen3_0_6b) == .idle)
        #expect(downloads.isDownloading(.qwen3_1_7b))
        #expect(downloads.phase(of: .fastVLM_0_5b_fp16) == .waiting)
        try await waitUntil { downloads.revision == 2 }
        #expect(await fake.started == [Model.qwen3_1_7b.id, Model.fastVLM_0_5b_fp16.id])
        #expect(await fake.maxConcurrent == 1)
        #expect(downloads.phase(of: .qwen3_1_7b) == .idle)
        #expect(downloads.phase(of: .fastVLM_0_5b_fp16) == .idle)
    }

    @Test("Two entries for one file share one transfer")
    func sharesOneFile() async throws {
        let catalog = Model.qwen2_5_7b_gguf
        let fileName = try #require(catalog.ggufFilename)
        let copy = try #require(Model.fromHuggingFaceURL(
            "https://huggingface.co/\(catalog.repoID)/resolve/main/\(fileName)"))
        #expect(copy.id != catalog.id)
        let fake = FakeTransfers()
        let downloads = OnDemandDownloads(io: fake.io)

        downloads.download([catalog])
        downloads.download([copy, catalog])

        #expect(downloads.isDownloading(copy))
        try await waitUntil { downloads.revision == 1 }
        #expect(await fake.started == [catalog.id])
    }

    @Test("Progress reports reach the phase")
    func reportsProgress() async throws {
        let fake = FakeTransfers(progress: (5, 10), holds: true)
        let downloads = OnDemandDownloads(io: fake.io)

        downloads.download([.qwen3_1_7b])

        try await waitUntil { downloads.phase(of: .qwen3_1_7b) == .downloading(done: 5, total: 10) }
        downloads.cancel([.qwen3_1_7b])
        try await waitUntil { downloads.phase(of: .qwen3_1_7b) == .idle && downloads.revision == 2 }
    }

    @Test("Cancel stops the transfer and drops the queued models")
    func cancelStopsAndDrops() async throws {
        let fake = FakeTransfers(holds: true)
        let downloads = OnDemandDownloads(io: fake.io)

        downloads.download([.qwen3_1_7b, .fastVLM_0_5b_fp16])
        try await waitUntil { await fake.started == [Model.qwen3_1_7b.id] }
        downloads.cancel([.qwen3_1_7b, .fastVLM_0_5b_fp16])

        #expect(downloads.phase(of: .fastVLM_0_5b_fp16) == .idle)
        try await waitUntil { downloads.phase(of: .qwen3_1_7b) == .idle && downloads.revision == 2 }
        try await Task.sleep(for: .milliseconds(50))
        #expect(await fake.started == [Model.qwen3_1_7b.id])
    }

    @Test("A failure keeps its message, and downloading again retries")
    func failureThenRetry() async throws {
        let fake = FakeTransfers(failures: 1)
        let downloads = OnDemandDownloads(io: fake.io)

        downloads.download([.qwen3_1_7b])
        try await waitUntil { downloads.revision == 1 }
        guard case .failed(let message) = downloads.phase(of: .qwen3_1_7b) else {
            Issue.record("expected .failed, got \(downloads.phase(of: .qwen3_1_7b))")
            return
        }
        #expect(message == URLError(.notConnectedToInternet).localizedDescription)
        #expect(!downloads.isDownloading(.qwen3_1_7b))

        downloads.download([.qwen3_1_7b])
        try await waitUntil { downloads.revision == 2 }
        #expect(downloads.phase(of: .qwen3_1_7b) == .idle)
        #expect(await fake.started.count == 2)
    }

    @Test("Approximate size counts a megabyte as 1,000,000 bytes")
    func approximateSize() {
        #expect(Model.qwen3_1_7b.approximateSizeBytes == Int64(Model.qwen3_1_7b.approximateSizeMB) * 1_000_000)
    }

    private func waitUntil(_ condition: @MainActor () async -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(5)
        while await !condition() {
            guard ContinuousClock.now < deadline else {
                Issue.record("condition not met within 5 s")
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}

/// Records each transfer instead of touching the network.
private actor FakeTransfers {
    private(set) var started: [String] = []
    private(set) var maxConcurrent = 0
    private var running = 0
    private var failuresLeft: Int
    private let onDisk: Set<String>
    private let progress: (Int64, Int64)?
    private let holds: Bool

    init(onDisk: Set<String> = [], progress: (Int64, Int64)? = nil, holds: Bool = false, failures: Int = 0) {
        self.onDisk = onDisk
        self.progress = progress
        self.holds = holds
        self.failuresLeft = failures
    }

    nonisolated var io: OnDemandDownloads.DiskAndNetwork {
        let onDisk = onDisk
        return OnDemandDownloads.DiskAndNetwork(
            isOnDisk: { onDisk.contains($0.id) },
            bytesOnDisk: { _ in 0 },
            fetch: { model, report in try await self.run(model, report: report) })
    }

    private func run(_ model: Model, report: @Sendable (Int64, Int64) -> Void) async throws {
        started.append(model.id)
        running += 1
        maxConcurrent = max(maxConcurrent, running)
        defer { running -= 1 }
        if let progress { report(progress.0, progress.1) }
        if failuresLeft > 0 {
            failuresLeft -= 1
            throw URLError(.notConnectedToInternet)
        }
        if holds {
            try await Task.sleep(for: .seconds(30))
        } else {
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}
