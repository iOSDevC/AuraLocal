import XCTest
import SQLite3
@testable import AuraCore
@testable import AuraDocs

final class DocumentLibraryIndexTests: XCTestCase {

    // MARK: - Provider switches

    func testSwitchingProvidersReembedsEveryChunkAndRecordsTheNewIdentity() async throws {
        let (library, folder) = try await makeLibrary()
        let first = try Self.writeDocument("first.txt", sentences: 12, in: folder)
        let second = try Self.writeDocument("second.txt", sentences: 12, in: folder)
        await library.configure(embeddingProvider: StubEmbedder("stub/a", dimensions: 4), llm: await Self.llm())
        try await library.add(url: first)

        await library.configure(embeddingProvider: StubEmbedder("stub/b", dimensions: 8), llm: await Self.llm())
        let staleBeforeAdd = try await library.indexNeedsReembedding()
        try await library.add(url: second)

        let store = try await Self.openStore(folder)
        XCTAssertTrue(staleBeforeAdd)
        let identity = try await store.embeddingIdentity()
        let lengths = try await store.storedVectorLengths()
        let needsReembedding = try await library.indexNeedsReembedding()
        XCTAssertEqual(identity, EmbeddingIndexIdentity(identifier: "stub/b", dimensions: 8))
        XCTAssertEqual(lengths, [8])
        XCTAssertFalse(needsReembedding)
    }

    func testReembeddingKeepsTheFullTextIndex() async throws {
        let (library, folder) = try await makeLibrary()
        let document = try Self.writeDocument("notes.txt", sentences: 20, in: folder)
        await library.configure(embeddingProvider: StubEmbedder("stub/a", dimensions: 4), llm: await Self.llm())
        try await library.add(url: document)
        let matchesBefore = try Self.scalar("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'invoices';", in: folder)

        await library.configure(embeddingProvider: StubEmbedder("stub/b", dimensions: 8), llm: await Self.llm())
        try await library.reembedAll()

        XCTAssertGreaterThan(matchesBefore, 0)
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'invoices';", in: folder),
                       matchesBefore)
        XCTAssertNoThrow(try Self.scalar("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check'); SELECT 1;",
                                         in: folder))
        let lengths = try await Self.openStore(folder).storedVectorLengths()
        XCTAssertEqual(lengths, [8])
    }

    func testProviderSwitchDuringAnAddDoesNotMixVectorSpaces() async throws {
        let (library, folder) = try await makeLibrary()
        let first = try Self.writeDocument("first.txt", sentences: 12, in: folder)
        let second = try Self.writeDocument("second.txt", sentences: 30, in: folder)
        let slow = StubEmbedder("stub/slow", dimensions: 4, delay: .milliseconds(400))
        await library.configure(embeddingProvider: slow, llm: await Self.llm())
        try await library.add(url: first)

        let adding = Task { try await library.add(url: second) }
        try await Task.sleep(for: .milliseconds(150))
        await library.configure(embeddingProvider: StubEmbedder("stub/b", dimensions: 8), llm: await Self.llm())
        try await library.reembedAll()
        try await adding.value

        let store = try await Self.openStore(folder)
        let lengths = try await store.storedVectorLengths()
        let identity = try await store.embeddingIdentity()
        let needsReembedding = try await library.indexNeedsReembedding()
        XCTAssertEqual(lengths, [8], "every stored vector comes from the provider the index records")
        XCTAssertEqual(identity?.identifier, "stub/b")
        XCTAssertFalse(needsReembedding)
    }

    func testInterruptedReembedLeavesAPendingMarkerAndIsRedone() async throws {
        let (library, folder) = try await makeLibrary(chunkTokens: 12)
        let document = try Self.writeDocument("long.txt", sentences: 160, in: folder)
        await library.configure(embeddingProvider: StubEmbedder("stub/a", dimensions: 4), llm: await Self.llm())
        try await library.add(url: document)
        let chunks = try await Self.openStore(folder).chunkCount()
        XCTAssertGreaterThan(chunks, 50, "needs more than one re-embed batch")

        let failing = StubEmbedder("stub/b", dimensions: 8, failAfterBatches: 1)
        await library.configure(embeddingProvider: failing, llm: await Self.llm())
        do {
            try await library.reembedAll()
            XCTFail("the second batch should throw")
        } catch {}
        let marker = try await Self.openStore(folder).embeddingIdentity()

        await library.configure(embeddingProvider: StubEmbedder("stub/b", dimensions: 8), llm: await Self.llm())
        let needsReembedding = try await library.indexNeedsReembedding()
        try await library.reembedAll()

        XCTAssertEqual(marker?.identifier, "pending:stub/b")
        XCTAssertTrue(needsReembedding)
        let store = try await Self.openStore(folder)
        let identity = try await store.embeddingIdentity()
        let lengths = try await store.storedVectorLengths()
        XCTAssertEqual(identity, EmbeddingIndexIdentity(identifier: "stub/b", dimensions: 8))
        XCTAssertEqual(lengths, [8])
    }

    // MARK: - Pre-tracking databases

    func testPreTrackingIndexIsAdoptedOnlyByTFIDF() async throws {
        let (library, folder) = try await makeLibrary()
        let document = try Self.writeDocument("old.txt", sentences: 12, in: folder)
        await library.configure(embeddingProvider: AutoEmbeddingProvider(), llm: await Self.llm())
        try await library.add(url: document)
        await library.close()
        _ = try Self.scalar("DELETE FROM metadata; SELECT 1;", in: folder)

        let reopened = DocumentLibrary(directory: folder)
        try await reopened.open()
        await reopened.configure(embeddingProvider: AutoEmbeddingProvider(), llm: await Self.llm())
        let tfidfNeedsReembedding = try await reopened.indexNeedsReembedding()
        let sameWidth = StubEmbedder("someone/hashing-4096", dimensions: TFIDFEmbeddingProvider.vocabSize)
        await reopened.configure(embeddingProvider: sameWidth, llm: await Self.llm())
        let otherNeedsReembedding = try await reopened.indexNeedsReembedding()

        XCTAssertFalse(tfidfNeedsReembedding)
        XCTAssertTrue(otherNeedsReembedding, "a same-width model cannot claim vectors it did not produce")
    }

    // MARK: - Configuration

    func testConfigureIfNeededKeepsTheProviderAlreadySet() async throws {
        let (library, folder) = try await makeLibrary()
        let document = try Self.writeDocument("kept.txt", sentences: 12, in: folder)
        await library.configure(embeddingProvider: StubEmbedder("stub/e5", dimensions: 4), llm: await Self.llm())

        let replaced = await library.configureIfNeeded(embeddingProvider: AutoEmbeddingProvider(), llm: await Self.llm())
        try await library.add(url: document)

        XCTAssertFalse(replaced)
        let identity = try await Self.openStore(folder).embeddingIdentity()
        XCTAssertEqual(identity?.identifier, "stub/e5")
    }

    func testDefaultIdentifierIsStableForPrivateTypes() {
        let identifier = PrivateProvider().identifier

        XCTAssertFalse(identifier.contains("unknown context"), identifier)
        XCTAssertFalse(identifier.contains("$"), identifier)
        XCTAssertTrue(identifier.hasSuffix("PrivateProvider/16"), identifier)
    }

    func testTFIDFIdentifierIsThePreTrackingIdentifier() {
        XCTAssertEqual(TFIDFEmbeddingProvider.vectorSpaceIdentifier, EmbeddingIndexIdentity.preTrackingIdentifier)
        XCTAssertEqual(AutoEmbeddingProvider().identifier, EmbeddingIndexIdentity.preTrackingIdentifier)
    }

    func testAutoProviderFallsBackToTFIDFWithoutAValidBundle() {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("no-such-bundle-\(UUID().uuidString)")

        let provider = AutoEmbeddingProvider(embeddingModelAt: missing)

        XCTAssertFalse(provider.usesDenseModel)
        XCTAssertNotNil(provider.denseModelProblem)
        XCTAssertEqual(provider.identifier, EmbeddingIndexIdentity.preTrackingIdentifier)
        XCTAssertEqual(provider.dimensions, TFIDFEmbeddingProvider.vocabSize)
    }

    // MARK: - Helpers

    private func makeLibrary(chunkTokens: Int = 24) async throws -> (DocumentLibrary, URL) {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("AuraDocsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        let library = DocumentLibrary(directory: folder, chunkTargetTokens: chunkTokens)
        try await library.open()
        return (library, folder)
    }

    /// Never loaded: indexing needs an LLM only to satisfy `configure`.
    @MainActor
    private static func llm() -> AuraLocal {
        AuraLocal(model: .qwen3_1_7b)
    }

    private static func writeDocument(_ name: String, sentences: Int, in folder: URL) throws -> URL {
        let text = (0..<sentences)
            .map { "Sentence \($0) says the invoices of \(name) are paid within thirty days." }
            .joined(separator: " ")
        let url = folder.appendingPathComponent(name)
        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private static func openStore(_ folder: URL) async throws -> VectorStore {
        let store = VectorStore(directory: folder)
        try await store.open()
        return store
    }

    /// Runs `sql` and returns the first column of the last statement's first row.
    @discardableResult
    private static func scalar(_ sql: String, in folder: URL) throws -> Int {
        var db: OpaquePointer?
        guard sqlite3_open(folder.appendingPathComponent("vectors.sqlite").path, &db) == SQLITE_OK else {
            throw SQLiteFailure(message: "open")
        }
        defer { sqlite3_close(db) }
        var remaining = sql
        var value = 0
        while !remaining.isEmpty {
            let (result, tail) = try step(remaining, in: db)
            value = result ?? value
            remaining = tail
        }
        return value
    }

    /// Runs the first statement of `sql`; returns its first integer column (if it yields a row) and the rest.
    private static func step(_ sql: String, in db: OpaquePointer?) throws -> (Int?, String) {
        var stmt: OpaquePointer?
        var tail: UnsafePointer<CChar>?
        let rest: String = try sql.withCString { text in
            guard sqlite3_prepare_v2(db, text, -1, &stmt, &tail) == SQLITE_OK else {
                throw SQLiteFailure(message: String(cString: sqlite3_errmsg(db)))
            }
            return tail.map { String(cString: $0) } ?? ""
        }
        defer { sqlite3_finalize(stmt) }
        guard let stmt else { return (nil, "") }
        switch sqlite3_step(stmt) {
        case SQLITE_ROW: return (Int(sqlite3_column_int64(stmt, 0)), rest.trimmingCharacters(in: .whitespacesAndNewlines))
        case SQLITE_DONE: return (nil, rest.trimmingCharacters(in: .whitespacesAndNewlines))
        default: throw SQLiteFailure(message: String(cString: sqlite3_errmsg(db)))
        }
    }
}

// MARK: - Stubs

private struct SQLiteFailure: Error {
    let message: String
}

private struct StubFailure: Error {}

/// Deterministic vectors derived from the text length; optionally slow, or failing after some batches.
private actor StubEmbedder: EmbeddingProvider {
    nonisolated let identifier: String
    nonisolated let dimensions: Int
    private let delay: Duration
    private let failAfterBatches: Int?
    private var batches = 0

    init(_ identifier: String, dimensions: Int, delay: Duration = .zero, failAfterBatches: Int? = nil) {
        self.identifier = identifier
        self.dimensions = dimensions
        self.delay = delay
        self.failAfterBatches = failAfterBatches
    }

    func embed(_ text: String) async throws -> [Float] {
        try await embedBatch([text]).first ?? []
    }

    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        batches += 1
        if let failAfterBatches, batches > failAfterBatches { throw StubFailure() }
        if delay > .zero { try await Task.sleep(for: delay) }
        return texts.map(vector(for:))
    }

    private nonisolated func vector(for text: String) -> [Float] {
        (0..<dimensions).map { Float(text.utf8.count % 97 + $0 + 1) }
    }
}

/// Uses the default `identifier`.
private struct PrivateProvider: EmbeddingProvider {
    let dimensions = 16

    func embed(_ text: String) async throws -> [Float] {
        [Float](repeating: 1, count: dimensions)
    }
}
