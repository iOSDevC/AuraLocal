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

    // MARK: - Document IDs

    func testDocumentIDsMatchTheGoldenValues() {
        let home = "/Users/golden"

        let inside = DocumentLibrary.stableDocumentID(
            for: URL(fileURLWithPath: "/Users/golden/Documents/contract.pdf"), homeDirectory: home)
        let outside = DocumentLibrary.stableDocumentID(
            for: URL(fileURLWithPath: "/Volumes/Shared/contract.pdf"), homeDirectory: home)

        XCTAssertEqual(inside.uuidString, "9C060E4C-7EC7-519B-9E7C-B562CE222889")
        XCTAssertEqual(outside.uuidString, "2C5DE524-A53A-57FF-AD75-AA5B78968D2A")
    }

    func testDocumentIDIsAVersion5UUIDWithTheRFCVariant() {
        let id = DocumentLibrary.stableDocumentID(
            for: URL(fileURLWithPath: "/Users/golden/notes.txt"), homeDirectory: "/Users/golden").uuid

        XCTAssertEqual(id.6 >> 4, 5)
        XCTAssertEqual(id.8 & 0xC0, 0x80)
    }

    func testEquivalentSpellingsOfAPathGiveTheSameDocumentID() throws {
        let folder = try makeFolder()
        let file = try Self.writeDocument("same.txt", sentences: 1, in: folder)
        let spellings = [
            file.path,
            folder.path + "/./same.txt",
            folder.path + "/missing/../same.txt",
            file.path + "/"
        ] + Self.privateVarSpellings(of: file.path)

        let ids = spellings.map {
            DocumentLibrary.stableDocumentID(for: URL(fileURLWithPath: $0), homeDirectory: "/Users/golden")
        }

        XCTAssertEqual(ids, Array(repeating: ids[0], count: ids.count), "\(spellings)")
    }

    func testHomeDirectorySpellingsAreCanonicalizedLikeFilePaths() throws {
        let folder = try makeFolder()
        let file = try Self.writeDocument("inside.txt", sentences: 1, in: folder)
        guard let otherHome = Self.privateVarSpellings(of: folder.path).first else {
            throw XCTSkip("the temporary directory is not reachable through /private/var on this host")
        }

        XCTAssertEqual(DocumentLibrary.documentIDName(for: file, homeDirectory: folder.path), "home:inside.txt")
        XCTAssertEqual(DocumentLibrary.documentIDName(for: file, homeDirectory: otherHome), "home:inside.txt")
    }

    func testDifferentFilesGetDifferentDocumentIDs() throws {
        let folder = try makeFolder()
        let first = try Self.writeDocument("a.txt", sentences: 1, in: folder)
        let second = try Self.writeDocument("b.txt", sentences: 1, in: folder)

        XCTAssertNotEqual(DocumentLibrary.stableDocumentID(for: first), DocumentLibrary.stableDocumentID(for: second))
    }

    func testTheSamePathInsideTwoHomeDirectoriesGivesTheSameDocumentID() {
        let oldContainer = "/Users/golden/Containers/0A1B2C3D-0000-4000-8000-000000000001"
        let newContainer = "/Users/golden/Containers/0A1B2C3D-0000-4000-8000-000000000002"
        let before = URL(fileURLWithPath: oldContainer + "/Documents/report.pdf")
        let after = URL(fileURLWithPath: newContainer + "/Documents/report.pdf")

        XCTAssertEqual(DocumentLibrary.documentIDName(for: before, homeDirectory: oldContainer), "home:Documents/report.pdf")
        XCTAssertEqual(DocumentLibrary.stableDocumentID(for: before, homeDirectory: oldContainer),
                       DocumentLibrary.stableDocumentID(for: after, homeDirectory: newContainer))
    }

    func testAPathOutsideTheHomeDirectoryUsesItsAbsoluteForm() {
        let shared = URL(fileURLWithPath: "/Volumes/Shared/report.pdf")
        let sibling = URL(fileURLWithPath: "/Users/golden-twin/report.pdf")

        XCTAssertEqual(DocumentLibrary.documentIDName(for: shared, homeDirectory: "/Users/golden"),
                       "path:/Volumes/Shared/report.pdf")
        XCTAssertEqual(DocumentLibrary.documentIDName(for: sibling, homeDirectory: "/Users/golden"),
                       "path:/Users/golden-twin/report.pdf")
        XCTAssertEqual(DocumentLibrary.stableDocumentID(for: shared, homeDirectory: "/Users/golden"),
                       DocumentLibrary.stableDocumentID(for: shared, homeDirectory: "/Users/someone-else"))
    }

    // MARK: - Re-adding and older databases

    func testAddingTheSameFileThroughTwoLibrariesStoresOneDocument() async throws {
        let (library, folder) = try await makeLibrary()
        let document = try Self.writeDocument("twice.txt", sentences: 12, in: folder)
        await library.configure(embeddingProvider: StubEmbedder("stub/a", dimensions: 4), llm: await Self.llm())
        let first = try await library.add(url: document)
        await library.close()

        let relaunched = DocumentLibrary(directory: folder, chunkTargetTokens: 24)
        try await relaunched.open()
        await relaunched.configure(embeddingProvider: StubEmbedder("stub/a", dimensions: 4), llm: await Self.llm())
        let second = try await relaunched.add(url: document)

        XCTAssertEqual(second.id, first.id)
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM documents;", in: folder), 1)
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM chunks;", in: folder), first.chunkCount)
    }

    func testAddReturnsALegacyEntryForTheSameURLWithoutIndexingAgain() async throws {
        let (library, folder) = try await makeLibrary()
        let document = try Self.writeDocument("legacy.txt", sentences: 12, in: folder)
        let legacyID = UUID()
        try Self.insertLegacyDocument(legacyID.uuidString, url: document.absoluteString,
                                      indexedAt: "2026-01-01T00:00:00.000Z", chunks: 1, in: folder)
        await library.configure(embeddingProvider: StubEmbedder("stub/a", dimensions: 4), llm: await Self.llm())

        let added = try await library.add(url: document)

        XCTAssertEqual(added.id, legacyID)
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM documents;", in: folder), 1)
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM chunks;", in: folder), 1)
    }

    func testAddAdoptsTheLegacyEntryWithChunksOverANewerOneWithout() async throws {
        let (library, folder) = try await makeLibrary()
        let document = try Self.writeDocument("legacy.txt", sentences: 12, in: folder)
        let complete = UUID()
        try Self.insertLegacyDocument(complete.uuidString, url: document.absoluteString,
                                      indexedAt: "2026-01-01T00:00:00.000Z", chunks: 2, in: folder)
        try Self.insertDocumentWithoutChunks(UUID().uuidString, url: document.absoluteString,
                                             indexedAt: "2026-02-01T00:00:00.000Z", in: folder)
        await library.configure(embeddingProvider: StubEmbedder("stub/a", dimensions: 4), llm: await Self.llm())

        let added = try await library.add(url: document)

        XCTAssertEqual(added.id, complete)
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM chunks;", in: folder), 2)
    }

    func testAddIndexesAgainAFileWhoseOnlyEntryHasNoChunks() async throws {
        let (library, folder) = try await makeLibrary()
        let document = try Self.writeDocument("interrupted.txt", sentences: 12, in: folder)
        let empty = UUID().uuidString
        try Self.insertDocumentWithoutChunks(empty, url: document.absoluteString,
                                             indexedAt: "2026-01-01T00:00:00.000Z", in: folder)
        await library.configure(embeddingProvider: StubEmbedder("stub/a", dimensions: 4), llm: await Self.llm())

        let added = try await library.add(url: document)

        XCTAssertEqual(added.id, DocumentLibrary.stableDocumentID(for: document))
        XCTAssertGreaterThan(added.chunkCount, 0)
        XCTAssertEqual(try Self.rowCounts(of: [empty, added.id.uuidString], in: folder), [[0, 0], [1, added.chunkCount]])
    }

    func testAFailedChunkInsertStoresNoDocumentSoTheNextAddIndexesTheFile() async throws {
        let (library, folder) = try await makeLibrary()
        let document = try Self.writeDocument("failing.txt", sentences: 12, in: folder)
        await library.configure(embeddingProvider: StubEmbedder("stub/a", dimensions: 4), llm: await Self.llm())
        try Self.scalar("CREATE TRIGGER fail_chunks BEFORE INSERT ON chunks BEGIN SELECT RAISE(ABORT, 'full'); END;",
                        in: folder)

        var failed = false
        do {
            try await library.add(url: document)
        } catch {
            failed = true
        }
        let documentsAfterFailure = try Self.scalar("SELECT count(*) FROM documents;", in: folder)
        try Self.scalar("DROP TRIGGER fail_chunks;", in: folder)
        let added = try await library.add(url: document)

        XCTAssertTrue(failed)
        XCTAssertEqual(documentsAfterFailure, 0)
        XCTAssertGreaterThan(added.chunkCount, 0)
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM chunks;", in: folder), added.chunkCount)
    }

    func testConcurrentAddsOfTheSameFileStoreItOnce() async throws {
        let (library, folder) = try await makeLibrary()
        let document = try Self.writeDocument("racing.txt", sentences: 12, in: folder)
        let slow = StubEmbedder("stub/a", dimensions: 4, delay: .milliseconds(50))
        await library.configure(embeddingProvider: slow, llm: await Self.llm())

        async let first = library.add(url: document)
        async let second = library.add(url: document)
        let added = try await [first, second]

        XCTAssertEqual(added[0].id, added[1].id)
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM documents;", in: folder), 1)
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM chunks;", in: folder), added[0].chunkCount)
    }

    func testOpeningKeepsOnlyTheNewestEntryPerURL() async throws {
        let (library, folder) = try await makeLibrary()
        await library.close()
        let url = folder.appendingPathComponent("dup.txt").absoluteString
        let older = "10000000-0000-4000-8000-000000000000"
        let tiedButLower = "20000000-0000-4000-8000-000000000000"
        let newest = "30000000-0000-4000-8000-000000000000"
        let unrelated = "40000000-0000-4000-8000-000000000000"
        try Self.insertLegacyDocument(older, url: url, indexedAt: "2026-01-01T00:00:00.000Z", chunks: 2, in: folder)
        try Self.insertLegacyDocument(newest, url: url, indexedAt: "2026-02-01T00:00:00.000Z", chunks: 3, in: folder)
        try Self.insertLegacyDocument(tiedButLower, url: url, indexedAt: "2026-02-01T00:00:00.000Z", chunks: 1, in: folder)
        try Self.insertLegacyDocument(unrelated, url: folder.appendingPathComponent("other.txt").absoluteString,
                                      indexedAt: "2025-06-01T00:00:00.000Z", chunks: 1, in: folder)
        let ids = [older, tiedButLower, newest, unrelated]

        try await Self.reopen(folder)
        let afterFirstOpen = try Self.rowCounts(of: ids, in: folder)
        let matches = try Self.scalar("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'alpha';", in: folder)
        try await Self.reopen(folder)

        XCTAssertEqual(afterFirstOpen, [[0, 0], [0, 0], [1, 3], [1, 1]], "[documents, chunks] per ID")
        XCTAssertEqual(matches, try Self.scalar("SELECT count(*) FROM chunks;", in: folder))
        XCTAssertNoThrow(try Self.scalar(Self.fullTextIntegrityCheck, in: folder))
        XCTAssertEqual(try Self.rowCounts(of: ids, in: folder), afterFirstOpen)
    }

    func testOpeningKeepsAnEntryWithChunksOverANewerOneWithout() async throws {
        let (library, folder) = try await makeLibrary()
        await library.close()
        let url = folder.appendingPathComponent("interrupted.txt").absoluteString
        let complete = "10000000-0000-4000-8000-000000000000"
        let empty = "20000000-0000-4000-8000-000000000000"
        try Self.insertLegacyDocument(complete, url: url, indexedAt: "2026-01-01T00:00:00.000Z", chunks: 2, in: folder)
        try Self.insertDocumentWithoutChunks(empty, url: url, indexedAt: "2026-02-01T00:00:00.000Z", in: folder)

        try await Self.reopen(folder)

        XCTAssertEqual(try Self.rowCounts(of: [complete, empty], in: folder), [[1, 2], [0, 0]], "[documents, chunks] per ID")
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'alpha';", in: folder), 2)
    }

    func testOpeningRepairsTheFullTextIndexOfADatabaseWithTheOlderTriggers() async throws {
        let (library, folder) = try await makeLibrary()
        await library.close()
        try Self.scalar("""
            DROP TRIGGER chunks_ai;
            DROP TRIGGER chunks_ad;
            CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(id, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, id, text) VALUES('delete', old.id, old.text);
            END;
            """, in: folder)
        let kept = UUID()
        let removed = UUID().uuidString
        try Self.insertLegacyDocument(kept.uuidString, url: "file:///kept.txt", indexedAt: "2026-01-01T00:00:00.000Z",
                                      chunks: 2, in: folder)
        try Self.insertLegacyDocument(removed, url: "file:///removed.txt", indexedAt: "2026-01-01T00:00:00.000Z",
                                      chunks: 3, in: folder)
        try Self.scalar("DELETE FROM chunks WHERE document_id = '\(removed)';", in: folder)
        XCTAssertThrowsError(try Self.scalar(Self.fullTextIntegrityCheck, in: folder), "the older delete trigger is a no-op")

        let reopened = DocumentLibrary(directory: folder)
        try await reopened.open()
        let matchesAfterRepair = try Self.scalar("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'alpha';", in: folder)
        try await reopened.removeDocument(id: kept)

        XCTAssertEqual(matchesAfterRepair, 2)
        XCTAssertEqual(try Self.scalar("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'alpha';", in: folder), 0)
        XCTAssertNoThrow(try Self.scalar(Self.fullTextIntegrityCheck, in: folder))
    }

    // MARK: - Helpers

    private func makeFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("AuraDocsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        return folder
    }

    private func makeLibrary(chunkTokens: Int = 24) async throws -> (DocumentLibrary, URL) {
        let folder = try makeFolder()
        let library = DocumentLibrary(directory: folder, chunkTargetTokens: chunkTokens)
        try await library.open()
        return (library, folder)
    }

    private static func reopen(_ folder: URL) async throws {
        let library = DocumentLibrary(directory: folder)
        try await library.open()
        await library.close()
    }

    /// Rank 1 also checks the index against the `chunks` content table.
    private static let fullTextIntegrityCheck = "INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1);"

    /// The same path through the other side of the /var -> /private/var symlink, when that exists on this host.
    private static func privateVarSpellings(of path: String) -> [String] {
        let other = path.hasPrefix("/private/") ? String(path.dropFirst("/private".count)) : "/private" + path
        return FileManager.default.fileExists(atPath: other) ? [other] : []
    }

    /// A document row plus `chunks` chunks mentioning "alpha", written straight to the database.
    private static func insertLegacyDocument(
        _ id: String, url: String, indexedAt: String, chunks: Int, in folder: URL
    ) throws {
        let chunkRows = (0..<chunks).map { index in
            """
            INSERT INTO chunks (id, document_id, document_title, text)
            VALUES ('\(UUID().uuidString)', '\(id)', 'legacy', 'alpha chunk \(index)');
            """
        }
        try scalar("""
            INSERT INTO documents (id, title, url, chunk_count, indexed_at)
            VALUES ('\(id)', 'legacy', '\(url)', \(chunks), '\(indexedAt)');
            """ + chunkRows.joined(separator: "\n"), in: folder)
    }

    /// A document row claiming 3 chunks with none stored, as an interrupted older version left one.
    private static func insertDocumentWithoutChunks(_ id: String, url: String, indexedAt: String, in folder: URL) throws {
        try scalar("""
            INSERT INTO documents (id, title, url, chunk_count, indexed_at)
            VALUES ('\(id)', 'legacy', '\(url)', 3, '\(indexedAt)');
            """, in: folder)
    }

    /// `[documents, chunks]` stored under each ID.
    private static func rowCounts(of ids: [String], in folder: URL) throws -> [[Int]] {
        try ids.map { id in
            [try scalar("SELECT count(*) FROM documents WHERE id = '\(id)';", in: folder),
             try scalar("SELECT count(*) FROM chunks WHERE document_id = '\(id)';", in: folder)]
        }
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
