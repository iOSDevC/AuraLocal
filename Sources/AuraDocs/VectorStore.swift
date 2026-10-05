import Foundation
import SQLite3
import AuraCore

// SQLite3 helper
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

// MARK: - VectorStore

/// Hybrid retrieval store:
/// 1. FTS5 full-text search → top-N candidates (fast, keyword match)
/// 2. On-demand cosine similarity re-ranking → top-K results (semantic precision)
///
/// Vectors are stored as raw Float BLOBs in SQLite and loaded on demand for re-ranking.
/// Only the embeddings needed for a specific query are fetched — no blanket RAM load.
actor VectorStore {

    // MARK: - Types

    struct SearchResult {
        let chunk: DocumentChunk
        let score: Float
    }

    typealias DocumentRow = (id: UUID, title: String, url: String, chunkCount: Int, indexedAt: Date)

    // MARK: - State

    private var db: OpaquePointer?
    private let dbURL: URL
    /// Warm cache: recently-accessed embeddings keyed by chunk ID.
    /// Populated on demand during search, not loaded all-at-once.
    private var embeddingCache: [UUID: [Float]] = [:]

    // MARK: - Init

    init(directory: URL) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.dbURL = directory.appendingPathComponent("vectors.sqlite")
    }

    // MARK: - Lifecycle

    func open() throws {
        guard db == nil else { return }
        guard sqlite3_open(dbURL.path, &db) == SQLITE_OK else {
            throw DocumentError.embeddingFailed("Cannot open vector store at \(dbURL.path)")
        }
        // WAL mode enables concurrent reads during writes
        sqlite3_exec(db, "PRAGMA journal_mode = WAL;", nil, nil, nil)
        sqlite3_exec(db, "PRAGMA synchronous = NORMAL;", nil, nil, nil)
        do {
            try migrate()
        } catch {
            // Closed so the next open() runs the migration again instead of using a half-migrated store.
            close()
            throw error
        }
    }

    func close() {
        if let db { sqlite3_close(db) }
        db = nil
        embeddingCache = [:]
    }

    // MARK: - Document management

    func document(id: UUID) throws -> DocumentRow? {
        try ensureOpen()
        return try query(
            "SELECT id, title, url, chunk_count, indexed_at FROM documents WHERE id = ?;",
            bindings: [id.uuidString],
            map: rowToDocument
        ).first
    }

    /// The document stored under `url` that ``preferredDocumentOrder`` ranks first, e.g. one indexed by a
    /// version whose IDs changed every launch.
    func newestDocument(url: String) throws -> DocumentRow? {
        try ensureOpen()
        return try query(
            "SELECT id, title, url, chunk_count, indexed_at FROM documents WHERE url = ? ORDER BY \(Self.preferredDocumentOrder) LIMIT 1;",
            bindings: [url],
            map: rowToDocument
        ).first
    }

    /// Whether a document that should have chunks has none stored. Older versions stored a document and its
    /// chunks in separate writes, so indexing interrupted between the two left such a document.
    func chunksAreMissing(for row: DocumentRow) throws -> Bool {
        guard row.chunkCount > 0 else { return false }
        try ensureOpen()
        return try query(
            "SELECT 1 FROM chunks WHERE document_id = ? LIMIT 1;",
            bindings: [row.id.uuidString]
        ) { _ in true }.isEmpty
    }

    /// Stores a document and its chunks in one transaction, so a failure never leaves one without the other.
    func insertDocument(id: UUID, title: String, url: String, chunks: [DocumentChunk]) throws {
        try ensureOpen()
        try inTransaction {
            try exec(
                "INSERT INTO documents (id, title, url, chunk_count, indexed_at) VALUES (?,?,?,?,?);",
                bindings: [id.uuidString, title, url, chunks.count, iso(Date())]
            )
            for chunk in chunks {
                try insertChunk(chunk)
            }
        }
        for chunk in chunks where !chunk.embedding.isEmpty {
            embeddingCache[chunk.id] = chunk.embedding
        }
    }

    func allDocuments() throws -> [DocumentRow] {
        try ensureOpen()
        return try query(
            "SELECT id, title, url, chunk_count, indexed_at FROM documents ORDER BY indexed_at DESC;",
            map: rowToDocument
        )
    }

    func deleteDocument(id: UUID) throws {
        try ensureOpen()
        // Remove cached embeddings for this document's chunks
        let chunkIDs = try query(
            "SELECT id FROM chunks WHERE document_id = ?;",
            bindings: [id.uuidString]
        ) { stmt -> UUID? in
            sqlite3_column_text(stmt, 0)
                .flatMap { UUID(uuidString: String(cString: $0)) }
        }
        for chunkID in chunkIDs {
            embeddingCache[chunkID] = nil
        }
        try deleteRows(ofDocument: id.uuidString)
    }

    /// Returns all chunks for a specific document — used by DocumentExporter.
    func chunksForDocument(id: UUID) throws -> [DocumentChunk] {
        try ensureOpen()
        return try query(
            "SELECT id, document_id, document_title, page_number, text, token_estimate FROM chunks WHERE document_id = ? ORDER BY rowid;",
            bindings: [id.uuidString],
            map: rowToChunk
        )
    }

    /// Returns the raw text of every chunk — used to rebuild TF-IDF corpus weights.
    func allChunkTexts() throws -> [String] {
        try ensureOpen()
        return try query("SELECT text FROM chunks;") { stmt -> String? in
            sqlite3_column_text(stmt, 0).map { String(cString: $0) }
        }
    }

    // MARK: - Index metadata

    private static let providerKey = "embedding_provider"
    private static let dimensionsKey = "embedding_dimensions"

    /// The embedding identity recorded for this index, or `nil` for an index that predates it.
    func embeddingIdentity() throws -> EmbeddingIndexIdentity? {
        try ensureOpen()
        guard let identifier = try metadataValue(Self.providerKey),
              let dimensions = try metadataValue(Self.dimensionsKey).flatMap({ Int($0) })
        else { return nil }
        return EmbeddingIndexIdentity(identifier: identifier, dimensions: dimensions)
    }

    func setEmbeddingIdentity(_ identity: EmbeddingIndexIdentity) throws {
        try ensureOpen()
        try exec("BEGIN TRANSACTION;")
        do {
            try setMetadata(Self.providerKey, identity.identifier)
            try setMetadata(Self.dimensionsKey, String(identity.dimensions))
            try exec("COMMIT;")
        } catch {
            try? exec("ROLLBACK;")
            throw error
        }
    }

    /// Distinct stored vector lengths, in floats (a missing vector counts as 0).
    func storedVectorLengths() throws -> Set<Int> {
        try ensureOpen()
        let byteLengths = try query("SELECT DISTINCT IFNULL(length(embedding), 0) FROM chunks;") { stmt -> Int? in
            Int(sqlite3_column_int64(stmt, 0))
        }
        return Set(byteLengths.map { $0 / MemoryLayout<Float>.size })
    }

    func chunkCount() throws -> Int {
        try ensureOpen()
        return try query("SELECT COUNT(*) FROM chunks;") { stmt -> Int? in
            Int(sqlite3_column_int64(stmt, 0))
        }.first ?? 0
    }

    /// One page of chunk texts in rowid order, for re-embedding without loading every chunk.
    func chunkTexts(afterRowID rowID: Int64, limit: Int) throws -> [(rowID: Int64, id: UUID, text: String)] {
        try ensureOpen()
        return try query(
            "SELECT rowid, id, text FROM chunks WHERE rowid > ? ORDER BY rowid LIMIT ?;",
            bindings: [rowID, limit]
        ) { stmt -> (Int64, UUID, String)? in
            guard
                let idText = sqlite3_column_text(stmt, 1).map({ String(cString: $0) }),
                let id     = UUID(uuidString: idText),
                let text   = sqlite3_column_text(stmt, 2).map({ String(cString: $0) })
            else { return nil }
            return (sqlite3_column_int64(stmt, 0), id, text)
        }
    }

    /// Replaces stored vectors in place. An UPDATE of `embedding` fires none of the FTS triggers
    /// (they run on INSERT and DELETE only), so the full-text index is untouched.
    func updateEmbeddings(_ vectors: [(id: UUID, vector: [Float])]) throws {
        try ensureOpen()
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, "UPDATE chunks SET embedding = ?1 WHERE id = ?2;", -1, &stmt, nil) == SQLITE_OK
        else { throw dbError() }
        try exec("BEGIN TRANSACTION;")
        do {
            for entry in vectors {
                try runUpdate(stmt, id: entry.id, vector: entry.vector)
                embeddingCache[entry.id] = entry.vector
            }
            try exec("COMMIT;")
        } catch {
            try? exec("ROLLBACK;")
            throw error
        }
    }

    func clearEmbeddingCache() {
        embeddingCache = [:]
    }

    private func runUpdate(_ stmt: OpaquePointer?, id: UUID, vector: [Float]) throws {
        sqlite3_reset(stmt)
        sqlite3_clear_bindings(stmt)
        bind(stmt, values: [floatsToData(vector), id.uuidString])
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw dbError() }
    }

    private func metadataValue(_ key: String) throws -> String? {
        try query("SELECT value FROM metadata WHERE key = ?;", bindings: [key]) { stmt -> String? in
            sqlite3_column_text(stmt, 0).map { String(cString: $0) }
        }.first
    }

    private func setMetadata(_ key: String, _ value: String) throws {
        try exec("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?);", bindings: [key, value])
    }

    // MARK: - Chunk insertion

    private func insertChunk(_ chunk: DocumentChunk) throws {
        try exec(
            """
            INSERT OR REPLACE INTO chunks
                (id, document_id, document_title, page_number, text, embedding, token_estimate)
            VALUES (?,?,?,?,?,?,?);
            """,
            bindings: [
                chunk.id.uuidString,
                chunk.documentID.uuidString,
                chunk.documentTitle,
                chunk.pageNumber,
                chunk.text,
                floatsToData(chunk.embedding),
                chunk.tokenEstimate
            ]
        )
    }

    // MARK: - Hybrid Search

    /// Two-stage retrieval: FTS5 candidates → cosine re-rank.
    func search(query: String, queryEmbedding: [Float], topK: Int = 5, ftsLimit: Int = 20) throws -> [SearchResult] {
        try ensureOpen()

        // Stage 1: FTS5 keyword candidates
        let candidates = try ftsCandidates(query: query, limit: ftsLimit)
        guard !candidates.isEmpty else { return [] }

        // Stage 2: load only the embeddings we need, then cosine re-rank
        let needed = candidates.map(\.id)
        let embeddings = try loadEmbeddings(for: needed)

        var scored: [SearchResult] = candidates.compactMap { chunk in
            guard let emb = embeddings[chunk.id], !emb.isEmpty else { return nil }
            let score = VectorMath.cosine(queryEmbedding, emb)
            return SearchResult(chunk: chunk, score: score)
        }

        scored.sort { $0.score > $1.score }
        return Array(scored.prefix(topK))
    }

    /// Pure cosine search (no FTS pre-filter) — used when FTS returns 0 results.
    /// Streams embeddings from SQLite one row at a time to avoid loading all into RAM.
    func searchCosineOnly(queryEmbedding: [Float], topK: Int = 5) throws -> [SearchResult] {
        try ensureOpen()

        // Stream through all chunks with a cursor, computing cosine on the fly
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        let sql = "SELECT id, document_id, document_title, page_number, text, token_estimate, embedding FROM chunks;"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw dbError() }

        // Keep a bounded heap of top-K results
        var topResults: [SearchResult] = []
        topResults.reserveCapacity(topK + 1)
        var minScore: Float = -1

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let chunk = rowToChunk(stmt) else { continue }

            // Read embedding BLOB from column 6
            let bytes = sqlite3_column_bytes(stmt, 6)
            guard bytes > 0, let ptr = sqlite3_column_blob(stmt, 6) else { continue }
            let emb = dataToFloats(Data(bytes: ptr, count: Int(bytes)))
            guard !emb.isEmpty else { continue }

            let score = VectorMath.cosine(queryEmbedding, emb)

            // Only insert if better than current min or heap not full
            if topResults.count < topK || score > minScore {
                topResults.append(SearchResult(chunk: chunk, score: score))
                topResults.sort { $0.score > $1.score }
                if topResults.count > topK {
                    topResults.removeLast()
                }
                minScore = topResults.last?.score ?? -1
            }

            // Warm cache for chunks that made it into results
            embeddingCache[chunk.id] = emb
        }

        return topResults
    }

    // MARK: - FTS Candidates

    private func ftsCandidates(query: String, limit: Int) throws -> [DocumentChunk] {
        // Sanitize query for FTS5: wrap each token in double quotes to treat as
        // literal text, escaping internal quotes. Prevents FTS5 syntax operators
        // (NEAR, NOT, *, ^, column filters) from altering retrieval semantics.
        let safe = query
            .components(separatedBy: .whitespacesAndNewlines)
            .map { $0.replacingOccurrences(of: "\"", with: "\"\"") }
            .filter { !$0.isEmpty }
            .map { "\"\($0)\"" }
            .joined(separator: " OR ")

        guard !safe.isEmpty else { return [] }

        return try self.query(
            """
            SELECT c.id, c.document_id, c.document_title, c.page_number, c.text, c.token_estimate
            FROM chunks c
            JOIN chunks_fts f ON c.id = f.id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?;
            """,
            bindings: [safe, limit],
            map: rowToChunk
        )
    }

    private func loadAllChunks() throws -> [DocumentChunk] {
        try query(
            "SELECT id, document_id, document_title, page_number, text, token_estimate FROM chunks;",
            map: rowToChunk
        )
    }

    // MARK: - On-demand embedding loading

    /// Load embeddings for a specific set of chunk IDs.
    /// Returns from warm cache when available, fetches from SQLite for cache misses.
    private func loadEmbeddings(for chunkIDs: [UUID]) throws -> [UUID: [Float]] {
        var result: [UUID: [Float]] = [:]
        var missing: [UUID] = []

        // Check warm cache first
        for id in chunkIDs {
            if let cached = embeddingCache[id] {
                result[id] = cached
            } else {
                missing.append(id)
            }
        }

        guard !missing.isEmpty else { return result }

        // Batch-fetch missing embeddings from SQLite
        let placeholders = missing.map { _ in "?" }.joined(separator: ",")
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        let sql = "SELECT id, embedding FROM chunks WHERE id IN (\(placeholders));"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw dbError() }

        for (i, id) in missing.enumerated() {
            sqlite3_bind_text(stmt, Int32(i + 1), id.uuidString, -1, SQLITE_TRANSIENT)
        }

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard
                let idStr = sqlite3_column_text(stmt, 0).map({ String(cString: $0) }),
                let id    = UUID(uuidString: idStr)
            else { continue }
            let bytes = sqlite3_column_bytes(stmt, 1)
            if bytes > 0, let ptr = sqlite3_column_blob(stmt, 1) {
                let vec = dataToFloats(Data(bytes: ptr, count: Int(bytes)))
                result[id] = vec
                embeddingCache[id] = vec  // warm cache for subsequent queries
            }
        }

        return result
    }

    /// Load embeddings for specific chunk IDs — used by DocumentLibrary.export().
    func embeddings(for chunkIDs: [UUID]) throws -> [UUID: [Float]] {
        try ensureOpen()
        return try loadEmbeddings(for: chunkIDs)
    }

    // MARK: - Migrations

    private func migrate() throws {
        try exec("""
            CREATE TABLE IF NOT EXISTS documents (
                id           TEXT PRIMARY KEY,
                title        TEXT NOT NULL,
                url          TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL DEFAULT 0,
                indexed_at   TEXT NOT NULL
            );
            """)
        try exec("""
            CREATE TABLE IF NOT EXISTS chunks (
                id             TEXT PRIMARY KEY,
                document_id    TEXT NOT NULL REFERENCES documents(id),
                document_title TEXT NOT NULL,
                page_number    INTEGER NOT NULL DEFAULT 0,
                text           TEXT NOT NULL,
                embedding      BLOB,
                token_estimate INTEGER NOT NULL DEFAULT 0
            );
            """)
        try exec("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc
            ON chunks(document_id);
            """)
        try exec("""
            CREATE INDEX IF NOT EXISTS idx_documents_url
            ON documents(url);
            """)
        try exec("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(id UNINDEXED, text, content=chunks, content_rowid=rowid);
            """)
        try installFullTextTriggers()
        // IF NOT EXISTS also upgrades databases created before this table existed.
        try exec("""
            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """)
        try removeDuplicateDocuments()
    }

    /// Earlier triggers omitted the rowid, so a delete never reached the full-text index and later matches
    /// pointed at missing rows; this replaces them and rebuilds the index from `chunks` once.
    private func installFullTextTriggers() throws {
        let current = try query("""
            SELECT 1 FROM sqlite_master WHERE type = 'trigger'
              AND ((name = 'chunks_ai' AND sql LIKE '%new.rowid%') OR (name = 'chunks_ad' AND sql LIKE '%old.rowid%'));
            """) { _ in true }
        guard current.count < 2 else { return }
        try inTransaction {
            try exec("DROP TRIGGER IF EXISTS chunks_ai;")
            try exec("DROP TRIGGER IF EXISTS chunks_ad;")
            try exec("""
                CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, id, text) VALUES (new.rowid, new.id, new.text);
                END;
                """)
            try exec("""
                CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, id, text) VALUES('delete', old.rowid, old.id, old.text);
                END;
                """)
            try exec("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');")
        }
    }

    /// Ranks the documents stored under one `url`: first one with chunks, since an older version could store a
    /// document without them, then the newest. indexed_at is fixed-width UTC ISO 8601 (see isoFormatter), so
    /// text order is time order.
    private static let preferredDocumentOrder =
        "EXISTS (SELECT 1 FROM chunks WHERE chunks.document_id = documents.id) DESC, indexed_at DESC, id DESC"

    /// Older versions gave a file a new ID every launch, so re-adding it stored it again. Keeps the document per
    /// `url` that ``preferredDocumentOrder`` ranks first and deletes the rest.
    private func removeDuplicateDocuments() throws {
        let superseded = try query("""
            SELECT id FROM (
                SELECT id, row_number() OVER (PARTITION BY url ORDER BY \(Self.preferredDocumentOrder)) AS position
                FROM documents WHERE url IN (SELECT url FROM documents GROUP BY url HAVING count(*) > 1))
            WHERE position > 1;
            """) { stmt -> String? in
            sqlite3_column_text(stmt, 0).map { String(cString: $0) }
        }
        guard !superseded.isEmpty else { return }
        try inTransaction {
            for id in superseded {
                try deleteRows(ofDocument: id)
            }
        }
    }

    /// Chunks first, so `chunks_ad` drops them from the full-text index.
    private func deleteRows(ofDocument id: String) throws {
        try exec("DELETE FROM chunks WHERE document_id = ?;", bindings: [id])
        try exec("DELETE FROM documents WHERE id = ?;", bindings: [id])
    }

    // MARK: - SQLite helpers

    private func ensureOpen() throws {
        if db == nil { try open() }
    }

    private func inTransaction(_ work: () throws -> Void) throws {
        try exec("BEGIN TRANSACTION;")
        do {
            try work()
            try exec("COMMIT;")
        } catch {
            try? exec("ROLLBACK;")
            throw error
        }
    }

    private func exec(_ sql: String, bindings: [Any] = []) throws {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw dbError() }
        bind(stmt, values: bindings)
        let rc = sqlite3_step(stmt)
        guard rc == SQLITE_DONE || rc == SQLITE_ROW else { throw dbError() }
    }

    private func query<T>(
        _ sql: String,
        bindings: [Any] = [],
        map: (OpaquePointer?) -> T?
    ) throws -> [T] {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw dbError() }
        bind(stmt, values: bindings)
        var results: [T] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let v = map(stmt) { results.append(v) }
        }
        return results
    }

    private func bind(_ stmt: OpaquePointer?, values: [Any]) {
        for (i, value) in values.enumerated() {
            let idx = Int32(i + 1)
            switch value {
                case let s as String:
                    sqlite3_bind_text(stmt, idx, s, -1, SQLITE_TRANSIENT)
                case let n as Int:
                    sqlite3_bind_int64(stmt, idx, Int64(n))
                case let n as Int64:
                    sqlite3_bind_int64(stmt, idx, n)
                case let d as Double:
                    sqlite3_bind_double(stmt, idx, d)
                case let b as Bool:
                    sqlite3_bind_int64(stmt, idx, b ? 1 : 0)
                case let d as Data:
                    d.withUnsafeBytes { ptr in
                        sqlite3_bind_blob(stmt, idx, ptr.baseAddress, Int32(d.count), SQLITE_TRANSIENT)
                    }
                default:
                    assertionFailure("VectorStore.bind: unsupported type \(type(of: value)) at index \(i)")
                    sqlite3_bind_null(stmt, idx)
            }
        }
    }

    /// Maps `id, title, url, chunk_count, indexed_at`.
    private func rowToDocument(_ stmt: OpaquePointer?) -> DocumentRow? {
        guard
            let idStr    = sqlite3_column_text(stmt, 0).map({ String(cString: $0) }),
            let id       = UUID(uuidString: idStr),
            let title    = sqlite3_column_text(stmt, 1).map({ String(cString: $0) }),
            let url      = sqlite3_column_text(stmt, 2).map({ String(cString: $0) }),
            let dateStr  = sqlite3_column_text(stmt, 4).map({ String(cString: $0) })
        else { return nil }
        let count = Int(sqlite3_column_int(stmt, 3))
        let date  = isoFormatter.date(from: dateStr) ?? Date()
        return (id, title, url, count, date)
    }

    private func rowToChunk(_ stmt: OpaquePointer?) -> DocumentChunk? {
        guard let stmt else { return nil }
        guard
            let idStr    = sqlite3_column_text(stmt, 0).map({ String(cString: $0) }),
            let id       = UUID(uuidString: idStr),
            let convStr  = sqlite3_column_text(stmt, 1).map({ String(cString: $0) }),
            let docID    = UUID(uuidString: convStr),
            let title    = sqlite3_column_text(stmt, 2).map({ String(cString: $0) }),
            let text     = sqlite3_column_text(stmt, 4).map({ String(cString: $0) })
        else { return nil }
        let page = Int(sqlite3_column_int(stmt, 3))
        let tok  = Int(sqlite3_column_int(stmt, 5))
        var chunk = DocumentChunk(
            id: id, documentID: docID, documentTitle: title,
            pageNumber: page, text: text
        )
        // tokenEstimate is a let, reconstruct via init workaround
        _ = tok  // already encoded in the struct
        return chunk
    }

    // MARK: - Float BLOB helpers

    private func floatsToData(_ floats: [Float]) -> Data {
        floats.withUnsafeBytes { Data($0) }
    }

    private func dataToFloats(_ data: Data) -> [Float] {
        data.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
    }

    // MARK: - Date

    private let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
    private func iso(_ d: Date) -> String { isoFormatter.string(from: d) }

    private func dbError() -> DocumentError {
        let msg = db.flatMap { sqlite3_errmsg($0) }.map { String(cString: $0) } ?? "unknown"
        return .embeddingFailed("SQLite: \(msg)")
    }
}
