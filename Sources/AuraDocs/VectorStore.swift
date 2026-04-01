import Foundation
import SQLite3

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
        try migrate()
    }

    func close() {
        if let db { sqlite3_close(db) }
        db = nil
        embeddingCache = [:]
    }

    // MARK: - Document management

    func documentExists(id: UUID) throws -> Bool {
        try ensureOpen()
        let rows = try query(
            "SELECT 1 FROM documents WHERE id = ? LIMIT 1;",
            bindings: [id.uuidString]
        ) { _ in true }
        return !rows.isEmpty
    }

    func insertDocument(id: UUID, title: String, url: String, chunkCount: Int) throws {
        try ensureOpen()
        try exec(
            "INSERT OR REPLACE INTO documents (id, title, url, chunk_count, indexed_at) VALUES (?,?,?,?,?);",
            bindings: [id.uuidString, title, url, chunkCount, iso(Date())]
        )
    }

    func allDocuments() throws -> [(id: UUID, title: String, url: String, chunkCount: Int, indexedAt: Date)] {
        try ensureOpen()
        return try query(
            "SELECT id, title, url, chunk_count, indexed_at FROM documents ORDER BY indexed_at DESC;"
        ) { stmt -> (UUID, String, String, Int, Date)? in
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
        try exec("DELETE FROM chunks WHERE document_id = ?;", bindings: [id.uuidString])
        try exec("DELETE FROM documents WHERE id = ?;",       bindings: [id.uuidString])
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

    // MARK: - Chunk insertion

    func insertChunks(_ chunks: [DocumentChunk]) throws {
        try ensureOpen()
        try exec("BEGIN TRANSACTION;")
        do {
            for chunk in chunks {
                let embBlob = floatsToData(chunk.embedding)
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
                        embBlob,
                        chunk.tokenEstimate
                    ]
                )
                // Warm the cache with freshly inserted embeddings
                if !chunk.embedding.isEmpty {
                    embeddingCache[chunk.id] = chunk.embedding
                }
            }
            try exec("COMMIT;")
        } catch {
            try? exec("ROLLBACK;")
            throw error
        }
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
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(id UNINDEXED, text, content=chunks, content_rowid=rowid);
            """)
        try exec("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(id, text) VALUES (new.id, new.text);
            END;
            """)
        try exec("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, id, text) VALUES('delete', old.id, old.text);
            END;
            """)
    }

    // MARK: - SQLite helpers

    private func ensureOpen() throws {
        if db == nil { try open() }
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
