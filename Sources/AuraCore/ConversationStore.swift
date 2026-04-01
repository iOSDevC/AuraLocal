import Foundation
import SQLite3


// SQLite3 helper — SQLITE_TRANSIENT is a C macro Swift can't import directly.
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

// MARK: - Domain Models

/// A single turn in a conversation (user or assistant message).
public struct Turn: Identifiable, Codable, Sendable, Equatable {
    public let id: UUID
    public let conversationID: UUID
    public let role: Role
    public let content: String
    public let createdAt: Date
    /// Token count estimate — used to enforce context window budgets.
    public let tokenEstimate: Int
    
    public enum Role: String, Codable, Sendable {
        case system
        case user
        case assistant
    }
    
    public init(
        id: UUID = UUID(),
        conversationID: UUID,
        role: Role,
        content: String,
        createdAt: Date = Date(),
        tokenEstimate: Int? = nil
    ) {
        self.id = id
        self.conversationID = conversationID
        self.role = role
        self.content = content
        self.createdAt = createdAt
        // ~4 chars per token — good enough for budget pruning
        self.tokenEstimate = tokenEstimate ?? max(1, content.count / 4)
    }
}

/// A named conversation thread, tied to a specific model.
public struct Conversation: Identifiable, Codable, Sendable, Equatable {
    public let id: UUID
    public var title: String
    public let model: String          // Model.rawValue
    public let createdAt: Date
    public var updatedAt: Date
    public var turnCount: Int
    
    public init(
        id: UUID = UUID(),
        title: String = "New conversation",
        model: String,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        turnCount: Int = 0
    ) {
        self.id = id
        self.title = title
        self.model = model
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.turnCount = turnCount
    }
}

// MARK: - ConversationStore

/// SQLite-backed persistent store for conversations and turns.
///
/// All read/write operations are async and actor-isolated.
/// The store is safe to share across the app — use `ConversationStore.shared`.
///
/// ```swift
/// // Create a conversation
/// var conv = try await ConversationStore.shared.createConversation(model: .qwen3_1_7b)
///
/// // Append turns as chat progresses
/// try await store.appendTurn(Turn(conversationID: conv.id, role: .user, content: prompt))
/// try await store.appendTurn(Turn(conversationID: conv.id, role: .assistant, content: reply))
///
/// // Load context window for next inference
/// let context = try await store.contextWindow(for: conv.id, maxTokens: 2048)
/// ```
public actor ConversationStore {
    
    // MARK: - Singleton
    
    public static let shared = ConversationStore()
    
    // MARK: - Private state
    
    private var db: OpaquePointer?
    private let dbURL: URL
    
    // MARK: - Init
    
    public init(directory: URL? = nil) {
        let base = directory ?? FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AuraLocal", isDirectory: true)
        
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.dbURL = base.appendingPathComponent("conversations.sqlite")
    }
    
    // MARK: - Lifecycle
    
    /// Open the database and apply migrations. Must be called once before use.
    public func open() throws {
        guard db == nil else { return }
        if sqlite3_open(dbURL.path, &db) != SQLITE_OK {
            throw StoreError.cannotOpen(dbURL.path)
        }
        // WAL mode enables concurrent reads during writes — prevents SQLITE_BUSY
        // when the voice pipeline writes turns while the UI reads conversation lists.
        sqlite3_exec(db, "PRAGMA journal_mode = WAL;", nil, nil, nil)
        sqlite3_exec(db, "PRAGMA synchronous = NORMAL;", nil, nil, nil)
        try migrate()
    }
    
    /// Close the database connection.
    public func close() {
        if let db { sqlite3_close(db) }
        db = nil
    }
    
    // MARK: - Conversations
    
    /// Insert a new conversation row and return it.
    @discardableResult
    public func createConversation(
        model: Model,
        title: String = "New conversation"
    ) throws -> Conversation {
        try ensureOpen()
        let conv = Conversation(title: title, model: model.rawValue)
        let sql = """
            INSERT INTO conversations (id, title, model, created_at, updated_at, turn_count)
            VALUES (?, ?, ?, ?, ?, 0);
            """
        try exec(sql, bindings: [
            conv.id.uuidString,
            conv.title,
            conv.model,
            iso(conv.createdAt),
            iso(conv.updatedAt)
        ])
        return conv
    }
    
    /// Fetch all conversations ordered by most recently updated.
    public func allConversations() throws -> [Conversation] {
        try ensureOpen()
        return try query(
            "SELECT * FROM conversations ORDER BY updated_at DESC;",
            map: rowToConversation
        )
    }
    
    /// Fetch a single conversation by ID.
    public func conversation(id: UUID) throws -> Conversation? {
        try ensureOpen()
        return try query(
            "SELECT * FROM conversations WHERE id = ? LIMIT 1;",
            bindings: [id.uuidString],
            map: rowToConversation
        ).first
    }
    
    /// Update a conversation's title.
    public func updateTitle(_ title: String, for id: UUID) throws {
        try ensureOpen()
        try exec(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?;",
            bindings: [title, iso(Date()), id.uuidString]
        )
    }
    
    /// Delete a conversation and all its turns.
    public func deleteConversation(id: UUID) throws {
        try ensureOpen()
        try exec("DELETE FROM turns WHERE conversation_id = ?;", bindings: [id.uuidString])
        try exec("DELETE FROM conversations WHERE id = ?;", bindings: [id.uuidString])
    }
    
    // MARK: - Turns
    
    /// Append a turn to a conversation.
    @discardableResult
    public func appendTurn(_ turn: Turn) throws -> Turn {
        try ensureOpen()
        let sql = """
            INSERT INTO turns (id, conversation_id, role, content, created_at, token_estimate)
            VALUES (?, ?, ?, ?, ?, ?);
            """
        try exec(sql, bindings: [
            turn.id.uuidString,
            turn.conversationID.uuidString,
            turn.role.rawValue,
            turn.content,
            iso(turn.createdAt),
            turn.tokenEstimate
        ])
        // Update conversation metadata
        try exec(
            """
            UPDATE conversations
            SET turn_count = turn_count + 1, updated_at = ?
            WHERE id = ?;
            """,
            bindings: [iso(Date()), turn.conversationID.uuidString]
        )
        return turn
    }
    
    /// Load all turns for a conversation in chronological order.
    public func turns(for conversationID: UUID) throws -> [Turn] {
        try ensureOpen()
        return try query(
            "SELECT * FROM turns WHERE conversation_id = ? ORDER BY created_at ASC;",
            bindings: [conversationID.uuidString],
            map: rowToTurn
        )
    }
    
    /// Delete a single turn.
    public func deleteTurn(id: UUID) throws {
        try ensureOpen()
        try exec("DELETE FROM turns WHERE id = ?;", bindings: [id.uuidString])
    }
    
    // MARK: - Context Window
    
    /// Return the most recent turns that fit within `maxTokens`, always
    /// including the system prompt if one exists.
    ///
    /// This is the primary method for building an inference context:
    /// ```swift
    /// let context = try await store.contextWindow(for: conv.id, maxTokens: 2048)
    /// let messages = context.map { ["role": $0.role.rawValue, "content": $0.content] }
    /// ```
    public func contextWindow(for conversationID: UUID, maxTokens: Int = 2048) throws -> [Turn] {
        try ensureOpen()
        
        // Always include system turns first
        let systemTurns = try query(
            "SELECT * FROM turns WHERE conversation_id = ? AND role = 'system' ORDER BY created_at ASC;",
            bindings: [conversationID.uuidString],
            map: rowToTurn
        )
        
        // Load non-system turns newest-first so we can budget from the end
        let otherTurns = try query(
            "SELECT * FROM turns WHERE conversation_id = ? AND role != 'system' ORDER BY created_at DESC;",
            bindings: [conversationID.uuidString],
            map: rowToTurn
        )
        
        let systemTokens = systemTurns.reduce(0) { $0 + $1.tokenEstimate }
        var budget = maxTokens - systemTokens
        var selected: [Turn] = []
        
        for turn in otherTurns {
            guard budget > 0 else { break }
            selected.append(turn)
            budget -= turn.tokenEstimate
        }
        
        // Restore chronological order and prepend system turns
        return systemTurns + selected.reversed()
    }
    
    /// Summarize and prune old turns to keep the conversation under a token budget.
    /// The summary turn replaces all pruned turns, preserving semantic continuity.
    public func pruneAndSummarize(
        conversationID: UUID,
        keepLastN: Int = 10,
        summary: String
    ) throws {
        try ensureOpen()
        
        // Find the IDs of turns to prune (all except last N non-system turns)
        let allTurns = try query(
            "SELECT * FROM turns WHERE conversation_id = ? AND role != 'system' ORDER BY created_at ASC;",
            bindings: [conversationID.uuidString],
            map: rowToTurn
        )
        
        let pruneCount = max(0, allTurns.count - keepLastN)
        guard pruneCount > 0 else { return }
        
        let toPrune = allTurns.prefix(pruneCount)
        let oldestKeptDate = allTurns[pruneCount].createdAt
        
        // Insert summary as a system turn before the kept turns
        let summaryTurn = Turn(
            conversationID: conversationID,
            role: .system,
            content: "[Summary of earlier conversation]: \(summary)",
            createdAt: oldestKeptDate.addingTimeInterval(-1)
        )
        try appendTurn(summaryTurn)
        
        // Delete pruned turns
        let placeholders = toPrune.map { _ in "?" }.joined(separator: ",")
        let ids = toPrune.map { $0.id.uuidString }
        try exec(
            "DELETE FROM turns WHERE id IN (\(placeholders));",
            bindings: ids
        )
        
        // Fix turn_count
        try exec(
            "UPDATE conversations SET turn_count = (SELECT COUNT(*) FROM turns WHERE conversation_id = ?) WHERE id = ?;",
            bindings: [conversationID.uuidString, conversationID.uuidString]
        )
    }
    
    // MARK: - Search
    
    /// Full-text search across turn content using SQLite FTS5.
    public func search(_ query: String, limit: Int = 20) throws -> [Turn] {
        try ensureOpen()
        // Sanitize for FTS5: wrap each token in double quotes to treat as literal,
        // preventing FTS5 operators (NEAR, NOT, *, ^) from altering query semantics.
        let safe = query
            .components(separatedBy: .whitespacesAndNewlines)
            .map { $0.replacingOccurrences(of: "\"", with: "\"\"") }
            .filter { !$0.isEmpty }
            .map { "\"\($0)\"" }
            .joined(separator: " OR ")
        guard !safe.isEmpty else { return [] }
        return try self.query(
            "SELECT t.* FROM turns t JOIN turns_fts f ON t.id = f.id WHERE turns_fts MATCH ? ORDER BY rank LIMIT ?;",
            bindings: [safe, limit],
            map: rowToTurn
        )
    }
    
    // MARK: - Statistics
    
    public struct ConversationStats: Sendable {
        public let turnCount: Int
        public let totalTokenEstimate: Int
        public let oldestTurn: Date?
        public let newestTurn: Date?
    }
    
    public func stats(for conversationID: UUID) throws -> ConversationStats {
        try ensureOpen()
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        
        let sql = """
            SELECT COUNT(*), SUM(token_estimate), MIN(created_at), MAX(created_at)
            FROM turns WHERE conversation_id = ?;
            """
        sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        sqlite3_bind_text(stmt, 1, conversationID.uuidString, -1, SQLITE_TRANSIENT)
        sqlite3_step(stmt)
        
        return ConversationStats(
            turnCount:          Int(sqlite3_column_int(stmt, 0)),
            totalTokenEstimate: Int(sqlite3_column_int(stmt, 1)),
            oldestTurn: (sqlite3_column_text(stmt, 2)).flatMap { parseISO(String(cString: $0)) },
            newestTurn: (sqlite3_column_text(stmt, 3)).flatMap { parseISO(String(cString: $0)) }
        )
    }
    
    // MARK: - Migrations
    
    private func migrate() throws {
        // Conversations table
        try exec("""
            CREATE TABLE IF NOT EXISTS conversations (
                id           TEXT PRIMARY KEY,
                title        TEXT NOT NULL,
                model        TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                turn_count   INTEGER NOT NULL DEFAULT 0
            );
            """)
        
        // Turns table
        try exec("""
            CREATE TABLE IF NOT EXISTS turns (
                id               TEXT PRIMARY KEY,
                conversation_id  TEXT NOT NULL REFERENCES conversations(id),
                role             TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
                content          TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                token_estimate   INTEGER NOT NULL DEFAULT 0
            );
            """)
        
        // Index for fast context window queries
        try exec("""
            CREATE INDEX IF NOT EXISTS idx_turns_conv_date
            ON turns(conversation_id, created_at DESC);
            """)
        
        // FTS5 virtual table for full-text search
        try exec("""
            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts
            USING fts5(id UNINDEXED, content, content=turns, content_rowid=rowid);
            """)
        
        // Keep FTS in sync via triggers
        try exec("""
            CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
                INSERT INTO turns_fts(id, content) VALUES (new.id, new.content);
            END;
            """)
        try exec("""
            CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
                INSERT INTO turns_fts(turns_fts, id, content) VALUES('delete', old.id, old.content);
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
        
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw dbError()
        }
        bind(stmt, values: bindings)
        let result = sqlite3_step(stmt)
        guard result == SQLITE_DONE || result == SQLITE_ROW else {
            throw dbError()
        }
    }
    
    private func query<T>(
        _ sql: String,
        bindings: [Any] = [],
        map: (OpaquePointer?) -> T?
    ) throws -> [T] {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw dbError()
        }
        bind(stmt, values: bindings)
        
        var results: [T] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let value = map(stmt) { results.append(value) }
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
                    assertionFailure("ConversationStore.bind: unsupported type \(type(of: value)) at index \(i)")
                    sqlite3_bind_null(stmt, idx)
            }
        }
    }
    
    // MARK: - Row mappers
    
    private func rowToConversation(_ stmt: OpaquePointer?) -> Conversation? {
        guard let stmt else { return nil }
        guard
            let idStr    = sqlite3_column_text(stmt, 0).map({ String(cString: $0) }),
            let id       = UUID(uuidString: idStr),
            let title    = sqlite3_column_text(stmt, 1).map({ String(cString: $0) }),
            let model    = sqlite3_column_text(stmt, 2).map({ String(cString: $0) }),
            let createdS = sqlite3_column_text(stmt, 3).map({ String(cString: $0) }),
            let updatedS = sqlite3_column_text(stmt, 4).map({ String(cString: $0) })
        else { return nil }
        
        return Conversation(
            id:        id,
            title:     title,
            model:     model,
            createdAt: parseISO(createdS) ?? Date(),
            updatedAt: parseISO(updatedS) ?? Date(),
            turnCount: Int(sqlite3_column_int(stmt, 5))
        )
    }
    
    private func rowToTurn(_ stmt: OpaquePointer?) -> Turn? {
        guard let stmt else { return nil }
        guard
            let idStr    = sqlite3_column_text(stmt, 0).map({ String(cString: $0) }),
            let id       = UUID(uuidString: idStr),
            let convStr  = sqlite3_column_text(stmt, 1).map({ String(cString: $0) }),
            let convID   = UUID(uuidString: convStr),
            let roleStr  = sqlite3_column_text(stmt, 2).map({ String(cString: $0) }),
            let role     = Turn.Role(rawValue: roleStr),
            let content  = sqlite3_column_text(stmt, 3).map({ String(cString: $0) }),
            let dateStr  = sqlite3_column_text(stmt, 4).map({ String(cString: $0) })
        else { return nil }
        
        return Turn(
            id:             id,
            conversationID: convID,
            role:           role,
            content:        content,
            createdAt:      parseISO(dateStr) ?? Date(),
            tokenEstimate:  Int(sqlite3_column_int(stmt, 5))
        )
    }
    
    // MARK: - Date helpers
    
    private let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
    
    private func iso(_ date: Date) -> String { isoFormatter.string(from: date) }
    private func parseISO(_ string: String) -> Date? { isoFormatter.date(from: string) }
    
    private func dbError() -> StoreError {
        let msg = db.flatMap { sqlite3_errmsg($0) }.map { String(cString: $0) } ?? "unknown"
        return .sqliteError(msg)
    }
}

// MARK: - Errors

/// Errors thrown by ``ConversationStore`` when database operations fail.
public enum StoreError: LocalizedError {
    /// The SQLite database file at the given path could not be opened.
    case cannotOpen(String)
    /// A raw SQLite error occurred.
    case sqliteError(String)
    /// No conversation exists with the given ID.
    case conversationNotFound(UUID)
    
    public var errorDescription: String? {
        switch self {
            case .cannotOpen(let path):       return "Cannot open database at \(path)"
            case .sqliteError(let msg):       return "SQLite error: \(msg)"
            case .conversationNotFound(let id): return "Conversation \(id) not found"
        }
    }
}
