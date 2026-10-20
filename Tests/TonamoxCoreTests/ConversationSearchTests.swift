import SQLite3
import XCTest
@testable import TonamoxCore

/// The full-text index is external-content FTS5 keyed by `turns.rowid`. A delete that does not reach it leaves
/// matches pointing at missing rows, or at a later turn that reused the rowid.
final class ConversationSearchTests: XCTestCase {

    private func makeDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("conversation-search-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    private func makeStore(in directory: URL) async throws -> ConversationStore {
        let store = ConversationStore(directory: directory)
        try await store.open()
        return store
    }

    private func append(_ content: String, role: Turn.Role = .user, to conversation: UUID,
                        in store: ConversationStore) async throws -> Turn {
        try await store.appendTurn(Turn(conversationID: conversation, role: role, content: content))
    }

    func testSearchFollowsDeletedTurnsAndReusedRowids() async throws {
        let directory = try makeDirectory()
        let store = try await makeStore(in: directory)
        let conversation = try await store.createConversation(model: .qwen3_1_7b, title: "search").id
        let apple = try await append("alpha apple", to: conversation, in: store)
        _ = try await append("bravo banana", role: .assistant, to: conversation, in: store)
        let cherry = try await append("charlie cherry", to: conversation, in: store)

        try await store.deleteTurn(id: cherry.id)
        let deletedHits = try await store.search("cherry")
        XCTAssertEqual(deletedHits.map(\.id), [])

        let delta = try await append("delta date", to: conversation, in: store)
        let staleHits = try await store.search("cherry")
        XCTAssertEqual(staleHits.map(\.id), [])
        let deltaHits = try await store.search("delta")
        XCTAssertEqual(deltaHits.map(\.id), [delta.id])
        let appleHits = try await store.search("apple")
        XCTAssertEqual(appleHits.map(\.id), [apple.id])

        XCTAssertNil(try SQLiteFile(directory).fullTextProblem())
        let count = try await store.conversation(id: conversation)?.turnCount
        XCTAssertEqual(count, 3)
    }

    func testSearchForgetsADeletedConversation() async throws {
        let directory = try makeDirectory()
        let store = try await makeStore(in: directory)
        let gone = try await store.createConversation(model: .qwen3_1_7b, title: "AgentCrew: gone").id
        let kept = try await store.createConversation(model: .qwen3_1_7b, title: "AgentCrew: kept").id
        _ = try await append("shared keyword one", role: .assistant, to: gone, in: store)
        _ = try await append("shared keyword two", role: .assistant, to: gone, in: store)
        let first = try await append("shared keyword three", role: .assistant, to: kept, in: store)

        try await store.deleteConversation(id: gone)
        let second = try await append("shared keyword four", role: .assistant, to: kept, in: store)

        let hits = try await store.search("keyword")
        XCTAssertEqual(Set(hits.map(\.id)), [first.id, second.id])
        let recalled = try await store.searchAgentMemory("keyword")
        XCTAssertEqual(Set(recalled.map(\.id)), [first.id, second.id])
        XCTAssertNil(try SQLiteFile(directory).fullTextProblem())
    }

    func testSearchAfterPruneAndSummarize() async throws {
        let directory = try makeDirectory()
        let store = try await makeStore(in: directory)
        let conversation = try await store.createConversation(model: .qwen3_1_7b, title: "prune").id
        var notes: [Turn] = []
        for index in 0..<6 {
            notes.append(try await append("zebra note \(index)", to: conversation, in: store))
        }

        try await store.pruneAndSummarize(conversationID: conversation, keepLastN: 2, summary: "zebra summary")
        let later = try await append("zebra later", to: conversation, in: store)

        let hits = try await store.search("zebra", limit: 50)
        let summary = try await store.turns(for: conversation).filter { $0.role == .system }.map(\.id)
        XCTAssertEqual(Set(hits.map(\.id)), Set(notes.suffix(2).map(\.id) + summary + [later.id]))
        XCTAssertNil(try SQLiteFile(directory).fullTextProblem())
    }

    func testOpeningRepairsAnIndexDamagedByOlderTriggers() async throws {
        let directory = try makeDirectory()
        let first = try await makeStore(in: directory)
        _ = try await first.createConversation(model: .qwen3_1_7b, title: "legacy")
        await first.close()

        try damageIndexWithLegacyTriggers(in: directory)
        XCTAssertNotNil(try SQLiteFile(directory).fullTextProblem(), "the legacy triggers should have damaged the index")

        let reopened = try await makeStore(in: directory)
        let cherry = try await reopened.search("cherry")
        XCTAssertEqual(cherry.map(\.content), [])
        let delta = try await reopened.search("delta")
        XCTAssertEqual(delta.map(\.content), ["delta date"])
        XCTAssertNil(try SQLiteFile(directory).fullTextProblem())
        let triggers = try SQLiteFile(directory)
            .strings("SELECT sql FROM sqlite_master WHERE type = 'trigger' ORDER BY name;")
        XCTAssertEqual(triggers.count, 2)
        XCTAssertTrue(triggers.allSatisfy { $0.contains("rowid") }, "\(triggers)")

        await reopened.close()
        let again = try await makeStore(in: directory)
        let deltaAgain = try await again.search("delta")
        XCTAssertEqual(deltaAgain.map(\.content), ["delta date"])
    }

    /// What an older version did to a store: its triggers, then a delete and an insert that reuses the rowid.
    private func damageIndexWithLegacyTriggers(in directory: URL) throws {
        let file = try SQLiteFile(directory)
        try file.run(Self.legacyTriggers)
        let conversation = try XCTUnwrap(file.strings("SELECT id FROM conversations;").first)
        try file.run("""
            INSERT INTO turns (id, conversation_id, role, content, created_at, token_estimate) VALUES
              ('\(UUID().uuidString)', '\(conversation)', 'user', 'alpha apple', '2026-10-15T10:00:00.000Z', 3),
              ('\(UUID().uuidString)', '\(conversation)', 'user', 'charlie cherry', '2026-10-15T10:00:01.000Z', 3);
            DELETE FROM turns WHERE content = 'charlie cherry';
            INSERT INTO turns (id, conversation_id, role, content, created_at, token_estimate) VALUES
              ('\(UUID().uuidString)', '\(conversation)', 'user', 'delta date', '2026-10-15T10:00:02.000Z', 3);
            """)
    }

    /// The triggers every version before the fix created: neither passes the rowid to the index.
    private static let legacyTriggers = """
        DROP TRIGGER IF EXISTS turns_ai;
        DROP TRIGGER IF EXISTS turns_ad;
        CREATE TRIGGER turns_ai AFTER INSERT ON turns BEGIN
            INSERT INTO turns_fts(id, content) VALUES (new.id, new.content);
        END;
        CREATE TRIGGER turns_ad AFTER DELETE ON turns BEGIN
            INSERT INTO turns_fts(turns_fts, id, content) VALUES('delete', old.id, old.content);
        END;
        """
}

struct SQLiteFailure: Error, CustomStringConvertible {
    let description: String
}

/// A second connection to a store's file, for what its API does not offer: older versions' SQL and FTS5's
/// integrity check.
final class SQLiteFile {
    private var db: OpaquePointer?

    init(_ directory: URL) throws {
        let path = directory.appendingPathComponent("conversations.sqlite").path
        guard sqlite3_open(path, &db) == SQLITE_OK else { throw SQLiteFailure(description: "cannot open \(path)") }
    }

    deinit { sqlite3_close(db) }

    func run(_ sql: String) throws {
        var message: UnsafeMutablePointer<CChar>?
        defer { sqlite3_free(message) }
        guard sqlite3_exec(db, sql, nil, nil, &message) == SQLITE_OK else {
            throw SQLiteFailure(description: message.map { String(cString: $0) } ?? "unknown")
        }
    }

    func strings(_ sql: String) throws -> [String] {
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw SQLiteFailure(description: String(cString: sqlite3_errmsg(db)))
        }
        var rows: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(Self.firstColumn(of: stmt))
        }
        return rows
    }

    private static func firstColumn(of stmt: OpaquePointer?) -> String {
        sqlite3_column_text(stmt, 0).map { String(cString: $0) } ?? ""
    }

    /// `nil` when the index matches the `turns` table; otherwise SQLite's complaint.
    func fullTextProblem() throws -> String? {
        do {
            try run("INSERT INTO turns_fts(turns_fts, rank) VALUES('integrity-check', 1);")
            return nil
        } catch let failure as SQLiteFailure {
            return failure.description
        }
    }
}
