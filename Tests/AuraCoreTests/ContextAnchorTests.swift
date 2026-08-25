import XCTest
@testable import AuraCore

/// llama.cpp reuses a prompt's KV cache only while the prompt's TEXT PREFIX is unchanged
/// (`chunkText.hasPrefix(cacheText)` in LocalLLMClient's Context). `contextWindow` used to refill
/// the budget newest-first on every call, so once a conversation outgrew the window its oldest kept
/// turn advanced by one *every turn* — moving the prefix and re-prefilling the whole history each
/// time. The boundary is now snapped to a stride so it stays put for several turns.
final class ContextAnchorTests: XCTestCase {

    private func makeStore() async throws -> (ConversationStore, URL) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ctxanchor-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let store = ConversationStore(directory: dir)
        try await store.open()
        return (store, dir)
    }

    /// 400 chars ≈ 100 tokens under Turn's `content.count / 4` estimate.
    private func body(_ tag: String) -> String {
        String(repeating: "\(tag) ", count: 100).prefix(400).description
    }

    /// The regression this exists for: the oldest kept turn must NOT advance on every single turn.
    func testPrefixAnchorHoldsAcrossTurns() async throws {
        let (store, dir) = try await makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }

        let conv = try await store.createConversation(model: .llama3_1_8b_gguf, title: "anchor")
        // Big enough window that anchoring engages (stride = fitting/4).
        let maxTokens = 2100   // ~20 turns of 100 tokens after the system turn

        _ = try await store.appendTurn(
            Turn(conversationID: conv.id, role: .system, content: body("SYS")))

        var anchors: [String] = []
        for i in 0..<40 {
            _ = try await store.appendTurn(
                Turn(conversationID: conv.id, role: .user, content: body("T\(i)")))
            let window = try await store.contextWindow(for: conv.id, maxTokens: maxTokens)
            let oldestNonSystem = window.first { $0.role != .system }
            anchors.append(oldestNonSystem?.content ?? "-")
        }

        // Once eviction starts the anchor must repeat — consecutive identical anchors are cache hits.
        let tail = anchors.suffix(20)
        let moves = zip(tail, tail.dropFirst()).filter { $0 != $1 }.count
        XCTAssertLessThan(moves, 10,
            "anchor moved \(moves)/19 times — the prefix is still sliding every turn")
        XCTAssertGreaterThan(moves, 0, "anchor never moved; the window isn't evicting at all")
    }

    /// Anchoring must never overflow the caller's budget.
    func testWindowNeverExceedsBudget() async throws {
        let (store, dir) = try await makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }

        let conv = try await store.createConversation(model: .llama3_1_8b_gguf, title: "budget")
        for i in 0..<30 {
            _ = try await store.appendTurn(
                Turn(conversationID: conv.id, role: .user, content: body("T\(i)")))
        }
        for budget in [400, 900, 1500, 2100] {
            let window = try await store.contextWindow(for: conv.id, maxTokens: budget)
            let used = window.reduce(0) { $0 + $1.tokenEstimate }
            XCTAssertLessThanOrEqual(used, budget, "window used \(used) > budget \(budget)")
            XCTAssertFalse(window.isEmpty, "must always return at least the newest turn")
        }
    }

    /// A short conversation that fits must come back whole — anchoring must not truncate it.
    func testShortConversationIsNotTruncated() async throws {
        let (store, dir) = try await makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }

        let conv = try await store.createConversation(model: .llama3_1_8b_gguf, title: "short")
        for i in 0..<3 {
            _ = try await store.appendTurn(
                Turn(conversationID: conv.id, role: .user, content: body("T\(i)")))
        }
        let window = try await store.contextWindow(for: conv.id, maxTokens: 2100)
        XCTAssertEqual(window.count, 3, "everything fits — nothing should be dropped")
    }
}
