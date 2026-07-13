import XCTest
@testable import AuraCore

/// Verifies ConversationStore.searchAgentMemory — the storage layer behind the
/// RecallTool. Critically: it must scope to agent/task conversations and NOT
/// surface the user's private direct-chat turns.
final class RecallSearchTests: XCTestCase {

    private func makeStore() async throws -> ConversationStore {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AuraRecallTests-\(UUID().uuidString)", isDirectory: true)
        let store = ConversationStore(directory: dir)
        try await store.open()
        return store
    }

    func testSearchAgentMemoryScopesToAgentTurnsOnly() async throws {
        let store = try await makeStore()

        // An agent/task conversation (title prefixed "AgentCrew:").
        let agentConv = try await store.createConversation(model: .qwen3_1_7b, title: "AgentCrew: authentication review")
        _ = try await store.appendTurn(Turn(conversationID: agentConv.id, role: .assistant,
            content: "We decided to require biometric authentication for login."))

        // The user's PRIVATE direct chat — same keyword, must NOT be recalled.
        let userConv = try await store.createConversation(model: .qwen3_1_7b, title: "New conversation")
        _ = try await store.appendTurn(Turn(conversationID: userConv.id, role: .user,
            content: "My private note about authentication passwords."))

        let hits = try await store.searchAgentMemory("authentication", limit: 8)

        XCTAssertFalse(hits.isEmpty, "should recall the agent-task turn")
        XCTAssertTrue(hits.allSatisfy { $0.conversationID == agentConv.id },
                      "recall must NOT surface the user's private direct-chat turns")
        XCTAssertTrue(hits.contains { $0.content.contains("biometric") })
    }

    func testSearchAgentMemoryEmptyQueryReturnsNothing() async throws {
        let store = try await makeStore()
        let hits = try await store.searchAgentMemory("   ", limit: 8)
        XCTAssertTrue(hits.isEmpty)
    }
}
