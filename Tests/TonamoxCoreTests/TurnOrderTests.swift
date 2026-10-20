import XCTest
@testable import TonamoxCore

/// Timestamps have millisecond resolution, and a reply and the next prompt are often stored within the same
/// millisecond. The GGUF backends send turns with their roles, so a swapped pair merges two prompts and two replies.
final class TurnOrderTests: XCTestCase {

    private func makeStore() async throws -> (ConversationStore, URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("turn-order-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let store = ConversationStore(directory: directory)
        try await store.open()
        return (store, directory)
    }

    private func turn(_ index: Int, in conversation: UUID, at stamp: Date) -> Turn {
        Turn(conversationID: conversation, role: index.isMultiple(of: 2) ? .user : .assistant, content: "turn \(index)",
             createdAt: stamp)
    }

    func testTurnsStoredInTheSameMillisecondKeepTheirOrder() async throws {
        let (store, directory) = try await makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }
        let conversation = try await store.createConversation(model: .llama3_1_8b_gguf, title: "order")
        let stamp = Date()
        for index in 0..<8 {
            _ = try await store.appendTurn(turn(index, in: conversation.id, at: stamp))
        }
        let expected = (0..<8).map { "turn \($0)" }
        let all = try await store.turns(for: conversation.id)
        XCTAssertEqual(all.map(\.content), expected)
        let window = try await store.contextWindow(for: conversation.id, maxTokens: 4096)
        XCTAssertEqual(window.map(\.content), expected)
        XCTAssertEqual(window.map(\.role), [.user, .assistant, .user, .assistant, .user, .assistant, .user, .assistant])
    }
}
