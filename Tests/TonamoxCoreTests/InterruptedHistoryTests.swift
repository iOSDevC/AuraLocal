import XCTest
@testable import TonamoxCore

/// A persistent conversation must stay one assistant turn per user turn when a generation fails, or the next turn
/// sends the model two user messages in a row.
@MainActor
final class InterruptedHistoryTests: XCTestCase {

    private let background = Notification.Name("history.didEnterBackground")
    private let foreground = Notification.Name("history.willEnterForeground")
    private let notifications = NotificationCenter()

    private struct Subject {
        let llm: Tonamox
        let backend: SlowFakeBackend
        let store: ConversationStore
        let conversation: UUID
    }

    private func makeSubject() async throws -> Subject {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let store = ConversationStore(directory: dir)
        try await store.open()
        let conversation = try await store.createConversation(model: .dolphin3_qwen25_1_5b_gguf, title: "history")

        let lifecycle = BackgroundLifecycle(notificationCenter: notifications, didEnterBackground: background,
                                            willEnterForeground: foreground)
        let backend = SlowFakeBackend()
        let engine = TonamoxEngine(model: .dolphin3_qwen25_1_5b_gguf, backend: backend, lifecycle: lifecycle)
        try await engine.load { _ in }
        return Subject(llm: Tonamox(engine: engine), backend: backend, store: store, conversation: conversation.id)
    }

    private func roles(_ store: ConversationStore, _ conversation: UUID) async throws -> [Turn.Role] {
        try await store.turns(for: conversation).map(\.role)
    }

    /// A stopped stream settles its turns after the consumer has moved on.
    private func turns(_ store: ConversationStore, _ conversation: UUID, atLeast count: Int) async throws -> [Turn] {
        let deadline = ContinuousClock.now + .seconds(5)
        var turns = try await store.turns(for: conversation)
        while turns.count < count, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(2))
            turns = try await store.turns(for: conversation)
        }
        return turns
    }

    func testBackgroundedChatSavesThePartialReplyAsTheAssistantTurn() async throws {
        let subject = try await makeSubject()
        let (llm, backend, store, conversation) = (subject.llm, subject.backend, subject.store, subject.conversation)
        let pending = Task { @MainActor in
            try await llm.chat("Tell me a long story.", in: conversation, maxTokens: 100_000, store: store)
        }
        try await waitUntil { backend.producedTokens >= 3 }
        notifications.post(name: background, object: nil)

        let result = await pending.result
        XCTAssertThrowsError(try result.get()) { XCTAssertTrue($0 is GenerationInterrupted, "\($0)") }
        let turns = try await store.turns(for: conversation)
        XCTAssertEqual(turns.map(\.role), [.user, .assistant])
        XCTAssertTrue(turns[1].content.hasPrefix("t0 t1 t2 "), turns[1].content)

        notifications.post(name: foreground, object: nil)
        // Turns are ordered by millisecond timestamps, which this fake backend is fast enough to tie.
        try await Task.sleep(for: .milliseconds(5))
        _ = try await llm.chat("Say hello.", in: conversation, maxTokens: backend.producedTokens + 3, store: store)
        let after = try await roles(store, conversation)
        XCTAssertEqual(after, [.user, .assistant, .user, .assistant])
    }

    func testBackgroundedStreamSavesWhatItYielded() async throws {
        let subject = try await makeSubject()
        let (llm, backend, store, conversation) = (subject.llm, subject.backend, subject.store, subject.conversation)
        let pending = Task { @MainActor in
            var shown = ""
            for try await delta in llm.stream("Tell me a long story.", in: conversation, maxTokens: 100_000,
                                              store: store) {
                shown += delta
            }
            return shown
        }
        try await waitUntil { backend.producedTokens >= 3 }
        notifications.post(name: background, object: nil)

        let result = await pending.result
        XCTAssertThrowsError(try result.get()) { XCTAssertTrue($0 is GenerationInterrupted, "\($0)") }
        let turns = try await store.turns(for: conversation)
        XCTAssertEqual(turns.map(\.role), [.user, .assistant])
        XCTAssertTrue(turns[1].content.hasPrefix("t0 t1 t2 "), turns[1].content)
    }

    func testStoppingAStreamCancelsItsGenerationAndKeepsWhatItProduced() async throws {
        let subject = try await makeSubject()
        let (llm, backend, store, conversation) = (subject.llm, subject.backend, subject.store, subject.conversation)
        var shown = ""
        var deltas = 0
        for try await delta in llm.stream("Tell me a long story.", in: conversation, maxTokens: 100_000, store: store) {
            shown += delta
            deltas += 1
            if deltas == 3 { break }
        }

        try await waitUntil { !backend.isGenerating }
        XCTAssertLessThan(backend.producedTokens, 1_000)
        let turns = try await turns(store, conversation, atLeast: 2)
        XCTAssertEqual(turns.map(\.role), [.user, .assistant])
        XCTAssertTrue(turns[1].content.hasPrefix(shown), "\(turns[1].content) does not start with \(shown)")
        let count = try await store.conversation(id: conversation)?.turnCount
        XCTAssertEqual(count, 2)
    }

    func testAPromptRightAfterAStoppedStreamFollowsItsReply() async throws {
        let subject = try await makeSubject()
        let (llm, backend, store, conversation) = (subject.llm, subject.backend, subject.store, subject.conversation)
        var deltas = 0
        for try await _ in llm.stream("Tell me a long story.", in: conversation, maxTokens: 100_000, store: store) {
            deltas += 1
            if deltas == 3 { break }
        }

        _ = try await llm.chat("Say hello.", in: conversation, maxTokens: backend.producedTokens + 50, store: store)
        let after = try await roles(store, conversation)
        XCTAssertEqual(after, [.user, .assistant, .user, .assistant])
    }

    func testAStreamStoppedBeforeItsFirstTokenStoresNothing() async throws {
        let subject = try await makeSubject()
        let (llm, backend, store, conversation) = (subject.llm, subject.backend, subject.store, subject.conversation)
        let consumer = Task { @MainActor in
            for try await _ in llm.stream("Hello?", in: conversation, maxTokens: 100_000, store: store) {}
        }
        consumer.cancel()
        _ = await consumer.result

        _ = try await llm.chat("Say hello.", in: conversation, maxTokens: backend.producedTokens + 3, store: store)
        let after = try await store.turns(for: conversation)
        XCTAssertEqual(after.map(\.role), [.user, .assistant])
        XCTAssertEqual(after.first?.content, "Say hello.")
        let count = try await store.conversation(id: conversation)?.turnCount
        XCTAssertEqual(count, 2)
    }

    func testRefusedChatRemovesItsUserTurn() async throws {
        let subject = try await makeSubject()
        let (llm, store, conversation) = (subject.llm, subject.store, subject.conversation)
        notifications.post(name: background, object: nil)

        do {
            _ = try await llm.chat("Hello?", in: conversation, maxTokens: 4, store: store)
            XCTFail("expected GenerationInterrupted")
        } catch is GenerationInterrupted {}

        let turns = try await roles(store, conversation)
        XCTAssertEqual(turns, [])
        let count = try await store.conversation(id: conversation)?.turnCount
        XCTAssertEqual(count, 0)
    }
}
