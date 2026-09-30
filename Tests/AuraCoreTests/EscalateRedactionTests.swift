import Foundation
import Synchronization
import XCTest
@testable import AuraCore

/// `escalate(redactPII:)` must cover the question too: `aura ask` and the Example
/// app send the whole prompt as `question` with an empty context.
@MainActor
final class EscalateRedactionTests: XCTestCase {

    private func send(_ question: String, redactPII: Bool) async throws -> (sent: String, result: HybridEscalator.Result) {
        let recorder = RequestRecorder()
        let target = RemoteTarget(
            provider: RecordingProvider(recorder: recorder), modelID: "m", contextLength: 8192, origin: .cloud)
        let result = try await HybridEscalator().escalate(
            to: target, context: "", question: question, maxTokens: 64, redactPII: redactPII)
        let sent = try XCTUnwrap(recorder.lastUserContent())
        return (sent, result)
    }

    func testRedactPIICoversTheQuestion() async throws {
        let question = "Email alice@example.com the report \(UUID().uuidString)"
        let (sent, result) = try await send(question, redactPII: true)
        XCTAssertFalse(sent.contains("alice@example.com"))
        XCTAssertTrue(sent.contains("[REDACTED_EMAIL]"))
        XCTAssertEqual(result.redactedPIICount, 1)
    }

    func testQuestionIsSentVerbatimWithoutRedaction() async throws {
        let question = "Email alice@example.com the report \(UUID().uuidString)"
        let (sent, result) = try await send(question, redactPII: false)
        XCTAssertEqual(sent, question)
        XCTAssertEqual(result.redactedPIICount, 0)
    }
}

private final class RequestRecorder: Sendable {
    private let messages = Mutex<[[String: String]]>([])

    func record(_ request: RemoteRequest) {
        messages.withLock { $0 = request.messages }
    }

    func lastUserContent() -> String? {
        messages.withLock { $0.last { $0["role"] == "user" }?["content"] }
    }
}

private struct RecordingProvider: RemoteLLMProvider {
    let id = "test.recording"
    let displayName = "Recording"
    let retentionNote = "test"
    let recorder: RequestRecorder

    func stream(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error> {
        recorder.record(request)
        return AsyncThrowingStream { continuation in
            continuation.yield(.token("ok"))
            continuation.finish()
        }
    }
}
