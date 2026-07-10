import XCTest
@testable import AuraCore

/// Locks the fix: `routeAndEscalate` now threads `localAnswer` into the router, so
/// a weak local draft can trigger low-confidence escalation (R5/R6). Previously
/// `localAnswer` was hardcoded to nil, making that path unreachable in production.
/// Uses a fake provider + injected target, so no network or model is needed.
@MainActor
final class RouteAndEscalateTests: XCTestCase {

    private func lanTarget(answer: String) -> RemoteTarget {
        RemoteTarget(
            provider: FakeRemoteProvider(answer: answer),
            modelID: "fake", contextLength: 8192,
            origin: .localNetwork(.llamaServer))
    }

    func testLowConfidenceLocalAnswerEscalates() async throws {
        let target = lanTarget(answer: "STRONGER REMOTE ANSWER")
        let result = try await HybridEscalator().routeAndEscalate(
            policy: EscalationPolicy(mode: .askEachTime),
            context: "short",
            question: "What is the safest fix here?",
            localAnswer: "I'm not sure, I cannot help with that.",   // low confidence
            consent: ApprovingConsentGate(),
            targets: [target])
        XCTAssertNotNil(result, "a low-confidence local answer must escalate")
        XCTAssertEqual(result?.answer, "STRONGER REMOTE ANSWER")
    }

    func testConfidentLocalAnswerStaysLocal() async throws {
        let target = lanTarget(answer: "REMOTE")
        let confident = String(repeating: "This is a thorough, confident, complete answer. ", count: 5)
        let result = try await HybridEscalator().routeAndEscalate(
            policy: EscalationPolicy(mode: .askEachTime),
            context: "short",
            question: "an easy question",
            localAnswer: confident,
            consent: ApprovingConsentGate(),
            targets: [target])
        XCTAssertNil(result, "a confident local answer should stay local")
    }

    func testNoLocalAnswerHasNoLowConfidenceTrigger() async throws {
        // Documents the fixed bug: with no localAnswer and a small prompt there is
        // no escalation trigger (no size overflow, no low-confidence) -> stays local.
        let target = lanTarget(answer: "REMOTE")
        let result = try await HybridEscalator().routeAndEscalate(
            policy: EscalationPolicy(mode: .askEachTime),
            context: "short",
            question: "another easy question",
            consent: ApprovingConsentGate(),
            targets: [target])
        XCTAssertNil(result)
    }
}

// MARK: - Test doubles

/// A ``RemoteLLMProvider`` that yields a canned answer + usage — no network.
private struct FakeRemoteProvider: RemoteLLMProvider {
    let id = "test.fake"
    let displayName = "Fake"
    let retentionNote = "test"
    let answer: String

    func stream(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error> {
        let answer = self.answer
        return AsyncThrowingStream { continuation in
            continuation.yield(.token(answer))
            continuation.yield(.usage(TokenUsage(inputTokens: 5, outputTokens: 7)))
            continuation.finish()
        }
    }
}

/// A consent gate that always approves (mirrors DenyingConsentGate's shape).
private struct ApprovingConsentGate: ConsentGate {
    func requestConsent(target: RemoteTarget, preview: CompressionResult, projectedCostUSD: Decimal) async -> Bool { true }
}
