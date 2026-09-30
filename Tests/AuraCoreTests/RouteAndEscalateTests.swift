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

    func testSensitiveDomainEscalatesAModerateAnswer() async throws {
        // The dead-code fix: passing `domain:` activates the sensitive-domain bias
        // (EscalationRouter raises the confidence bar for .security/.medicine), so a
        // moderate answer that stays local for a general domain escalates for security.
        let target = lanTarget(answer: "SPECIALIST ANSWER")
        let moderate = "A reasonably complete answer that is over forty characters long and confident."

        let asGeneral = try await HybridEscalator().routeAndEscalate(
            policy: EscalationPolicy(mode: .askEachTime),
            context: "short", question: "a moderately hard systems question",
            domain: nil, localAnswer: moderate,
            consent: ApprovingConsentGate(), targets: [target])
        XCTAssertNil(asGeneral, "a moderate answer should stay local for a general domain")

        let asSecurity = try await HybridEscalator().routeAndEscalate(
            policy: EscalationPolicy(mode: .askEachTime),
            context: "short", question: "a moderately hard security question",
            domain: .security, localAnswer: moderate,
            consent: ApprovingConsentGate(), targets: [target])
        XCTAssertEqual(asSecurity?.answer, "SPECIALIST ANSWER",
                       "a .security step must escalate the same moderate answer")
    }

    // MARK: - Cost

    private func cloudTarget() -> RemoteTarget {
        RemoteTarget(provider: FakeRemoteProvider(answer: "CLOUD"), modelID: "fake", contextLength: 8192, origin: .cloud)
    }

    func testRoutingInputCarriesSessionSpendAndLedgerProjection() {
        let ledger = CostLedger()
        ledger.setPrice(TokenPrice(inputUSDPerMillion: 1, outputUSDPerMillion: 1), provider: "test.fake", model: "fake")
        ledger.record(provider: "test.fake", model: "fake",
                      usage: TokenUsage(inputTokens: 950_000, outputTokens: 0), origin: .cloud, compressionRatio: nil)
        let target = cloudTarget()

        var input = HybridEscalator(ledger: ledger).routingInput(
            policy: EscalationPolicy(mode: .askEachTime, allowCloud: true, costCapUSDPerSession: 1),
            target: target, promptTokens: 8000, maxTokens: 100_000,
            localContextWindow: 8192, localAnswer: nil, domain: nil)

        XCTAssertEqual(input.sessionSpentUSD, Decimal(string: "0.95"))
        XCTAssertEqual(input.projectedCostUSD, Decimal(string: "0.108"))
        input.online = true   // R3 would otherwise depend on the test host
        XCTAssertEqual(EscalationRouter.decide(input), .offer(reason: .costCapped),
                       "0.95 spent + 0.108 projected crosses the 1.00 cap although the request alone does not")
    }

    func testUnpricedCloudTargetAsksConsentWithUnknownCost() async throws {
        try XCTSkipUnless(NetworkMonitor.shared.isOnline, "cloud routing needs a network path (R3)")
        let gate = RecordingConsentGate()
        do {
            _ = try await HybridEscalator(ledger: CostLedger()).routeAndEscalate(
                policy: EscalationPolicy(mode: .askEachTime, allowCloud: true),
                context: "short", question: "What is the safest fix here?",
                localAnswer: "I'm not sure, I cannot help with that.",
                consent: gate, targets: [cloudTarget()])
            XCTFail("the recording gate declines, so routeAndEscalate must throw")
        } catch AuraError.escalationDeclined {}
        XCTAssertEqual(gate.projectedCosts, [nil])
    }

    func testPricedCloudTargetAsksConsentWithLedgerProjection() async throws {
        try XCTSkipUnless(NetworkMonitor.shared.isOnline, "cloud routing needs a network path (R3)")
        let ledger = CostLedger()
        ledger.setPrice(TokenPrice(inputUSDPerMillion: 1, outputUSDPerMillion: 2), provider: "test.fake")
        let gate = RecordingConsentGate()
        do {
            _ = try await HybridEscalator(ledger: ledger).routeAndEscalate(
                policy: EscalationPolicy(mode: .askEachTime, allowCloud: true),
                context: "short", question: "What is the safest fix here?",
                localAnswer: "I'm not sure, I cannot help with that.",
                consent: gate, maxTokens: 1000, targets: [cloudTarget()])
            XCTFail("the recording gate declines, so routeAndEscalate must throw")
        } catch AuraError.escalationDeclined {}
        let projected = try XCTUnwrap(gate.projectedCosts.first ?? nil)
        XCTAssertGreaterThan(projected, Decimal(string: "0.002")!, "1000 output tokens at $2/1M alone cost $0.002")
    }

    func testEscalationRecordsModelAndFreeLANCost() async throws {
        let ledger = CostLedger()
        _ = try await HybridEscalator(ledger: ledger).escalate(
            to: lanTarget(answer: "LAN"), context: "", question: "unique \(UUID().uuidString)")
        let entry = try XCTUnwrap(ledger.records.last)
        XCTAssertEqual(entry.model, "fake")
        XCTAssertTrue(entry.isLocalNetwork)
        XCTAssertEqual(entry.costUSD, 0)
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
    func requestConsent(target: RemoteTarget, preview: CompressionResult, projectedCostUSD: Decimal?) async -> Bool { true }
}

/// Declines every offer and keeps the projected cost it was shown.
@MainActor
private final class RecordingConsentGate: ConsentGate {
    private(set) var projectedCosts: [Decimal?] = []

    func requestConsent(target: RemoteTarget, preview: CompressionResult, projectedCostUSD: Decimal?) async -> Bool {
        projectedCosts.append(projectedCostUSD)
        return false
    }
}
