import XCTest
@testable import AuraCore

final class HybridRoutingTests: XCTestCase {

    // MARK: - Router decision table

    func testOffModeAlwaysStaysLocal() {
        // R1 dominance: even with a huge overflow and a target, OFF wins.
        let d = EscalationRouter.decide(RoutingInput(
            policy: .off, hasCandidateTarget: true, candidateIsCloud: false,
            promptTokens: 100_000, localContextWindow: 8192))
        XCTAssertEqual(d, .stayLocal)
    }

    func testNoTargetStaysLocal() {
        let d = EscalationRouter.decide(RoutingInput(
            policy: EscalationPolicy(mode: .askEachTime), hasCandidateTarget: false,
            candidateIsCloud: false, promptTokens: 100_000, localContextWindow: 8192))
        XCTAssertEqual(d, .stayLocal)
    }

    func testCloudOfflineStaysLocal() {
        let d = EscalationRouter.decide(RoutingInput(
            policy: EscalationPolicy(mode: .askEachTime, allowCloud: true),
            hasCandidateTarget: true, candidateIsCloud: true, online: false,
            promptTokens: 100_000, localContextWindow: 8192))
        XCTAssertEqual(d, .stayLocal)
    }

    func testSizeOverflowOffersInAskEachTime() {
        let d = EscalationRouter.decide(RoutingInput(
            policy: EscalationPolicy(mode: .askEachTime), hasCandidateTarget: true,
            candidateIsCloud: false, promptTokens: 8000, localContextWindow: 8192))
        XCTAssertEqual(d, .offer(reason: .sizeOverflow))
    }

    func testSizeOverflowAutoEscalatesToLAN() {
        let d = EscalationRouter.decide(RoutingInput(
            policy: EscalationPolicy(mode: .autoWithConsentMemory), hasCandidateTarget: true,
            candidateIsCloud: false, promptTokens: 8000, localContextWindow: 8192))
        XCTAssertEqual(d, .escalate(reason: .sizeOverflow))
    }

    func testAutoModeCloudStillOffers() {
        let d = EscalationRouter.decide(RoutingInput(
            policy: EscalationPolicy(mode: .autoWithConsentMemory, allowCloud: true),
            hasCandidateTarget: true, candidateIsCloud: true,
            promptTokens: 8000, localContextWindow: 8192))
        XCTAssertEqual(d, .offer(reason: .sizeOverflow))
    }

    func testLowConfidenceAnswerOffers() {
        let d = EscalationRouter.decide(RoutingInput(
            policy: EscalationPolicy(mode: .askEachTime), hasCandidateTarget: true,
            candidateIsCloud: false, promptTokens: 10, localContextWindow: 8192,
            localAnswer: "I'm not sure, I cannot help with that."))
        XCTAssertEqual(d, .offer(reason: .lowConfidence))
    }

    func testConfidentAnswerStaysLocal() {
        let d = EscalationRouter.decide(RoutingInput(
            policy: EscalationPolicy(mode: .askEachTime), hasCandidateTarget: true,
            candidateIsCloud: false, promptTokens: 10, localContextWindow: 8192,
            localAnswer: String(repeating: "This is a thorough, confident answer. ", count: 5)))
        XCTAssertEqual(d, .stayLocal)
    }

    func testCostCapForcesOffer() {
        // A trigger fires but projected cost exceeds the cap -> offer(costCapped), never silent.
        let d = EscalationRouter.decide(RoutingInput(
            policy: EscalationPolicy(mode: .autoWithConsentMemory, allowCloud: true, costCapUSDPerSession: 0.01),
            hasCandidateTarget: true, candidateIsCloud: false,
            promptTokens: 8000, localContextWindow: 8192, projectedCostUSD: 0.50))
        XCTAssertEqual(d, .offer(reason: .costCapped))
    }

    private func overflowInput(
        cloud: Bool, projected: Decimal?, spent: Decimal, cap: Decimal = 1.0
    ) -> RoutingInput {
        RoutingInput(
            policy: EscalationPolicy(mode: .autoWithConsentMemory, allowCloud: true, costCapUSDPerSession: cap),
            hasCandidateTarget: true, candidateIsCloud: cloud,
            promptTokens: 8000, localContextWindow: 8192,
            projectedCostUSD: projected, sessionSpentUSD: spent)
    }

    func testSessionSpendCrossingTheCapForcesOffer() {
        XCTAssertEqual(EscalationRouter.decide(overflowInput(cloud: true, projected: 0.10, spent: 0.95)),
                       .offer(reason: .costCapped))
        XCTAssertEqual(EscalationRouter.decide(overflowInput(cloud: true, projected: 0.10, spent: 0.50)),
                       .offer(reason: .sizeOverflow))
    }

    func testSessionSpendCrossingTheCapStopsAutoEscalation() {
        XCTAssertEqual(EscalationRouter.decide(overflowInput(cloud: false, projected: 0.10, spent: 0.95)),
                       .offer(reason: .costCapped))
    }

    func testUnknownCloudCostOffersCostUnknown() {
        XCTAssertEqual(EscalationRouter.decide(overflowInput(cloud: true, projected: nil, spent: 0)),
                       .offer(reason: .costUnknown))
    }

    func testUnknownCostWithoutTriggerStaysLocal() {
        let d = EscalationRouter.decide(RoutingInput(
            policy: EscalationPolicy(mode: .askEachTime, allowCloud: true),
            hasCandidateTarget: true, candidateIsCloud: true,
            promptTokens: 10, localContextWindow: 8192, projectedCostUSD: nil))
        XCTAssertEqual(d, .stayLocal)
    }

    func testFreeLANIgnoresSessionSpendAndUnknownCost() {
        XCTAssertEqual(EscalationRouter.decide(overflowInput(cloud: false, projected: 0, spent: 5)),
                       .escalate(reason: .sizeOverflow))
        XCTAssertEqual(EscalationRouter.decide(overflowInput(cloud: false, projected: nil, spent: 5)),
                       .escalate(reason: .sizeOverflow))
    }

    // MARK: - Heuristic

    func testLowConfidenceHeuristic() {
        XCTAssertTrue(EscalationRouter.isLowConfidence("short", domain: nil))
        XCTAssertTrue(EscalationRouter.isLowConfidence("I don't know the answer to this question at all really", domain: nil))
        let medium = "A reasonably complete answer that is over forty characters long and confident."
        XCTAssertFalse(EscalationRouter.isLowConfidence(medium, domain: nil))
        // Sensitive domains raise the bar.
        XCTAssertTrue(EscalationRouter.isLowConfidence(medium, domain: .security))
    }

    func testSpanishRefusalsAreLowConfidence() {
        let refusals = [
            "No estoy seguro de cuál es la respuesta correcta a esta pregunta.",
            "No estoy segura de poder responder con precisión a lo que preguntas.",
            "Lo siento, no lo sé; esa información no aparece en el documento.",
            "Como modelo de lenguaje, no tengo acceso a datos en tiempo real sobre esto.",
            "No puedo confirmar esa cifra sin revisar el contrato original completo.",
            "Desconozco el importe exacto del recibo que mencionas en tu mensaje.",
        ]
        for answer in refusals {
            XCTAssertTrue(EscalationRouter.isLowConfidence(answer, domain: nil), answer)
        }
    }

    func testSpanishImpersonalPhrasingIsNotARefusal() {
        let confident = "El total del recibo es 42,50 €; no se incluyen propinas ni cargos por servicio."
        XCTAssertFalse(EscalationRouter.isLowConfidence(confident, domain: nil))
    }

    // MARK: - Policy defaults

    func testPolicyDefaultsToOff() {
        XCTAssertEqual(EscalationPolicy.off.mode, .off)
        XCTAssertFalse(EscalationPolicy.off.allowCloud)
    }

    func testPolicyCodableRoundtrip() throws {
        let policy = EscalationPolicy(mode: .askEachTime, allowCloud: true, costCapUSDPerSession: 2.5, keepRatio: 0.4)
        let data = try JSONEncoder().encode(policy)
        let decoded = try JSONDecoder().decode(EscalationPolicy.self, from: data)
        XCTAssertEqual(policy, decoded)
    }

    // MARK: - Keychain (tolerant of test-host restrictions)

    func testKeychainRoundtrip() throws {
        let account = "test.provider.\(UUID().uuidString)"
        do {
            try KeychainStore.save("secret-key-123", for: account)
        } catch {
            throw XCTSkip("Keychain unavailable in this test host: \(error)")
        }
        XCTAssertEqual(KeychainStore.read(for: account), "secret-key-123")
        XCTAssertTrue(KeychainStore.hasKey(for: account))
        try KeychainStore.save("updated-key", for: account)   // update path
        XCTAssertEqual(KeychainStore.read(for: account), "updated-key")
        KeychainStore.delete(for: account)
        XCTAssertNil(KeychainStore.read(for: account))
    }
}
