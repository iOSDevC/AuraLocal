import XCTest
@testable import AuraCore

final class HybridEscalationTests: XCTestCase {

    func testCompressorShrinksAndKeepsOrder() {
        let compressor = ContextCompressor()
        // Padded, low-relevance context + one relevant sentence.
        let context = String(repeating: "Bananas are yellow fruit. ", count: 60)
            + "The capital of France is Paris."
        let result = compressor.compress(
            context: context,
            question: "What is the capital of France?",
            budgetTokens: 40)
        XCTAssertLessThan(result.compressedTokens, result.originalTokens)
        XCTAssertTrue(result.keptText.lowercased().contains("paris"),
                      "the relevant sentence should survive compression")
    }

    func testCompressorNoOpWhenUnderBudget() {
        let compressor = ContextCompressor()
        let short = "Short context."
        let r = compressor.compress(context: short, question: "q", budgetTokens: 1000)
        XCTAssertEqual(r.keptText, short)
        XCTAssertEqual(r.ratio, 1.0, accuracy: 0.0001)
    }

    /// Live end-to-end escalation against the user's running llama-server / Ollama.
    /// Opt-in via AURA_LIVE_PROVIDER_TESTS=1.
    @MainActor
    func testLiveHybridEscalation() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["AURA_LIVE_PROVIDER_TESTS"] == "1",
                          "Set AURA_LIVE_PROVIDER_TESTS=1 with a local llama-server/Ollama running.")
        guard let target = await HybridEscalator.bestLocalTarget() else {
            XCTFail("No local provider detected — is llama-server on :8080 running?")
            return
        }
        let escalator = HybridEscalator()
        let context = String(repeating: "Bananas are yellow. The sky is blue. ", count: 40)
            + "The capital of France is Paris."
        let result = try await escalator.escalate(
            to: target,
            systemPrompt: "Answer in one short sentence.",
            context: context,
            question: "What is the capital of France?",
            maxTokens: 300)   // room past a thinking model's reasoning

        XCTAssertFalse(result.answer.isEmpty)
        // Compression is unit-tested separately; a huge remote context window makes
        // it a legitimate no-op here. Just assert the receipt is populated.
        XCTAssertGreaterThan(result.compression.originalTokens, 0)
        print("── HYBRID MVP LIVE ──")
        print("target:", result.providerName, "| model:", target.modelID)
        print("compression:", result.compression.originalTokens, "→", result.compression.compressedTokens,
              "tokens (\(String(format: "%.1f", result.compression.factor))× fewer)")
        print("usage:", result.usage.map { "\($0.inputTokens) in / \($0.outputTokens) out" } ?? "n/a")
        print("answer:", result.answer.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}
