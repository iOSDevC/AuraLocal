import XCTest
@testable import AuraCore

final class HybridCompressionTests: XCTestCase {

    // MARK: - PII redaction

    func testRedactsSecrets() {
        let r = PIIRedactor().redact("Mail me at john.doe@example.com with key sk-ABCDEFGHIJKLMNOP1234.")
        XCTAssertTrue(r.redacted.contains("[REDACTED_EMAIL]"))
        XCTAssertTrue(r.redacted.contains("[REDACTED_KEY]"))
        XCTAssertFalse(r.redacted.contains("john.doe@example.com"))
        XCTAssertEqual(r.count, 2)
    }

    func testLeavesCodeAlone() {
        // High-precision patterns must not mangle source code (the security domain sends code).
        let code = "func add(a: Int, b: Int) -> Int { return a + b }  // id 12345, ratio 3.14"
        let r = PIIRedactor().redact(code)
        XCTAssertEqual(r.redacted, code)
        XCTAssertEqual(r.count, 0)
    }

    // MARK: - Pluggable scorer

    func testCompressorUsesInjectedScorer() {
        struct KeepScorer: SelfInfoScorer {
            func scores(for sentences: [String], question: String) -> [Double] {
                sentences.map { $0.contains("keep") ? 100 : 0 }
            }
        }
        let compressor = ContextCompressor(scorer: KeepScorer())
        let ctx = "drop this one. keep this important sentence. drop that too."
        let result = compressor.compress(context: ctx, question: "x", budgetTokens: 8)
        XCTAssertLessThan(result.compressedTokens, result.originalTokens)
        XCTAssertTrue(result.keptText.contains("keep"))
        XCTAssertFalse(result.keptText.contains("drop that"))
    }

    func testHeuristicScorerRanksRelevantHigher() {
        let scores = HeuristicScorer().scores(
            for: ["bananas are yellow fruit", "the capital of france is paris"],
            question: "what is the capital of france")
        XCTAssertGreaterThan(scores[1], scores[0])
    }
}
