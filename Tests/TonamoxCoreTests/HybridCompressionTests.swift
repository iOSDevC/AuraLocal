import XCTest
@testable import TonamoxCore

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

    // MARK: - Response cache

    @MainActor
    func testResponseCacheHitAndMiss() {
        let cache = ResponseCache(capacity: 4)
        XCTAssertNil(cache.lookup(provider: "p", model: "m", prompt: "hi"))
        cache.insert("hello", provider: "p", model: "m", prompt: "hi")
        XCTAssertEqual(cache.lookup(provider: "p", model: "m", prompt: "hi"), "hello")
        XCTAssertNil(cache.lookup(provider: "p", model: "m", prompt: "other"))
        XCTAssertNil(cache.lookup(provider: "p2", model: "m", prompt: "hi"))  // different key
    }

    @MainActor
    func testResponseCacheEviction() {
        let cache = ResponseCache(capacity: 2)
        cache.insert("a", provider: "p", model: "m", prompt: "1")
        cache.insert("b", provider: "p", model: "m", prompt: "2")
        cache.insert("c", provider: "p", model: "m", prompt: "3")   // evicts "1"
        XCTAssertNil(cache.lookup(provider: "p", model: "m", prompt: "1"))
        XCTAssertEqual(cache.lookup(provider: "p", model: "m", prompt: "3"), "c")
        XCTAssertEqual(cache.count, 2)
    }

    @MainActor
    func testResponseCacheKeysOnSystemPromptAndMaxTokens() {
        let target = RemoteTarget(provider: AnthropicProvider(apiKey: "unused"), modelID: "m", origin: .cloud)
        let escalator = HybridEscalator()
        func payload(_ system: String?, _ maxTokens: Int) -> EscalationPayload {
            escalator.preparePayload(
                for: target, systemPrompt: system, context: "", question: "hi", maxTokens: maxTokens)
        }
        let cache = ResponseCache(capacity: 8)
        cache.insert("bonjour", provider: "p", model: "m", payload: payload("Answer in French.", 8))

        XCTAssertEqual(cache.lookup(provider: "p", model: "m", payload: payload("Answer in French.", 8)), "bonjour")
        XCTAssertNil(cache.lookup(provider: "p", model: "m", payload: payload("Answer in English.", 8)))
        XCTAssertNil(cache.lookup(provider: "p", model: "m", payload: payload("Answer in French.", 512)))
        XCTAssertNil(cache.lookup(provider: "p", model: "m", payload: payload(nil, 8)))
        XCTAssertNil(cache.lookup(provider: "p", model: "m", payload: payload("", 8)))
        XCTAssertNil(cache.lookup(provider: "p", model: "m", prompt: "hi"))
    }
}
