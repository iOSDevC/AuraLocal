#if canImport(MLXLLM)
import MLXLMCommon
import XCTest
@testable import AuraCore

/// Which prompts reuse the stored KV cache, and how far. Layer offsets are plain counters, so no model or GPU runs.
final class MLXPromptCacheTests: XCTestCase {

    private func layers(holding count: Int, depth: Int = 3) -> [KVCache] {
        (0..<depth).map { _ in Self.simpleLayer(offset: count) }
    }

    private static func simpleLayer(offset: Int) -> KVCache {
        let layer = KVCacheSimple()
        layer.offset = offset
        return layer
    }

    private func cache(storing tokens: [Int], in layers: [KVCache]? = nil) -> MLXPromptCache {
        let cache = MLXPromptCache()
        cache.keep(layers ?? self.layers(holding: tokens.count), holding: tokens)
        return cache
    }

    // MARK: Prefix length

    func testTheSharedPrefixLeavesOnePromptTokenToPrefill() {
        XCTAssertEqual(MLXPromptCache.reusableLength(cached: [1, 2, 3], prompt: [1, 2, 3, 4, 5]), 3)
        XCTAssertEqual(MLXPromptCache.reusableLength(cached: [1, 2, 3], prompt: [1, 2, 3]), 2)
        XCTAssertEqual(MLXPromptCache.reusableLength(cached: [1, 2, 3, 9], prompt: [1, 2, 7, 8]), 2)
        XCTAssertEqual(MLXPromptCache.reusableLength(cached: [1, 2, 3, 4], prompt: [1, 2]), 1)
        XCTAssertEqual(MLXPromptCache.reusableLength(cached: [9], prompt: [1, 2]), 0)
        XCTAssertEqual(MLXPromptCache.reusableLength(cached: [], prompt: [1, 2]), 0)
        XCTAssertEqual(MLXPromptCache.reusableLength(cached: [1], prompt: [1]), 0)
        XCTAssertEqual(MLXPromptCache.reusableLength(cached: [1], prompt: []), 0)
    }

    // MARK: Leases

    func testAPromptThatExtendsTheCacheReusesAllOfIt() {
        let stored = layers(holding: 4)
        let cache = cache(storing: [1, 2, 3, 4], in: stored)
        let lease = cache.lease(for: [1, 2, 3, 4, 5, 6], window: nil) { XCTFail("no new cache"); return [] }
        XCTAssertEqual(lease.reused, 4)
        XCTAssertTrue(lease.cache.first as? KVCacheSimple === stored.first as? KVCacheSimple)
        XCTAssertEqual(lease.cache.map(\.offset), [4, 4, 4])
        XCTAssertEqual(cache.stats, .init(hits: 1, misses: 0, reusedTokens: 4, prefilledTokens: 2))
    }

    func testADivergingPromptTrimsEveryLayerToTheSharedPrefix() {
        let cache = cache(storing: [1, 2, 3, 4, 5])
        let lease = cache.lease(for: [1, 2, 9, 9], window: nil) { XCTFail("no new cache"); return [] }
        XCTAssertEqual(lease.reused, 2)
        XCTAssertEqual(lease.cache.map(\.offset), [2, 2, 2])
    }

    func testAnUnrelatedPromptGetsANewCache() {
        let fresh = layers(holding: 0)
        let cache = cache(storing: [1, 2, 3])
        let lease = cache.lease(for: [7, 8, 9], window: nil) { fresh }
        XCTAssertEqual(lease.reused, 0)
        XCTAssertTrue(lease.cache.first as? KVCacheSimple === fresh.first as? KVCacheSimple)
        XCTAssertEqual(cache.stats, .init(hits: 0, misses: 1, reusedTokens: 0, prefilledTokens: 3))
    }

    func testTheCacheIsGivenAwayByEveryLease() {
        let cache = cache(storing: [1, 2, 3])
        _ = cache.lease(for: [1, 2, 3, 4], window: nil) { [] }
        let second = cache.lease(for: [1, 2, 3, 4], window: nil) { self.layers(holding: 0) }
        XCTAssertEqual(second.reused, 0, "nothing is stored until the generation keeps its cache")
    }

    func testALayerThatCannotTrimMisses() {
        let rotated = RotatingKVCache(maxSize: 4)
        rotated.offset = 6
        let stored = layers(holding: 6, depth: 2) + [rotated]
        let cache = cache(storing: [1, 2, 3, 4, 5, 6], in: stored)
        let lease = cache.lease(for: [1, 2, 3, 9], window: nil) { self.layers(holding: 0) }
        XCTAssertEqual(lease.reused, 0)
        XCTAssertEqual(stored.map(\.offset), [6, 6, 6], "nothing was trimmed")
    }

    func testAPromptAtTheWindowMisses() {
        let cache = cache(storing: [1, 2, 3])
        XCTAssertEqual(cache.lease(for: [1, 2, 3, 4], window: 4) { self.layers(holding: 0) }.reused, 0)
        let below = self.cache(storing: [1, 2, 3])
        XCTAssertEqual(below.lease(for: [1, 2, 3, 4], window: 5) { [] }.reused, 3)
    }

    func testLayersThatDisagreeWithTheTokensAreNotKept() {
        let cache = MLXPromptCache()
        cache.keep(layers(holding: 5), holding: [1, 2, 3])
        XCTAssertEqual(cache.lease(for: [1, 2, 3, 4], window: nil) { self.layers(holding: 0) }.reused, 0)

        let uneven = layers(holding: 3, depth: 2) + [Self.simpleLayer(offset: 2)]
        cache.keep(uneven, holding: [1, 2, 3])
        XCTAssertEqual(cache.lease(for: [1, 2, 3, 4], window: nil) { self.layers(holding: 0) }.reused, 0)
    }

    func testAnEmptyCacheMissesAndCountsThePrompt() {
        let cache = MLXPromptCache()
        XCTAssertEqual(cache.lease(for: [1, 2], window: nil) { self.layers(holding: 0) }.reused, 0)
        XCTAssertEqual(cache.stats, .init(hits: 0, misses: 1, reusedTokens: 0, prefilledTokens: 2))
    }
}
#endif
