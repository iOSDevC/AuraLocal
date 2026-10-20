// MLX prompt cache — compiled in only when the MLX products (mlx-swift-lm) are linked.
#if canImport(MLXLLM)
import Foundation
import MLXLMCommon

// MARK: - MLXPromptCache

/// The KV cache an `MLXBackend` text generation left behind, with the exact tokens it holds. The next prompt that
/// starts with some of those tokens prefills only the rest; any other prompt gets a new cache, so a hit never changes
/// which tokens the model has seen.
///
/// Used only inside `ModelContainer.perform`, which serialises every access.
final class MLXPromptCache: @unchecked Sendable {
    struct ReuseStats: Sendable, Equatable {
        /// Prompts that reused part of the cache.
        var hits = 0
        /// Prompts that got a new cache.
        var misses = 0
        /// Prompt tokens taken from the cache instead of prefilled.
        var reusedTokens = 0
        /// Prompt tokens prefilled.
        var prefilledTokens = 0
    }

    /// A cache handed to one generation, and how many of its prompt's tokens it already holds.
    struct Lease {
        let cache: [KVCache]
        let reused: Int
    }

    private var layers: [KVCache]?
    private var tokens: [Int] = []
    private(set) var stats = ReuseStats()

    /// The stored cache trimmed to the longest prefix it shares with `prompt`, or a new one from `fresh`. The
    /// stored cache is given away either way; ``keep(_:holding:)`` stores what the generation leaves.
    func lease(for prompt: [Int], window: Int?, fresh: () -> [KVCache]) -> Lease {
        let stored = layers
        let held = tokens
        discard()
        let reused = stored.map { Self.reuse($0, holding: held, for: prompt, window: window) } ?? 0
        record(reused: reused, promptTokens: prompt.count)
        guard reused > 0, let stored else { return Lease(cache: fresh(), reused: 0) }
        return Lease(cache: stored, reused: reused)
    }

    /// Stores `cache` for the next prompt when every layer holds exactly `held`.
    func keep(_ cache: [KVCache], holding held: [Int]) {
        guard !cache.isEmpty, cache.allSatisfy({ $0.offset == held.count }) else { return discard() }
        layers = cache
        tokens = held
    }

    func discard() {
        layers = nil
        tokens = []
    }

    /// Trims `cache` to the prefix it shares with `prompt` and returns its length; 0 when it cannot be reused.
    ///
    /// Misses: a prompt at or past the KV window (a rotating cache would drop tokens differently than a full
    /// prefill), a cache any layer cannot trim (recurrent layers, a rotating cache that wrapped), and layers whose
    /// offsets disagree with the tokens recorded.
    static func reuse(_ cache: [KVCache], holding held: [Int], for prompt: [Int], window: Int?) -> Int {
        guard prompt.count < (window ?? .max), !cache.isEmpty, canTrimPromptCache(cache),
              cache.allSatisfy({ $0.offset == held.count }) else { return 0 }
        let shared = reusableLength(cached: held, prompt: prompt)
        guard shared > 0 else { return 0 }
        trimPromptCache(cache, numTokens: held.count - shared)
        return cache.allSatisfy { $0.offset == shared } ? shared : 0
    }

    /// The tokens `prompt` shares with `cached` from the start, leaving at least one prompt token to prefill: the
    /// first sampled token needs the logits of the prompt's last one.
    static func reusableLength(cached: [Int], prompt: [Int]) -> Int {
        let limit = min(cached.count, prompt.count - 1)
        guard limit > 0 else { return 0 }
        return zip(cached.prefix(limit), prompt).prefix { $0 == $1 }.count
    }

    private func record(reused: Int, promptTokens: Int) {
        if reused > 0 {
            stats.hits += 1
        } else {
            stats.misses += 1
        }
        stats.reusedTokens += reused
        stats.prefilledTokens += promptTokens - reused
    }
}
#endif
