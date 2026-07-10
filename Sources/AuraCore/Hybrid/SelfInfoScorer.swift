import Foundation

/// Scores how much each sentence should be KEPT when compressing context toward a
/// token budget (higher score = more likely kept). This is the pluggable seam the
/// ``ContextCompressor`` sits on, so scoring can evolve without touching callers.
///
/// - `HeuristicScorer` (default) — keyword overlap with the question + recency.
///   Cheap, dependency-free, ships today.
/// - A true **self-information** scorer (Selective Context: keep low-probability,
///   high-surprisal sentences) needs per-token logprobs from the local model.
///   BLOCKER: `LocalLLMClient`'s `llama_context` is `package`-private, so AuraCore
///   cannot reuse the loaded context; a real scorer would load its own
///   `llama_context` via the C API (`llama.h` is available in `LocalLLMClientLlamaC`)
///   and run a windowed prefill (`llama_decode` with per-position logits →
///   `llama_get_logits_ith` → log-softmax → surprisal), mirroring llama.cpp's
///   `perplexity.cpp`. That is the intended `LlamaCppSelfInfoScorer` — deferred
///   because it doubles model memory and needs careful iOS-side chunking.
public protocol SelfInfoScorer: Sendable {
    /// Return one score per sentence (same order). Higher = keep.
    func scores(for sentences: [String], question: String) -> [Double]
}

/// Default scorer: relevance to the question (keyword overlap) plus a recency bias.
public struct HeuristicScorer: SelfInfoScorer {
    public init() {}

    public func scores(for sentences: [String], question: String) -> [Double] {
        let questionWords = ContextCompressor.keywords(question)
        let lastIndex = max(sentences.count - 1, 1)
        return sentences.enumerated().map { index, sentence in
            let overlap = Double(ContextCompressor.keywords(sentence).intersection(questionWords).count)
            let recency = Double(index) / Double(lastIndex)
            return overlap * 2.0 + recency
        }
    }
}
