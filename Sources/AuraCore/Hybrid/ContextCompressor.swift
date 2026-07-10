import Foundation

/// The receipt of a compression pass — also the exact payload shown for consent.
public struct CompressionResult: Sendable {
    public let originalTokens: Int
    public let compressedTokens: Int
    public let keptText: String

    /// compressed / original (lower is better). 1.0 when nothing was dropped.
    public var ratio: Double { Double(compressedTokens) / Double(max(originalTokens, 1)) }
    /// e.g. "3.2×" fewer tokens.
    public var factor: Double { Double(originalTokens) / Double(max(compressedTokens, 1)) }
}

/// Selectively keeps the most relevant sentences of a context within a token
/// budget, so the remote model receives fewer (cheaper) tokens.
///
/// Stage A-lite: relevance = keyword overlap with the question + recency bias.
/// Upgradeable behind this same interface to TF-IDF (AuraDocs) or self-information
/// scoring (llama.cpp logits) without changing callers.
public struct ContextCompressor: Sendable {
    private let scorer: SelfInfoScorer

    public init(scorer: SelfInfoScorer = HeuristicScorer()) {
        self.scorer = scorer
    }

    /// Compress `context` to `budgetTokens`, keeping sentences most relevant to
    /// `question`. Original order is preserved. If `context` already fits, it is
    /// returned unchanged.
    public func compress(context: String, question: String, budgetTokens: Int) -> CompressionResult {
        let originalTokens = Self.estimateTokens(context)
        guard originalTokens > budgetTokens, budgetTokens > 0 else {
            return CompressionResult(originalTokens: originalTokens, compressedTokens: originalTokens, keptText: context)
        }

        let sentences = Self.sentences(context)
        guard sentences.count > 1 else {
            return CompressionResult(originalTokens: originalTokens, compressedTokens: originalTokens, keptText: context)
        }

        // Score each sentence via the pluggable scorer (higher = keep).
        let scoreValues = scorer.scores(for: sentences, question: question)
        let scored = sentences.enumerated().map { index, sentence -> (index: Int, text: String, score: Double) in
            (index, sentence, scoreValues[index])
        }

        // Greedily keep highest-scored sentences within budget, then restore order.
        var kept: [(index: Int, text: String)] = []
        var used = 0
        for item in scored.sorted(by: { $0.score > $1.score }) {
            let tokens = Self.estimateTokens(item.text)
            if used + tokens > budgetTokens { continue }
            kept.append((item.index, item.text))
            used += tokens
        }

        let keptText = kept.sorted { $0.index < $1.index }.map(\.text).joined(separator: ". ")
        return CompressionResult(
            originalTokens: originalTokens,
            compressedTokens: Self.estimateTokens(keptText),
            keptText: keptText)
    }

    // MARK: - Helpers

    /// Rough token estimate (~chars/4), matching AuraDocs' heuristic.
    static func estimateTokens(_ text: String) -> Int { max(1, text.count / 4) }

    static func sentences(_ text: String) -> [String] {
        text.split(whereSeparator: { ".!?\n".contains($0) })
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    static func keywords(_ text: String) -> Set<String> {
        Set(text.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count > 3 })
    }
}
