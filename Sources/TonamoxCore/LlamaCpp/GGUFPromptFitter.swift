import Foundation
import LocalLLMClient

// MARK: - GGUFPromptFitter

/// Keeps a rendered conversation inside the KV window. LocalLLMClient 0.5.0 exposes no tokenizer, so tokens are
/// estimated from UTF-8 bytes with a ratio learned from the overflows llama.cpp reports. Until the first overflow
/// nothing is trimmed ahead of a prefill, so a prompt that fits but leaves less than the reply reserve gets its reply
/// cut by the window; the exact count after a prefill, `Context.position`, is `package`.
struct GGUFPromptFitter: Sendable {
    let windowSize: Int
    /// Tokens per UTF-8 byte, a lower bound taken from the largest overflow seen.
    private(set) var tokensPerByte: Double?

    init(windowSize: Int) {
        self.windowSize = windowSize
    }

    /// Room kept for the reply when trimming ahead of a prefill: the budget, at most a quarter of the window.
    func replyReserve(forBudget maxTokens: Int) -> Int {
        min(max(maxTokens, 0), windowSize / 4)
    }

    func estimatedTokens(in prompt: String) -> Int? {
        tokensPerByte.map { Int((Double(prompt.utf8.count) * $0).rounded(.up)) }
    }

    /// Renders `history`, dropping its oldest exchanges while the estimate plus the reply reserve exceeds the window.
    /// The system turn and the newest user turn always stay.
    func fitted(
        _ history: [LlamaChatMessage],
        replyBudget: Int,
        render: ([LlamaChatMessage]) throws -> String
    ) throws -> (history: [LlamaChatMessage], prompt: String) {
        let reserve = replyReserve(forBudget: replyBudget)
        var kept = history
        var prompt = try render(kept)
        while let estimate = estimatedTokens(in: prompt), estimate + reserve > windowSize,
              let shorter = LlamaChatNormalizer.droppingOldestExchange(kept) {
            kept = shorter
            prompt = try render(kept)
        }
        return (kept, prompt)
    }

    /// After an overflow: what ``fitted(_:replyBudget:render:)`` keeps under the updated ratio, or at least one
    /// exchange fewer. `nil` when only the newest user turn is left.
    func shortened(
        _ history: [LlamaChatMessage],
        replyBudget: Int,
        render: ([LlamaChatMessage]) throws -> String
    ) throws -> (history: [LlamaChatMessage], prompt: String)? {
        let refitted = try fitted(history, replyBudget: replyBudget, render: render)
        if refitted.history.count < history.count { return refitted }
        guard let shorter = LlamaChatNormalizer.droppingOldestExchange(history) else { return nil }
        return (shorter, try render(shorter))
    }

    /// Updates the ratio from a prefill llama.cpp refused for the window; `false` for any other error.
    mutating func learn(from error: any Error, promptBytes: Int) -> Bool {
        guard let needed = Self.overflowedTokens(in: error, windowSize: windowSize) else { return false }
        if promptBytes > 0 {
            tokensPerByte = max(tokensPerByte ?? 0, Double(needed) / Double(promptBytes))
        }
        return true
    }

    /// The positions LocalLLMClient's `context size exceeded[window < needed]` reports: a lower bound on the
    /// prompt's tokens, exact when the chunk that overflowed was its last. `nil` for any other error.
    static func overflowedTokens(in error: any Error, windowSize: Int) -> Int? {
        guard case .failedToDecode(let reason) = error as? LLMError,
              reason.contains("context size exceeded") else { return nil }
        if let match = reason.firstMatch(of: #/exceeded\[\d+\s*<\s*(\d+)\]/#), let needed = Int(match.1) {
            return max(needed, windowSize + 1)
        }
        return windowSize + 1
    }
}
