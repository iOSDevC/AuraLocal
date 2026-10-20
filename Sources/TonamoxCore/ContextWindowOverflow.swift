import Foundation

/// A generation whose prompt plus output needed more tokens than the KV window the backend allotted.
///
/// MLX bounds its cache with a rotating window that keeps the first 4 tokens and overwrites the oldest of the
/// rest, without an error: the model stopped seeing ``droppedTokens`` tokens of the conversation.
public struct ContextWindowOverflow: Sendable, Equatable {
    public let promptTokens: Int
    public let generatedTokens: Int
    /// The KV window, in tokens.
    public let windowSize: Int

    public init(promptTokens: Int, generatedTokens: Int, windowSize: Int) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.windowSize = windowSize
    }

    /// Tokens that no longer fit the window.
    public var droppedTokens: Int {
        max(0, promptTokens + generatedTokens - windowSize)
    }

    /// `nil` when everything fit.
    static func detect(promptTokens: Int, generatedTokens: Int, windowSize: Int) -> ContextWindowOverflow? {
        guard windowSize > 0, promptTokens + generatedTokens > windowSize else { return nil }
        return ContextWindowOverflow(promptTokens: promptTokens, generatedTokens: generatedTokens,
                                     windowSize: windowSize)
    }
}
