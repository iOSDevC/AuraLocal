import Foundation

/// The exact request an escalation sends: built once by ``HybridEscalator/preparePayload(for:systemPrompt:context:question:maxTokens:keepRatio:redactPII:)``,
/// shown to the ``ConsentGate`` and then sent unchanged by ``HybridEscalator/escalate(_:to:onToken:)``.
public struct EscalationPayload: Sendable, Equatable {
    /// Sent as the system message, verbatim: never compressed or redacted.
    public let systemPrompt: String?
    /// The single user message: the compressed context and the question, after redaction.
    public let userMessage: String
    /// The request's `max_tokens`.
    public let maxTokens: Int
    /// Whether ``PIIRedactor`` ran on the context and the question.
    public let redactionEnabled: Bool
    /// Secrets/PII replaced in the question and the whole context, including any that compression then dropped;
    /// 0 when redaction is off.
    public let redactedPIICount: Int
    /// The compression receipt. Its `keptText` is the context as sent; redaction runs before compression.
    public let compression: CompressionResult

    /// The messages in `role`/`content` form, the system message first when there is one.
    /// OpenAI-compatible providers send them as is; Anthropic sends the system message as its top-level `system`.
    public var messages: [[String: String]] {
        let user = ["role": "user", "content": userMessage]
        guard let systemPrompt else { return [user] }
        return [["role": "system", "content": systemPrompt], user]
    }

    /// The system prompt and the user message at ~4 characters per token: what `routeAndEscalate` prices.
    public var estimatedInputTokens: Int {
        ContextCompressor.estimateTokens((systemPrompt ?? "") + userMessage)
    }
}
