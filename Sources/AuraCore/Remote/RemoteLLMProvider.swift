import Foundation

// MARK: - RemoteLLMProvider

/// A pluggable remote LLM the hybrid pipeline can escalate to.
///
/// Providers are `Sendable` strategy values (mirrors ``InferenceBackend`` /
/// ``BackendRouter``). Concrete conformers: `OpenAICompatibleProvider`
/// (hosted OpenAI, the user's llama-server, Ollama `/v1`) and `AnthropicProvider`.
public protocol RemoteLLMProvider: Sendable {
    /// Stable identifier, also used as the Keychain account for the API key.
    var id: String { get }
    /// Human-readable name for UI.
    var displayName: String { get }
    /// One-line data-retention note surfaced in the consent sheet.
    var retentionNote: String { get }
    /// Stream a response as a sequence of ``RemoteEvent``s. Throws on transport
    /// or non-success HTTP status.
    func stream(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error>
}

// MARK: - Wire types

/// A remote generation request. `messages` reuses AuraCore's `[[String: String]]`
/// shape (keys `role` / `content`), which maps 1:1 to both OpenAI and Anthropic bodies.
public struct RemoteRequest: Sendable {
    public let model: String
    public let system: String?
    public let messages: [[String: String]]
    public let maxTokens: Int
    public let temperature: Float

    public init(
        model: String,
        system: String? = nil,
        messages: [[String: String]],
        maxTokens: Int = 1024,
        temperature: Float = 0.7
    ) {
        self.model = model
        self.system = system
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
    }
}

/// A streamed event from a remote provider.
public enum RemoteEvent: Sendable {
    /// An incremental text delta.
    case token(String)
    /// Final (or cumulative) token accounting, when the provider reports it.
    case usage(TokenUsage)
}

/// Token accounting for a remote call.
public struct TokenUsage: Sendable, Hashable {
    public let inputTokens: Int
    public let outputTokens: Int

    public init(inputTokens: Int, outputTokens: Int) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
    }

    public var totalTokens: Int { inputTokens + outputTokens }
}

// MARK: - RemoteTarget

/// A concrete escalation target: a provider + a specific model + where it lives.
public struct RemoteTarget: Sendable {
    public let provider: any RemoteLLMProvider
    public let modelID: String
    /// Trained context length, when known (feeds the compression budget).
    public let contextLength: Int?
    public let origin: Origin

    /// Where the target runs — drives consent friction and cost.
    public enum Origin: Sendable, Hashable {
        /// A hosted cloud provider — full consent + retention disclosure.
        case cloud
        /// The user's own machine (llama-server / Ollama) — light-touch consent, $0.
        case localNetwork(LocalProviderKind)
    }

    public init(provider: any RemoteLLMProvider, modelID: String, contextLength: Int? = nil, origin: Origin) {
        self.provider = provider
        self.modelID = modelID
        self.contextLength = contextLength
        self.origin = origin
    }

    /// `true` when the target runs on the user's own hardware (no data leaves the trust boundary, no cost).
    public var isLocalNetwork: Bool {
        if case .localNetwork = origin { return true }
        return false
    }
}
