import Foundation
import LocalLLMClientCore

// MARK: - AuraProfile

/// A **profile** bundles everything that defines a conversational "mode": a persona (system prompt), the tools
/// available, the model/backend to run on, and the sampling parameters. A session can switch between profiles
/// mid-conversation while preserving the transcript (see `AuraSession`, increment 2).
///
/// This mirrors the *anatomy* of Apple's `LanguageModelSession.DynamicProfile` (instructions + tools + model +
/// sampling) but is **engine-agnostic** and does NOT depend on that beta API — it drives the on-device GGUF
/// engine and works anywhere, including where Apple Intelligence is unavailable.
///
/// Pure value type (`Sendable`, no C++/`LLMSession` references) so it lives in the Simulator-safe layer and is
/// referenceable from any target. `[any LLMTool]` is `Sendable` (from `LocalLLMClientCore`), so the whole
/// profile is.
public struct AuraProfile: Sendable, Identifiable {
    /// Stable identity; also the discriminator a session/cache uses to tell profiles apart.
    public let id: String
    public var displayName: String
    /// The persona / system prompt. Held here (NOT persisted as a conversation turn) so switching a profile
    /// actually changes the persona instead of being clobbered by a written-once system turn.
    public var instructions: String
    /// Model + backend hint — `model.format` drives backend selection.
    public var model: Model
    public var sampling: SamplingParams
    /// Empty ⇒ plain chat; non-empty ⇒ agentic (function-calling). Only the GGUF backend honors tools.
    public var tools: [any LLMTool]
    /// Optional structured-output constraint. Carried now; grammar/JSON-mode plumbing is a later increment.
    public var outputSchema: OutputSchema?

    public init(id: String,
                displayName: String,
                instructions: String,
                model: Model,
                sampling: SamplingParams = .default,
                tools: [any LLMTool] = [],
                outputSchema: OutputSchema? = nil) {
        self.id = id
        self.displayName = displayName
        self.instructions = instructions
        self.model = model
        self.sampling = sampling
        self.tools = tools
        self.outputSchema = outputSchema
    }

    /// Whether this profile enables function-calling (agentic) vs. plain chat.
    public var isAgentic: Bool { !tools.isEmpty }
}

// `[any LLMTool]` isn't `Equatable`, so compare tools by their (stable) name set. Two profiles are equal when
// identity, persona, model, sampling, tool-name-set, and output schema all match.
extension AuraProfile: Equatable {
    public static func == (lhs: AuraProfile, rhs: AuraProfile) -> Bool {
        lhs.id == rhs.id
            && lhs.displayName == rhs.displayName
            && lhs.instructions == rhs.instructions
            && lhs.model == rhs.model
            && lhs.sampling == rhs.sampling
            && lhs.tools.map(\.name) == rhs.tools.map(\.name)
            && lhs.outputSchema == rhs.outputSchema
    }
}

// MARK: - SamplingParams

/// The knobs that shape generation. In increment 1 only `temperature` is threaded end-to-end; the rest are
/// carried so profiles can express them and later increments widen `LlamaClient.Parameter` to honor them.
public struct SamplingParams: Sendable, Equatable, Codable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int?
    public var maxTokens: Int
    /// Override the derived context window (tokens). Deferred plumbing.
    public var contextOverride: Int?

    public init(temperature: Float = 0.7, topP: Float = 0.95, topK: Int? = nil,
                maxTokens: Int = 1024, contextOverride: Int? = nil) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.maxTokens = maxTokens
        self.contextOverride = contextOverride
    }

    /// General-purpose chat sampling (matches today's engine default).
    public static let `default` = SamplingParams()
    /// Low-temperature sampling for deterministic tool-calling / structured output (agent, security review).
    public static let precise = SamplingParams(temperature: 0.2)
}

// MARK: - OutputSchema

/// An optional structured-output constraint carried by a profile. Only `.json` is defined for now; enforcement
/// (GBNF / JSON-mode into the sampler) is a later increment — no JSON-mode seam exists in the backend yet.
public enum OutputSchema: Sendable, Equatable, Codable {
    /// A JSON schema / GBNF grammar string the output must conform to.
    case json(String)
}

// MARK: - AuraProfileCatalog

/// Built-in profiles for the common modes. Pure values (no engine). Instructions default to sensible library
/// personas but callers (e.g. Deckrix) override them with app-specific prompts.
public enum AuraProfileCatalog {
    public static let defaultChatInstructions =
        "You are a helpful, concise on-device assistant."
    public static let defaultAgentInstructions =
        "You complete the user's task by calling the provided tools, one step at a time. "
        + "Respond with the single next tool call; do not narrate."
    public static let defaultSecurityInstructions =
        "You are a security reviewer. Analyze the provided input for risks and report findings concisely."

    /// Plain chat — no tools, general sampling.
    public static func chat(model: Model,
                            instructions: String = defaultChatInstructions,
                            sampling: SamplingParams = .default) -> AuraProfile {
        AuraProfile(id: "chat", displayName: "Chat", instructions: instructions,
                    model: model, sampling: sampling, tools: [])
    }

    /// Agentic — function-calling, low-temperature for deterministic tool selection.
    public static func agent(model: Model,
                             tools: [any LLMTool],
                             instructions: String = defaultAgentInstructions,
                             sampling: SamplingParams = .precise) -> AuraProfile {
        AuraProfile(id: "agent", displayName: "Agent", instructions: instructions,
                    model: model, sampling: sampling, tools: tools)
    }

    /// Security review — routes to the (security) model with a review persona; precise sampling.
    public static func securityReview(model: Model,
                                      tools: [any LLMTool] = [],
                                      instructions: String = defaultSecurityInstructions,
                                      sampling: SamplingParams = .precise) -> AuraProfile {
        AuraProfile(id: "security", displayName: "Security review", instructions: instructions,
                    model: model, sampling: sampling, tools: tools)
    }
}
