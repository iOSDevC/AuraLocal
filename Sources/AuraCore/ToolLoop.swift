import Foundation
import LocalLLMClientCore

// MARK: - ToolLoopOptions

/// How a turn runs the tool calls the model makes. A **round** is every call in one reply: the calls run in the
/// order the model wrote them, then the model replies again with their results. The loop ends at a reply that asks
/// for no tools, at a cut reply (`maxTokens`, shared by every round of the turn, or cancellation), or at a reply
/// that asks for tools after `maxToolRounds` rounds have run; those last calls are reported as skipped, not run.
///
/// Read when a turn starts, so a change applies from the next turn. GGUF models only.
public struct ToolLoopOptions: Sendable, Equatable {
    /// One round: as many as ran before `ToolLoopOptions` existed.
    public static let defaultMaxToolRounds = 1
    public static let `default` = ToolLoopOptions()

    /// Rounds of tool calls one turn may run. Negative counts as 0; 0 advertises the tools but runs none, so every
    /// call the model makes is reported as skipped.
    public var maxToolRounds: Int
    /// What a failing call does to the turn. Default ``ToolErrorHandling/reportToModel``.
    public var toolErrors: ToolErrorHandling

    public init(maxToolRounds: Int = ToolLoopOptions.defaultMaxToolRounds,
                toolErrors: ToolErrorHandling = .reportToModel) {
        self.maxToolRounds = maxToolRounds
        self.toolErrors = toolErrors
    }
}

/// What happens when a tool throws, its arguments do not decode, or the model names a tool that does not exist.
public enum ToolErrorHandling: Sendable, Equatable {
    /// The model receives `{"error": "…"}` as the tool's result and the turn goes on; the result has `isError` set.
    case reportToModel
    /// The turn throws the error, as before `ToolLoopOptions` existed: the tool's own error, LocalLLMClient's
    /// `ToolError.argumentDecodingFailed`, or `LLMError.invalidParameter` for an unknown tool.
    case failTurn
}

// MARK: - Events and results

/// One tool call the model asked for.
public struct ToolInvocation: Sendable, Equatable, Identifiable {
    /// The call's id from the parser; a UUID when the model gave none.
    public let id: String
    /// 1-based. For a skipped call, the round that did not run.
    public let round: Int
    public let name: String
    /// The arguments JSON the parser extracted from the reply.
    public let arguments: String

    public init(id: String, round: Int, name: String, arguments: String) {
        self.id = id
        self.round = round
        self.name = name
        self.arguments = arguments
    }
}

/// What one call returned to the model.
public struct ToolCallResult: Sendable, Equatable {
    public let call: ToolInvocation
    /// Exactly what the model received: the tool's output as JSON with sorted keys (`key: value` pairs, sorted, when
    /// it is not JSON-serialisable), or `{"error": "…"}` when the call failed.
    public let output: String
    public let isError: Bool

    public init(call: ToolInvocation, output: String, isError: Bool) {
        self.call = call
        self.output = output
        self.isError = isError
    }
}

/// Reported on the main actor while a turn runs its tools, in order; not ordered relative to the streamed text.
public enum ToolEvent: Sendable, Equatable {
    /// A call is about to run. Every `started` is followed by its `finished` unless the turn throws.
    case started(ToolInvocation)
    case finished(ToolCallResult)
    /// The model asked for more calls after `maxToolRounds` rounds; none of them ran.
    case roundLimitReached(skipped: [ToolInvocation])
}

/// What the tool loop did in one turn.
public struct ToolTurnSummary: Sendable, Equatable {
    /// Rounds that ran, at most `maxToolRounds`.
    public let rounds: Int
    /// Every call that ran, in order.
    public let results: [ToolCallResult]
    /// Calls the model asked for after the last allowed round; non-empty only when `maxToolRounds` ended the loop.
    public let skipped: [ToolInvocation]

    public init(rounds: Int, results: [ToolCallResult], skipped: [ToolInvocation]) {
        self.rounds = rounds
        self.results = results
        self.skipped = skipped
    }
}
