import Foundation
import LocalLLMClientCore

// MARK: - ToolLoopHost

/// A backend (or `AuraLocal`) whose generations run AuraLocal's tool loop.
@MainActor
protocol ToolLoopHost: AnyObject {
    var toolLoop: ToolLoopOptions { get set }
    var onToolEvent: (@MainActor (ToolEvent) -> Void)? { get set }
    var lastToolTurn: ToolTurnSummary? { get }
}

/// Captured when a turn starts, so changes apply to later turns only.
struct ToolLoopContext {
    let options: ToolLoopOptions
    let onEvent: (@MainActor (ToolEvent) -> Void)?

    static var standard: ToolLoopContext { ToolLoopContext(options: .default, onEvent: nil) }
}

// MARK: - ToolLoopDriver

/// The round loop every backend that runs tools shares. A backend owns only how a reply is streamed and parsed and
/// how a round is written into its conversation.
@MainActor
struct ToolLoopDriver {
    let tools: [AnyLLMTool]
    let context: ToolLoopContext

    /// - Parameters:
    ///   - reply: streams the next reply over everything recorded so far; `[]` is a final answer, `nil` a cut reply.
    ///   - record: appends one executed round (its calls and results) to the conversation.
    func run(reply: () async throws -> [LLMToolCall]?,
             record: ([LLMToolCall], [ToolCallResult]) -> Void) async throws -> ToolTurnSummary {
        let limit = max(0, context.options.maxToolRounds)
        var rounds = 0
        var results: [ToolCallResult] = []
        while let calls = try await reply(), !calls.isEmpty {
            let invocations = Self.invocations(of: calls, round: rounds + 1)
            guard rounds < limit else {
                emitLimit(invocations)
                return Self.summary(rounds: rounds, results: results, skipped: invocations)
            }
            rounds += 1
            let executed = try await execute(invocations)
            results += executed
            record(calls, executed)
            try Task.checkCancellation()
        }
        return Self.summary(rounds: rounds, results: results, skipped: [])
    }

    private func execute(_ invocations: [ToolInvocation]) async throws -> [ToolCallResult] {
        var executed: [ToolCallResult] = []
        for invocation in invocations {
            try Task.checkCancellation()
            emitStarted(invocation)
            let outcome = try await ToolRunner.run(invocation, among: tools)
            emitFinished(outcome.result)
            if context.options.toolErrors == .failTurn, let failure = outcome.failure { throw failure }
            executed.append(outcome.result)
        }
        return executed
    }

    private func emitStarted(_ invocation: ToolInvocation) {
        context.onEvent?(.started(invocation))
    }

    private func emitFinished(_ result: ToolCallResult) {
        context.onEvent?(.finished(result))
    }

    private func emitLimit(_ skipped: [ToolInvocation]) {
        context.onEvent?(.roundLimitReached(skipped: skipped))
    }

    static func invocations(of calls: [LLMToolCall], round: Int) -> [ToolInvocation] {
        calls.map { invocation(of: $0, round: round) }
    }

    private static func invocation(of call: LLMToolCall, round: Int) -> ToolInvocation {
        ToolInvocation(id: call.id, round: round, name: call.name, arguments: call.arguments)
    }

    private static func summary(rounds: Int, results: [ToolCallResult],
                                skipped: [ToolInvocation]) -> ToolTurnSummary {
        ToolTurnSummary(rounds: rounds, results: results, skipped: skipped)
    }
}

// MARK: - ToolRunner

/// Runs one call and renders what the model receives. Nonisolated, so tools run off the main actor.
enum ToolRunner {
    struct Outcome: Sendable {
        let result: ToolCallResult
        /// What ``ToolErrorHandling/failTurn`` throws; `nil` when the call succeeded.
        let failure: (any Error)?
    }

    /// Throws only `CancellationError`; every other failure is an error result.
    static func run(_ call: ToolInvocation, among tools: [AnyLLMTool]) async throws -> Outcome {
        guard let tool = tools.first(where: { $0.name == call.name }) else {
            let available = tools.map(\.name).joined(separator: ", ")
            return failed(call, message: "Unknown tool \"\(call.name)\". Available tools: \(available).",
                          failure: LLMError.invalidParameter(reason: "Tool '\(call.name)' not found"))
        }
        let output: ToolOutput
        do {
            output = try await tool.call(argumentsJSON: call.arguments)
        } catch {
            if isCancellation(error) { throw CancellationError() }
            let explained = explanation(of: error, tool: call.name)
            return failed(call, message: explained.message, failure: explained.failure)
        }
        return Outcome(result: ToolCallResult(call: call, output: rendered(output), isError: false), failure: nil)
    }

    /// Sorted-keys JSON; the fallback covers values JSON cannot hold (dates, custom types, NaN).
    static func rendered(_ output: ToolOutput) -> String {
        // An invalid object makes `data(withJSONObject:)` raise an Objective-C exception, not throw.
        guard JSONSerialization.isValidJSONObject(output.data),
              let data = try? JSONSerialization.data(withJSONObject: output.data,
                                                     options: [.sortedKeys, .withoutEscapingSlashes]) else {
            return output.data.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value)" }.joined(separator: ", ")
        }
        return data.isEmpty ? "{}" : String(decoding: data, as: UTF8.self)
    }

    /// `{"error":"…"}`.
    static func errorOutput(_ message: String) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: ["error": message],
                                                     options: [.sortedKeys, .withoutEscapingSlashes]) else {
            return "{\"error\":" + JSONValue.quote(message) + "}"
        }
        return String(decoding: data, as: UTF8.self)
    }

    /// A decoding error as the model can act on it: which argument, and what was wrong with it.
    static func describe(_ error: any Error) -> String {
        guard let decoding = error as? DecodingError else { return error.localizedDescription }
        switch decoding {
        case .typeMismatch(let type, let context):
            return "\(path(of: context)): expected \(type)"
        case .valueNotFound(let type, let context):
            return "\(path(of: context)): missing \(type) value"
        case .keyNotFound(let key, _):
            return "missing argument \"\(key.stringValue)\""
        case .dataCorrupted(let context):
            return context.debugDescription
        @unknown default:
            return decoding.localizedDescription
        }
    }

    private static func path(of context: DecodingError.Context) -> String {
        let path = context.codingPath.map(\.stringValue).joined(separator: ".")
        return path.isEmpty ? "arguments" : path
    }

    // A tool cancelled with its task reaches here wrapped by AnyLLMTool; cancellation must still end the turn.
    private static func isCancellation(_ error: any Error) -> Bool {
        if error is CancellationError { return true }
        guard case .executionFailed(_, let underlying)? = error as? ToolError else { return false }
        return underlying is CancellationError
    }

    private static func explanation(of error: any Error, tool name: String) -> (message: String, failure: any Error) {
        switch error as? ToolError {
        case .argumentDecodingFailed(_, let underlying)?:
            return ("Invalid arguments for tool \"\(name)\": \(describe(underlying))", error)
        case .invalidArgumentEncoding?:
            return ("Invalid arguments for tool \"\(name)\": not UTF-8 text.", error)
        case .executionFailed(_, let underlying)?:
            return ("Tool \"\(name)\" failed: \(underlying.localizedDescription)", underlying)
        case nil:
            return ("Tool \"\(name)\" failed: \(error.localizedDescription)", error)
        }
    }

    private static func failed(_ call: ToolInvocation, message: String, failure: any Error) -> Outcome {
        Outcome(result: ToolCallResult(call: call, output: errorOutput(message), isError: true), failure: failure)
    }
}
