// MLX tool rounds — compiled in only when the MLX products (mlx-swift-lm) are linked.
#if canImport(MLXLLM)
import Foundation
import LocalLLMClientCore
import MLXLMCommon

/// A call's arguments as mlx-swift-lm's parser produced them.
typealias ParsedArguments = [String: MLXLMCommon.JSONValue]

// MARK: - MLXToolRound

/// One executed round of an MLX tool turn: the reply's visible text, its calls and what each call returned.
struct MLXToolRound: Sendable {
    let text: String
    let calls: [LLMToolCall]
    /// Each call's arguments as the parser produced them, by call id; the native form sends them as an object.
    let arguments: [String: ParsedArguments]
    let results: [ToolCallResult]
}

/// A render of the conversation, and the round shape it used.
struct MLXRenderAttempt: Sendable {
    let shape: MLXToolTranscript.RoundShape
    let messages: [[String: any Sendable]]
}

/// One reply of a tool turn and the round shape its prompt was rendered with.
struct MLXToolReply: Sendable {
    let shape: MLXToolTranscript.RoundShape
    let outcome: MLXDecodeOutcome
}

// MARK: - MLXToolTranscript

/// The conversation one MLX tool turn renders: the messages the turn started from, then every round run so far.
struct MLXToolTranscript: Sendable {
    /// How the rounds reach the chat template.
    enum RoundShape: Sendable, Equatable {
        /// An assistant turn with `tool_calls`, then one `tool` turn per result, which templates with tool support
        /// render in their own format.
        case native
        /// Every round in one assistant turn: the calls as `<tool_call>` text, the results in `<tool_response>`.
        case text
    }

    let base: [[String: any Sendable]]
    private(set) var rounds: [MLXToolRound] = []
    private(set) var shape = RoundShape.native

    init(base: [[String: any Sendable]]) {
        self.base = base
    }

    /// Renders to try in order: the native shape first, the text form when the template throws on it.
    var attempts: [MLXRenderAttempt] {
        guard !rounds.isEmpty, shape == .native else { return [attempt(shape)] }
        return [attempt(.native), attempt(.text)]
    }

    func messages(_ shape: RoundShape) -> [[String: any Sendable]] {
        guard !rounds.isEmpty else { return base }
        switch shape {
        case .native: return base + rounds.flatMap(MLXToolMessages.native)
        case .text: return base + [MLXToolMessages.text(rounds)]
        }
    }

    mutating func record(_ round: MLXToolRound) {
        rounds.append(round)
    }

    /// Keeps the shape that rendered: once the native shape has thrown, later replies of the turn skip it.
    mutating func settle(on rendered: RoundShape) {
        if !rounds.isEmpty { shape = rendered }
    }

    private func attempt(_ shape: RoundShape) -> MLXRenderAttempt {
        MLXRenderAttempt(shape: shape, messages: messages(shape))
    }
}

// MARK: - MLXToolMessages

/// Builds the chat-template messages of a round and converts mlx-swift-lm's parsed calls.
enum MLXToolMessages {
    /// `String?.none`, which the template engine reads as JSON `null`.
    private static let null: String? = nil

    static func native(_ round: MLXToolRound) -> [[String: any Sendable]] {
        let calls = round.calls.map { nativeCall($0, arguments: round.arguments[$0.id] ?? [:]) }
        let reply: [String: any Sendable] = ["role": "assistant", "content": visible(round.text), "tool_calls": calls]
        return [reply] + round.results.map(toolMessage)
    }

    /// The OpenAI shape chat templates read, with `arguments` as an object so `tojson` writes it back.
    static func nativeCall(_ call: LLMToolCall, arguments: ParsedArguments) -> [String: any Sendable] {
        let function: [String: any Sendable] = ["name": call.name, "arguments": arguments.mapValues(nativeValue)]
        return ["type": "function", "id": call.id, "function": function]
    }

    static func toolMessage(_ result: ToolCallResult) -> [String: any Sendable] {
        ["role": "tool", "content": result.output, "tool_call_id": result.call.id, "name": result.call.name]
    }

    /// Native Swift values: a `Bool` must reach the template as a boolean, not as the number `NSNumber` holds.
    static func nativeValue(_ value: MLXLMCommon.JSONValue) -> any Sendable {
        switch value {
        case .null: return null
        case .bool(let flag): return flag
        case .int(let integer): return integer
        case .double(let number): return number
        case .string(let text): return text
        case .array(let elements): return elements.map(nativeValue)
        case .object(let members): return members.mapValues(nativeValue)
        }
    }

    /// One assistant turn for every round so far, so templates that require alternating roles accept it.
    static func text(_ rounds: [MLXToolRound]) -> [String: any Sendable] {
        ["role": "assistant", "content": rounds.map(transcript).joined(separator: "\n")]
    }

    static func transcript(_ round: MLXToolRound) -> String {
        let reply = visible(round.text)
        let parts = (reply.isEmpty ? [] : [reply]) + round.calls.map(callText) + round.results.map(responseText)
        return parts.joined(separator: "\n")
    }

    static func callText(_ call: LLMToolCall) -> String {
        let arguments = call.arguments.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "{}" : call.arguments
        let name = AuraCore.JSONValue.quote(call.name)
        return "<tool_call>\n{\"name\": \(name), \"arguments\": \(arguments)}\n</tool_call>"
    }

    private static func responseText(_ result: ToolCallResult) -> String {
        "<tool_response>\n\(result.output)\n</tool_response>"
    }

    /// Whitespace between calls is left as text by the parser; kept, it would precede the calls in the prompt.
    private static func visible(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "" : text
    }

    // MARK: Parsed calls

    /// mlx-swift-lm's calls carry no id, so each gets a UUID; the arguments become sorted-keys JSON.
    static func calls(from parsed: [ToolCall]) -> (calls: [LLMToolCall], arguments: [String: ParsedArguments]) {
        let converted = parsed.map(call)
        let arguments = Dictionary(converted.map { ($0.id, $0.arguments) }) { first, _ in first }
        return (converted.map(\.call), arguments)
    }

    private static func call(_ parsed: ToolCall) -> (id: String, call: LLMToolCall, arguments: ParsedArguments) {
        let id = UUID().uuidString
        let arguments = parsed.function.arguments
        return (id, LLMToolCall(id: id, name: parsed.function.name, arguments: argumentsJSON(arguments)), arguments)
    }

    static func argumentsJSON(_ arguments: ParsedArguments) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(arguments), !data.isEmpty else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }
}
#endif
