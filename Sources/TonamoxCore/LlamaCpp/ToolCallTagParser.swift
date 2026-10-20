import Foundation
import LocalLLMClientCore

// MARK: - ToolCallTagParser

/// Splits a streamed reply into the text a reader sees and the calls written as
/// `<tool_call>{"name": …, "arguments": {…}}</tool_call>`: the format of Qwen2.5, Qwen3 and Hermes templates and of
/// the tool instructions sent to templates without tool support.
///
/// Every call is kept, two to the same tool included. A block whose body is not such an object stays text, tags
/// included. Arguments come out as compact JSON with sorted keys.
struct ToolCallTagParser {
    static let openTag = "<tool_call>"
    static let closeTag = "</tool_call>"

    /// Text that may still be the start of a tag, or the body of an open call.
    private var held = ""
    private var insideCall = false
    private(set) var calls: [LLMToolCall] = []

    /// The text `piece` completes; text that may still open a tag, and a call still open, are held back.
    mutating func consume(_ piece: String) -> String {
        held += piece
        var visible = ""
        while true {
            if insideCall {
                guard let close = held.range(of: Self.closeTag) else { return visible }
                visible += closed(body: Self.text(of: held, upTo: close.lowerBound))
                held = Self.text(of: held, from: close.upperBound)
                insideCall = false
            } else if let open = held.range(of: Self.openTag) {
                visible += Self.text(of: held, upTo: open.lowerBound)
                held = Self.text(of: held, from: open.upperBound)
                insideCall = true
            } else {
                let kept = Self.partialTagLength(at: held)
                visible += Self.text(of: held, droppingLast: kept)
                held = Self.text(of: held, keepingLast: kept)
                return visible
            }
        }
    }

    /// The text still held back once the reply is over. A call left open is text when the reply ended inside it
    /// (`includingOpenCall`), and dropped when the reply was cut there.
    mutating func finish(includingOpenCall: Bool) -> String {
        defer {
            held = ""
            insideCall = false
        }
        guard insideCall else { return held }
        return includingOpenCall ? Self.openTag + held : ""
    }

    /// Records the call a closed block holds; a body that is not one comes back as text.
    private mutating func closed(body: String) -> String {
        guard let call = Self.call(in: body) else { return Self.openTag + body + Self.closeTag }
        calls.append(call)
        return ""
    }

    /// `{"name": …, "arguments": …}`; arguments given as a JSON string holding an object are unwrapped.
    static func call(in body: String) -> LLMToolCall? {
        guard case .object(let fields)? = JSONValue.parse(body),
              case .string(let name)? = fields["name"], !name.isEmpty else { return nil }
        let id: String = if case .string(let given)? = fields["id"], !given.isEmpty { given } else { UUID().uuidString }
        return LLMToolCall(id: id, name: name, arguments: arguments(fields["arguments"]))
    }

    private static func arguments(_ value: JSONValue?) -> String {
        guard let value, value != .null else { return "{}" }
        if case .string(let text) = value, let parsed = JSONValue.parse(text), case .object = parsed {
            return parsed.serialized
        }
        return value.serialized
    }

    // MARK: Text helpers

    /// How many characters at the end of `text` could begin ``openTag``.
    private static func partialTagLength(at text: String) -> Int {
        let longest = min(text.count, openTag.count - 1)
        guard longest > 0 else { return 0 }
        return (1...longest).reversed().first { text.hasSuffix(openTag.prefix($0)) } ?? 0
    }

    private static func text(of text: String, upTo end: String.Index) -> String {
        String(text[..<end])
    }

    private static func text(of text: String, from start: String.Index) -> String {
        String(text[start...])
    }

    private static func text(of text: String, droppingLast count: Int) -> String {
        String(text.dropLast(count))
    }

    private static func text(of text: String, keepingLast count: Int) -> String {
        String(text.suffix(count))
    }
}
