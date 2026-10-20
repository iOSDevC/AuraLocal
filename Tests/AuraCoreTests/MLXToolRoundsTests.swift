#if canImport(MLXLLM)
import Jinja
import MLXLMCommon
import XCTest
@testable import AuraCore

/// How an MLX tool round is converted and written into the conversation. No model.
final class MLXToolRoundsTests: XCTestCase {

    /// Qwen-style rendering of `tool_calls` and `tool` turns, enough to see what `tojson` writes.
    private static let template = """
        {%- for message in messages %}{%- if message.role == "assistant" and message.tool_calls %}\
        {{- '<|im_start|>assistant\\n' + message.content }}{%- for entry in message.tool_calls %}\
        {%- set entry = entry.function %}{{- '<tool_call>\\n{"name": "' + entry.name + '", "arguments": ' \
        + (entry.arguments | tojson) + '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}\
        {%- elif message.role == "tool" %}{{- '<|im_start|>user\\n<tool_response>\\n' + message.content \
        + '\\n</tool_response><|im_end|>\\n' }}{%- else %}{{- '<|im_start|>' + message.role + '\\n' \
        + message.content + '<|im_end|>\\n' }}{%- endif %}{%- endfor %}
        """

    private static let arguments: [String: MLXLMCommon.JSONValue] = [
        "y": .int(9), "x": .int(5), "ok": .bool(true), "ratio": .double(0.5), "note": .null,
        "tags": .array([.string("a"), .bool(false)]), "nested": .object(["b": .int(2), "a": .string("z/y")]),
    ]

    private func round(text: String = "") -> MLXToolRound {
        let converted = MLXToolMessages.calls(from: [ToolCall(function: .init(name: "tap", arguments: Self.arguments))])
        let call = converted.calls[0]
        let invocation = ToolInvocation(id: call.id, round: 1, name: call.name, arguments: call.arguments)
        let result = ToolCallResult(call: invocation, output: #"{"ok":true}"#, isError: false)
        return MLXToolRound(text: text, calls: converted.calls, arguments: converted.arguments, results: [result])
    }

    // MARK: Parsed calls

    func testParsedArgumentsBecomeSortedCompactJSON() {
        let converted = MLXToolMessages.calls(from: [ToolCall(function: .init(name: "tap", arguments: Self.arguments))])
        XCTAssertEqual(converted.calls.map(\.name), ["tap"])
        XCTAssertEqual(converted.calls[0].arguments, #"""
            {"nested":{"a":"z/y","b":2},"note":null,"ok":true,"ratio":0.5,"tags":["a",false],"x":5,"y":9}
            """#)
        XCTAssertEqual(converted.arguments[converted.calls[0].id], Self.arguments)
    }

    func testEveryCallGetsItsOwnIdInModelOrder() {
        let parsed = [ToolCall(function: .init(name: "a", arguments: [String: MLXLMCommon.JSONValue]())),
                      ToolCall(function: .init(name: "b", arguments: [String: MLXLMCommon.JSONValue]())),
                      ToolCall(function: .init(name: "a", arguments: [String: MLXLMCommon.JSONValue]()))]
        let converted = MLXToolMessages.calls(from: parsed)
        XCTAssertEqual(converted.calls.map(\.name), ["a", "b", "a"], "no per-name dedupe")
        XCTAssertEqual(Set(converted.calls.map(\.id)).count, 3)
        XCTAssertEqual(converted.calls.map(\.arguments), ["{}", "{}", "{}"])
    }

    // MARK: Round messages

    func testTheNativeShapeIsACallTurnThenAToolTurnPerResult() throws {
        let round = round(text: "Let me tap.")
        let messages = MLXToolMessages.native(round)
        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0]["role"] as? String, "assistant")
        XCTAssertEqual(messages[0]["content"] as? String, "Let me tap.")
        let calls = try XCTUnwrap(messages[0]["tool_calls"] as? [[String: any Sendable]])
        XCTAssertEqual(calls.first?["type"] as? String, "function")
        XCTAssertEqual(calls.first?["id"] as? String, round.calls[0].id)
        let function = try XCTUnwrap(calls.first?["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "tap")
        let arguments = try XCTUnwrap(function["arguments"] as? [String: any Sendable])
        XCTAssertTrue(arguments["ok"] is Bool, "a boolean stays a boolean")
        XCTAssertEqual(arguments["x"] as? Int, 5)

        XCTAssertEqual(messages[1]["role"] as? String, "tool")
        XCTAssertEqual(messages[1]["content"] as? String, #"{"ok":true}"#)
        XCTAssertEqual(messages[1]["tool_call_id"] as? String, round.calls[0].id)
        XCTAssertEqual(messages[1]["name"] as? String, "tap")
    }

    func testTheTemplateWritesTheArgumentsBackAsJSON() throws {
        let messages: [[String: any Sendable]] = [["role": "user", "content": "tap"]] + MLXToolMessages.native(round())
        let rendered = try Template(Self.template).render(["messages": .array(try messages.map { try Value(any: $0) })])
        let arguments = #"{"nested":{"a":"z\/y","b":2},"note":null,"ok":true,"ratio":0.5,"tags":["a",false],"#
            + #""x":5,"y":9}"#
        XCTAssertEqual(rendered, """
            <|im_start|>user
            tap<|im_end|>
            <|im_start|>assistant
            <tool_call>
            {"name": "tap", "arguments": \(arguments)}
            </tool_call><|im_end|>
            <|im_start|>user
            <tool_response>
            {"ok":true}
            </tool_response><|im_end|>

            """)
    }

    func testABlankReplyIsSentAsEmptyContent() {
        XCTAssertEqual(MLXToolMessages.native(round(text: "\n\n"))[0]["content"] as? String, "")
    }

    func testTheTextFormIsOneAssistantTurnForEveryRound() {
        let first = round(text: "Tapping.")
        let second = round()
        let message = MLXToolMessages.text([first, second])
        XCTAssertEqual(message["role"] as? String, "assistant")
        let call = "<tool_call>\n{\"name\": \"tap\", \"arguments\": \(first.calls[0].arguments)}\n</tool_call>"
        let response = "<tool_response>\n{\"ok\":true}\n</tool_response>"
        let secondCall = "<tool_call>\n{\"name\": \"tap\", \"arguments\": \(second.calls[0].arguments)}\n</tool_call>"
        XCTAssertEqual(message["content"] as? String,
                       ["Tapping.", call, response, secondCall, response].joined(separator: "\n"))
    }

    // MARK: Transcript

    func testATurnWithoutRoundsRendersItsMessagesOnce() {
        let transcript = MLXToolTranscript(base: [["role": "user", "content": "hi"]])
        XCTAssertEqual(transcript.attempts.map(\.shape), [.native])
        XCTAssertEqual(transcript.attempts.first?.messages.count, 1)
    }

    func testRoundsTryTheNativeShapeFirstAndKeepTheTextFormOnceItWasNeeded() {
        var transcript = MLXToolTranscript(base: [["role": "user", "content": "hi"]])
        transcript.settle(on: .text)
        XCTAssertEqual(transcript.shape, .native, "nothing to settle before a round")
        transcript.record(round())
        XCTAssertEqual(transcript.attempts.map(\.shape), [.native, .text])
        XCTAssertEqual(transcript.attempts.map(\.messages.count), [3, 2])

        transcript.settle(on: .native)
        XCTAssertEqual(transcript.attempts.map(\.shape), [.native, .text])
        transcript.settle(on: .text)
        transcript.record(round())
        XCTAssertEqual(transcript.attempts.map(\.shape), [.text])
        XCTAssertEqual(transcript.attempts.first?.messages.count, 2, "every round in the one assistant turn")
    }
}
#endif
