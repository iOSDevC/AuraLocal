import XCTest
@testable import TonamoxCore

final class ToolCallTagParserTests: XCTestCase {

    private static func parse(_ pieces: [String], ended: Bool = true) -> (text: String, calls: [LLMToolCall]) {
        var parser = ToolCallTagParser()
        let text = pieces.map { parser.consume($0) }.joined() + parser.finish(includingOpenCall: ended)
        return (text, parser.calls)
    }

    func testTextOutsideCallsPassesAndCallsAreCollected() {
        let parsed = Self.parse(["Sure. ", "<tool_call>", "\n{\"name\": \"echo\", \"arguments\": {\"text\": \"hi\"}}\n",
                                 "</tool_call>", " Done."])
        XCTAssertEqual(parsed.text, "Sure.  Done.")
        XCTAssertEqual(parsed.calls.map(\.name), ["echo"])
        XCTAssertEqual(parsed.calls.first?.arguments, #"{"text":"hi"}"#)
    }

    func testTagsSplitAcrossPiecesAreFound() {
        let pieces = ["A<", "tool", "_ca", "ll>{\"name\": \"e", "cho\", \"arguments\": {}}</to", "ol_call", ">B"]
        let parsed = Self.parse(pieces)
        XCTAssertEqual(parsed.text, "AB")
        XCTAssertEqual(parsed.calls.map(\.name), ["echo"])
        XCTAssertEqual(parsed.calls.first?.arguments, "{}")
    }

    func testEveryCallIsKeptTheSameToolTwiceIncluded() {
        let call = "<tool_call>\n{\"name\": \"echo\", \"arguments\": {\"text\": \"%@\"}}\n</tool_call>"
        let parsed = Self.parse([call.replacingOccurrences(of: "%@", with: "a") + "\n"
                                 + call.replacingOccurrences(of: "%@", with: "b")])
        XCTAssertEqual(parsed.calls.map(\.arguments), [#"{"text":"a"}"#, #"{"text":"b"}"#])
        XCTAssertEqual(parsed.text, "\n")
        XCTAssertNotEqual(parsed.calls[0].id, parsed.calls[1].id)
    }

    func testABlockThatIsNotACallStaysText() {
        for body in ["{\"name\": \"echo\", \"arguments\": {}", "get_secret", "{\"arguments\": {}}", "{\"name\": \"\"}"] {
            let block = "<tool_call>" + body + "</tool_call>"
            let parsed = Self.parse(["x", block, "y"])
            XCTAssertEqual(parsed.text, "x" + block + "y", body)
            XCTAssertEqual(parsed.calls, [], body)
        }
    }

    func testTextThatOnlyLooksLikeATagStartIsReleased() {
        var parser = ToolCallTagParser()
        XCTAssertEqual(parser.consume("1 <"), "1 ", "a `<` may start a tag")
        XCTAssertEqual(parser.consume(" 2 and <b>"), "< 2 and <b>")
        XCTAssertEqual(parser.consume("<tool"), "")
        XCTAssertEqual(parser.finish(includingOpenCall: false), "<tool", "held text outside a call is text")
    }

    func testACallLeftOpenIsTextOnlyWhenTheReplyEndedInsideIt() {
        let open = ["Hm ", "<tool_call>", "{\"name\": \"echo\""]
        XCTAssertEqual(Self.parse(open, ended: true).text, "Hm <tool_call>{\"name\": \"echo\"")
        XCTAssertEqual(Self.parse(open, ended: false).text, "Hm ", "a cut reply drops the unfinished call")
        XCTAssertEqual(Self.parse(open).calls, [])
    }

    @MainActor
    func testARoundIsRecordedAsTheModelWroteIt() {
        let raw = "<tool_call>\n{\"name\": \"echo\", \"arguments\": {\"text\": \"hi\"}}\n</tool_call>"
        let call = ToolInvocation(id: "c1", round: 1, name: "echo", arguments: #"{"text":"hi"}"#)
        let results = [ToolCallResult(call: call, output: #"{"text":"hi"}"#, isError: false),
                       ToolCallResult(call: call, output: #"{"error":"x"}"#, isError: true)]
        XCTAssertEqual(LlamaTurnEngine.roundMessages(reply: raw, results: results),
                       [.assistant(raw), .tool(#"{"text":"hi"}"#), .tool(#"{"error":"x"}"#)],
                       "the reply keeps its exact text, so the next prompt extends what the KV cache holds")
    }

    func testArgumentsComeOutSortedAndStringArgumentsAreUnwrapped() throws {
        let sorted = try XCTUnwrap(ToolCallTagParser.call(in: #"{"name": "t", "arguments": {"b": 1, "a": [true, null]}}"#))
        XCTAssertEqual(sorted.arguments, #"{"a":[true,null],"b":1}"#)
        let wrapped = try XCTUnwrap(ToolCallTagParser.call(in: #"{"name": "t", "arguments": "{\"x\": 2}"}"#))
        XCTAssertEqual(wrapped.arguments, #"{"x":2}"#)
        let odd = try XCTUnwrap(ToolCallTagParser.call(in: #"{"name": "t", "arguments": "left"}"#))
        XCTAssertEqual(odd.arguments, #""left""#, "passed on, so the tool's decoding reports it to the model")
        XCTAssertEqual(ToolCallTagParser.call(in: #"{"name": "t", "arguments": null}"#)?.arguments, "{}")
        XCTAssertEqual(ToolCallTagParser.call(in: #" {"name": "t", "id": "call_7"} "#)?.id, "call_7")
    }
}
