import XCTest
@testable import TonamoxCore

// swiftlint:disable identifier_name
private struct TapTool: LLMTool {
    let name = "tap"
    let description = "Tap the screen at a pixel coordinate."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let x: Int
        let y: Int
        static var argumentsSchema: LLMToolArgumentsSchema {
            ["x": .integer(description: "X pixel"), "y": .integer(description: "Y pixel")]
        }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        ToolOutput(data: ["ok": true, "x": arguments.x, "y": arguments.y])
    }
}
// swiftlint:enable identifier_name

private struct FailingTool: LLMTool {
    struct Failure: LocalizedError {
        var errorDescription: String? { "the disk is full" }
    }
    let name = "fail"
    let description = "Always fails."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput { throw Failure() }
}

/// Returns whatever output it was built with.
private struct FixedTool: LLMTool {
    let name = "fixed"
    let description = "Returns a fixed output."
    let output: ToolOutput
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput { output }
}

private struct CancelledTool: LLMTool {
    let name = "cancelled"
    let description = "Throws CancellationError."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput { throw CancellationError() }
}

private struct Opaque: Sendable, CustomStringConvertible {
    var description: String { "opaque" }
}

/// What the model receives from one tool call, and which error `.failTurn` rethrows. No model needed.
final class ToolRunnerTests: XCTestCase {

    private func run(_ name: String, _ arguments: String = "{}", among tools: [AnyLLMTool]) async throws
        -> ToolRunner.Outcome {
        try await ToolRunner.run(ToolInvocation(id: "c1", round: 1, name: name, arguments: arguments), among: tools)
    }

    func testSuccessfulOutputIsSortedCompactJSON() async throws {
        let outcome = try await run("tap", #"{"x": 5, "y": 9}"#, among: [AnyLLMTool(TapTool())])
        XCTAssertEqual(outcome.result.output, #"{"ok":true,"x":5,"y":9}"#)
        XCTAssertFalse(outcome.result.isError)
        XCTAssertNil(outcome.failure)
        XCTAssertEqual(outcome.result.call, ToolInvocation(id: "c1", round: 1, name: "tap", arguments: #"{"x": 5, "y": 9}"#))
    }

    func testNestedOutputIsDeterministic() async throws {
        let nested = ToolOutput(data: [
            "zeta": ["b": 2, "a": [1, 2, 3]] as [String: any Sendable],
            "alpha": "a/b",
            "mid": [["y": true, "x": false]] as [[String: Bool]],
        ])
        let tool = AnyLLMTool(FixedTool(output: nested))
        let first = try await run("fixed", among: [tool]).result.output
        for _ in 0..<5 {
            let again = try await run("fixed", among: [tool]).result.output
            XCTAssertEqual(again, first)
        }
        XCTAssertEqual(first, #"{"alpha":"a/b","mid":[{"x":false,"y":true}],"zeta":{"a":[1,2,3],"b":2}}"#)
    }

    func testNonJSONOutputFallsBackToSortedPairs() async throws {
        let date = Date(timeIntervalSince1970: 0)
        let tool = AnyLLMTool(FixedTool(output: ToolOutput(data: ["when": date, "what": Opaque(), "count": 3])))
        let outcome = try await run("fixed", among: [tool])
        XCTAssertEqual(outcome.result.output, "count: 3, what: opaque, when: \(date)")
        XCTAssertFalse(outcome.result.isError)
    }

    func testNaNOutputFallsBack() async throws {
        let tool = AnyLLMTool(FixedTool(output: ToolOutput(data: ["value": Double.nan, "name": "x"])))
        let outcome = try await run("fixed", among: [tool])
        XCTAssertEqual(outcome.result.output, "name: x, value: nan")
    }

    func testThrowingToolIsReportedAsAnError() async throws {
        let outcome = try await run("fail", among: [AnyLLMTool(FailingTool())])
        XCTAssertTrue(outcome.result.isError)
        XCTAssertEqual(outcome.result.output, #"{"error":"Tool \"fail\" failed: the disk is full"}"#)
        XCTAssertTrue(outcome.result.output.contains("failed"))
        XCTAssertTrue(outcome.failure is FailingTool.Failure, "\(String(describing: outcome.failure))")
    }

    func testUndecodableArgumentsNameTheArgument() async throws {
        let outcome = try await run("tap", #"{"x":"left","y":1}"#, among: [AnyLLMTool(TapTool())])
        XCTAssertTrue(outcome.result.isError)
        XCTAssertTrue(outcome.result.output.contains("x: expected Int"), outcome.result.output)
        guard case .argumentDecodingFailed(let toolName, _)? = outcome.failure as? ToolError else {
            return XCTFail("\(String(describing: outcome.failure))")
        }
        XCTAssertEqual(toolName, "tap")
    }

    func testAMissingArgumentIsNamed() async throws {
        let outcome = try await run("tap", #"{"x":1}"#, among: [AnyLLMTool(TapTool())])
        XCTAssertTrue(outcome.result.output.contains(#"missing argument \"y\""#), outcome.result.output)
    }

    func testUnknownToolListsTheAvailableOnes() async throws {
        let outcome = try await run("swipe", among: [AnyLLMTool(TapTool()), AnyLLMTool(FailingTool())])
        XCTAssertTrue(outcome.result.isError)
        XCTAssertEqual(outcome.result.output, #"{"error":"Unknown tool \"swipe\". Available tools: tap, fail."}"#)
        guard case .invalidParameter(let reason)? = outcome.failure as? LLMError else {
            return XCTFail("\(String(describing: outcome.failure))")
        }
        XCTAssertEqual(reason, "Tool 'swipe' not found")
    }

    func testCancellationAlwaysPropagates() async {
        do {
            _ = try await run("cancelled", among: [AnyLLMTool(CancelledTool())])
            XCTFail("a cancelled tool became a result")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
    }

    func testErrorOutputIsAnErrorObject() {
        XCTAssertEqual(ToolRunner.errorOutput("a/b \"c\""), #"{"error":"a/b \"c\""}"#)
    }
}
