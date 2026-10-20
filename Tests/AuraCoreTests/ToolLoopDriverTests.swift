import Synchronization
import XCTest
@testable import AuraCore

/// Names of the tool calls that actually ran, in order.
private final class RunLog: Sendable {
    private let names = Mutex<[String]>([])
    var ran: [String] { names.withLock { $0 } }
    func record(_ name: String) { names.withLock { $0.append(name) } }
}

/// A tool that records each run; `fails` makes it throw instead.
private struct RecordingTool: LLMTool {
    struct Failure: Error {}
    let name: String
    let description = "Records its calls."
    let log: RunLog
    var fails = false
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        log.record(name)
        if fails { throw Failure() }
        return ToolOutput(data: ["tool": name])
    }
}

/// Runs until its task is cancelled.
private struct HangingTool: LLMTool {
    let name = "hang"
    let description = "Waits for cancellation."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        while true { try await Task.sleep(for: .milliseconds(2)) }
    }
}

/// Hands out scripted replies in order, then final answers.
@MainActor
private final class Script {
    private var replies: [[LLMToolCall]?]
    private(set) var asked = 0
    private(set) var recorded: [[String]] = []
    private(set) var events: [ToolEvent] = []

    init(_ replies: [[LLMToolCall]?]) { self.replies = replies }

    func next() -> [LLMToolCall]? {
        asked += 1
        return replies.isEmpty ? [] : replies.removeFirst()
    }

    func record(_ calls: [LLMToolCall], _ results: [ToolCallResult]) {
        recorded.append(results.map(\.call.name))
    }

    func context(_ options: ToolLoopOptions) -> ToolLoopContext {
        ToolLoopContext(options: options) { [weak self] event in self?.events.append(event) }
    }
}

private func call(_ name: String) -> LLMToolCall {
    LLMToolCall(id: "id-\(name)", name: name, arguments: "{}")
}

/// The round loop with scripted replies: no model needed.
@MainActor
final class ToolLoopDriverTests: XCTestCase {

    private func tools(_ names: [String], log: RunLog, failing: Set<String> = []) -> [AnyLLMTool] {
        names.map { AnyLLMTool(RecordingTool(name: $0, log: log, fails: failing.contains($0))) }
    }

    private func run(_ script: Script, tools: [AnyLLMTool],
                     options: ToolLoopOptions = .default) async throws -> ToolTurnSummary {
        let driver = ToolLoopDriver(tools: tools, context: script.context(options))
        return try await driver.run(reply: { script.next() }, record: { script.record($0, $1) })
    }

    func testAFinalAnswerRunsNoRound() async throws {
        let log = RunLog()
        let script = Script([[]])
        let summary = try await run(script, tools: tools(["a"], log: log))
        XCTAssertEqual(summary, ToolTurnSummary(rounds: 0, results: [], skipped: []))
        XCTAssertEqual(script.asked, 1)
        XCTAssertTrue(script.events.isEmpty)
        XCTAssertEqual(log.ran, [])
    }

    func testOneRoundThenAnswer() async throws {
        let log = RunLog()
        let script = Script([[call("a")], []])
        let summary = try await run(script, tools: tools(["a"], log: log))

        let invocation = ToolInvocation(id: "id-a", round: 1, name: "a", arguments: "{}")
        let result = ToolCallResult(call: invocation, output: #"{"tool":"a"}"#, isError: false)
        XCTAssertEqual(script.events, [.started(invocation), .finished(result)])
        XCTAssertEqual(script.recorded, [["a"]])
        XCTAssertEqual(summary, ToolTurnSummary(rounds: 1, results: [result], skipped: []))
        XCTAssertEqual(script.asked, 2)
    }

    func testDefaultLimitSkipsTheSecondRoundAndSaysSo() async throws {
        let log = RunLog()
        let script = Script([[call("a")], [call("b")], []])
        let summary = try await run(script, tools: tools(["a", "b"], log: log))

        let skipped = ToolInvocation(id: "id-b", round: 2, name: "b", arguments: "{}")
        XCTAssertEqual(log.ran, ["a"], "b never runs")
        XCTAssertEqual(script.events.last, .roundLimitReached(skipped: [skipped]))
        XCTAssertEqual(summary.rounds, 1)
        XCTAssertEqual(summary.skipped, [skipped])
        XCTAssertEqual(script.recorded, [["a"]])
        XCTAssertEqual(script.asked, 2, "no reply after the limit")
    }

    func testChainRunsInOrderUpToTheLimit() async throws {
        let log = RunLog()
        let script = Script([[call("a")], [call("b")], [call("c")], []])
        let summary = try await run(script, tools: tools(["a", "b", "c"], log: log),
                                    options: ToolLoopOptions(maxToolRounds: 3))
        XCTAssertEqual(log.ran, ["a", "b", "c"])
        XCTAssertEqual(summary.results.map(\.call.round), [1, 2, 3])
        XCTAssertEqual(summary.rounds, 3)
        XCTAssertEqual(summary.skipped, [])
        XCTAssertEqual(script.recorded, [["a"], ["b"], ["c"]])
    }

    func testZeroRoundsRunsNothing() async throws {
        let log = RunLog()
        let script = Script([[call("a"), call("b")]])
        let summary = try await run(script, tools: tools(["a", "b"], log: log),
                                    options: ToolLoopOptions(maxToolRounds: 0))
        XCTAssertEqual(log.ran, [])
        XCTAssertEqual(summary.rounds, 0)
        XCTAssertEqual(summary.skipped.map(\.name), ["a", "b"])
        XCTAssertEqual(summary.skipped.map(\.round), [1, 1])
        XCTAssertEqual(script.events.count, 1)
        XCTAssertEqual(script.recorded, [])
    }

    func testNegativeRoundsCountAsZero() async throws {
        let log = RunLog()
        let script = Script([[call("a")]])
        let summary = try await run(script, tools: tools(["a"], log: log),
                                    options: ToolLoopOptions(maxToolRounds: -3))
        XCTAssertEqual(log.ran, [])
        XCTAssertEqual(summary, ToolTurnSummary(rounds: 0, results: [],
                                                skipped: [ToolInvocation(id: "id-a", round: 1, name: "a",
                                                                         arguments: "{}")]))
    }

    func testCallsInOneReplyRunInOrderInOneRound() async throws {
        let log = RunLog()
        let script = Script([[call("b"), call("a"), call("c")], []])
        let summary = try await run(script, tools: tools(["a", "b", "c"], log: log))
        XCTAssertEqual(log.ran, ["b", "a", "c"])
        XCTAssertEqual(summary.rounds, 1)
        XCTAssertEqual(summary.results.map(\.call.round), [1, 1, 1])
        XCTAssertEqual(script.recorded, [["b", "a", "c"]])
    }

    func testACutReplyEndsTheLoopWithoutSkips() async throws {
        let log = RunLog()
        let script = Script([[call("a")], nil, [call("b")]])
        let summary = try await run(script, tools: tools(["a", "b"], log: log),
                                    options: ToolLoopOptions(maxToolRounds: 5))
        XCTAssertEqual(log.ran, ["a"])
        XCTAssertEqual(summary.rounds, 1)
        XCTAssertEqual(summary.skipped, [])
        XCTAssertEqual(script.asked, 2)
    }

    func testAnErrorIsReportedToTheModelByDefault() async throws {
        let log = RunLog()
        let script = Script([[call("a"), call("b")], []])
        let summary = try await run(script, tools: tools(["a", "b"], log: log, failing: ["a"]))
        XCTAssertEqual(log.ran, ["a", "b"], "the turn goes on after the failure")
        XCTAssertEqual(summary.results.map(\.isError), [true, false])
        XCTAssertTrue(summary.results[0].output.hasPrefix(#"{"error":"#), summary.results[0].output)
        XCTAssertEqual(script.recorded, [["a", "b"]])
    }

    func testFailTurnEmitsFinishedThenThrows() async throws {
        let log = RunLog()
        let script = Script([[call("a"), call("b")], []])
        do {
            _ = try await run(script, tools: tools(["a", "b"], log: log, failing: ["a"]),
                              options: ToolLoopOptions(toolErrors: .failTurn))
            XCTFail("the tool's error was not thrown")
        } catch {
            XCTAssertTrue(error is RecordingTool.Failure, "\(error)")
        }
        XCTAssertEqual(log.ran, ["a"])
        XCTAssertEqual(script.events.count, 2)
        guard case .finished(let result)? = script.events.last else { return XCTFail("\(script.events)") }
        XCTAssertTrue(result.isError)
        XCTAssertEqual(script.recorded, [], "record is not called")
    }

    func testFailTurnRethrowsAnUnknownToolAsInvalidParameter() async throws {
        let script = Script([[call("missing")], []])
        do {
            _ = try await run(script, tools: tools(["a"], log: RunLog()),
                              options: ToolLoopOptions(toolErrors: .failTurn))
            XCTFail("an unknown tool did not fail the turn")
        } catch LLMError.invalidParameter(let reason) {
            XCTAssertEqual(reason, "Tool 'missing' not found")
        }
    }

    func testCancellationDuringAToolStopsTheLoop() async throws {
        let script = Script([[call("hang")], [call("hang")], []])
        let tools = [AnyLLMTool(HangingTool())]
        let options = ToolLoopOptions(maxToolRounds: 3)
        let turn = Task { @MainActor in
            let driver = ToolLoopDriver(tools: tools, context: script.context(options))
            return try await driver.run(reply: { script.next() }, record: { script.record($0, $1) })
        }
        try await waitUntil(timeout: .seconds(5)) { script.events.count == 1 }
        turn.cancel()
        do {
            _ = try await turn.value
            XCTFail("a cancelled turn returned")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        XCTAssertEqual(script.asked, 1, "no further reply")
        XCTAssertEqual(script.recorded, [])
        guard case .started? = script.events.last else { return XCTFail("\(script.events)") }
    }
}
