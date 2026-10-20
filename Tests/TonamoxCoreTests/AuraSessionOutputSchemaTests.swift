import XCTest
@testable import TonamoxCore

/// Replays scripted replies (the last one repeats), each as two deltas so buffering is exercised.
@MainActor private final class ScriptedEngine: TonamoxProfileEngine {
    private let replies: [String]
    /// 1-based call that blocks until cancelled instead of replying.
    var hangOnCall: Int?
    /// Like llama.cpp stopping early: the blocked call returns its partial reply instead of throwing.
    var returnsOnCancel = false
    private(set) var prompts: [String] = []
    private(set) var systemPrompts: [String?] = []
    private(set) var didTeardown = false

    init(replies: [String]) { self.replies = replies }

    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onDelta: @escaping @MainActor (String) -> Void) async throws {
        prompts.append(prompt)
        systemPrompts.append(systemPrompt)
        if prompts.count == hangOnCall {
            while !Task.isCancelled { try? await Task.sleep(for: .milliseconds(2)) }
            if returnsOnCancel { return onDelta("{") }
            throw CancellationError()
        }
        let reply = replies[min(prompts.count, replies.count) - 1]
        let middle = reply.index(reply.startIndex, offsetBy: reply.count / 2)
        onDelta(String(reply[..<middle]))
        onDelta(String(reply[middle...]))
    }

    func teardown() { didTeardown = true }
}

/// Records each call's constraint (nil when unconstrained) and replays scripted replies; the last one repeats.
@MainActor private final class ConstrainedScriptedEngine: TonamoxProfileEngine, ConstrainedProfileEngine {
    private let replies: [String]
    /// Thrown by constrained calls instead of replying.
    var constrainedError: (any Error)?
    /// 1-based call that blocks until cancelled, then throws `CancellationError`.
    var hangOnCall: Int?
    private(set) var constraints: [GenerationConstraint?] = []
    private(set) var systemPrompts: [String?] = []
    private var answered = 0

    init(replies: [String]) { self.replies = replies }

    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onDelta: @escaping @MainActor (String) -> Void) async throws {
        try await reply(constraint: nil, systemPrompt: systemPrompt, onDelta: onDelta)
    }

    func generate(prompt: String, systemPrompt: String?, maxTokens: Int, constraint: GenerationConstraint,
                  onDelta: @escaping @MainActor (String) -> Void) async throws {
        try await reply(constraint: constraint, systemPrompt: systemPrompt, onDelta: onDelta)
    }

    private func reply(constraint: GenerationConstraint?, systemPrompt: String?,
                       onDelta: @MainActor (String) -> Void) async throws {
        constraints.append(constraint)
        systemPrompts.append(systemPrompt)
        if constraints.count == hangOnCall {
            while !Task.isCancelled { try? await Task.sleep(for: .milliseconds(2)) }
            throw CancellationError()
        }
        if constraint != nil, let constrainedError { throw constrainedError }
        answered += 1
        onDelta(replies[min(answered, replies.count) - 1])
    }

    func teardown() {}
}

private struct EchoTool: LLMTool {
    let name = "echo"
    let description = "echo"
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput { ToolOutput(data: ["ok": true]) }
}

private func collect(_ stream: AsyncThrowingStream<String, Error>) async -> (elements: [String], failure: (any Error)?) {
    var elements: [String] = []
    do {
        for try await element in stream { elements.append(element) }
        return (elements, nil)
    } catch {
        return (elements, error)
    }
}

final class TonamoxSessionOutputSchemaTests: XCTestCase {

    private static let personSchema = #"""
        {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0}},
         "required":["name","age"],"additionalProperties":false}
        """#

    private func ggufModel() -> Model {
        Model.fromURL("https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf")!
    }

    private func extractor(schema: String = personSchema) -> TonamoxProfile {
        TonamoxProfile(id: "extract", displayName: "Extract", instructions: "You extract people.",
                    model: ggufModel(), sampling: .precise, outputSchema: .json(schema))
    }

    @MainActor
    func testConformingFirstReplyYieldsOneElement() async throws {
        let engine = ScriptedEngine(replies: ["Sure:\n```json\n{\"name\":\"Ana\",\"age\":30}\n```"])
        let session = try await TonamoxSession(profile: extractor()) { _ in engine }

        let result = await collect(session.stream("Ana is 30."))

        XCTAssertNil(result.failure)
        XCTAssertEqual(result.elements, [#"{"name":"Ana","age":30}"#])
        XCTAssertEqual(engine.prompts, ["Ana is 30."])
        let system = try XCTUnwrap(engine.systemPrompts.first ?? nil)
        XCTAssertTrue(system.hasPrefix("You extract people.\n\n"))
        XCTAssertTrue(system.contains("Reply with a single JSON value that conforms to this JSON Schema, no prose:"))
        XCTAssertTrue(system.hasSuffix(Self.personSchema))
    }

    @MainActor
    func testInvalidReplyIsRepairedWithItsViolations() async throws {
        let engine = ScriptedEngine(replies: [#"{"name": 7, "age": 30}"#, #"{"name":"Ana","age":30}"#])
        let session = try await TonamoxSession(profile: extractor()) { _ in engine }

        let result = await collect(session.stream("Ana is 30."))

        XCTAssertNil(result.failure)
        XCTAssertEqual(result.elements, [#"{"name":"Ana","age":30}"#])
        XCTAssertEqual(engine.prompts.count, 2)
        let repair = engine.prompts[1]
        XCTAssertTrue(repair.hasPrefix("Ana is 30."))
        XCTAssertTrue(repair.contains(#"{"name": 7, "age": 30}"#))
        XCTAssertTrue(repair.contains("- /name: expected string, got integer"))
        XCTAssertEqual(engine.systemPrompts[1], engine.systemPrompts[0])
    }

    @MainActor
    func testPersistentViolationsThrowAfterEveryRepair() async throws {
        let engine = ScriptedEngine(replies: ["I don't know.", #"{"name":"Ana","age":-1}"#])
        let session = try await TonamoxSession(profile: extractor()) { _ in engine }
        XCTAssertEqual(session.maxRepairAttempts, 2)

        let result = await collect(session.stream("Who?"))

        XCTAssertEqual(result.elements, [])
        XCTAssertEqual(engine.prompts.count, 3)                 // 1 + maxRepairAttempts
        XCTAssertEqual(result.failure as? OutputSchemaError, .violations(
            output: #"{"name":"Ana","age":-1}"#,
            violations: [OutputSchemaViolation(path: "/age", message: "must be >= 0, got -1")]))
        XCTAssertTrue(engine.prompts[1].contains("- (root): no JSON value found in the reply"))
    }

    @MainActor
    func testZeroOrNegativeRepairAttemptsGenerateOnce() async throws {
        let engine = ScriptedEngine(replies: ["nope"])
        let session = try await TonamoxSession(profile: extractor()) { _ in engine }

        session.maxRepairAttempts = 0
        let none = await collect(session.stream("Who?"))
        session.maxRepairAttempts = -3
        let negative = await collect(session.stream("Who?"))

        XCTAssertTrue(none.failure is OutputSchemaError)
        XCTAssertTrue(negative.failure is OutputSchemaError)
        XCTAssertEqual(engine.prompts.count, 2)
    }

    @MainActor
    func testInvalidSchemaFailsInitBeforeBuildingAnEngine() async {
        var builds = 0
        let make: @MainActor (TonamoxProfile) async throws -> any TonamoxProfileEngine = { _ in
            builds += 1
            return ScriptedEngine(replies: ["{}"])
        }

        do {
            _ = try await TonamoxSession(profile: extractor(schema: #"root ::= "yes" | "no""#), makeEngine: make)
            XCTFail("a GBNF schema must be rejected")
        } catch {
            guard case .invalidSchema = error as? OutputSchemaError else { return XCTFail("unexpected \(error)") }
        }
        do {
            _ = try await TonamoxSession(profile: extractor(schema: #"{"type":"string","format":"email"}"#), makeEngine: make)
            XCTFail("an unsupported keyword must be rejected")
        } catch {
            XCTAssertEqual(error as? OutputSchemaError, .unsupportedKeywords(["/format"]))
        }
        XCTAssertEqual(builds, 0)
    }

    @MainActor
    func testSwitchToInvalidSchemaKeepsTheCurrentEngine() async throws {
        var builds = 0
        let engine = ScriptedEngine(replies: ["Hello world"])
        let chat = TonamoxProfileCatalog.chat(model: ggufModel())
        let session = try await TonamoxSession(profile: chat) { _ in builds += 1; return engine }

        do {
            try await session.switchProfile(to: extractor(schema: ##"{"$ref":"#/x"}"##))
            XCTFail("an unsupported keyword must be rejected")
        } catch {
            XCTAssertEqual(error as? OutputSchemaError, .unsupportedKeywords(["/$ref"]))
        }

        XCTAssertEqual(builds, 1)
        XCTAssertFalse(engine.didTeardown)
        XCTAssertEqual(session.profile, chat)
        let result = await collect(session.stream("hi"))
        XCTAssertEqual(result.elements, ["Hello", " world"])
    }

    @MainActor
    func testSwitchAppliesTheNewProfilesSchema() async throws {
        let chatEngine = ScriptedEngine(replies: ["Hello world"])
        let schemaEngine = ScriptedEngine(replies: [#"{"name":"Ana","age":30}"#])
        var engines: [ScriptedEngine] = [chatEngine, schemaEngine]
        let session = try await TonamoxSession(profile: TonamoxProfileCatalog.chat(model: ggufModel())) { _ in
            engines.removeFirst()
        }

        try await session.switchProfile(to: extractor())
        let result = await collect(session.stream("Ana is 30."))

        XCTAssertEqual(result.elements, [#"{"name":"Ana","age":30}"#])
        XCTAssertTrue(chatEngine.didTeardown)
    }

    @MainActor
    func testWithoutSchemaDeltasStreamUnchanged() async throws {
        let engine = ScriptedEngine(replies: ["Hello world"])
        let chat = TonamoxProfileCatalog.chat(model: ggufModel(), instructions: "You are CHAT.")
        let session = try await TonamoxSession(profile: chat) { _ in engine }

        let result = await collect(session.stream("hi"))

        XCTAssertNil(result.failure)
        XCTAssertEqual(result.elements, ["Hello", " world"])
        XCTAssertEqual(engine.systemPrompts, ["You are CHAT."])
    }

    @MainActor
    func testCancellationDuringRepairStopsFurtherGenerations() async throws {
        try await assertCancellationDuringRepairStops(returnsOnCancel: false)
    }

    @MainActor
    func testCancellationDuringRepairStopsEvenWhenTheEngineReturnsEarly() async throws {
        try await assertCancellationDuringRepairStops(returnsOnCancel: true)
    }

    @MainActor
    private func assertCancellationDuringRepairStops(returnsOnCancel: Bool) async throws {
        let engine = ScriptedEngine(replies: ["not json"])
        engine.hangOnCall = 2
        engine.returnsOnCancel = returnsOnCancel
        let session = try await TonamoxSession(profile: extractor()) { _ in engine }
        session.maxRepairAttempts = 5

        let stream = session.stream("Who?")
        let clock = ContinuousClock()
        let deadline = clock.now + .seconds(5)
        while engine.prompts.count < 2, clock.now < deadline {
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertEqual(engine.prompts.count, 2)

        session.cancel()
        let result = await collect(stream)

        XCTAssertTrue(result.failure is CancellationError, "got \(String(describing: result.failure))")
        XCTAssertEqual(result.elements, [])
        XCTAssertEqual(engine.prompts.count, 2)
    }
}

// MARK: - Schema grammar

extension TonamoxSessionOutputSchemaTests {

    private static let ana = #"{"name":"Ana","age":30}"#

    private func personGrammar(_ schema: String = personSchema) throws -> GenerationConstraint {
        .grammar(gbnf: try XCTUnwrap(OutputSchemaValidator(.json(schema)).grammar), root: "root")
    }

    @MainActor
    private func waitForCalls(_ count: Int, on engine: ConstrainedScriptedEngine) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now + .seconds(5)
        while engine.constraints.count < count, clock.now < deadline {
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertEqual(engine.constraints.count, count)
    }

    @MainActor
    func testGrammarConstrainsEveryAttemptIncludingRepairs() async throws {
        let engine = ConstrainedScriptedEngine(replies: [#"{"name":7,"age":30}"#, Self.ana])
        let session = try await TonamoxSession(profile: extractor()) { _ in engine }
        XCTAssertTrue(session.usesSchemaGrammar)

        let result = await collect(session.stream("Ana is 30."))

        XCTAssertNil(result.failure)
        XCTAssertEqual(result.elements, [Self.ana])
        let grammar = try personGrammar()
        XCTAssertEqual(engine.constraints, [grammar, grammar])
        let system = try XCTUnwrap(engine.systemPrompts.first ?? nil)
        XCTAssertTrue(system.contains("Reply with a single JSON value that conforms to this JSON Schema, no prose:"))
        XCTAssertEqual(engine.systemPrompts[1], engine.systemPrompts[0])
    }

    @MainActor
    func testUsesSchemaGrammarFalseValidatesOnly() async throws {
        let engine = ConstrainedScriptedEngine(replies: ["nope", Self.ana])
        let session = try await TonamoxSession(profile: extractor()) { _ in engine }
        session.usesSchemaGrammar = false

        let result = await collect(session.stream("Ana is 30."))

        XCTAssertEqual(result.elements, [Self.ana])
        XCTAssertEqual(engine.constraints, [nil, nil])
    }

    @MainActor
    func testProfilesWithToolsAreNotConstrained() async throws {
        let engine = ConstrainedScriptedEngine(replies: [Self.ana])
        let profile = TonamoxProfile(id: "agent", displayName: "Agent", instructions: "", model: ggufModel(),
                                  tools: [EchoTool()], outputSchema: .json(Self.personSchema))
        let session = try await TonamoxSession(profile: profile) { _ in engine }

        let result = await collect(session.stream("Ana is 30."))

        XCTAssertEqual(result.elements, [Self.ana])
        XCTAssertEqual(engine.constraints, [nil])
    }

    @MainActor
    func testUnsupportedBackendFallsBackInTheSameAttempt() async throws {
        let engine = ConstrainedScriptedEngine(replies: [Self.ana])
        engine.constrainedError = GenerationConstraintError.unsupportedBackend(.mlx)
        let session = try await TonamoxSession(profile: extractor()) { _ in engine }
        session.maxRepairAttempts = 0

        let first = await collect(session.stream("Ana is 30."))
        let second = await collect(session.stream("Ana is 30."))

        XCTAssertEqual(first.elements, [Self.ana])
        XCTAssertEqual(second.elements, [Self.ana])
        XCTAssertEqual(engine.constraints, [try personGrammar(), nil, nil])
    }

    @MainActor
    func testRejectedGrammarIsNotRetriedUntilASuccessfulSwitch() async throws {
        let rejecting = ConstrainedScriptedEngine(replies: [Self.ana])
        rejecting.constrainedError = GenerationConstraintError.invalidGrammar(root: "root")
        let accepting = ConstrainedScriptedEngine(replies: [Self.ana])
        var engines = [rejecting, accepting]
        let session = try await TonamoxSession(profile: extractor()) { _ in engines.removeFirst() }
        let grammar = try personGrammar()

        _ = await collect(session.stream("Ana is 30."))
        _ = await collect(session.stream("Ana is 30."))
        try await session.switchProfile(to: extractor())          // same profile: a no-op, still refused
        _ = await collect(session.stream("Ana is 30."))
        XCTAssertEqual(rejecting.constraints, [grammar, nil, nil, nil])

        var renamed = extractor()
        renamed.displayName = "Extract again"
        try await session.switchProfile(to: renamed)
        let result = await collect(session.stream("Ana is 30."))

        XCTAssertEqual(result.elements, [Self.ana])
        XCTAssertEqual(accepting.constraints, [grammar])
    }

    @MainActor
    func testCancellationOnTheConstrainedPathIsNotAFallback() async throws {
        let engine = ConstrainedScriptedEngine(replies: [Self.ana])
        engine.hangOnCall = 1
        let session = try await TonamoxSession(profile: extractor()) { _ in engine }
        let grammar = try personGrammar()

        let stream = session.stream("Ana is 30.")
        try await waitForCalls(1, on: engine)
        session.cancel()
        let cancelled = await collect(stream)

        XCTAssertTrue(cancelled.failure is CancellationError, "got \(String(describing: cancelled.failure))")
        XCTAssertEqual(engine.constraints, [grammar])

        let next = await collect(session.stream("Ana is 30."))
        XCTAssertEqual(next.elements, [Self.ana])
        XCTAssertEqual(engine.constraints, [grammar, grammar])
    }

    @MainActor
    func testUnsatisfiableSchemaRunsUnconstrained() async throws {
        let engine = ConstrainedScriptedEngine(replies: [Self.ana])
        let session = try await TonamoxSession(profile: extractor(schema: "false")) { _ in engine }
        session.maxRepairAttempts = 0

        let result = await collect(session.stream("Ana is 30."))

        XCTAssertTrue(result.failure is OutputSchemaError)
        XCTAssertEqual(engine.constraints, [nil])
    }

    @MainActor
    func testPlainEnginesKeepTheValidationOnlyPath() async throws {
        let engine = ScriptedEngine(replies: ["nope", Self.ana])
        let session = try await TonamoxSession(profile: extractor()) { _ in engine }
        XCTAssertTrue(session.usesSchemaGrammar)

        let result = await collect(session.stream("Ana is 30."))

        XCTAssertNil(result.failure)
        XCTAssertEqual(result.elements, [Self.ana])
        XCTAssertEqual(engine.prompts.count, 2)
        XCTAssertEqual(engine.systemPrompts[1], engine.systemPrompts[0])
    }
}
