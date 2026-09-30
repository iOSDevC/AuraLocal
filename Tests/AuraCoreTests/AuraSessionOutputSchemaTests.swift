import XCTest
@testable import AuraCore

/// Replays scripted replies (the last one repeats), each as two deltas so buffering is exercised.
@MainActor private final class ScriptedEngine: AuraProfileEngine {
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

private func collect(_ stream: AsyncThrowingStream<String, Error>) async -> (elements: [String], failure: (any Error)?) {
    var elements: [String] = []
    do {
        for try await element in stream { elements.append(element) }
        return (elements, nil)
    } catch {
        return (elements, error)
    }
}

final class AuraSessionOutputSchemaTests: XCTestCase {

    private static let personSchema = #"""
        {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0}},
         "required":["name","age"],"additionalProperties":false}
        """#

    private func ggufModel() -> Model {
        Model.fromURL("https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf")!
    }

    private func extractor(schema: String = personSchema) -> AuraProfile {
        AuraProfile(id: "extract", displayName: "Extract", instructions: "You extract people.",
                    model: ggufModel(), sampling: .precise, outputSchema: .json(schema))
    }

    @MainActor
    func testConformingFirstReplyYieldsOneElement() async throws {
        let engine = ScriptedEngine(replies: ["Sure:\n```json\n{\"name\":\"Ana\",\"age\":30}\n```"])
        let session = try await AuraSession(profile: extractor()) { _ in engine }

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
        let session = try await AuraSession(profile: extractor()) { _ in engine }

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
        let session = try await AuraSession(profile: extractor()) { _ in engine }
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
        let session = try await AuraSession(profile: extractor()) { _ in engine }

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
        let make: @MainActor (AuraProfile) async throws -> any AuraProfileEngine = { _ in
            builds += 1
            return ScriptedEngine(replies: ["{}"])
        }

        do {
            _ = try await AuraSession(profile: extractor(schema: #"root ::= "yes" | "no""#), makeEngine: make)
            XCTFail("a GBNF schema must be rejected")
        } catch {
            guard case .invalidSchema = error as? OutputSchemaError else { return XCTFail("unexpected \(error)") }
        }
        do {
            _ = try await AuraSession(profile: extractor(schema: #"{"type":"string","format":"email"}"#), makeEngine: make)
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
        let chat = AuraProfileCatalog.chat(model: ggufModel())
        let session = try await AuraSession(profile: chat) { _ in builds += 1; return engine }

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
        let session = try await AuraSession(profile: AuraProfileCatalog.chat(model: ggufModel())) { _ in
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
        let chat = AuraProfileCatalog.chat(model: ggufModel(), instructions: "You are CHAT.")
        let session = try await AuraSession(profile: chat) { _ in engine }

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
        let session = try await AuraSession(profile: extractor()) { _ in engine }
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
