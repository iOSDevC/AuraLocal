import XCTest
@testable import AuraCore

private struct TapTool: LLMTool {
    let name = "tap"
    let description = "tap"
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput { ToolOutput(data: ["ok": true]) }
}

/// Shared ordered event log so tests can assert cancel-before-teardown across engines.
@MainActor private final class Log { var events: [String] = [] }

/// A stub engine that records what it was asked, in what order, without any live GGUF model.
@MainActor private final class SpyEngine: AuraProfileEngine {
    let id: String
    let log: Log
    var hangUntilCancelled = false
    private(set) var lastSystemPrompt: String?
    private(set) var generateCount = 0
    private(set) var didTeardown = false

    init(id: String, log: Log) { self.id = id; self.log = log }

    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onDelta: @escaping @MainActor (String) -> Void) async throws {
        lastSystemPrompt = systemPrompt
        generateCount += 1
        onDelta("hi from \(id)")
        if hangUntilCancelled {
            while !Task.isCancelled { try? await Task.sleep(for: .milliseconds(2)) }
            log.events.append("\(id).cancelled")   // logged when the awaited decode observes cancellation
            throw CancellationError()
        }
    }

    func teardown() { didTeardown = true; log.events.append("\(id).teardown") }
}

/// Consume one delta so the (deferred) `engine.stream(...)` call actually runs before we assert on the spy.
/// File-scope (no `self`) to avoid sending the non-Sendable test instance across the isolation boundary.
private func drainOne(_ stream: AsyncThrowingStream<String, Error>) async throws {
    for try await _ in stream { break }
}

/// Hands out engines in order so a makeEngine factory returns spyA, then spyB, …
@MainActor private final class EngineQueue {
    private var engines: [SpyEngine]
    private var i = 0
    init(_ engines: [SpyEngine]) { self.engines = engines }
    func next() -> any AuraProfileEngine { defer { i += 1 }; return engines[i] }
}

final class AuraSessionTests: XCTestCase {

    private func ggufModel() -> Model {
        Model.fromURL("https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf")!
    }

    @MainActor
    func testSwitchRebuildsEngineAndKeepsConversationID() async throws {
        let log = Log()
        let spyA = SpyEngine(id: "A", log: log), spyB = SpyEngine(id: "B", log: log)
        let queue = EngineQueue([spyA, spyB])
        let m = ggufModel()
        let session = try await AuraSession(profile: AuraProfileCatalog.chat(model: m)) { _ in queue.next() }
        let convo = session.conversationID

        try await session.switchProfile(to: AuraProfileCatalog.agent(model: m, tools: [TapTool()]))

        XCTAssertEqual(session.profile.id, "agent")
        XCTAssertEqual(session.conversationID, convo)   // stable across the switch
        XCTAssertTrue(spyA.didTeardown)                 // old engine freed
        // The new engine (spyB) is the one now used:
        try await drainOne(session.stream("go"))
        XCTAssertEqual(spyB.generateCount, 1)
        XCTAssertEqual(spyA.generateCount, 0)
        session.cancel()
    }

    @MainActor
    func testStreamInjectsActiveProfilePersona() async throws {
        let log = Log()
        let spyA = SpyEngine(id: "A", log: log), spyB = SpyEngine(id: "B", log: log)
        let queue = EngineQueue([spyA, spyB])
        let m = ggufModel()
        let chat = AuraProfileCatalog.chat(model: m, instructions: "You are CHAT.")
        let session = try await AuraSession(profile: chat) { _ in queue.next() }

        try await drainOne(session.stream("hello"))
        XCTAssertEqual(spyA.lastSystemPrompt, "You are CHAT.")   // persona injected as the system prompt

        try await session.switchProfile(to: AuraProfileCatalog.agent(model: m, tools: [TapTool()], instructions: "You are AGENT."))
        try await drainOne(session.stream("act"))
        XCTAssertEqual(spyB.lastSystemPrompt, "You are AGENT.")  // switched persona takes effect
        session.cancel()
    }

    @MainActor
    func testSwitchDuringActiveStreamCancelsBeforeTeardown() async throws {
        let log = Log()
        let spyA = SpyEngine(id: "A", log: log); spyA.hangUntilCancelled = true
        let spyB = SpyEngine(id: "B", log: log)
        let queue = EngineQueue([spyA, spyB])
        let m = ggufModel()
        let session = try await AuraSession(profile: AuraProfileCatalog.chat(model: m)) { _ in queue.next() }

        let hold = session.stream("long")                // hold it (a real consumer would) so discarding it
        try? await Task.sleep(for: .milliseconds(10))    // doesn't prematurely cancel; let the decode reach the hang

        try await session.switchProfile(to: AuraProfileCatalog.agent(model: m, tools: [TapTool()]))

        XCTAssertTrue(spyA.didTeardown)
        let cancelled = try XCTUnwrap(log.events.firstIndex(of: "A.cancelled"))
        let torndown = try XCTUnwrap(log.events.firstIndex(of: "A.teardown"))
        XCTAssertLessThan(cancelled, torndown)           // the SAFETY invariant: cancel+await BEFORE teardown
        session.cancel()
        withExtendedLifetime(hold) {}
    }

    @MainActor
    func testRebuildFailureLeavesSessionRecoverable() async throws {
        struct BuildError: Error {}
        let log = Log()
        let good = SpyEngine(id: "good", log: log)
        let recovered = SpyEngine(id: "recovered", log: log)
        var call = 0
        let make: @MainActor (AuraProfile) async throws -> any AuraProfileEngine = { _ in
            defer { call += 1 }
            switch call {
            case 0: return good            // initial build
            case 1: throw BuildError()     // the switch's rebuild fails
            default: return recovered      // a retry succeeds
            }
        }
        let m = ggufModel()
        let session = try await AuraSession(profile: AuraProfileCatalog.chat(model: m), makeEngine: make)

        // A switch whose rebuild throws → surfaces the error, and the old engine was torn down.
        let target = AuraProfileCatalog.agent(model: m, tools: [TapTool()])
        do { try await session.switchProfile(to: target); XCTFail("expected the rebuild to throw") }
        catch is BuildError { XCTAssertTrue(good.didTeardown) }

        // Recovery back to the CURRENT profile (chat) must re-enter — without the switchFailed relaxation the
        // equality guard would no-op it and the torn-down session would stay wedged.
        try await session.switchProfile(to: AuraProfileCatalog.chat(model: m))
        XCTAssertEqual(session.profile.id, "chat")
        try await drainOne(session.stream("go"))
        XCTAssertEqual(recovered.generateCount, 1)
        session.cancel()
    }

    @MainActor
    func testSwitchToIdenticalProfileIsNoOp() async throws {
        let log = Log()
        let spyA = SpyEngine(id: "A", log: log), spyB = SpyEngine(id: "B", log: log)
        let queue = EngineQueue([spyA, spyB])
        let m = ggufModel()
        let chat = AuraProfileCatalog.chat(model: m)
        let session = try await AuraSession(profile: chat) { _ in queue.next() }

        try await session.switchProfile(to: AuraProfileCatalog.chat(model: m))  // identical profile

        XCTAssertFalse(spyA.didTeardown)   // no rebuild, no teardown
        XCTAssertEqual(spyB.generateCount, 0)
        session.cancel()
    }
}
