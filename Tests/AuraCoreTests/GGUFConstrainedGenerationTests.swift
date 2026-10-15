import XCTest
@testable import AuraCore

/// Needs AURA_GGUF_TEST_MODEL. A reused 0.5.0 client returns empty output after the first constrained call.
@MainActor
final class GGUFConstrainedGenerationTests: XCTestCase {

    private static let modelVariable = "AURA_GGUF_TEST_MODEL"

    func testTenConsecutiveJSONCallsAreAllValidObjects() async throws {
        let llm = try await loadModel()
        for index in 0..<10 {
            let reply = try await llm.chat(personPrompt(index), maxTokens: 128, constraint: .json)
            XCTAssertTrue(isJSONObject(reply), "call \(index + 1) returned \(reply.debugDescription)")
        }
    }

    func testConsecutiveCustomGrammarCallsAllMatchTheGrammar() async throws {
        let llm = try await loadModel()
        for index in 0..<5 {
            let reply = try await llm.chat("Answer yes or no: is \(index + 2) an even number?", maxTokens: 8,
                                           constraint: .grammar(gbnf: #"root ::= "yes" | "no""#))
            XCTAssertTrue(["yes", "no"].contains(reply), "call \(index + 1) returned \(reply.debugDescription)")
        }
    }

    func testStreamedConstrainedOutputIsValidJSON() async throws {
        let llm = try await loadModel()
        for _ in 0..<2 {
            var reply = ""
            for try await delta in llm.stream(personPrompt(3), maxTokens: 128, constraint: .json) { reply += delta }
            XCTAssertTrue(isJSONObject(reply), reply.debugDescription)
        }
    }

    func testMainSessionKeepsAnsweringAroundConstrainedCalls() async throws {
        let llm = try await loadModel()
        let before = try await llm.chat("Name one primary color.", maxTokens: 24)
        XCTAssertFalse(before.isEmpty)
        let constrained = try await llm.chat(personPrompt(1), maxTokens: 128, constraint: .json)
        XCTAssertTrue(isJSONObject(constrained), constrained.debugDescription)
        let after = try await llm.chat("Name one primary color.", maxTokens: 24)
        XCTAssertFalse(after.isEmpty, "the main session must not inherit the grammar")
    }

    /// 0.5.0 would crash on the first sampled token for each of these.
    func testGrammarsLlamaCppRejectsThrowInsteadOfCrashing() async throws {
        let llm = try await loadModel()
        let rejected = [
            "root ::= (\"a\"",              // unbalanced group
            #"answer ::= "yes""#,           // no root rule
            "root ::= other",               // undefined rule
            #"root ::= root "a" | "a""#,    // left recursion
        ]
        for gbnf in rejected {
            do {
                let reply = try await llm.chat("hi", maxTokens: 8, constraint: .grammar(gbnf: gbnf))
                XCTFail("\(gbnf) was accepted and returned \(reply.debugDescription)")
            } catch let error as GenerationConstraintError {
                XCTAssertEqual(error, .invalidGrammar(root: "root"))
            }
        }
        let reply = try await llm.chat(personPrompt(2), maxTokens: 128, constraint: .json)
        XCTAssertTrue(isJSONObject(reply), "a rejected grammar must not poison later calls: \(reply.debugDescription)")
    }

    func testCancelledConstrainedCallThrowsInsteadOfReturningPartialOutput() async throws {
        let llm = try await loadModel()
        let prompt = personPrompt(0)
        let call = Task {
            try await llm.engine.generate(prompt: prompt, maxTokens: 128, constraint: .json) { _ in
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        do {
            let reply = try await call.value
            XCTFail("a cancelled call returned \(reply.debugDescription)")
        } catch is CancellationError {}
        let reply = try await llm.chat(personPrompt(1), maxTokens: 128, constraint: .json)
        XCTAssertTrue(isJSONObject(reply), reply.debugDescription)
    }

    func testCancelledChatThrowsBeforeStartingADecode() async throws {
        let llm = try await loadModel()
        let call = Task { try await llm.chat("Name one primary color.", maxTokens: 24) }
        call.cancel()
        do {
            let reply = try await call.value
            XCTFail("a cancelled chat returned \(reply.debugDescription)")
        } catch is CancellationError {}
        let after = try await llm.chat("Name one primary color.", maxTokens: 24)
        XCTAssertFalse(after.isEmpty)
    }

    func testConstrainedCallQueuedAcrossAnUnloadThrowsModelNotLoaded() async throws {
        let llm = try await loadModel()
        let probe = QueueProbe()
        let (holderPrompt, queuedPrompt) = (personPrompt(0), personPrompt(1))
        let holder = Task {
            try await llm.engine.generate(prompt: holderPrompt, maxTokens: 128, constraint: .json) { _ in
                probe.tokens += 1
                if probe.tokens == 1 {
                    probe.queued = Task { try await llm.chat(queuedPrompt, maxTokens: 128, constraint: .json) }
                } else if probe.tokens == 3 {
                    llm.engine.unload()
                }
            }
        }
        let first = try await holder.value
        XCTAssertTrue(isJSONObject(first), first.debugDescription)
        let queued = try XCTUnwrap(probe.queued)
        do {
            let reply = try await queued.value
            XCTFail("a call queued across unload() returned \(reply.debugDescription)")
        } catch AuraError.modelNotLoaded {}
    }

    // MARK: - Helpers

    private func loadModel() async throws -> AuraLocal {
        guard let path = ProcessInfo.processInfo.environment[Self.modelVariable], !path.isEmpty else {
            throw XCTSkip("Set \(Self.modelVariable) to a local .gguf file to run the constrained-generation tests.")
        }
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("\(Self.modelVariable) points at a missing file: \(path)")
        }
        let model = try XCTUnwrap(Model.from(.localFile(URL(fileURLWithPath: path))))
        let llm = AuraLocal(model: model)
        XCTAssertTrue(llm.engine.backend is LlamaCppBackend, "a constraint needs the llama.cpp backend")
        try await llm.engine.load { _ in }
        addTeardownBlock { @MainActor in
            llm.engine.unload()
            await (llm.engine.backend as? LlamaCppBackend)?.awaitRelease()
        }
        return llm
    }

    private func personPrompt(_ index: Int) -> String {
        let people = [("Ana", 31, "Lisbon"), ("Luis", 45, "Quito"), ("Mei", 27, "Taipei"), ("Omar", 52, "Cairo")]
        let (name, age, city) = people[index % people.count]
        return "Return a JSON object with the keys \"name\" (string), \"age\" (number) and \"city\" (string) "
            + "for this person: \(name) is \(age) years old and lives in \(city)."
    }

    private func isJSONObject(_ text: String) -> Bool {
        (try? JSONSerialization.jsonObject(with: Data(text.utf8))) is [String: Any]
    }
}

@MainActor
private final class QueueProbe {
    var tokens = 0
    var queued: Task<String, Error>?
}
