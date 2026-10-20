import XCTest
@testable import AuraCore

/// Needs AURA_GGUF_TEST_MODEL. Replies and wall times are printed with a `[schema-grammar]` prefix.
@MainActor
final class GGUFSchemaGrammarLiveTests: XCTestCase {

    private static let modelVariable = "AURA_GGUF_TEST_MODEL"
    private static let maxTokens = 200

    private static let personSchema = #"""
        {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0}},
         "required":["name","age"],"additionalProperties":false}
        """#

    private struct Case {
        let name: String
        let schema: String
        let prompt: String
        /// Every keyword is enforced by the grammar, so a reply that is not cut must validate.
        let enforced: Bool
    }

    private static let corpus = [
        Case(name: "person", schema: personSchema,
             prompt: "Ana Souza is 31 years old and lives in Lisbon.", enforced: true),
        Case(name: "enum-only", schema: #"{"enum":["red","green","blue"]}"#,
             prompt: "Which of red, green or blue is the color of fresh grass?", enforced: true),
        Case(name: "nested-order", schema: #"""
            {"type":"object","required":["customer","items"],"additionalProperties":false,
             "properties":{"customer":{"type":"string","minLength":1,"maxLength":40},
              "items":{"type":"array","minItems":1,"items":{"type":"object","required":["sku","qty"],
               "additionalProperties":false,
               "properties":{"sku":{"type":"string"},"qty":{"type":"integer","minimum":1}}}}}}
            """#, prompt: "Luis ordered 2 of item AB-1 and 1 of item ZX-9.", enforced: true),
        Case(name: "nullable-field", schema: #"""
            {"type":"object","required":["name","nickname"],"additionalProperties":false,
             "properties":{"name":{"type":"string"},"nickname":{"type":["string","null"]}}}
            """#, prompt: "Margaret, whom everyone calls Peggy, joined the team.", enforced: true),
        Case(name: "string-bounds", schema: #"""
            {"type":"object","required":["code"],"properties":{"code":{"type":"string","minLength":3,"maxLength":5}}}
            """#, prompt: "Give the three-letter IATA code of Lisbon's airport.", enforced: true),
        Case(name: "any-of", schema: #"{"anyOf":[{"type":"integer","minimum":0},{"type":"string","maxLength":10}]}"#,
             prompt: "How many legs does a spider have?", enforced: true),
        // An open prompt here makes the 0.5B model repeat members until maxTokens.
        Case(name: "empty-schema", schema: "{}", prompt: #"Return {"sum": n} where n is 2 + 2."#, enforced: true),
        Case(name: "root-array", schema: #"""
            {"type":"array","minItems":2,"maxItems":4,"items":{"type":"string","maxLength":20}}
            """#, prompt: "List three fruits.", enforced: true),
        // The schema `aura receipt --model` decodes under (ReceiptPrompt.schema).
        Case(name: "receipt-fields", schema: #"""
            {"type":"object","additionalProperties":false,
             "required":["merchant","date","total","tax","currency"],
             "properties":{"merchant":{"type":["string","null"],"minLength":1},
              "date":{"type":["string","null"],"minLength":10,"maxLength":10},
              "total":{"type":["number","null"],"minimum":0},"tax":{"type":["number","null"],"minimum":0},
              "currency":{"type":["string","null"],"minLength":3,"maxLength":3}}}
            """#, prompt: "Receipt text, line by line:\nSUPERMERCADO LA PLAZA\n03/03/2026\nIVA 6.00\nTOTAL $43.50 MXN"
                + "\n\nReturn the JSON object for this receipt.", enforced: true),
        Case(name: "unenforced-maximum", schema: #"""
            {"type":"object","required":["score"],"properties":{"score":{"type":"integer","minimum":0,"maximum":10}}}
            """#, prompt: "Rate the movie Casablanca from 0 to 10.", enforced: false),
    ]

    private static let personPrompts = [
        "Ana Souza is 31 years old.",
        "Luis turned 45 last week.",
        "Mei, aged 27, works in Taipei.",
        "Omar is fifty-two.",
    ]

    // MARK: Grammars

    func testEveryCorpusGrammarLoadsAndDecodes() async throws {
        let llm = try await loadModel()
        for entry in Self.corpus {
            try await decode(entry, on: llm)
        }
    }

    private func decode(_ entry: Case, on llm: AuraLocal) async throws {
        let validator = try OutputSchemaValidator(.json(entry.schema))
        let grammar = try XCTUnwrap(validator.grammar, entry.name)
        let clock = ContinuousClock()
        let start = clock.now
        let reply: String
        do {
            reply = try await llm.chat(entry.prompt, systemPrompt: Self.contract(entry.schema),
                                       maxTokens: Self.maxTokens,
                                       constraint: .grammar(gbnf: grammar, root: OutputSchemaGrammar.rootRule))
        } catch let error as GenerationConstraintError {
            return XCTFail("\(entry.name): llama.cpp refused the grammar: \(error)")
        }
        let elapsed = clock.now - start
        let generated = llm.lastGenerationStats?.generatedTokens ?? 0
        print("[schema-grammar] \(entry.name): \(generated) tokens in \(elapsed): \(reply.debugDescription)")
        guard generated < Self.maxTokens else {
            return XCTFail("\(entry.name): cut by maxTokens (\(Self.maxTokens)), not a grammar failure: "
                           + reply.debugDescription)
        }
        guard JSONValue.parse(reply) != nil else {
            return XCTFail("\(entry.name): not JSON: \(reply.debugDescription)")
        }
        let violations = validator.validate(reply)
        if entry.enforced {
            XCTAssertEqual(violations, [], "\(entry.name): \(reply.debugDescription)")
        } else if !violations.isEmpty {
            print("[schema-grammar] \(entry.name): not enforced by the grammar, validation found \(violations)")
        }
    }

    // MARK: Session

    func testSessionRepliesConformWithoutRepairs() async throws {
        let model = try localModel()
        let holder = EngineHolder()
        let session = try await AuraSession(profile: extractor(model, schema: Self.personSchema)) { profile in
            try await holder.load(profile)
        }
        addTeardownBlock { @MainActor in await holder.release() }
        session.maxRepairAttempts = 0

        var constrained: [Duration] = []
        for prompt in Self.personPrompts {
            constrained.append(try await conformingReply(to: prompt, in: session, startingWith: "name", holder: holder))
        }

        session.usesSchemaGrammar = false
        var unconstrained: [Duration] = []
        for prompt in Self.personPrompts {
            unconstrained.append(await unconstrainedReply(to: prompt, in: session, holder: holder))
        }
        print("[schema-grammar] session wall time per reply, grammar: \(constrained), "
              + "validation only: \(unconstrained)")
    }

    /// A model writes `name` before `age` unprompted; only the grammar puts `age` first.
    func testSessionFollowsTheRequiredOrder() async throws {
        let model = try localModel()
        let holder = EngineHolder()
        let ageFirst = Self.personSchema.replacingOccurrences(of: #"["name","age"]"#, with: #"["age","name"]"#)
        let session = try await AuraSession(profile: extractor(model, schema: ageFirst)) { profile in
            try await holder.load(profile)
        }
        addTeardownBlock { @MainActor in await holder.release() }
        session.maxRepairAttempts = 0

        _ = try await conformingReply(to: Self.personPrompts[0], in: session, startingWith: "age", holder: holder)
    }

    private func conformingReply(to prompt: String, in session: AuraSession, startingWith key: String,
                                 holder: EngineHolder) async throws -> Duration {
        let clock = ContinuousClock()
        let start = clock.now
        let reply = try await collect(session.stream(prompt))
        let elapsed = clock.now - start
        print("[schema-grammar] session, grammar, \(holder.generatedTokens) tokens in \(elapsed): "
              + reply.debugDescription)
        let opening = try Regex("^\\{(|[ ]|\\n[ \\t]{0,20})\"\(key)\"")
        XCTAssertNotNil(reply.prefixMatch(of: opening), "the grammar fixes the member order: \(reply.debugDescription)")
        return elapsed
    }

    private func unconstrainedReply(to prompt: String, in session: AuraSession,
                                    holder: EngineHolder) async -> Duration {
        let clock = ContinuousClock()
        let start = clock.now
        let outcome: String
        do {
            outcome = try await collect(session.stream(prompt))
        } catch {
            outcome = "failed: \(error)"
        }
        let elapsed = clock.now - start
        print("[schema-grammar] session, validation only, \(holder.generatedTokens) tokens in \(elapsed): "
              + outcome.debugDescription)
        return elapsed
    }

    // MARK: Helpers

    private static func contract(_ schema: String) -> String {
        "Reply with a single JSON value that conforms to this JSON Schema, no prose:\n\(schema)"
    }

    private func extractor(_ model: Model, schema: String) -> AuraProfile {
        AuraProfile(id: "extract", displayName: "Extract", instructions: "You extract people from text.",
                    model: model, sampling: SamplingParams(temperature: 0, maxTokens: 96), outputSchema: .json(schema))
    }

    private func collect(_ stream: AsyncThrowingStream<String, Error>) async throws -> String {
        var text = ""
        for try await element in stream { text += element }
        return text
    }

    private func localModel() throws -> Model {
        guard let path = ProcessInfo.processInfo.environment[Self.modelVariable], !path.isEmpty else {
            throw XCTSkip("Set \(Self.modelVariable) to a local .gguf file to run the schema-grammar tests.")
        }
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("\(Self.modelVariable) points at a missing file: \(path)")
        }
        return try XCTUnwrap(Model.from(.localFile(URL(fileURLWithPath: path))))
    }

    private func loadModel() async throws -> AuraLocal {
        let model = try localModel()
        let llm = AuraLocal(model: model, sampling: try SamplingSettings.resolve(temperature: 0, topP: nil, for: model))
        XCTAssertTrue(llm.engine.backend is LlamaCppBackend, "a constraint needs the llama.cpp backend")
        try await llm.engine.load { _ in }
        addTeardownBlock { @MainActor in
            llm.engine.unload()
            await (llm.engine.backend as? LlamaCppBackend)?.awaitRelease()
        }
        return llm
    }
}

/// Builds the session's engine outside `ModelManager`, so no shared instance outlives the test.
@MainActor
private final class EngineHolder {
    private var llm: AuraLocal?

    var generatedTokens: String { llm?.lastGenerationStats.map { "\($0.generatedTokens)" } ?? "?" }

    func load(_ profile: AuraProfile) async throws -> any AuraProfileEngine {
        let sampling = try SamplingSettings.resolve(temperature: profile.sampling.temperature,
                                                    topP: profile.sampling.topP, for: profile.model)
        let loaded = AuraLocal(model: profile.model, sampling: sampling)
        XCTAssertTrue(loaded.engine.backend is LlamaCppBackend, "the schema grammar needs the llama.cpp backend")
        try await loaded.engine.load { _ in }
        llm = loaded
        return loaded
    }

    func release() async {
        llm?.engine.unload()
        await (llm?.engine.backend as? LlamaCppBackend)?.awaitRelease()
    }
}
