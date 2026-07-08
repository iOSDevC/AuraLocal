import XCTest
@testable import AuraCore

// Two minimal tools so tool-name-set comparisons have something to differ on.
private struct TapTool: LLMTool {
    let name = "tap"
    let description = "tap"
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput { ToolOutput(data: ["ok": true]) }
}
private struct SwipeTool: LLMTool {
    let name = "swipe"
    let description = "swipe"
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput { ToolOutput(data: ["ok": true]) }
}

/// Value semantics + built-in catalog for `AuraProfile` (pure — no engine, no live model). Locks the vocabulary
/// the AuraSession switching layer (increment 2) composes onto.
final class AuraProfileTests: XCTestCase {

    // A real GGUF `Model` parsed from a Hugging Face URL (parse only — no network).
    private func ggufModel() -> Model {
        Model.fromURL(
            "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
        )!
    }

    func testSamplingParamsCodableRoundTripAndPresets() throws {
        let s = SamplingParams(temperature: 0.3, topP: 0.9, topK: 40, maxTokens: 512, contextOverride: 4096)
        let back = try JSONDecoder().decode(SamplingParams.self, from: try JSONEncoder().encode(s))
        XCTAssertEqual(s, back)
        XCTAssertEqual(SamplingParams.default.temperature, 0.7)
        XCTAssertEqual(SamplingParams.precise.temperature, 0.2)
    }

    func testOutputSchemaCodableRoundTrip() throws {
        let o = OutputSchema.json(#"{"type":"object"}"#)
        XCTAssertEqual(try JSONDecoder().decode(OutputSchema.self, from: try JSONEncoder().encode(o)), o)
    }

    func testEquatableHonorsEveryField() {
        let m = ggufModel()
        let base = AuraProfile(id: "agent", displayName: "Agent", instructions: "do it", model: m, tools: [TapTool()])

        // Identical → equal (tools compared by name set, since [any LLMTool] isn't Equatable).
        XCTAssertEqual(base, AuraProfile(id: "agent", displayName: "Agent", instructions: "do it", model: m, tools: [TapTool()]))

        // Any single differing field → unequal.
        XCTAssertNotEqual(base, AuraProfile(id: "agent", displayName: "Agent", instructions: "do it", model: m, tools: [SwipeTool()])) // tool name
        XCTAssertNotEqual(base, AuraProfile(id: "agent", displayName: "Agent", instructions: "other", model: m, tools: [TapTool()]))   // instructions
        XCTAssertNotEqual(base, AuraProfile(id: "chat", displayName: "Agent", instructions: "do it", model: m, tools: [TapTool()]))    // id
        var hot = SamplingParams.default; hot.temperature = 1.0
        XCTAssertNotEqual(base, AuraProfile(id: "agent", displayName: "Agent", instructions: "do it", model: m, sampling: hot, tools: [TapTool()])) // sampling
        XCTAssertNotEqual(base, AuraProfile(id: "agent", displayName: "Agent", instructions: "do it", model: m, tools: [TapTool()], outputSchema: .json("{}"))) // schema
    }

    func testCatalogChatIsPlainAndLocal() {
        let p = AuraProfileCatalog.chat(model: ggufModel())
        XCTAssertEqual(p.id, "chat")
        XCTAssertTrue(p.tools.isEmpty)
        XCTAssertFalse(p.isAgentic)
        XCTAssertFalse(p.instructions.isEmpty)
        XCTAssertEqual(p.model.format, .gguf)   // Personal-tier invariant: local GGUF, no cloud backend hint
    }

    func testCatalogAgentAndSecurityAreAgenticAndDistinct() {
        let m = ggufModel()
        let agent = AuraProfileCatalog.agent(model: m, tools: [TapTool()])
        let security = AuraProfileCatalog.securityReview(model: m, tools: [SwipeTool()])
        XCTAssertTrue(agent.isAgentic)
        XCTAssertTrue(security.isAgentic)
        XCTAssertGreaterThan(agent.tools.count, 0)
        XCTAssertNotEqual(agent.id, security.id)                     // "agent" vs "security"
        XCTAssertNotEqual(agent.instructions, security.instructions) // distinct personas
        XCTAssertEqual(agent.sampling.temperature, 0.2)             // precise preset → deterministic tool use
    }
}
