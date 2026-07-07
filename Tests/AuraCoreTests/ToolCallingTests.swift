import XCTest
@testable import AuraCore

// A minimal tool defined using only AuraCore's re-exported surface (no `@ToolArguments` macro). File-scope so
// its `Arguments` stays within the nesting limit. `x`/`y` mirror a real pixel-coordinate tool API.
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

/// GGUF function-calling plumbing: a host-defined `LLMTool` threads from `AuraLocal(model:tools:)` /
/// `ModelManager.load(tools:)` all the way to the llama.cpp backend's session. The model-driven tool loop
/// itself is LocalLLMClient's and is exercised live with a function-calling GGUF; here we verify the wiring
/// and that a tool declared against AuraCore's re-exported API is well-formed.
@MainActor
final class ToolCallingTests: XCTestCase {

    private var ggufModel: Model {
        Model.fromURL("https://huggingface.co/o/r/resolve/main/tiny.Q4_K_M.gguf")!
    }

    /// Read tool names off whichever llama backend the router picked.
    private func toolNames(_ backend: any InferenceBackend) -> [String]? {
        (backend as? LlamaCppBackend)?.registeredToolNames
            ?? (backend as? LayerStreamingBackend)?.registeredToolNames
    }

    func testToolThreadsIntoLlamaBackends() {
        let tap = TapTool()
        XCTAssertEqual(LlamaCppBackend(model: ggufModel, tools: [tap]).registeredToolNames, ["tap"])
        XCTAssertEqual(LayerStreamingBackend(model: ggufModel, tools: [tap]).registeredToolNames, ["tap"])
        // Default is no tools (plain chat path).
        XCTAssertEqual(LlamaCppBackend(model: ggufModel).registeredToolNames, [])
    }

    func testRouterThreadsToolsToTheSelectedBackend() {
        let backend = BackendRouter.selectBackend(for: ggufModel, tools: [TapTool()])
        XCTAssertEqual(toolNames(backend), ["tap"])
    }

    func testAuraLocalThreadsToolsEndToEnd() {
        let aura = AuraLocal(model: ggufModel, tools: [TapTool()])
        XCTAssertEqual(toolNames(aura.engine.backend), ["tap"])
        // No tools → the backend registers none.
        XCTAssertEqual(toolNames(AuraLocal(model: ggufModel).engine.backend), [])
    }

    func testToolDeclarationIsWellFormed() async throws {
        // The schema advertises exactly the declared arguments…
        XCTAssertEqual(Set(TapTool.Arguments.argumentsSchema.keys), ["x", "y"])
        // …and calling the tool returns structured output.
        let output = try await TapTool().call(arguments: .init(x: 5, y: 9))
        XCTAssertEqual(output.data["ok"] as? Bool, true)
        XCTAssertEqual(output.data["x"] as? Int, 5)
    }
}
