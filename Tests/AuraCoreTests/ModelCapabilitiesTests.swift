import Foundation
import Testing
@testable import AuraCore

/// `Model.capabilities` and `HFPipelineFilter`: what the Example's pickers show and filter by.
struct ModelCapabilitiesTests {

    private static func model(category: String, format: String, documentPrompt: String? = nil) throws -> Model {
        let prompt = documentPrompt.map { "\"\($0)\"" } ?? "null"
        let json = """
            {"id": "caps_test", "repoID": "org/caps-test", "displayName": "Caps", "category": "\(category)",
             "docTags": false, "format": "\(format)", "approximateSizeMB": 100, "isUncensored": false,
             "ggufFilename": \(format == "gguf" ? "\"caps.gguf\"" : "null"), "defaultDocumentPrompt": \(prompt),
             "numLayers": 1, "kvHeads": 1, "headDim": 1}
            """
        return try JSONDecoder().decode(Model.self, from: Data(json.utf8))
    }

    @Test func ggufTextModelChatsAndCallsTools() throws {
        #expect(try Self.model(category: "text", format: "gguf").capabilities == [.text, .tools])
    }

    @Test func mlxTextModelCallsToolsWhereMLXRuns() throws {
        #if canImport(MLXLLM) && !targetEnvironment(simulator)
        #expect(try Self.model(category: "text", format: "mlx").capabilities == [.text, .tools])
        #else
        #expect(try Self.model(category: "text", format: "mlx").capabilities == [.text])
        #endif
    }

    @Test func visionModelChatsAndTakesImages() throws {
        #expect(try Self.model(category: "vision", format: "mlx").capabilities == [.text, .vision])
    }

    @Test func documentModelExtractsButDoesNotChat() throws {
        let caps = try Self.model(category: "visionSpecialized", format: "mlx", documentPrompt: "Read it.")
            .capabilities
        #expect(caps == [.vision, .documentExtraction])
    }

    @Test func ggufVisionModelTakesNoImages() throws {
        #expect(try Self.model(category: "vision", format: "gguf").capabilities == [.text, .tools])
        let document = try Self.model(category: "visionSpecialized", format: "gguf", documentPrompt: "Read it.")
        #expect(document.capabilities == [.tools])
    }

    @Test func documentPromptOnATextModelIsNoExtraction() throws {
        let caps = try Self.model(category: "text", format: "gguf", documentPrompt: "Read it.").capabilities
        #expect(!caps.contains(.documentExtraction))
    }

    @Test func toolsMatchTheRouterRule() throws {
        let tools: [any LLMTool] = [CapsTool()]
        for category in ["vision", "visionSpecialized"] {
            let mlx = try Self.model(category: category, format: "mlx", documentPrompt: "Read it.")
            #expect(!mlx.capabilities.contains(.tools))
            #expect(throws: ModelConfigurationError.toolsUnsupported(.mlx)) {
                try BackendRouter.requireToolSupport(tools, for: mlx)
            }
        }
        for format in ["gguf", "mlx"] {
            let model = try Self.model(category: "text", format: format)
            let runs = model.capabilities.contains(.tools)
            #expect(runs == ((try? BackendRouter.requireToolSupport(tools, for: model)) != nil))
        }
        #expect(try Self.model(category: "text", format: "gguf").capabilities.contains(.tools))
    }

    @Test func catalogDocumentModelsExtract() {
        let documents = Model.specializedModels
        #expect(!documents.isEmpty)
        #expect(documents.allSatisfy { $0.capabilities.contains(.documentExtraction) })
        #expect(Model.textModels.allSatisfy { !$0.capabilities.contains(.vision) })
    }

    // MARK: HFPipelineFilter

    private static func hit(_ pipeline: String?) -> HFModelHit {
        HFModelHit(id: "org/repo", likes: 0, downloads: 0, pipelineTag: pipeline, tags: [], gated: false)
    }

    @Test func chatFilterKeepsChatPipelinesAndUndeclaredRepos() {
        let filter = HFPipelineFilter.chat
        #expect(filter.admits(Self.hit("text-generation")))
        #expect(filter.admits(Self.hit("image-text-to-text")))
        #expect(filter.admits(Self.hit("any-to-any")))
        #expect(filter.admits(Self.hit(nil)))
        #expect(!filter.admits(Self.hit("text-to-image")))
        #expect(!filter.admits(Self.hit("feature-extraction")))
    }

    @Test func strictFilterDropsUndeclaredRepos() {
        let filter = HFPipelineFilter(pipelineTags: ["text-generation"], admitsUndeclared: false)
        #expect(!filter.admits(Self.hit(nil)))
        #expect(filter.admits(Self.hit("text-generation")))
    }
}

private struct CapsTool: LLMTool {
    let name = "caps"
    let description = "caps"
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput { ToolOutput(data: ["ok": true]) }
}
