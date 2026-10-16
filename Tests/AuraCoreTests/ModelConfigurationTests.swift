import XCTest
@testable import AuraCore
#if canImport(MLXLLM)
import MLXLMCommon
#endif

private struct TapTool: LLMTool {
    let name = "tap"
    let description = "tap"
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput { ToolOutput(data: ["ok": true]) }
}

@MainActor private final class StubEngine: AuraProfileEngine {
    private(set) var didTeardown = false
    private(set) var generateCount = 0

    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onDelta: @escaping @MainActor (String) -> Void) async throws {
        generateCount += 1
        onDelta("ok")
    }

    func teardown() { didTeardown = true }
}

@MainActor private final class BuildCounter {
    var builds = 0
}

/// Tools on MLX and sampling values are refused before any download, engine build or teardown; resolved sampling
/// reaches every backend. Nothing here loads weights or touches the network.
@MainActor
final class ModelConfigurationTests: XCTestCase {

    private var gguf: Model {
        Model.fromURL("https://huggingface.co/o/r/resolve/main/tiny.Q4_K_M.gguf")!
    }

    /// A file that does not exist, so a load that got past validation would fail without the network.
    private var missingLocalGGUF: Model {
        Model.from(.localFile(URL(fileURLWithPath: "/nonexistent/aura-missing.gguf")))!
    }

    private func mlxText() throws -> Model {
        try XCTUnwrap(ModelRegistry.shared.models.first { $0.format == .mlx && $0.category == .text })
    }

    private func mlxModel(_ category: Model.Category) throws -> Model {
        try XCTUnwrap(ModelRegistry.shared.models.first { $0.format == .mlx && $0.category == category })
    }

    // MARK: - Tools on MLX

    func testModelManagerRefusesToolsForAnMLXModelBeforeLoading() async throws {
        let mlx = try mlxText()
        do {
            _ = try await ModelManager.shared.load(mlx, tools: [TapTool()])
            XCTFail("tools on an MLX model were accepted")
        } catch let error as ModelConfigurationError {
            XCTAssertEqual(error, .toolsUnsupported(.mlx))
        }
        XCTAssertEqual(ModelManager.shared.state(for: mlx), .idle)
        XCTAssertFalse(ModelManager.shared.isLoaded(mlx))
    }

    func testRouterGivesAnMLXModelWithToolsABackendThatThrowsOnLoad() async throws {
        let backend = BackendRouter.selectBackend(for: try mlxText(), tools: [TapTool()])
        XCTAssertTrue(backend is UnavailableBackend)
        do {
            try await backend.load { _ in }
            XCTFail("the backend loaded")
        } catch let error as ModelConfigurationError {
            XCTAssertEqual(error, .toolsUnsupported(.mlx))
        }
        XCTAssertTrue(AuraLocal(model: try mlxText(), tools: [TapTool()]).engine.backend is UnavailableBackend)
    }

    func testToolSupportCheckAcceptsGGUFAndToolFreeMLX() throws {
        XCTAssertNoThrow(try BackendRouter.requireToolSupport([TapTool()], for: gguf))
        XCTAssertNoThrow(try BackendRouter.requireToolSupport([], for: try mlxText()))
        XCTAssertThrowsError(try BackendRouter.requireToolSupport([TapTool()], for: try mlxText())) { error in
            XCTAssertEqual(error as? ModelConfigurationError, .toolsUnsupported(.mlx))
        }
    }

    func testSessionRefusesAnMLXAgentProfileBeforeBuildingAnEngine() async throws {
        let counter = BuildCounter()
        let profile = AuraProfileCatalog.agent(model: try mlxText(), tools: [TapTool()])
        do {
            _ = try await AuraSession(profile: profile) { _ in
                counter.builds += 1
                return StubEngine()
            }
            XCTFail("an MLX agent profile was accepted")
        } catch let error as ModelConfigurationError {
            XCTAssertEqual(error, .toolsUnsupported(.mlx))
        }
        XCTAssertEqual(counter.builds, 0)
    }

    func testSwitchToAnMLXAgentProfileThrowsAndKeepsTheCurrentEngine() async throws {
        let engine = StubEngine()
        let counter = BuildCounter()
        let chat = AuraProfileCatalog.chat(model: gguf)
        let session = try await AuraSession(profile: chat) { _ in
            counter.builds += 1
            return engine
        }
        do {
            try await session.switchProfile(to: AuraProfileCatalog.agent(model: try mlxText(), tools: [TapTool()]))
            XCTFail("an MLX agent profile was accepted")
        } catch let error as ModelConfigurationError {
            XCTAssertEqual(error, .toolsUnsupported(.mlx))
        }
        XCTAssertFalse(engine.didTeardown)
        XCTAssertEqual(counter.builds, 1)
        XCTAssertEqual(session.profile, chat)
        for try await _ in session.stream("hi") {}
        XCTAssertEqual(engine.generateCount, 1)
    }

    // MARK: - Sampling validation

    func testOutOfRangeSamplingIsRejected() {
        for temperature: Float in [-0.1, -.infinity, .infinity, .nan] {
            XCTAssertThrowsError(try SamplingSettings.resolve(temperature: temperature, topP: nil, for: gguf)) {
                guard case .invalidTemperature? = $0 as? ModelConfigurationError else { return XCTFail("\($0)") }
            }
        }
        for topP: Float in [0, -0.5, 1.01, .infinity, .nan] {
            XCTAssertThrowsError(try SamplingSettings.resolve(temperature: nil, topP: topP, for: gguf)) {
                guard case .invalidTopP? = $0 as? ModelConfigurationError else { return XCTFail("\($0)") }
            }
        }
        XCTAssertNoThrow(try SamplingSettings.resolve(temperature: 0, topP: 1, for: gguf))
        XCTAssertNoThrow(try SamplingSettings.resolve(temperature: 2.5, topP: 0.01, for: gguf))
    }

    func testLoadAndFactoriesRejectBadSamplingBeforeTouchingTheFile() async {
        await assertConfigurationError(.invalidTemperature(-1)) {
            _ = try await ModelManager.shared.load(self.missingLocalGGUF, temperature: -1)
        }
        await assertConfigurationError(.invalidTopP(0)) {
            _ = try await ModelManager.shared.load(self.missingLocalGGUF, topP: 0)
        }
        await assertConfigurationError(.invalidTemperature(-2)) {
            _ = try await AuraLocal.text(self.missingLocalGGUF, temperature: -2)
        }
        await assertConfigurationError(.invalidTopP(1.5)) {
            _ = try await AuraLocal.text(self.missingLocalGGUF, topP: 1.5)
        }
        XCTAssertEqual(ModelManager.shared.state(for: missingLocalGGUF), .idle)
    }

    /// The live factory validates nothing itself, so a rejection proves the profile's values reach the load.
    func testSessionLiveEngineThreadsTheProfileSamplingIntoTheLoad() async {
        await assertConfigurationError(.invalidTemperature(-1)) {
            let profile = AuraProfileCatalog.chat(model: self.missingLocalGGUF,
                                                  sampling: SamplingParams(temperature: -1))
            _ = try await AuraSession.liveEngine(profile)
        }
        await assertConfigurationError(.invalidTopP(2)) {
            let profile = AuraProfileCatalog.chat(model: self.missingLocalGGUF, sampling: SamplingParams(topP: 2))
            _ = try await AuraSession.liveEngine(profile)
        }
    }

    func testSessionRefusesBadSamplingBeforeBuildingOrSwitching() async throws {
        let counter = BuildCounter()
        let engine = StubEngine()
        let make: @MainActor (AuraProfile) async throws -> any AuraProfileEngine = { _ in
            counter.builds += 1
            return engine
        }
        await assertConfigurationError(.invalidTemperature(-1)) {
            _ = try await AuraSession(profile: AuraProfileCatalog.chat(model: self.gguf,
                                                                      sampling: SamplingParams(temperature: -1)),
                                      makeEngine: make)
        }
        XCTAssertEqual(counter.builds, 0)

        let session = try await AuraSession(profile: AuraProfileCatalog.chat(model: gguf), makeEngine: make)
        await assertConfigurationError(.invalidTopP(0)) {
            try await session.switchProfile(to: AuraProfileCatalog.chat(model: self.gguf,
                                                                       sampling: SamplingParams(topP: 0)))
        }
        XCTAssertFalse(engine.didTeardown)
        XCTAssertEqual(counter.builds, 1)
    }

    // MARK: - Resolution

    func testNilKeepsEachBackendsPreviousDefault() throws {
        let ggufDefault = try SamplingSettings.resolve(temperature: nil, topP: nil, for: gguf)
        XCTAssertEqual(ggufDefault, SamplingSettings(temperature: 0.7, topP: 0.95))
        XCTAssertEqual(try SamplingSettings.resolve(temperature: nil, topP: nil, for: try mlxModel(.text)),
                       SamplingSettings(temperature: 0.7, topP: 1))
        XCTAssertEqual(try SamplingSettings.resolve(temperature: nil, topP: nil, for: try mlxModel(.vision)),
                       SamplingSettings(temperature: 0.1, topP: 1))
        XCTAssertEqual(try SamplingSettings.resolve(temperature: nil, topP: nil,
                                                    for: try mlxModel(.visionSpecialized)),
                       SamplingSettings(temperature: 0, topP: 1))
        // Explicit defaults resolve to the same cache key as nil.
        XCTAssertEqual(try SamplingSettings.resolve(temperature: 0.7, topP: 0.95, for: gguf), ggufDefault)
        XCTAssertEqual(try SamplingSettings.resolve(temperature: 0.2, topP: nil, for: gguf),
                       SamplingSettings(temperature: 0.2, topP: 0.95))
    }

    func testLlamaBackendsBuildTheirContextWithTheResolvedSampling() throws {
        let greedy = try SamplingSettings.resolve(temperature: 0, topP: 0.5, for: gguf)
        for parameter in [LlamaCppBackend(model: gguf, sampling: greedy).loadParameter(),
                          LayerStreamingBackend(model: gguf, sampling: greedy).loadParameter()] {
            XCTAssertEqual(parameter.temperature, 0)
            XCTAssertEqual(parameter.topP, 0.5)
        }
        let plain = LlamaCppBackend(model: gguf).loadParameter()
        XCTAssertEqual(plain.temperature, 0.7)
        XCTAssertEqual(plain.topP, 0.95)
    }

    func testSamplingThreadsFromAuraLocalThroughTheRouter() throws {
        let greedy = try SamplingSettings.resolve(temperature: 0, topP: nil, for: gguf)
        let llm = AuraLocal(model: gguf, sampling: greedy)
        XCTAssertEqual(llm.sampling, greedy)
        let backend = llm.engine.backend
        let routed = (backend as? LlamaCppBackend)?.sampling ?? (backend as? LayerStreamingBackend)?.sampling
        XCTAssertEqual(routed, greedy)
        XCTAssertEqual(AuraLocal(model: gguf).sampling, SamplingSettings.defaults(for: gguf))
    }

    #if canImport(MLXLLM) && !targetEnvironment(simulator)
    func testMLXTemperatureZeroSelectsTheArgmaxSampler() throws {
        let mlx = try mlxText()
        let greedy = MLXBackend(model: mlx,
                                sampling: try SamplingSettings.resolve(temperature: 0, topP: nil, for: mlx))
        XCTAssertEqual(greedy.parameters.temperature, 0)
        XCTAssertTrue(greedy.parameters.sampler() is ArgMaxSampler)

        let nucleus = MLXBackend(model: mlx,
                                 sampling: try SamplingSettings.resolve(temperature: 1, topP: 0.9, for: mlx))
        XCTAssertEqual(nucleus.parameters.temperature, 1)
        XCTAssertEqual(nucleus.parameters.topP, 0.9)
        XCTAssertTrue(nucleus.parameters.sampler() is TopPSampler)
    }
    #endif

    // MARK: - Helpers

    private func assertConfigurationError(_ expected: ModelConfigurationError,
                                          file: StaticString = #filePath, line: UInt = #line,
                                          _ body: @MainActor () async throws -> Void) async {
        do {
            try await body()
            XCTFail("expected \(expected)", file: file, line: line)
        } catch let error as ModelConfigurationError {
            XCTAssertEqual(error, expected, file: file, line: line)
        } catch {
            XCTFail("expected \(expected), got \(error)", file: file, line: line)
        }
    }
}
