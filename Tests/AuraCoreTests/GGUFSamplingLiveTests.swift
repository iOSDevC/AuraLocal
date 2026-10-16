import XCTest
@testable import AuraCore

/// Needs AURA_GGUF_TEST_MODEL. A public temperature reaches llama.cpp's sampler, and 0 repeats its output.
@MainActor
final class GGUFSamplingLiveTests: XCTestCase {

    private static let modelVariable = "AURA_GGUF_TEST_MODEL"
    private let prompt = "Describe the sea in two sentences."

    func testTemperatureZeroRepeatsItselfAndEachTemperatureGetsItsOwnInstance() async throws {
        let model = try localModel()
        let manager = ModelManager.shared
        let greedy = try await loaded(model, temperature: 0)
        XCTAssertEqual(greedy.sampling, SamplingSettings(temperature: 0, topP: 0.95))

        let first = try await greedy.chat(prompt, maxTokens: 48)
        let second = try await greedy.chat(prompt, maxTokens: 48)
        XCTAssertFalse(first.isEmpty)
        XCTAssertEqual(first, second)
        let cached = try await manager.load(model, temperature: 0)
        XCTAssertTrue(cached === greedy, "the same setting must be served from the cache")

        // A fresh context at 0 decodes the same tokens.
        manager.evict(model)
        await (greedy.engine.backend as? LlamaCppBackend)?.awaitRelease()
        let reloaded = try await loaded(model, temperature: 0)
        XCTAssertFalse(reloaded === greedy)
        let afterReload = try await reloaded.chat(prompt, maxTokens: 48)
        XCTAssertEqual(afterReload, first)

        let hot = try await loaded(model, temperature: 1)
        XCTAssertFalse(hot === reloaded, "another temperature must not share the greedy instance")
        XCTAssertEqual(hot.sampling.temperature, 1)
        let hotReply = try await hot.chat(prompt, maxTokens: 48)
        XCTAssertFalse(hotReply.isEmpty)
        if manager.memoryBudget >= 2 {
            let stillGreedy = try await manager.load(model, temperature: 0)
            XCTAssertTrue(stillGreedy === reloaded)
        }
    }

    func testCancelLoadKeepsALoadedModelReady() async throws {
        let model = try localModel()
        let manager = ModelManager.shared
        _ = try await loaded(model, temperature: 0)
        XCTAssertEqual(manager.state(for: model), .ready)

        manager.cancelLoad(model)

        XCTAssertTrue(manager.isReady(model))
        XCTAssertEqual(manager.state(for: model), .ready)
    }

    func testSessionLoadsTheProfileTemperature() async throws {
        let model = try localModel()
        let profile = AuraProfileCatalog.chat(model: model, sampling: SamplingParams(temperature: 0, maxTokens: 48))
        let engine = try await AuraSession.liveEngine(profile)
        let llm = try XCTUnwrap(engine as? AuraLocal)
        track(llm)
        XCTAssertEqual(llm.sampling, SamplingSettings(temperature: 0, topP: 0.95))

        let session = try await AuraSession(profile: profile)
        let first = try await reply(session)
        let second = try await reply(session)
        XCTAssertFalse(first.isEmpty)
        XCTAssertEqual(first, second)
    }

    // MARK: - Helpers

    private func reply(_ session: AuraSession) async throws -> String {
        var text = ""
        for try await delta in session.stream(prompt) { text += delta }
        return text
    }

    private func loaded(_ model: Model, temperature: Float) async throws -> AuraLocal {
        let llm = try await ModelManager.shared.load(model, temperature: temperature)
        track(llm)
        return llm
    }

    /// Evicts every instance of the model and waits for its context to be freed.
    private func track(_ llm: AuraLocal) {
        addTeardownBlock { @MainActor in
            ModelManager.shared.evict(llm.model)
            await (llm.engine.backend as? LlamaCppBackend)?.awaitRelease()
        }
    }

    private func localModel() throws -> Model {
        guard let path = ProcessInfo.processInfo.environment[Self.modelVariable], !path.isEmpty else {
            throw XCTSkip("Set \(Self.modelVariable) to a local .gguf file to run the sampling tests.")
        }
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("\(Self.modelVariable) points at a missing file: \(path)")
        }
        return try XCTUnwrap(Model.from(.localFile(URL(fileURLWithPath: path))))
    }
}
