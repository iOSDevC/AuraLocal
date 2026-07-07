import XCTest
@testable import AuraCore

final class LocalProviderDetectionTests: XCTestCase {

    func testDefaultEndpoints() {
        XCTAssertEqual(LocalProviderEndpoint.ollamaDefault.baseURL.absoluteString, "http://localhost:11434")
        XCTAssertEqual(LocalProviderEndpoint.llamaDefault.baseURL.absoluteString, "http://127.0.0.1:8080/v1")
        XCTAssertEqual(LocalProviderEndpoint.defaults.count, 2)
        XCTAssertEqual(LocalProviderEndpoint.ollamaDefault.kind, .ollama)
        XCTAssertEqual(LocalProviderEndpoint.llamaDefault.kind, .llamaServer)
    }

    func testIsAvailableFlag() {
        let url = URL(string: "http://localhost:11434")!
        let withModel = LocalProviderStatus(
            kind: .ollama, baseURL: url, reachable: true,
            version: "1.0", models: [LocalProviderModel(name: "llama3:latest")])
        XCTAssertTrue(withModel.isAvailable)

        let upNoModels = LocalProviderStatus(
            kind: .ollama, baseURL: url, reachable: true, version: "1.0", models: [])
        XCTAssertFalse(upNoModels.isAvailable)

        let down = LocalProviderStatus(
            kind: .ollama, baseURL: url, reachable: false, version: nil, models: [])
        XCTAssertFalse(down.isAvailable)
    }

    @MainActor
    func testDetectDeadPortIsGraceful() async {
        // Port 1 is closed → connection refused → graceful reachable:false, no throw.
        let dead = LocalProviderEndpoint(kind: .llamaServer, baseURL: URL(string: "http://127.0.0.1:1/v1")!)
        let status = await LocalProviderDetector.detect(dead, timeout: 0.5)
        XCTAssertFalse(status.reachable)
        XCTAssertTrue(status.models.isEmpty)
        XCTAssertNil(status.version)
        XCTAssertEqual(status.kind, .llamaServer)
    }

    @MainActor
    func testDetectAllPreservesOrderAndNeverThrows() async {
        let a = LocalProviderEndpoint(kind: .ollama, baseURL: URL(string: "http://127.0.0.1:1")!)
        let b = LocalProviderEndpoint(kind: .llamaServer, baseURL: URL(string: "http://127.0.0.1:2/v1")!)
        let statuses = await LocalProviderDetector.detectAll(endpoints: [a, b], timeout: 0.5)
        XCTAssertEqual(statuses.count, 2)
        XCTAssertEqual(statuses[0].kind, .ollama)   // input order preserved
        XCTAssertEqual(statuses[1].kind, .llamaServer)
        XCTAssertTrue(statuses.allSatisfy { !$0.reachable })
    }

    @MainActor
    func testFacadeAvailableFiltersUnreachable() async {
        let dead = [
            LocalProviderEndpoint(kind: .ollama, baseURL: URL(string: "http://127.0.0.1:1")!),
            LocalProviderEndpoint(kind: .llamaServer, baseURL: URL(string: "http://127.0.0.1:2/v1")!),
        ]
        let available = await AuraLocal.availableLocalProviders(endpoints: dead, timeout: 0.5)
        XCTAssertTrue(available.isEmpty)
    }

    /// Live check against the real machine — opt-in via AURA_LIVE_PROVIDER_TESTS=1.
    @MainActor
    func testLiveOllamaDetection() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["AURA_LIVE_PROVIDER_TESTS"] == "1",
                          "Set AURA_LIVE_PROVIDER_TESTS=1 to run against a live Ollama server.")
        let status = await LocalProviderDetector.detect(.ollamaDefault, timeout: 3.0)
        XCTAssertTrue(status.reachable)
        XCTAssertNotNil(status.version)
    }

    /// Live check against a running llama.cpp `llama-server` — opt-in via AURA_LIVE_PROVIDER_TESTS=1.
    @MainActor
    func testLiveLlamaServerDetection() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["AURA_LIVE_PROVIDER_TESTS"] == "1",
                          "Set AURA_LIVE_PROVIDER_TESTS=1 to run against a live llama-server.")
        let status = await LocalProviderDetector.detect(.llamaDefault, timeout: 3.0)
        XCTAssertTrue(status.reachable)
        XCTAssertFalse(status.models.isEmpty)
        XCTAssertNil(status.version)   // llama-server exposes no version
    }
}
