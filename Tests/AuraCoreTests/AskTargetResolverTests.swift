import XCTest
@testable import AuraCore

/// `AskTargetResolver` with an injected key reader: no test
/// here touches the real Keychain or the network.
final class AskTargetResolverTests: XCTestCase {

    private typealias Failure = AskTargetResolver.Failure

    // MARK: - Fixtures

    private func status(
        _ kind: LocalProviderKind, models: [String], reachable: Bool = true, contextLength: Int? = nil
    ) -> LocalProviderStatus {
        LocalProviderStatus(
            kind: kind,
            baseURL: kind == .ollama ? LocalProviderEndpoint.ollamaDefault.baseURL : LocalProviderEndpoint.llamaDefault.baseURL,
            reachable: reachable, version: nil,
            models: models.map { LocalProviderModel(name: $0, contextLength: contextLength) })
    }

    private let allCloudKeys = ["OPENAI_API_KEY": "env-openai", "ANTHROPIC_API_KEY": "env-anthropic"]

    /// Records which Keychain accounts were read.
    private final class FakeKeychain {
        var requested: [String] = []
        let keys: [String: String]
        init(_ keys: [String: String]) { self.keys = keys }
        func read(_ account: String) -> String? {
            requested.append(account)
            return keys[account]
        }
    }

    private func everyAccountKeychain() -> FakeKeychain {
        FakeKeychain([
            "cloud.openai": "kc-openai", "cloud.anthropic": "kc-anthropic",
            "cloud.github-models": "kc-ghm",
        ])
    }

    private func resolve(
        _ choice: AskTargetResolver.Choice, model: String? = nil, baseURL: String? = nil,
        environment: [String: String] = [:], keychain: FakeKeychain = FakeKeychain([:]),
        local: [LocalProviderStatus] = []
    ) throws(Failure) -> RemoteTarget {
        try AskTargetResolver.resolve(
            choice, model: model, baseURL: baseURL, environment: environment,
            readKey: keychain.read, localProviders: local)
    }

    private func failure(_ body: () throws(Failure) -> RemoteTarget) -> Failure? {
        do {
            _ = try body()
            return nil
        } catch {
            return error
        }
    }

    /// The API key a provider was built with (stored privately).
    private func storedKey(_ provider: any RemoteLLMProvider) -> String? {
        Mirror(reflecting: provider).children.first { $0.label == "apiKey" }?.value as? String
    }

    /// The URL a target's chat request is posted to.
    private func requestURL(_ target: RemoteTarget) throws -> String? {
        let provider = try XCTUnwrap(target.provider as? OpenAICompatibleProvider)
        let request = RemoteRequest(model: target.modelID, messages: [["role": "user", "content": "hi"]])
        return try provider.makeRequest(request).url?.absoluteString
    }

    // MARK: - auto / local

    func testAutoNeverPicksCloudEvenWithKeys() {
        let keychain = everyAccountKeychain()
        for choice in [AskTargetResolver.Choice.auto, .local] {
            XCTAssertEqual(
                failure { () throws(Failure) in try resolve(choice, environment: allCloudKeys, keychain: keychain) },
                .noLocalProvider)
        }
        XCTAssertTrue(keychain.requested.isEmpty, "auto/local must not even read cloud keys")
    }

    func testAutoWithNoRunningLocalProviderFails() {
        let down = [status(.ollama, models: [], reachable: false), status(.llamaServer, models: [], reachable: false)]
        let error = failure { () throws(Failure) in try resolve(.auto, local: down) }
        XCTAssertEqual(error, .noLocalProvider)
        XCTAssertEqual(error?.isUsageError, false)
        XCTAssertTrue(error?.errorDescription?.contains("--provider openai") == true)
    }

    func testAutoPrefersLlamaServerOverOllama() throws {
        let target = try resolve(.auto, local: [
            status(.ollama, models: ["llama3:latest"]),
            status(.llamaServer, models: ["qwen3-32b"], contextLength: 32_768),
        ])
        XCTAssertEqual(target.origin, .localNetwork(.llamaServer))
        XCTAssertEqual(target.provider.id, "local.llamaServer")
        XCTAssertEqual(target.modelID, "qwen3-32b")
        XCTAssertEqual(target.contextLength, 32_768)
        XCTAssertEqual(try requestURL(target), "http://127.0.0.1:8080/v1/chat/completions")
    }

    func testAutoFallsBackToOllama() throws {
        let target = try resolve(.local, local: [
            status(.ollama, models: ["llama3:latest"]),
            status(.llamaServer, models: [], reachable: false),
        ])
        XCTAssertEqual(target.origin, .localNetwork(.ollama))
        XCTAssertEqual(target.modelID, "llama3:latest")
        XCTAssertEqual(try requestURL(target), "http://localhost:11434/v1/chat/completions")
    }

    func testAutoMatchesBestLocalTarget() throws {
        let statuses = [status(.ollama, models: ["a", "b"]), status(.llamaServer, models: ["c"])]
        let expected = try XCTUnwrap(HybridEscalator.bestLocalTarget(from: statuses))
        let target = try resolve(.auto, local: statuses)
        XCTAssertEqual(target.provider.id, expected.provider.id)
        XCTAssertEqual(target.modelID, expected.modelID)
        XCTAssertEqual(target.origin, expected.origin)
    }

    func testLocalModelPicksTheProviderServingIt() throws {
        let target = try resolve(.auto, model: "llama3", local: [
            status(.llamaServer, models: ["qwen3-32b"]),
            status(.ollama, models: ["llama3:latest"]),
        ])
        XCTAssertEqual(target.origin, .localNetwork(.ollama))
        XCTAssertEqual(target.modelID, "llama3:latest")
        XCTAssertEqual(try requestURL(target), "http://localhost:11434/v1/chat/completions")
    }

    func testLocalModelNotServedListsAvailableModels() {
        let error = failure { () throws(Failure) in
            try resolve(.local, model: "gpt-4o", local: [status(.llamaServer, models: ["qwen3-32b"])])
        }
        XCTAssertEqual(error, .modelNotServedLocally(model: "gpt-4o", available: ["qwen3-32b"]))
    }

    // MARK: - Ollama cloud models (forwarded off the machine)

    private func ollama(local: [String], cloud: [String]) -> LocalProviderStatus {
        LocalProviderStatus(
            kind: .ollama, baseURL: LocalProviderEndpoint.ollamaDefault.baseURL,
            reachable: true, version: "0.34.4",
            models: local.map { LocalProviderModel(name: $0) }
                + cloud.map { LocalProviderModel(name: $0, remoteHost: "https://ollama.com:443") })
    }

    func testAutoSkipsOllamaCloudModels() {
        let onlyCloud = [ollama(local: [], cloud: ["deepseek-v4-pro:cloud"])]
        XCTAssertNil(HybridEscalator.bestLocalTarget(from: onlyCloud))
        XCTAssertEqual(failure { () throws(Failure) in try resolve(.auto, local: onlyCloud) }, .noLocalProvider)
        XCTAssertEqual(
            failure { () throws(Failure) in try resolve(.local, model: "deepseek-v4-pro:cloud", local: onlyCloud) },
            .noLocalProvider)
    }

    func testAutoPicksTheOnMachineModelNextToACloudOne() throws {
        let mixed = [ollama(local: ["llama3:latest"], cloud: ["deepseek-v4-pro:cloud"])]
        XCTAssertEqual(try resolve(.auto, local: mixed).modelID, "llama3:latest")
        XCTAssertEqual(
            failure { () throws(Failure) in try resolve(.auto, model: "deepseek-v4-pro:cloud", local: mixed) },
            .modelNotServedLocally(model: "deepseek-v4-pro:cloud", available: ["llama3:latest"]))
    }

    func testRunsLocally() {
        XCTAssertTrue(LocalProviderModel(name: "llama3:latest").runsLocally)
        XCTAssertTrue(LocalProviderModel(name: "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M").runsLocally)
        XCTAssertTrue(LocalProviderModel(name: "qwen3-cloud-edition").runsLocally)
        XCTAssertFalse(LocalProviderModel(name: "llama3:latest", remoteHost: "https://ollama.com:443").runsLocally)
        XCTAssertFalse(LocalProviderModel(name: "deepseek-v4-pro:cloud").runsLocally)
        XCTAssertFalse(LocalProviderModel(name: "gpt-oss:120b-cloud").runsLocally)
    }

    // MARK: - Named cloud providers

    func testOpenAIKeyFromEnvironmentWinsOverKeychain() throws {
        let keychain = everyAccountKeychain()
        let target = try resolve(.openAI, environment: allCloudKeys, keychain: keychain)
        XCTAssertEqual(storedKey(target.provider), "env-openai")
        XCTAssertTrue(keychain.requested.isEmpty)
        XCTAssertEqual(target.origin, .cloud)
    }

    func testAnthropicKeyFromEnvironmentWinsOverKeychain() throws {
        let target = try resolve(.anthropic, environment: allCloudKeys, keychain: everyAccountKeychain())
        XCTAssertEqual(storedKey(target.provider), "env-anthropic")
        XCTAssertEqual(target.provider.id, "cloud.anthropic")
    }

    func testKeychainIsTheFallback() throws {
        let keychain = everyAccountKeychain()
        let openAI = try resolve(.openAI, environment: ["OPENAI_API_KEY": "  "], keychain: keychain)
        let anthropic = try resolve(.anthropic, keychain: keychain)
        XCTAssertEqual(storedKey(openAI.provider), "kc-openai")
        XCTAssertEqual(storedKey(anthropic.provider), "kc-anthropic")
        XCTAssertEqual(keychain.requested, ["cloud.openai", "cloud.anthropic"])
    }

    func testMissingKeyNamesTheVariableAndAccount() {
        let openAI = failure { () throws(Failure) in try resolve(.openAI) }
        XCTAssertEqual(openAI, .missingAPIKey(
            provider: .openAI, environmentVariable: "OPENAI_API_KEY", keychainAccount: "cloud.openai"))
        XCTAssertEqual(openAI?.isUsageError, false)
        let anthropic = failure { () throws(Failure) in try resolve(.anthropic) }
        XCTAssertEqual(anthropic, .missingAPIKey(
            provider: .anthropic, environmentVariable: "ANTHROPIC_API_KEY", keychainAccount: "cloud.anthropic"))
    }

    func testExplicitModelOverridesTheCloudDefault() throws {
        let target = try resolve(.openAI, model: "gpt-4.1", environment: allCloudKeys)
        XCTAssertEqual(target.modelID, "gpt-4.1")
    }

    // MARK: - Base URL

    func testBaseURLOnThePrivateNetworkIsLocal() throws {
        let keychain = everyAccountKeychain()
        let target = try resolve(
            .auto, model: "qwen", baseURL: "http://192.168.1.20:8080/v1",
            environment: ["AURA_API_KEY": "lan-key"], keychain: keychain)
        XCTAssertEqual(target.origin, .localNetwork(.llamaServer))
        XCTAssertTrue(target.isLocalNetwork)
        XCTAssertEqual(target.provider.id, "custom.192.168.1.20:8080")
        XCTAssertEqual(try requestURL(target), "http://192.168.1.20:8080/v1/chat/completions")
        XCTAssertEqual(storedKey(target.provider), "lan-key")
        XCTAssertTrue(keychain.requested.isEmpty)
    }

    func testBaseURLOnAPublicHostIsCloud() throws {
        let target = try resolve(.auto, model: "m", baseURL: "https://llm.example.com/v1")
        XCTAssertEqual(target.origin, .cloud)
        XCTAssertNil(storedKey(target.provider))
    }

    func testLocalChoiceRejectsAPublicBaseURL() {
        let error = failure { () throws(Failure) in
            try resolve(.local, model: "m", baseURL: "https://llm.example.com/v1")
        }
        guard case .conflictingOptions = error else { return XCTFail("got \(error as Any)") }
        XCTAssertEqual(error?.isUsageError, true)
    }

    func testBaseURLAndCloudProviderAreMutuallyExclusive() {
        let keychain = everyAccountKeychain()
        for choice in [AskTargetResolver.Choice.openAI, .anthropic] {
            let error = failure { () throws(Failure) in
                try resolve(choice, model: "m", baseURL: "http://127.0.0.1:8080/v1",
                            environment: allCloudKeys, keychain: keychain)
            }
            guard case .conflictingOptions = error else { return XCTFail("got \(error as Any)") }
            XCTAssertEqual(error?.isUsageError, true)
        }
        XCTAssertTrue(keychain.requested.isEmpty)
    }

    func testBaseURLNeedsAModel() {
        let error = failure { () throws(Failure) in try resolve(.auto, baseURL: "http://127.0.0.1:8080/v1") }
        XCTAssertEqual(error, .modelRequired)
        XCTAssertEqual(error?.isUsageError, true)
    }

    func testInvalidBaseURLs() {
        for value in ["ftp://host/v1", "127.0.0.1:8080", "http://", "not a url"] {
            let error = failure { () throws(Failure) in try resolve(.auto, model: "m", baseURL: value) }
            guard case .invalidBaseURL(let rejected) = error else { return XCTFail("\(value): \(error as Any)") }
            XCTAssertEqual(rejected, value)
            XCTAssertEqual(error?.isUsageError, true)
        }
    }

    func testLocalNetworkHostClassification() {
        let local = [
            "localhost", "127.0.0.1", "127.8.9.10", "0.0.0.0", "::1", "[::1]",
            "10.0.0.5", "172.16.0.1", "172.31.255.254", "192.168.1.20", "169.254.3.4",
            "fd12:3456::1", "fe80::1", "fe80::1%en0", "::ffff:192.168.1.2",
            "mac-studio.local", "box.home.arpa", "app.localhost", "localhost.", "mac-studio.local.",
        ]
        let publicHosts = [
            "8.8.8.8", "172.32.0.1", "172.15.0.1", "100.64.0.1", "192.169.0.1",
            "api.openai.com", "llm.example.com", "colibri", "2001:4860:4860::8888", "::ffff:8.8.8.8",
            "local.example.com",
        ]
        for host in local {
            XCTAssertTrue(AskTargetResolver.isLocalNetworkHost(host), host)
        }
        for host in publicHosts {
            XCTAssertFalse(AskTargetResolver.isLocalNetworkHost(host), host)
        }
    }
}
