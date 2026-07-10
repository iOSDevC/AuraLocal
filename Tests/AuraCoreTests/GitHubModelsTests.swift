import XCTest
@testable import AuraCore

final class GitHubModelsTests: XCTestCase {

    // MARK: - Provider factory

    func testGitHubModelsFactoryIdentity() {
        let provider = OpenAICompatibleProvider.gitHubModels(apiKey: "ghp_test")
        XCTAssertEqual(provider.id, "cloud.github-models")
        XCTAssertEqual(provider.displayName, "GitHub Models")
        XCTAssertFalse(provider.retentionNote.isEmpty)
    }

    // MARK: - Cost: free tier prices at $0 by default (no priceTable row needed)

    @MainActor
    func testGitHubModelsPricesAtZero() {
        let cost = CostLedger.cost(
            usage: TokenUsage(inputTokens: 10_000, outputTokens: 10_000),
            origin: .cloud, provider: "cloud.github-models")
        XCTAssertEqual(cost, 0)
    }

    // MARK: - cloudTargets wiring (tolerant of test-host Keychain restrictions)

    @MainActor
    func testCloudTargetsIncludeGitHubModelsWhenKeyPresent() throws {
        let account = "cloud.github-models"
        do {
            try KeychainStore.save("ghp_scratch_key", for: account)
        } catch {
            throw XCTSkip("Keychain unavailable in this test host: \(error)")
        }
        defer { KeychainStore.delete(for: account) }

        let targets = HybridEscalator.cloudTargets(allowCloud: true)
        guard let ghm = targets.first(where: { $0.provider.id == "cloud.github-models" }) else {
            return XCTFail("GitHub Models target missing when a key is configured")
        }
        XCTAssertTrue(ghm.modelID.hasPrefix("openai/"))
        XCTAssertEqual(ghm.contextLength, 128_000)
        if case .cloud = ghm.origin {} else {
            XCTFail("GitHub Models must be a .cloud origin (data leaves the device)")
        }
        XCTAssertFalse(ghm.isLocalNetwork)
    }

    @MainActor
    func testCloudTargetsEmptyWhenCloudDisallowed() {
        XCTAssertTrue(HybridEscalator.cloudTargets(allowCloud: false).isEmpty)
    }
}
