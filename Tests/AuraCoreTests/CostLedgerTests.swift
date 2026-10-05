import XCTest
@testable import AuraCore

@MainActor
final class CostLedgerTests: XCTestCase {

    private let oneMillion = TokenUsage(inputTokens: 1_000_000, outputTokens: 1_000_000)

    // MARK: - Price lookup

    func testDefaultsPriceOnlyTheDefaultModels() {
        let ledger = CostLedger()
        XCTAssertEqual(
            ledger.price(provider: "cloud.anthropic", model: RemoteTarget.defaultAnthropicModel),
            TokenPrice(inputUSDPerMillion: 3, outputUSDPerMillion: 15))
        XCTAssertEqual(
            ledger.price(provider: "cloud.openai", model: RemoteTarget.defaultOpenAIModel),
            TokenPrice(inputUSDPerMillion: 2.5, outputUSDPerMillion: 10))
        XCTAssertNil(ledger.price(provider: "cloud.anthropic", model: "claude-opus-4-1"))
        XCTAssertNil(ledger.price(provider: "cloud.openai"))
    }

    func testModelPriceWinsOverProviderWidePrice() {
        let ledger = CostLedger()
        let providerWide = TokenPrice(inputUSDPerMillion: 1, outputUSDPerMillion: 2)
        let modelSpecific = TokenPrice(inputUSDPerMillion: 5, outputUSDPerMillion: 6)
        ledger.setPrice(providerWide, provider: "custom")
        ledger.setPrice(modelSpecific, provider: "custom", model: "big")

        XCTAssertEqual(ledger.price(provider: "custom", model: "big"), modelSpecific)
        XCTAssertEqual(ledger.price(provider: "custom", model: "small"), providerWide)
        XCTAssertEqual(ledger.price(provider: "custom"), providerWide)
        XCTAssertNil(ledger.price(provider: "other", model: "big"))
    }

    func testSettingNilRemovesOnlyThatEntry() {
        let ledger = CostLedger()
        let providerWide = TokenPrice(inputUSDPerMillion: 1, outputUSDPerMillion: 2)
        ledger.setPrice(providerWide, provider: "custom")
        ledger.setPrice(TokenPrice(inputUSDPerMillion: 5, outputUSDPerMillion: 6), provider: "custom", model: "big")

        ledger.setPrice(nil, provider: "custom", model: "big")
        XCTAssertEqual(ledger.price(provider: "custom", model: "big"), providerWide)
        ledger.setPrice(nil, provider: "custom")
        XCTAssertNil(ledger.price(provider: "custom", model: "big"))
    }

    func testPricesArePerInstance() {
        let priced = CostLedger()
        priced.setPrice(TokenPrice(inputUSDPerMillion: 1, outputUSDPerMillion: 1), provider: "custom")
        XCTAssertNil(CostLedger().price(provider: "custom"))
    }

    // MARK: - Recorded cost

    func testPricedCloudRecordUsesTheModelPrice() {
        let ledger = CostLedger()
        ledger.record(
            provider: "cloud.anthropic", model: RemoteTarget.defaultAnthropicModel,
            usage: TokenUsage(inputTokens: 1000, outputTokens: 1000), origin: .cloud, compressionRatio: nil)
        XCTAssertEqual(ledger.records.last?.costUSD, Decimal(string: "0.018"))
        XCTAssertEqual(ledger.records.last?.model, RemoteTarget.defaultAnthropicModel)
        XCTAssertEqual(ledger.records.last?.isLocalNetwork, false)
        XCTAssertEqual(ledger.sessionCostUSD, Decimal(string: "0.018"))
        XCTAssertEqual(ledger.unpricedRecordCount, 0)
    }

    func testUnpricedCloudModelIsUnknownNotFree() {
        let ledger = CostLedger()
        ledger.record(provider: "custom", model: "m", usage: oneMillion, origin: .cloud, compressionRatio: nil)
        ledger.record(provider: "cloud.anthropic", model: "claude-opus-4-1", usage: oneMillion,
                      origin: .cloud, compressionRatio: nil)
        XCTAssertEqual(ledger.records.map(\.costUSD), [nil, nil])
        XCTAssertEqual(ledger.unpricedRecordCount, 2)
        XCTAssertEqual(ledger.sessionCostUSD, 0)
    }

    func testMissingUsageIsUnknownEvenWhenPriced() {
        let ledger = CostLedger()
        ledger.record(provider: "cloud.openai", model: RemoteTarget.defaultOpenAIModel,
                      usage: nil, origin: .cloud, compressionRatio: nil)
        XCTAssertNil(ledger.records.last?.costUSD)
        XCTAssertEqual(ledger.unpricedRecordCount, 1)
    }

    func testRecordWithoutModelFallsBackToProviderWidePrice() {
        let ledger = CostLedger()
        ledger.setPrice(TokenPrice(inputUSDPerMillion: 1, outputUSDPerMillion: 1), provider: "custom")
        ledger.record(provider: "custom", usage: oneMillion, origin: .cloud, compressionRatio: nil)
        XCTAssertEqual(ledger.records.last?.costUSD, 2)
        XCTAssertNil(ledger.records.last?.model)
    }

    func testLocalNetworkIsExactlyFree() {
        let ledger = CostLedger()
        ledger.record(provider: "llama-server", model: "qwen", usage: oneMillion,
                      origin: .localNetwork(.llamaServer), compressionRatio: nil)
        ledger.record(provider: "ollama", usage: nil, origin: .localNetwork(.ollama), compressionRatio: nil)
        XCTAssertEqual(ledger.records.map(\.costUSD), [0, 0])
        XCTAssertEqual(ledger.records.map(\.isLocalNetwork), [true, true])
        XCTAssertEqual(ledger.unpricedRecordCount, 0)
    }

    // MARK: - Projection

    func testProjectedCostUsesTheInstanceTable() {
        let ledger = CostLedger()
        let target = RemoteTarget(provider: StubProvider(), modelID: "m", origin: .cloud)
        XCTAssertNil(ledger.projectedCost(target: target, inputTokens: 1_000_000, maxOutput: 1_000_000))

        ledger.setPrice(TokenPrice(inputUSDPerMillion: 1, outputUSDPerMillion: 4), provider: "test.stub", model: "m")
        XCTAssertEqual(ledger.projectedCost(target: target, inputTokens: 1_000_000, maxOutput: 500_000), 3)
    }

    func testProjectedCostForLocalNetworkIsZero() {
        let target = RemoteTarget(provider: StubProvider(), modelID: "m", origin: .localNetwork(.ollama))
        XCTAssertEqual(CostLedger().projectedCost(target: target, inputTokens: 10_000, maxOutput: 1024), 0)
    }

    // MARK: - Session boundary

    func testNewSessionKeepsHistoryAndRestartsTotals() {
        let ledger = CostLedger()
        ledger.setPrice(TokenPrice(inputUSDPerMillion: 1, outputUSDPerMillion: 1), provider: "custom", model: "m")
        ledger.record(provider: "custom", model: "m", usage: oneMillion, origin: .cloud, compressionRatio: nil)
        ledger.record(provider: "unpriced", usage: oneMillion, origin: .cloud, compressionRatio: nil)
        XCTAssertEqual(ledger.sessionCostUSD, 2)
        XCTAssertEqual(ledger.unpricedRecordCount, 1)
        XCTAssertEqual(ledger.sessionTokens, 4_000_000)
        XCTAssertEqual(ledger.sessionRecordCount, 2)

        ledger.startNewSession()
        XCTAssertEqual(ledger.records.count, 2)
        XCTAssertEqual(ledger.sessionCostUSD, 0)
        XCTAssertEqual(ledger.unpricedRecordCount, 0)
        XCTAssertEqual(ledger.sessionTokens, 0)
        XCTAssertEqual(ledger.sessionRecordCount, 0)

        ledger.record(provider: "custom", model: "m",
                      usage: TokenUsage(inputTokens: 500_000, outputTokens: 0), origin: .cloud, compressionRatio: nil)
        XCTAssertEqual(ledger.sessionCostUSD, Decimal(string: "0.5"))
        XCTAssertEqual(ledger.records.count, 3)
        XCTAssertEqual(ledger.sessionRecordCount, 1)
    }

    // MARK: - Provider usage

    func testOpenAIUsageMissingACountIsNotReported() throws {
        let partial = OpenAICompatibleProvider.parse(line: #"data: {"choices":[],"usage":{"prompt_tokens":12}}"#)
        XCTAssertTrue(partial.isEmpty, "a missing count is unknown, not 0")

        let complete = OpenAICompatibleProvider.parse(line: #"data: {"usage":{"prompt_tokens":12,"completion_tokens":3}}"#)
        guard case .usage(let usage)? = complete.first else { return XCTFail("expected a usage event, got \(complete)") }
        XCTAssertEqual(usage, TokenUsage(inputTokens: 12, outputTokens: 3))
    }
}

private struct StubProvider: RemoteLLMProvider {
    let id = "test.stub"
    let displayName = "Stub"
    let retentionNote = "test"

    func stream(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}
