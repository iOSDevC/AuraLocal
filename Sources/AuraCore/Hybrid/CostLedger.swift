import Foundation
import Combine

/// A model's price in USD per 1M tokens.
public struct TokenPrice: Sendable, Hashable {
    public var inputUSDPerMillion: Decimal
    public var outputUSDPerMillion: Decimal

    public init(inputUSDPerMillion: Decimal, outputUSDPerMillion: Decimal) {
        self.inputUSDPerMillion = inputUSDPerMillion
        self.outputUSDPerMillion = outputUSDPerMillion
    }

    func cost(of usage: TokenUsage) -> Decimal {
        let million = Decimal(1_000_000)
        return Decimal(usage.inputTokens) / million * inputUSDPerMillion
             + Decimal(usage.outputTokens) / million * outputUSDPerMillion
    }
}

/// Records token usage and cost for every remote escalation. Local-network
/// targets cost exactly `$0`. A cloud call is priced from this ledger's table
/// (``setPrice(_:provider:model:)``); without a matching price or reported usage
/// its cost is `nil` — unknown, never assumed free.
@MainActor
public final class CostLedger: ObservableObject {
    public static let shared = CostLedger()

    public init() {
        prices = Self.defaultPrices
    }

    public struct Record: Identifiable, Sendable {
        public let id = UUID()
        public let provider: String
        /// The model ID, when the caller passed one.
        public let model: String?
        /// `true` for the user's own machine (always `$0`).
        public let isLocalNetwork: Bool
        public let usage: TokenUsage?
        /// `nil` when unknown: a cloud call with no matching price or no reported usage.
        public let costUSD: Decimal?
        /// compressed / original token ratio (nil when no compression ran).
        public let compressionRatio: Double?
    }

    @Published public private(set) var records: [Record] = []
    /// Index of the current session's first record; `records` is append-only.
    @Published private var sessionStart = 0

    private var sessionRecords: ArraySlice<Record> { records[sessionStart...] }

    public var sessionTokens: Int {
        sessionRecords.reduce(0) { $0 + ($1.usage?.totalTokens ?? 0) }
    }

    /// Spend of the priced records since the session started. A lower bound
    /// while ``unpricedRecordCount`` is above zero.
    public var sessionCostUSD: Decimal {
        sessionRecords.reduce(Decimal(0)) { $0 + ($1.costUSD ?? 0) }
    }

    /// Records since the session started.
    public var sessionRecordCount: Int { sessionRecords.count }

    /// Records since the session started whose cost is unknown.
    public var unpricedRecordCount: Int {
        sessionRecords.reduce(0) { $0 + ($1.costUSD == nil ? 1 : 0) }
    }

    /// Starts a new cost session: `records` keeps the history, while `sessionTokens`,
    /// `sessionCostUSD`, `unpricedRecordCount` and `sessionRecordCount` restart at zero.
    public func startNewSession() {
        sessionStart = records.count
    }

    public func record(
        provider: String,
        model: String? = nil,
        usage: TokenUsage?,
        origin: RemoteTarget.Origin,
        compressionRatio: Double?
    ) {
        let isLocal = if case .localNetwork = origin { true } else { false }
        records.append(Record(
            provider: provider, model: model, isLocalNetwork: isLocal, usage: usage,
            costUSD: cost(provider: provider, model: model, usage: usage, origin: origin),
            compressionRatio: compressionRatio))
    }

    // MARK: - Pricing

    private struct PriceKey: Hashable, Sendable {
        let provider: String
        let model: String?
    }

    private var prices: [PriceKey: TokenPrice]

    /// Approximate list prices for the default cloud models only; set your plan's
    /// with ``setPrice(_:provider:model:)``. Other models stay unpriced, not guessed.
    private nonisolated static let defaultPrices: [PriceKey: TokenPrice] = [
        PriceKey(provider: CloudAccount.anthropic, model: RemoteTarget.defaultAnthropicModel):
            TokenPrice(inputUSDPerMillion: 3, outputUSDPerMillion: 15),
        PriceKey(provider: CloudAccount.openAI, model: RemoteTarget.defaultOpenAIModel):
            TokenPrice(inputUSDPerMillion: 2.5, outputUSDPerMillion: 10),
    ]

    /// Sets the price for `model` on `provider` (a ``RemoteLLMProvider/id``), or for
    /// every model of `provider` when `model` is nil. A nil `price` removes the entry.
    public func setPrice(_ price: TokenPrice?, provider: String, model: String? = nil) {
        prices[PriceKey(provider: provider, model: model)] = price
    }

    /// The model-specific price, else the provider-wide one, else nil.
    public func price(provider: String, model: String? = nil) -> TokenPrice? {
        if let model, let modelPrice = prices[PriceKey(provider: provider, model: model)] {
            return modelPrice
        }
        return prices[PriceKey(provider: provider, model: nil)]
    }

    func cost(provider: String, model: String?, usage: TokenUsage?, origin: RemoteTarget.Origin) -> Decimal? {
        if case .localNetwork = origin { return 0 }
        guard let usage, let matched = price(provider: provider, model: model) else { return nil }
        return matched.cost(of: usage)
    }

    /// Pre-flight estimate for `target` (input + max output at this ledger's price):
    /// `0` for the local network, `nil` for an unpriced cloud model.
    public func projectedCost(target: RemoteTarget, inputTokens: Int, maxOutput: Int) -> Decimal? {
        cost(provider: target.provider.id, model: target.modelID,
             usage: TokenUsage(inputTokens: inputTokens, outputTokens: maxOutput),
             origin: target.origin)
    }

    @available(*, deprecated, message: "Prices live on the ledger instance: call ledger.projectedCost(target:inputTokens:maxOutput:) on the CostLedger your HybridEscalator uses.")
    public static func projectedCost(target: RemoteTarget, inputTokens: Int, maxOutput: Int) -> Decimal? {
        shared.projectedCost(target: target, inputTokens: inputTokens, maxOutput: maxOutput)
    }
}
