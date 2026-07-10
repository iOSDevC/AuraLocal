import Foundation
import Combine

/// Records token usage and cost for every remote escalation (the one idea worth
/// borrowing from llm-lang: cost telemetry, in dollars). Local-network targets
/// cost `$0`. Cloud pricing is filled in Phase 2.
@MainActor
public final class CostLedger: ObservableObject {
    public static let shared = CostLedger()
    public init() {}

    public struct Record: Identifiable, Sendable {
        public let id = UUID()
        public let provider: String
        public let usage: TokenUsage?
        public let costUSD: Decimal
        /// compressed / original token ratio (nil when no compression ran).
        public let compressionRatio: Double?
    }

    @Published public private(set) var records: [Record] = []

    public var sessionTokens: Int {
        records.reduce(0) { $0 + ($1.usage?.totalTokens ?? 0) }
    }
    public var sessionCostUSD: Decimal {
        records.reduce(Decimal(0)) { $0 + $1.costUSD }
    }

    public func record(provider: String, usage: TokenUsage?, origin: RemoteTarget.Origin, compressionRatio: Double?) {
        let cost = Self.cost(usage: usage, origin: origin, provider: provider)
        records.append(Record(provider: provider, usage: usage, costUSD: cost, compressionRatio: compressionRatio))
    }

    // MARK: - Pricing

    struct Price { let input: Decimal; let output: Decimal }  // USD per 1M tokens

    /// Cloud price table — populated in Phase 2. Local targets are always free.
    static let priceTable: [String: Price] = [:]

    static func cost(usage: TokenUsage?, origin: RemoteTarget.Origin, provider: String) -> Decimal {
        if case .localNetwork = origin { return 0 }
        guard let usage, let price = priceTable[provider] else { return 0 }
        let million = Decimal(1_000_000)
        return Decimal(usage.inputTokens) / million * price.input
             + Decimal(usage.outputTokens) / million * price.output
    }
}
