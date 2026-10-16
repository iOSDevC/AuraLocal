import Foundation

/// The fields the Recibo spike extracts from one receipt. `nil` means the receipt does not show the field.
public struct ReceiptFields: Sendable, Equatable, Codable {
    public var merchant: String?
    /// The purchase date as `YYYY-MM-DD`.
    public var date: String?
    /// The final amount paid.
    public var total: Decimal?
    /// The total tax (IVA, VAT, sales tax).
    public var tax: Decimal?
    /// ISO 4217 code (`MXN`, `USD`, `EUR`…).
    public var currency: String?

    public init(merchant: String? = nil, date: String? = nil, total: Decimal? = nil, tax: Decimal? = nil,
                currency: String? = nil) {
        self.merchant = merchant
        self.date = date
        self.total = total
        self.tax = tax
        self.currency = currency
    }

    /// Decode the JSON object a model returned under ``ReceiptPrompt/schema``.
    public static func decode(_ json: String) throws -> ReceiptFields {
        try JSONDecoder().decode(ReceiptFields.self, from: Data(json.utf8))
    }
}

/// One field the spike scores.
public enum ReceiptField: String, Sendable, CaseIterable, Codable {
    case merchant, date, total, tax
}
