import Foundation

/// One hand-labelled receipt: the image's file name and the fields printed on it.
public struct ReceiptLabel: Sendable, Equatable {
    public let file: String
    public let fields: ReceiptFields

    public init(file: String, fields: ReceiptFields) {
        self.file = file
        self.fields = fields
    }
}

public enum ReceiptLabelError: LocalizedError, Equatable {
    case missingColumn(String)
    case badValue(line: Int, column: String, value: String)

    public var errorDescription: String? {
        switch self {
        case .missingColumn(let name):
            "The labels CSV has no \(name) column (needed: file, merchant, date, total; optional: tax, currency)."
        case .badValue(let line, let column, let value):
            "Line \(line): \(column) \"\(value)\" is not valid (dates are YYYY-MM-DD, amounts use a dot: 1234.50)."
        }
    }
}

public enum ReceiptLabels {

    /// Parse a CSV with a header row naming `file`, `merchant`, `date` and `total`, and optionally `tax` and
    /// `currency`. Fields may be quoted (`"OXXO, S.A."`, `""` for a quote). An empty cell means the receipt does not
    /// show the field.
    public static func parse(_ csv: String) throws -> [ReceiptLabel] {
        var rows = records(in: csv)
        guard !rows.isEmpty else { return [] }
        let header = rows.removeFirst().map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
        func column(_ name: String) throws -> Int {
            guard let index = header.firstIndex(of: name) else { throw ReceiptLabelError.missingColumn(name) }
            return index
        }
        let file = try column("file"), merchant = try column("merchant"), date = try column("date")
        let total = try column("total")
        let tax = header.firstIndex(of: "tax"), currency = header.firstIndex(of: "currency")

        var labels: [ReceiptLabel] = []
        for (offset, row) in rows.enumerated() where !row.allSatisfy({ $0.trimmingCharacters(in: .whitespaces).isEmpty }) {
            let line = offset + 2
            func cell(_ index: Int?) -> String? {
                guard let index, index < row.count else { return nil }
                let value = row[index].trimmingCharacters(in: .whitespaces)
                return value.isEmpty ? nil : value
            }
            func amount(_ index: Int?, _ name: String) throws -> Decimal? {
                guard let raw = cell(index) else { return nil }
                guard let value = Decimal(string: raw, locale: Locale(identifier: "en_US_POSIX")),
                      raw.allSatisfy({ $0.isNumber || $0 == "." }) else {
                    throw ReceiptLabelError.badValue(line: line, column: name, value: raw)
                }
                return value
            }
            guard let name = cell(file) else { throw ReceiptLabelError.badValue(line: line, column: "file", value: "") }
            let day = cell(date)
            if let day, !isISODate(day) { throw ReceiptLabelError.badValue(line: line, column: "date", value: day) }
            labels.append(ReceiptLabel(file: name, fields: ReceiptFields(
                merchant: cell(merchant), date: day, total: try amount(total, "total"), tax: try amount(tax, "tax"),
                currency: cell(currency)?.uppercased())))
        }
        return labels
    }

    static func isISODate(_ text: String) -> Bool {
        let parts = text.split(separator: "-", omittingEmptySubsequences: false)
        return parts.count == 3 && parts.map(\.count) == [4, 2, 2] && parts.allSatisfy { $0.allSatisfy(\.isNumber) }
    }

    /// RFC 4180-style records: commas separate fields, quotes allow commas, newlines and `""` inside a field.
    static func records(in csv: String) -> [[String]] {
        var records: [[String]] = []
        var record: [String] = []
        var field = ""
        var quoted = false
        func consume(_ character: Character) {
            switch character {
            case "\"": quoted = true
            case ",": record.append(field); field = ""
            case "\n", "\r\n", "\r": record.append(field); field = ""; records.append(record); record = []
            default: field.append(character)
            }
        }
        var iterator = csv.makeIterator()
        while let character = iterator.next() {
            guard quoted else {
                consume(character)
                continue
            }
            guard character == "\"" else {
                field.append(character)
                continue
            }
            // A quote inside a quoted field: `""` is a literal quote, anything else ends the field.
            let next = iterator.next()
            if next == "\"" {
                field.append("\"")
            } else {
                quoted = false
                if let next { consume(next) }
            }
        }
        if !field.isEmpty || !record.isEmpty {
            record.append(field)
            records.append(record)
        }
        return records
    }
}

/// Per-field accuracy over a set of receipts, the spike's go/no-go measure.
public struct ReceiptScorecard: Sendable, Equatable {
    public private(set) var receipts = 0
    public private(set) var correct: [ReceiptField: Int] = [:]

    public init() {}

    /// Score one receipt and return which fields matched its label.
    @discardableResult
    public mutating func add(_ predicted: ReceiptFields, label: ReceiptFields) -> [ReceiptField: Bool] {
        receipts += 1
        var result: [ReceiptField: Bool] = [:]
        for field in ReceiptField.allCases {
            let match = Self.matches(field, predicted, label)
            result[field] = match
            if match { correct[field, default: 0] += 1 }
        }
        return result
    }

    /// 0…1; 0 before any receipt.
    public func accuracy(_ field: ReceiptField) -> Double {
        receipts == 0 ? 0 : Double(correct[field, default: 0]) / Double(receipts)
    }

    /// The fields whose accuracy is below their threshold. Fields without a threshold always pass.
    public func failures(against thresholds: [ReceiptField: Double]) -> [ReceiptField] {
        ReceiptField.allCases.filter { field in thresholds[field].map { accuracy(field) < $0 } ?? false }
    }

    /// A field the receipt does not show matches only an empty prediction. Amounts compare by value, dates as
    /// `YYYY-MM-DD`, and merchants by their letters and digits, where either name may contain the other
    /// ("OXXO" for "OXXO Tienda 123").
    static func matches(_ field: ReceiptField, _ predicted: ReceiptFields, _ label: ReceiptFields) -> Bool {
        switch field {
        case .total: return predicted.total == label.total
        case .tax: return predicted.tax == label.tax
        case .date: return predicted.date == label.date
        case .merchant:
            switch (predicted.merchant.map(ReceiptText.key), label.merchant.map(ReceiptText.key)) {
            case (nil, nil): return true
            case let (guess?, truth?): return !guess.isEmpty && !truth.isEmpty && (guess.contains(truth) || truth.contains(guess))
            default: return false
            }
        }
    }
}
