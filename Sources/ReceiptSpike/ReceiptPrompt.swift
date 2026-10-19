import Foundation
import AuraCore

/// What the model sees: the receipt's OCR text and Data Detection's candidates. The app reads the image; the model
/// only assigns the text it is given to fields.
public enum ReceiptPrompt {

    public static let instructions = """
        You read the text of one shopping receipt and fill in its fields. Copy values from the text and never invent \
        them; use null for a field the text does not show.
        - merchant: the store or business name as printed.
        - date: the purchase date as YYYY-MM-DD.
        - total: the final amount paid, as a number without currency symbols or thousands separators.
        - tax: the total tax (IVA, VAT, sales tax) as a number, or null.
        - currency: the ISO 4217 code such as MXN, USD or EUR, or null.
        """

    /// Every field is required and may be null, so a reply always names all five.
    public static let schema = """
        {
          "type": "object",
          "properties": {
            "merchant": {"type": ["string", "null"], "minLength": 1},
            "date": {"type": ["string", "null"], "minLength": 10, "maxLength": 10},
            "total": {"type": ["number", "null"], "minimum": 0},
            "tax": {"type": ["number", "null"], "minimum": 0},
            "currency": {"type": ["string", "null"], "minLength": 3, "maxLength": 3}
          },
          "required": ["merchant", "date", "total", "tax", "currency"],
          "additionalProperties": false
        }
        """

    /// Greedy (temperature 0) and short: the reply is one small JSON object.
    public static func profile(model: Model) -> AuraProfile {
        AuraProfile(id: "receipt-spike", displayName: "Receipt fields", instructions: instructions, model: model,
                    sampling: SamplingParams(temperature: 0, topP: 1, maxTokens: 256), outputSchema: .json(schema))
    }

    /// The receipt's lines, then the amounts and dates Data Detection found in them.
    public static func prompt(for document: VisionDocumentTool.Document, timeZone: TimeZone = .current) -> String {
        var sections = ["Receipt text, line by line:\n" + ReceiptHeuristic.lines(of: document).joined(separator: "\n")]
        let amounts = document.amounts.compactMap { match -> String? in
            guard let amount = match.amount else { return nil }
            let value = "\(amount)" + (match.currency.map { " \($0)" } ?? "")
            return match.text.isEmpty ? value : "\(match.text) (\(value))"
        }
        if !amounts.isEmpty {
            sections.append("Amounts found in the text: " + amounts.joined(separator: ", "))
        }
        let dates = document.detections.compactMap { match -> String? in
            guard match.kind == .date, let date = match.startDate else { return nil }
            let day = ReceiptText.isoDate(date, in: timeZone)
            return match.text.isEmpty ? day : "\(match.text) (\(day))"
        }
        if !dates.isEmpty {
            sections.append("Dates found in the text: " + dates.joined(separator: ", "))
        }
        sections.append("Return the JSON object for this receipt.")
        return sections.joined(separator: "\n\n")
    }
}

/// Whether each extracted value appears in the receipt itself, so an app can send an unanchored field to review
/// instead of trusting it. `nil` for a field that was not extracted.
public struct ReceiptAnchors: Sendable, Equatable {
    public var merchant: Bool?
    public var date: Bool?
    public var total: Bool?
    public var tax: Bool?

    public init(merchant: Bool? = nil, date: Bool? = nil, total: Bool? = nil, tax: Bool? = nil) {
        self.merchant = merchant
        self.date = date
        self.total = total
        self.tax = tax
    }

    /// - merchant: its letters and digits occur in the text's.
    /// - date: a detected date has that day, or the text spells it as day/month/year, month/day/year or
    ///   year-month-day with `/`, `-` or `.`.
    /// - total and tax: a detected amount or a number written in the text has that value.
    public static func check(_ fields: ReceiptFields, in document: VisionDocumentTool.Document,
                             timeZone: TimeZone = .current) -> ReceiptAnchors {
        let written = Set(ReceiptText.numbers(in: document.text).map(\.value))
        let detected = Set(document.amounts.compactMap(\.amount))
        func amountAnchored(_ value: Decimal?) -> Bool? {
            value.map { written.contains($0) || detected.contains($0) }
        }
        let textKey = ReceiptText.key(document.text)
        return ReceiptAnchors(
            merchant: fields.merchant.map { name in
                let key = ReceiptText.key(name)
                return !key.isEmpty && textKey.contains(key)
            },
            date: fields.date.map { dateAnchored($0, in: document, timeZone: timeZone) },
            total: amountAnchored(fields.total),
            tax: amountAnchored(fields.tax))
    }

    static func dateAnchored(_ iso: String, in document: VisionDocumentTool.Document, timeZone: TimeZone) -> Bool {
        let detected = document.detections.compactMap { $0.kind == .date ? $0.startDate : nil }
        if detected.contains(where: { ReceiptText.isoDate($0, in: timeZone) == iso }) { return true }
        let parts = iso.split(separator: "-").map(String.init)
        guard parts.count == 3, let month = Int(parts[1]), let day = Int(parts[2]) else { return false }
        let year = parts[0]
        let shortYear = String(year.suffix(2))
        let days = Set([parts[2], String(day)])
        let months = Set([parts[1], String(month)])
        var spellings: [String] = []
        for separator in ["/", "-", "."] {
            for dd in days {
                for mm in months {
                    for yy in [year, shortYear] {
                        spellings.append([dd, mm, yy].joined(separator: separator))
                        spellings.append([mm, dd, yy].joined(separator: separator))
                    }
                }
            }
            spellings.append([year, parts[1], parts[2]].joined(separator: separator))
        }
        let text = document.text
        return spellings.contains { spelling in
            var searchRange = text.startIndex..<text.endIndex
            while let found = text.range(of: spelling, range: searchRange) {
                let before = found.lowerBound > text.startIndex ? text[text.index(before: found.lowerBound)] : " "
                let after = found.upperBound < text.endIndex ? text[found.upperBound] : " "
                if !before.isNumber && !after.isNumber { return true }
                searchRange = found.upperBound..<text.endIndex
            }
            return false
        }
    }
}
