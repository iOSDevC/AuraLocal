import Foundation
import TonamoxCore

/// The spike's baseline: receipt fields read from a document by rule, with no model. A model has to beat it to be
/// worth its memory on a 6 GB iPhone.
public enum ReceiptHeuristic {

    static let totalWords = ["total", "importe", "a pagar", "monto", "amount due", "balance due", "grand total"]
    static let subtotalWords = ["subtotal", "sub total", "sub-total"]
    static let taxWords = ["iva", "i v a", "tax", "taxes", "impuesto", "impuestos", "vat", "igv", "ieps"]
    static let dateWords = ["fecha", "date"]

    /// - total: the last amount on the last line naming a total (not a subtotal or a tax), else the largest
    ///   amount with cents on the receipt.
    /// - tax: the last amount on the first line naming a tax.
    /// - date: the first date Data Detection found, as `YYYY-MM-DD` in `timeZone`.
    /// - currency: that of the detected amount equal to the total, else of the first detected amount.
    /// - merchant: the document's title, else the first line with letters that holds no amount or date and names no
    ///   total or tax.
    public static func extract(from document: VisionDocumentTool.Document,
                               timeZone: TimeZone = .current) -> ReceiptFields {
        let lines = Self.lines(of: document)
        let totalLine = lines.last {
            ReceiptText.containsWords($0, totalWords) && !ReceiptText.containsWords($0, subtotalWords)
                && !ReceiptText.containsWords($0, taxWords)
        }
        let total = totalLine.flatMap(lastAmount(on:)) ?? largestAmount(in: lines)
        let tax = lines.first { ReceiptText.containsWords($0, taxWords) }.flatMap(lastAmount(on:))
        let amounts = document.amounts
        let currency = amounts.first { total != nil && $0.amount == total }?.currency ?? amounts.first?.currency
        let date = document.detections.first { $0.kind == .date }?.startDate.map { ReceiptText.isoDate($0, in: timeZone) }
        return ReceiptFields(merchant: merchant(in: document, lines: lines), date: date, total: total, tax: tax,
                             currency: currency)
    }

    /// The recognized lines, or the text split into lines when Vision gave none.
    static func lines(of document: VisionDocumentTool.Document) -> [String] {
        let lines = document.lines.map(\.text)
        return lines.isEmpty ? document.text.components(separatedBy: .newlines) : lines
    }

    /// Receipts print the value last on its line; prefer numbers with cents over quantities and percentages.
    static func lastAmount(on line: String) -> Decimal? {
        let numbers = ReceiptText.numbers(in: line)
        return (numbers.last { $0.hasFraction } ?? numbers.last)?.value
    }

    static func largestAmount(in lines: [String]) -> Decimal? {
        lines.flatMap { ReceiptText.numbers(in: $0).filter(\.hasFraction).map(\.value) }.max()
    }

    static func merchant(in document: VisionDocumentTool.Document, lines: [String]) -> String? {
        if let title = document.title?.trimmingCharacters(in: .whitespacesAndNewlines), !title.isEmpty {
            return title
        }
        let dates = document.detections.filter { $0.kind == .date && !$0.text.isEmpty }.map(\.text)
        return lines.first { line in
            line.contains(where: \.isLetter) && ReceiptText.numbers(in: line).allSatisfy { !$0.hasFraction }
                && !ReceiptText.containsWords(line, totalWords + subtotalWords + taxWords + dateWords)
                && !dates.contains { line.contains($0) }
        }?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
