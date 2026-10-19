import Foundation

/// A number as written on a receipt. `hasFraction` tells "12.50" (an amount) from "2" or "2026".
struct ReceiptNumber: Equatable {
    let value: Decimal
    let hasFraction: Bool
}

/// Text helpers shared by the heuristic, the anchors and the scoring.
enum ReceiptText {

    /// Lowercased, without diacritics: "IMPORTE TÓTAL" and "importe total" compare equal.
    static func folded(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
    }

    /// Letters and digits only, folded, for comparing merchant names ("OXXO, S.A." → "oxxosa").
    static func key(_ text: String) -> String {
        String(folded(text).unicodeScalars.filter(CharacterSet.alphanumerics.contains))
    }

    /// Whether `text` contains one of `phrases` as whole words: "iva" matches "IVA 16%" but not "activa".
    static func containsWords(_ text: String, _ phrases: [String]) -> Bool {
        let words = folded(text).split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        let padded = " " + words.joined(separator: " ") + " "
        return phrases.contains { padded.contains(" \($0) ") }
    }

    /// Numbers written in `text`, decimal or not: "1,234.50", "1.234,50", "12,75", "2026".
    static func numbers(in text: String) -> [ReceiptNumber] {
        let range = NSRange(text.startIndex..., in: text)
        return numberPattern.matches(in: text, range: range).compactMap { match in
            Range(match.range, in: text).flatMap { decimal(fromToken: String(text[$0])) }
        }
    }

    /// A number token's value. With both separators the last one is the decimal point; with one, it is the
    /// decimal point when one or two digits follow it ("12,75") and a thousands separator otherwise ("1.234").
    static func decimal(fromToken token: String) -> ReceiptNumber? {
        let lastDot = token.lastIndex(of: ".")
        let lastComma = token.lastIndex(of: ",")
        var point: String.Index?
        switch (lastDot, lastComma) {
        case let (dot?, comma?): point = max(dot, comma)
        case let (dot?, nil): point = token[token.index(after: dot)...].count <= 2 ? dot : nil
        case let (nil, comma?): point = token[token.index(after: comma)...].count <= 2 ? comma : nil
        case (nil, nil): point = nil
        }
        var plain = ""
        for index in token.indices {
            if index == point {
                plain.append(".")
            } else if token[index].isNumber {
                plain.append(token[index])
            }
        }
        guard let value = Decimal(string: plain, locale: posix) else { return nil }
        return ReceiptNumber(value: value, hasFraction: point != nil)
    }

    /// `YYYY-MM-DD` for `date` in `timeZone`.
    static func isoDate(_ date: Date, in timeZone: TimeZone) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        let day = calendar.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04ld-%02ld-%02ld", day.year ?? 0, day.month ?? 0, day.day ?? 0)
    }

    /// Parses the dot-decimal strings the spike builds and the labels CSV uses, whatever the user's locale.
    static let posix = Locale(identifier: "en_US_POSIX")

    /// Thousands-grouped numbers first, so "1,234.50" is one token, not "1,23" and "4.50".
    private static let numberPattern = try! NSRegularExpression(
        pattern: #"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?(?!\d)|\d+(?:[.,]\d{1,2})?(?!\d)"#)
}
