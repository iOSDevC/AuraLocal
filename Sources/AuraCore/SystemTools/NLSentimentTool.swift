import Foundation
import NaturalLanguage

/// On-device sentiment scoring via Apple's **NaturalLanguage** framework: a score from
/// -1 (negative) to 1 (positive) with no model download. For a language without a model the
/// framework answers 0.0, reported here as `nil` so it is never mistaken for "neutral".
public struct NLSentimentTool: SystemTool {
    public let id = "system.nl.sentiment"
    public let displayName = "Sentiment (NaturalLanguage)"
    public let summary = "Score how positive or negative a text is (-1…1) using Apple's NaturalLanguage — no model download, works offline."
    public let category = SystemToolCategory.language

    public init() {}

    public func availability() async -> SystemToolAvailability {
        if supports(language: NLLanguage.english.rawValue) {
            return .available
        }
        return .unavailable(reason: "No sentiment model is installed on this device.")
    }

    /// Whether this device has a sentiment model for `language` (a NaturalLanguage code such as `es`).
    public func supports(language: String) -> Bool {
        NLTagger.availableTagSchemes(for: .paragraph, language: NLLanguage(rawValue: language))
            .contains(.sentimentScore)
    }

    /// Sentiment of `text` in -1…1: each paragraph is scored, then averaged weighted by length.
    /// Returns `nil` when the language is unsupported or undetermined, or the text is blank.
    /// A wrong `language` hint skews the score, so pass one only when it is known.
    public func score(_ text: String, language: String? = nil) -> Double? {
        guard let code = language ?? NLLanguageRecognizer.dominantLanguage(for: text)?.rawValue,
              supports(language: code) else {
            return nil
        }
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = text
        let whole = text.startIndex..<text.endIndex
        tagger.setLanguage(NLLanguage(rawValue: code), range: whole)

        let paragraphs = tagger
            .tags(in: whole, unit: .paragraph, scheme: .sentimentScore)
            .compactMap { Self.weightedScore(tag: $0.0, range: $0.1, in: text) }
        let totalWeight = paragraphs.reduce(0) { $0 + $1.weight }
        guard totalWeight > 0 else { return nil }
        return paragraphs.reduce(0) { $0 + $1.value * $1.weight } / totalWeight
    }

    private static func weightedScore(
        tag: NLTag?, range: Range<String.Index>, in text: String
    ) -> (value: Double, weight: Double)? {
        let paragraph = text[range]
        // Blank separator paragraphs still get a (negative) score; they carry no sentiment.
        guard !paragraph.allSatisfy(\.isWhitespace),
              let raw = tag?.rawValue, let value = Double(raw) else {
            return nil
        }
        return (value, Double(paragraph.utf16.count))
    }
}
