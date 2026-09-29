import Foundation
import NaturalLanguage

/// On-device language identification via Apple's **NaturalLanguage** framework.
/// Tells the SLM (or a router) which language a text is in before choosing a
/// prompt, a voice or a model — no model download. Resilient: text with no
/// identifiable language returns a `nil` dominant language and no hypotheses.
public struct NLLanguageIdentificationTool: SystemTool {
    public let id = "system.nl.language"
    public let displayName = "Language identification (NaturalLanguage)"
    public let summary = "Detect the language of a text with probabilities using Apple's NaturalLanguage — no model download, works offline."
    public let category = SystemToolCategory.language

    public init() {}

    /// One candidate language with its probability (0…1).
    public struct Hypothesis: Sendable, Equatable {
        /// BCP-47 code as NaturalLanguage names it, e.g. `es`, `en`, `zh-Hans`.
        public let language: String
        public let probability: Double
    }

    public struct Identification: Sendable, Equatable {
        /// Most likely language, or `nil` when none can be determined (empty text, digits only…).
        public let dominantLanguage: String?
        /// Sorted by probability, highest first.
        public let hypotheses: [Hypothesis]
    }

    public func availability() async -> SystemToolAvailability {
        // NLLanguageRecognizer ships on every OS version AuraCore targets.
        .available
    }

    /// Identify the language of `text`.
    /// - Parameters:
    ///   - maxHypotheses: upper bound on returned candidates; `0` or less returns none.
    ///   - constraints: restrict the answer to these codes (e.g. `["es", "en"]`). They must be
    ///     NaturalLanguage codes (`en`, not `en-US`); unknown codes match nothing.
    public func identify(
        _ text: String,
        maxHypotheses: Int = 3,
        constraints: [String] = []
    ) -> Identification {
        let recognizer = NLLanguageRecognizer()
        if !constraints.isEmpty {
            recognizer.languageConstraints = constraints.map(NLLanguage.init(rawValue:))
        }
        recognizer.processString(text)

        // languageHypotheses(withMaximum: 0) returns every candidate, not none.
        let raw = maxHypotheses > 0 ? recognizer.languageHypotheses(withMaximum: maxHypotheses) : [:]
        // Under constraints the framework pads the list with excluded languages at 0.
        let hypotheses = raw
            .filter { $0.value > 0 }
            .map { Hypothesis(language: $0.key.rawValue, probability: $0.value) }
            .sorted {
                $0.probability != $1.probability ? $0.probability > $1.probability : $0.language < $1.language
            }
        return Identification(
            dominantLanguage: recognizer.dominantLanguage?.rawValue,
            hypotheses: hypotheses)
    }
}
