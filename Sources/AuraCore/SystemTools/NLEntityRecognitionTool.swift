import Foundation
import NaturalLanguage

/// On-device named-entity recognition via Apple's **NaturalLanguage** framework: people,
/// places and organizations as exact spans, with no model download. Text without names, or
/// in a language without a name model, returns an empty list.
public struct NLEntityRecognitionTool: SystemTool {
    public let id = "system.nl.entities"
    public let displayName = "Named entities (NaturalLanguage)"
    public let summary = "Find people, places and organizations in text using Apple's NaturalLanguage — no model download, works offline."
    public let category = SystemToolCategory.language

    public init() {}

    public enum EntityKind: String, Sendable, Equatable, CaseIterable {
        case person
        case place
        case organization
    }

    public struct Entity: Sendable, Equatable {
        public let text: String
        public let kind: EntityKind
        /// UTF-16 range in the input, ready for `NSString` / `NSAttributedString`.
        public let range: NSRange
    }

    public func availability() async -> SystemToolAvailability {
        if supports(language: NLLanguage.english.rawValue) {
            return .available
        }
        return .unavailable(reason: "No named-entity model is installed on this device.")
    }

    /// Whether this device has a name model for `language` (a NaturalLanguage code such as `es`).
    public func supports(language: String) -> Bool {
        NLTagger.availableTagSchemes(for: .word, language: NLLanguage(rawValue: language)).contains(.nameType)
    }

    /// Entities in reading order. Multi-word names are joined ("Tim Cook", "Banco Santander").
    /// `language` skips automatic detection when the caller already knows it.
    public func entities(in text: String, language: String? = nil) -> [Entity] {
        guard !text.isEmpty else { return [] }
        let tagger = NLTagger(tagSchemes: [.nameType])
        tagger.string = text
        let whole = text.startIndex..<text.endIndex
        if let language {
            tagger.setLanguage(NLLanguage(rawValue: language), range: whole)
        }
        return tagger
            .tags(in: whole, unit: .word, scheme: .nameType,
                  options: [.omitPunctuation, .omitWhitespace, .joinNames])
            .compactMap { Self.makeEntity(tag: $0.0, range: $0.1, in: text) }
    }

    private static func makeEntity(tag: NLTag?, range: Range<String.Index>, in text: String) -> Entity? {
        guard let tag, let kind = entityKind(for: tag) else { return nil }
        return Entity(text: String(text[range]), kind: kind, range: NSRange(range, in: text))
    }

    private static func entityKind(for tag: NLTag) -> EntityKind? {
        switch tag {
        case .personalName: .person
        case .placeName: .place
        case .organizationName: .organization
        default: nil
        }
    }
}
