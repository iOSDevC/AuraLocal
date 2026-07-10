import Foundation
import NaturalLanguage

/// On-device text embeddings via Apple's **NaturalLanguage** framework — a
/// semantic vector for a sentence with no model download. Gives the Library's RAG
/// a dense-embedding option that ships with the OS. Resilient: unsupported
/// languages return `nil` rather than failing.
public struct NLEmbeddingTool: SystemTool {
    public let id = "system.nl.embedding"
    public let displayName = "On-device embeddings (NaturalLanguage)"
    public let summary = "Produce a semantic vector for text using Apple's NaturalLanguage — no model download, for offline semantic search."

    public init() {}

    public func availability() async -> SystemToolAvailability {
        // Sentence embeddings exist for a subset of languages; require English at minimum.
        if NLEmbedding.sentenceEmbedding(for: .english) != nil {
            return .available
        }
        return .unavailable(reason: "No sentence embedding is installed for a supported language.")
    }

    /// The vector dimension for `language`, or `nil` if unsupported.
    public func dimension(for language: NLLanguage = .english) -> Int? {
        NLEmbedding.sentenceEmbedding(for: language)?.dimension
    }

    /// Embed `text`. Returns `nil` (not an error) when the language has no
    /// installed embedding — the caller falls back to another provider.
    public func embed(_ text: String, language: NLLanguage = .english) -> [Float]? {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: language),
              let vector = embedding.vector(for: text) else {
            return nil
        }
        return vector.map { Float($0) }
    }

    /// Cosine distance between two strings (0 = identical), or `nil` if unsupported.
    public func distance(_ a: String, _ b: String, language: NLLanguage = .english) -> Double? {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: language) else { return nil }
        return embedding.distance(between: a, and: b)
    }
}
