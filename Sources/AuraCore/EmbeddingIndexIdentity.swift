import Foundation

/// Which embedding model produced the vectors in a stored index. Vectors from different models
/// (or of different lengths) are not comparable, so an index records this and is re-embedded
/// when the configured model changes.
public struct EmbeddingIndexIdentity: Sendable, Equatable {
    public let identifier: String
    public let dimensions: Int

    public init(identifier: String, dimensions: Int) {
        self.identifier = identifier
        self.dimensions = dimensions
    }

    /// The only provider indexes were built with before identity tracking: AuraDocs' TF-IDF.
    public static let preTrackingIdentifier = "aura.tfidf-hash/4096"

    /// What an index should do before it is searched or extended with `configured`.
    public enum Reconciliation: Sendable, Equatable {
        /// The stored vectors came from the configured model.
        case upToDate
        /// Record the configured identity without touching vectors: the index is empty, or it
        /// predates identity tracking, the configured model is TF-IDF and the vector lengths match.
        case adopt
        /// Recompute every stored vector with the configured model.
        case reembed
    }

    /// - Parameters:
    ///   - stored: the identity recorded in the index; `nil` for an index that predates tracking.
    ///   - storedVectorLengths: the distinct vector lengths (in floats) found in the index.
    ///   - storedVectorCount: how many vectors the index holds.
    public static func reconciliation(
        stored: EmbeddingIndexIdentity?,
        configured: EmbeddingIndexIdentity,
        storedVectorLengths: Set<Int>,
        storedVectorCount: Int
    ) -> Reconciliation {
        if stored == configured { return .upToDate }
        if storedVectorCount == 0 { return .adopt }
        // Another model of the same width would otherwise take over vectors it did not produce.
        if stored == nil, configured.identifier == preTrackingIdentifier,
           storedVectorLengths == [configured.dimensions] { return .adopt }
        return .reembed
    }
}
