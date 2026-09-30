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

    /// What an index should do before it is searched or extended with `configured`.
    public enum Reconciliation: Sendable, Equatable {
        /// The stored vectors came from the configured model.
        case upToDate
        /// Record the configured identity without touching vectors: the index is empty, or it
        /// predates identity tracking and its vectors already have the configured length.
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
        if stored == nil, storedVectorLengths == [configured.dimensions] { return .adopt }
        return .reembed
    }
}
