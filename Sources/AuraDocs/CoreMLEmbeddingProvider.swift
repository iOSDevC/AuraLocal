import Foundation
import CoreML
import AuraCore

/// ``EmbeddingProvider`` backed by ``CoreMLTextEmbeddingTool``: dense multilingual vectors (e.g.
/// multilingual-e5-small) computed on-device. Queries get the model's query prefix, documents
/// its passage prefix; ``embed(_:)`` and ``embedBatch(_:)`` treat text as documents.
public struct CoreMLEmbeddingProvider: EmbeddingProvider, TruncationReporting {
    public let tool: CoreMLTextEmbeddingTool
    public let dimensions: Int
    /// `model_id@revision` from the bundle's manifest.
    public let identifier: String

    /// Throws when the bundle is missing or invalid, so a provider always has a valid manifest.
    /// Loading the model is deferred to ``warmUp()`` or the first call.
    public init(
        bundleAt url: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        compiledModelsDirectory: URL? = nil
    ) throws {
        let manifest = try CoreMLTextEmbeddingTool.validateBundle(at: url)
        tool = CoreMLTextEmbeddingTool(bundleAt: url, computeUnits: computeUnits,
                                       compiledModelsDirectory: compiledModelsDirectory)
        dimensions = manifest.dimensions
        identifier = manifest.identifier
    }

    /// The model's short name, e.g. `multilingual-e5-small`.
    public var modelName: String {
        tool.manifest.map { ($0.modelID as NSString).lastPathComponent } ?? tool.bundleURL.lastPathComponent
    }

    public func embed(_ text: String) async throws -> [Float] {
        try await tool.embed(text, role: .passage).vector
    }

    public func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        try await tool.embed(texts, role: .passage).map(\.vector)
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        try await tool.embed(text, role: .query).vector
    }

    /// See ``CoreMLTextEmbeddingTool/warmUp()``.
    @discardableResult
    public func warmUp() async throws -> Duration {
        try await tool.warmUp()
    }

    public func truncatedInputCount() async -> Int {
        await tool.truncatedInputCount()
    }
}
