import Foundation
import CoreML
import Accelerate

/// Which side of a retrieval pair a text is. Asymmetric models such as e5 were trained with a
/// different prefix per side; using the wrong one lowers retrieval quality.
public enum TextEmbeddingRole: String, Sendable, CaseIterable {
    /// A search query, prefixed with the manifest's `query_prefix`.
    case query
    /// A document or chunk to be found, prefixed with `passage_prefix`.
    case passage
    /// Text used exactly as given: already prefixed, or a model without prefixes.
    case raw
}

/// One embedded text.
public struct TextEmbedding: Sendable, Equatable {
    public let vector: [Float]
    /// Tokens the model saw, `<s>` and `</s>` included.
    public let tokenCount: Int
    /// True when the text exceeded `max_tokens` and its tail was dropped before embedding.
    public let isTruncated: Bool
}

/// Token ids for one text after prefixing, normalisation and truncation (no padding).
public struct TextEmbeddingTokens: Sendable, Equatable {
    public let ids: [Int]
    /// Length before truncation.
    public let originalCount: Int

    public var isTruncated: Bool { originalCount > ids.count }
}

// MARK: - Manifest

/// `embedding-model.json`, schema `aura.text-embedding/1`: how to run the Core ML encoder in a
/// text-embedding bundle. Unknown keys are ignored.
public struct TextEmbeddingManifest: Codable, Sendable, Equatable {
    public static let schemaName = "aura.text-embedding/1"
    public static let fileName = "embedding-model.json"
    static let supportedPooling = "mean"
    static let modelExtensions: Set<String> = ["mlpackage", "mlmodelc", "mlmodel"]

    public let schema: String
    public let modelID: String
    public let revision: String
    /// File name of the `.mlpackage` / `.mlmodelc` inside the bundle.
    public let modelFile: String
    public let inputName: String
    public let outputName: String
    /// Sequence lengths the model accepts, ascending.
    public let buckets: [Int]
    public let padTokenID: Int
    public let pooling: String
    public let normalize: Bool
    public let dimensions: Int
    public let queryPrefix: String
    public let passagePrefix: String
    /// Longest input in tokens, `<s>` and `</s>` included.
    public let maxTokens: Int
    public let license: String

    enum CodingKeys: String, CodingKey {
        case schema, revision, buckets, pooling, normalize, dimensions, license
        case modelID = "model_id"
        case modelFile = "model_file"
        case inputName = "input_name"
        case outputName = "output_name"
        case padTokenID = "pad_token_id"
        case queryPrefix = "query_prefix"
        case passagePrefix = "passage_prefix"
        case maxTokens = "max_tokens"
    }

    public init(
        schema: String = TextEmbeddingManifest.schemaName,
        modelID: String,
        revision: String,
        modelFile: String,
        inputName: String,
        outputName: String,
        buckets: [Int],
        padTokenID: Int,
        pooling: String = "mean",
        normalize: Bool = true,
        dimensions: Int,
        queryPrefix: String,
        passagePrefix: String,
        maxTokens: Int,
        license: String
    ) {
        self.schema = schema
        self.modelID = modelID
        self.revision = revision
        self.modelFile = modelFile
        self.inputName = inputName
        self.outputName = outputName
        self.buckets = buckets
        self.padTokenID = padTokenID
        self.pooling = pooling
        self.normalize = normalize
        self.dimensions = dimensions
        self.queryPrefix = queryPrefix
        self.passagePrefix = passagePrefix
        self.maxTokens = maxTokens
        self.license = license
    }

    /// Identity of the vectors this model produces: `model_id@revision`.
    public var identifier: String { "\(modelID)@\(revision)" }

    public func prefix(for role: TextEmbeddingRole) -> String {
        switch role {
        case .query: queryPrefix
        case .passage: passagePrefix
        case .raw: ""
        }
    }

    /// Reads and validates `embedding-model.json` from a bundle directory.
    public static func load(fromBundle bundleURL: URL) throws -> TextEmbeddingManifest {
        let url = bundleURL.appendingPathComponent(fileName)
        guard let data = FileManager.default.contents(atPath: url.path) else {
            throw CoreMLTextEmbeddingTool.ToolError.manifestNotFound(url.path)
        }
        return try decode(data)
    }

    /// Decodes and validates manifest JSON.
    public static func decode(_ data: Data) throws -> TextEmbeddingManifest {
        let decoded: TextEmbeddingManifest
        do {
            decoded = try JSONDecoder().decode(TextEmbeddingManifest.self, from: data)
        } catch let error as DecodingError {
            throw CoreMLTextEmbeddingTool.ToolError.invalidManifest(describe(error))
        } catch {
            throw CoreMLTextEmbeddingTool.ToolError.invalidManifest(error.localizedDescription)
        }
        try decoded.validate()
        return decoded
    }

    /// Throws ``CoreMLTextEmbeddingTool/ToolError/invalidManifest(_:)`` naming the first problem.
    public func validate() throws {
        if let problem = firstProblem() {
            throw CoreMLTextEmbeddingTool.ToolError.invalidManifest(problem)
        }
    }

    private func firstProblem() -> String? {
        if schema != Self.schemaName {
            return "unsupported schema “\(schema)” (expected “\(Self.schemaName)”)."
        }
        let required = [("model_id", modelID), ("revision", revision), ("model_file", modelFile),
                        ("input_name", inputName), ("output_name", outputName)]
        if let empty = required.first(where: { $0.1.trimmingCharacters(in: .whitespaces).isEmpty }) {
            return "“\(empty.0)” is empty."
        }
        if modelFile.contains("/") || modelFile.hasPrefix(".") {
            return "“model_file” must be a file name inside the bundle, not “\(modelFile)”."
        }
        let modelExtension = (modelFile as NSString).pathExtension.lowercased()
        if !Self.modelExtensions.contains(modelExtension) {
            return "“model_file” must be a .mlpackage, .mlmodelc or .mlmodel, not “\(modelFile)”."
        }
        if buckets.isEmpty || buckets.contains(where: { $0 <= 0 }) || buckets != buckets.sorted()
            || Set(buckets).count != buckets.count {
            return "“buckets” must be positive sequence lengths in ascending order, got \(buckets)."
        }
        if padTokenID < 0 {
            return "“pad_token_id” must not be negative."
        }
        if pooling != Self.supportedPooling {
            return "unsupported pooling “\(pooling)” (only “\(Self.supportedPooling)”)."
        }
        if dimensions <= 0 {
            return "“dimensions” must be positive."
        }
        if let longest = buckets.last, maxTokens < 2 || maxTokens > longest {
            return "“max_tokens” must be between 2 and the largest bucket (\(longest)), got \(maxTokens)."
        }
        return nil
    }

    private static func describe(_ error: DecodingError) -> String {
        switch error {
        case .keyNotFound(let key, _):
            "missing “\(key.stringValue)”."
        case .typeMismatch(_, let context), .valueNotFound(_, let context):
            "“\(context.codingPath.map(\.stringValue).joined(separator: "."))” has the wrong type."
        case .dataCorrupted(let context):
            "not valid JSON (\(context.debugDescription))."
        @unknown default:
            error.localizedDescription
        }
    }
}

// MARK: - Pipeline steps

/// The pure steps around the Core ML call, kept separate so they are testable without a model.
enum TextEmbeddingPipeline {
    /// Zero-width code points that swift-transformers' approximate normalizer keeps but Python's drops.
    static let ignoredScalars: Set<Unicode.Scalar> = ["\u{200B}", "\u{200C}", "\u{200D}", "\u{2060}", "\u{FEFF}"]

    /// NFC plus zero-width removal, which keeps Swift token ids in line with the Python tokenizer.
    static func preprocess(_ text: String) -> String {
        let composed = text.precomposedStringWithCanonicalMapping
        guard composed.unicodeScalars.contains(where: { ignoredScalars.contains($0) }) else { return composed }
        return String(String.UnicodeScalarView(composed.unicodeScalars.filter { !ignoredScalars.contains($0) }))
    }

    /// The smallest bucket that holds `tokenCount` tokens, or `nil` if none does.
    static func bucket(forTokenCount tokenCount: Int, in buckets: [Int]) -> Int? {
        buckets.first { $0 >= tokenCount }
    }

    /// Cuts `ids` to `maxTokens`, keeping the final token (`</s>`) so the sequence stays closed.
    static func truncate(_ ids: [Int], maxTokens: Int) -> [Int] {
        guard ids.count > maxTokens, maxTokens > 0, let closing = ids.last else { return ids }
        return Array(ids.prefix(maxTokens - 1)) + [closing]
    }

    /// Right-pads `ids` with `padTokenID` to `length`.
    static func pad(_ ids: [Int], to length: Int, padTokenID: Int) -> [Int32] {
        ids.prefix(length).map { Int32(truncatingIfNeeded: $0) }
            + [Int32](repeating: Int32(truncatingIfNeeded: padTokenID), count: max(0, length - ids.count))
    }

    /// A `[1, n]` Int32 array holding `ids`.
    static func makeInputArray(_ ids: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        array.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, strides in
            let step = strides.last ?? 1
            for (position, id) in ids.enumerated() {
                buffer[position * step] = id
            }
        }
        return array
    }

    /// Mean of the token states at non-padding positions, then L2-normalised when `normalize`.
    /// `hiddenStates` is `[1, S, D]` or `[S, D]` with any strides and element type.
    static func meanPool(
        _ hiddenStates: MLMultiArray,
        tokenIDs: [Int32],
        padTokenID: Int32,
        normalize: Bool
    ) throws -> [Float] {
        let shape = hiddenStates.shape.map(\.intValue)
        guard shape.count == 2 || (shape.count == 3 && shape[0] == 1) else {
            throw CoreMLTextEmbeddingTool.ToolError.unexpectedOutput("token states of shape \(shape)")
        }
        let length = shape[shape.count - 2]
        let width = shape[shape.count - 1]
        let kept = tokenIDs.prefix(length).indices.filter { tokenIDs[$0] != padTokenID }

        let sum: [Float]
        if hiddenStates.dataType == .float32 {
            let strides = hiddenStates.strides.map(\.intValue)
            sum = hiddenStates.withUnsafeBufferPointer(ofType: Float.self) { buffer in
                addRows(kept, of: buffer, rowStride: strides[strides.count - 2],
                        columnStride: strides[strides.count - 1], width: width)
            }
        } else {
            let converted = MLShapedArray<Float>(converting: hiddenStates)
            sum = converted.withUnsafeShapedBufferPointer { buffer, _, strides in
                addRows(kept, of: buffer, rowStride: strides[strides.count - 2],
                        columnStride: strides[strides.count - 1], width: width)
            }
        }
        guard !kept.isEmpty else { return sum }
        let mean = vDSP.divide(sum, Float(kept.count))
        return normalize ? l2Normalized(mean) : mean
    }

    static func l2Normalized(_ vector: [Float]) -> [Float] {
        let norm = vDSP.sumOfSquares(vector).squareRoot()
        return norm > 0 ? vDSP.divide(vector, norm) : vector
    }

    private static func addRows(
        _ rows: [Int],
        of buffer: UnsafeBufferPointer<Float>,
        rowStride: Int,
        columnStride: Int,
        width: Int
    ) -> [Float] {
        var sum = [Float](repeating: 0, count: width)
        guard let base = buffer.baseAddress else { return sum }
        let count = vDSP_Length(width)
        let columnStep = vDSP_Stride(columnStride)
        sum.withUnsafeMutableBufferPointer { accumulator in
            guard let total = accumulator.baseAddress else { return }
            for row in rows {
                vDSP_vadd(total, 1, base + row * rowStride, columnStep, total, 1, count)
            }
        }
        return sum
    }
}
