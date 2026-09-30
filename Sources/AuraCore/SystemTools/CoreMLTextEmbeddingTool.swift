import Foundation
import CoreML
import Tokenizers

/// Dense sentence embeddings from a Core ML encoder (e.g. multilingual-e5-small) shipped as a
/// **bundle directory**: `embedding-model.json` (``TextEmbeddingManifest``), the `.mlpackage` or
/// `.mlmodelc`, and the Hugging Face `tokenizer.json` + `tokenizer_config.json`.
///
/// Text is prefixed per ``TextEmbeddingRole``, tokenized, cut to `max_tokens` (keeping `</s>`),
/// padded to the smallest bucket, run through the model and mean-pooled over non-padding tokens.
/// Call ``warmUp()`` before reporting the tool as ready: the first Neural Engine load compiles
/// the model on-device, which can take tens of seconds.
public struct CoreMLTextEmbeddingTool: SystemTool {
    public static let requiredTokenizerFiles = ["tokenizer.json", "tokenizer_config.json"]

    public let id: String
    public let displayName: String
    public let summary: String
    public let category = SystemToolCategory.customModel
    public let bundleURL: URL
    public let computeUnits: MLComputeUnits
    /// The bundle's manifest as read at init, or `nil` when missing or invalid (see ``availability()``).
    public let manifest: TextEmbeddingManifest?

    private let manifestProblem: ToolError?
    private let host: EmbeddingHost

    /// - Parameter compiledModelsDirectory: where the compiled model is cached;
    ///   `nil` uses `Caches/AuraLocal/CompiledModels`.
    public init(
        bundleAt url: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        compiledModelsDirectory: URL? = nil
    ) {
        let name = url.lastPathComponent
        self.id = "coreml.text-embedding.\(name)"
        self.displayName = "Text embeddings “\(name)”"
        self.summary = "Turn text into semantic vectors with the Core ML model “\(name)” on-device — no network."
        self.bundleURL = url
        self.computeUnits = computeUnits
        do {
            self.manifest = try TextEmbeddingManifest.load(fromBundle: url)
            self.manifestProblem = nil
        } catch {
            self.manifest = nil
            self.manifestProblem = FileManager.default.fileExists(atPath: url.path)
                ? error as? ToolError ?? .invalidManifest(error.localizedDescription)
                : .bundleNotFound(url.path)
        }
        self.host = EmbeddingHost(bundleURL: url, computeUnits: computeUnits, cacheDirectory: compiledModelsDirectory)
    }

    public enum ToolError: LocalizedError, Equatable {
        case bundleNotFound(String)
        case manifestNotFound(String)
        case invalidManifest(String)
        case modelNotFound(String)
        case tokenizerNotFound(String)
        case tokenizerLoadFailed(String)
        case modelLoadFailed(String)
        case predictionFailed(String)
        case unexpectedOutput(String)

        public var errorDescription: String? {
            switch self {
            case .bundleNotFound(let path): "No embedding model bundle at \(path)."
            case .manifestNotFound(let path): "The bundle has no manifest at \(path)."
            case .invalidManifest(let reason): "Invalid embedding-model.json: \(reason)"
            case .modelNotFound(let path): "The bundle has no model file at \(path)."
            case .tokenizerNotFound(let path): "The bundle has no tokenizer file at \(path)."
            case .tokenizerLoadFailed(let reason): "The tokenizer could not be loaded: \(reason)"
            case .modelLoadFailed(let reason): "The embedding model could not be loaded: \(reason)"
            case .predictionFailed(let reason): "Embedding failed: \(reason)"
            case .unexpectedOutput(let what): "The model returned unexpected output: \(what)."
            }
        }
    }

    /// Vector length, or 0 when the manifest is missing or invalid.
    public var dimensions: Int { manifest?.dimensions ?? 0 }

    /// Identity of the vectors (`model_id@revision`), for detecting that stored vectors came from
    /// another model. Falls back to ``id`` when the manifest is missing or invalid.
    public var identifier: String { manifest?.identifier ?? id }

    /// Checks everything that can be checked without loading the model: the manifest, the model
    /// file and the tokenizer files. Returns the manifest.
    @discardableResult
    public static func validateBundle(at url: URL) throws -> TextEmbeddingManifest {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw ToolError.bundleNotFound(url.path)
        }
        let manifest = try TextEmbeddingManifest.load(fromBundle: url)
        let modelURL = url.appendingPathComponent(manifest.modelFile)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ToolError.modelNotFound(modelURL.path)
        }
        let missing = requiredTokenizerFiles
            .map { url.appendingPathComponent($0) }
            .first { !FileManager.default.fileExists(atPath: $0.path) }
        if let missing {
            throw ToolError.tokenizerNotFound(missing.path)
        }
        return manifest
    }

    public func availability() async -> SystemToolAvailability {
        do {
            let current = try Self.validateBundle(at: bundleURL)
            if let reason = await host.failureReason(modelURL: bundleURL.appendingPathComponent(current.modelFile)) {
                return .unavailable(reason: reason)
            }
            return .available
        } catch {
            return .unavailable(reason: error.localizedDescription)
        }
    }

    /// Embeds one text. `role` picks the manifest's query or passage prefix.
    public func embed(_ text: String, role: TextEmbeddingRole = .passage) async throws -> TextEmbedding {
        try await host.embed([text], role: role, manifest: validManifest())[0]
    }

    /// Embeds several texts in one hop to the model's queue; results are in input order.
    public func embed(_ texts: [String], role: TextEmbeddingRole = .passage) async throws -> [TextEmbedding] {
        guard !texts.isEmpty else { return [] }
        return try await host.embed(texts, role: role, manifest: validManifest())
    }

    /// The token ids ``embed(_:role:)`` would feed the model, before padding.
    public func tokenize(_ text: String, role: TextEmbeddingRole = .passage) async throws -> TextEmbeddingTokens {
        try await host.tokenize(text, role: role, manifest: validManifest())
    }

    /// Loads the tokenizer, compiles and loads the model, and runs every bucket once so the first
    /// real call is fast. Returns how long it took (the first Neural Engine load compiles on-device).
    @discardableResult
    public func warmUp() async throws -> Duration {
        try await host.warmUp(manifest: validManifest())
    }

    /// How many inputs so far exceeded `max_tokens` and were truncated (shared by copies of this tool).
    public func truncatedInputCount() async -> Int {
        await host.truncatedInputs
    }

    /// Releases the loaded model; the next call loads it again from the compiled cache.
    public func unload() async {
        await host.unload()
    }

    private func validManifest() throws -> TextEmbeddingManifest {
        guard let manifest else { throw manifestProblem ?? .manifestNotFound(bundleURL.path) }
        return manifest
    }
}

// MARK: - Host

private actor EmbeddingHost {
    private let queue = DispatchSerialQueue(label: "AuraLocal.CoreMLTextEmbeddingTool")
    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

    private let bundleURL: URL
    private let computeUnits: MLComputeUnits
    private let cacheDirectory: URL?
    private var model: MLModel?
    private var cachedTokenizer: (any Tokenizers.Tokenizer)?
    private var lastFailure: ModelLoadFailure?
    private(set) var truncatedInputs = 0

    init(bundleURL: URL, computeUnits: MLComputeUnits, cacheDirectory: URL?) {
        self.bundleURL = bundleURL
        self.computeUnits = computeUnits
        self.cacheDirectory = cacheDirectory
    }

    func failureReason(modelURL: URL) -> String? {
        lastFailure?.reportedReason(for: modelURL)
    }

    func unload() {
        model = nil
    }

    func tokenize(_ text: String, role: TextEmbeddingRole, manifest: TextEmbeddingManifest) async throws
        -> TextEmbeddingTokens {
        let tokenizer = try await loadedTokenizer()
        return Self.tokens(for: text, role: role, manifest: manifest, tokenizer: tokenizer)
    }

    func embed(_ texts: [String], role: TextEmbeddingRole, manifest: TextEmbeddingManifest) async throws
        -> [TextEmbedding] {
        let tokenizer = try await loadedTokenizer()
        let model = try await loadedModel(manifest: manifest)
        var results: [TextEmbedding] = []
        results.reserveCapacity(texts.count)
        for text in texts {
            let tokens = Self.tokens(for: text, role: role, manifest: manifest, tokenizer: tokenizer)
            if tokens.isTruncated { truncatedInputs += 1 }
            results.append(try Self.embedding(of: tokens, model: model, manifest: manifest))
        }
        return results
    }

    func warmUp(manifest: TextEmbeddingManifest) async throws -> Duration {
        let clock = ContinuousClock()
        let start = clock.now
        let tokenizer = try await loadedTokenizer()
        let model = try await loadedModel(manifest: manifest)
        let empty = tokenizer.encode(text: "")
        for bucket in manifest.buckets {
            _ = try Self.hiddenStates(for: TextEmbeddingPipeline.pad(empty, to: bucket, padTokenID: manifest.padTokenID),
                                      model: model, manifest: manifest)
        }
        return clock.now - start
    }

    // MARK: Loading

    private func loadedTokenizer() async throws -> any Tokenizers.Tokenizer {
        if let cachedTokenizer { return cachedTokenizer }
        do {
            let loaded = try await Tokenizers.AutoTokenizer.from(modelFolder: bundleURL)
            // Another call may have finished loading while this one awaited.
            if let cachedTokenizer { return cachedTokenizer }
            cachedTokenizer = loaded
            return loaded
        } catch {
            throw CoreMLTextEmbeddingTool.ToolError.tokenizerLoadFailed(error.localizedDescription)
        }
    }

    private func loadedModel(manifest: TextEmbeddingManifest) async throws -> MLModel {
        if let model { return model }
        let source = bundleURL.appendingPathComponent(manifest.modelFile)
        let compiled: URL
        do {
            compiled = try await CompiledModelCache.compiledURL(for: source, in: cacheDirectory)
        } catch {
            lastFailure = ModelLoadFailure(source: source, error: error)
            throw CoreMLTextEmbeddingTool.ToolError.modelLoadFailed(error.localizedDescription)
        }
        if let model { return model }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        do {
            let loaded = try MLModel(contentsOf: compiled, configuration: configuration)
            model = loaded
            lastFailure = nil
            return loaded
        } catch {
            lastFailure = ModelLoadFailure(source: source, error: error)
            throw CoreMLTextEmbeddingTool.ToolError.modelLoadFailed(error.localizedDescription)
        }
    }

    // MARK: Steps

    private static func tokens(
        for text: String,
        role: TextEmbeddingRole,
        manifest: TextEmbeddingManifest,
        tokenizer: any Tokenizers.Tokenizer
    ) -> TextEmbeddingTokens {
        let prepared = TextEmbeddingPipeline.preprocess(manifest.prefix(for: role) + text)
        let ids = tokenizer.encode(text: prepared)
        return TextEmbeddingTokens(ids: TextEmbeddingPipeline.truncate(ids, maxTokens: manifest.maxTokens),
                                   originalCount: ids.count)
    }

    private static func embedding(
        of tokens: TextEmbeddingTokens,
        model: MLModel,
        manifest: TextEmbeddingManifest
    ) throws -> TextEmbedding {
        guard let bucket = TextEmbeddingPipeline.bucket(forTokenCount: tokens.ids.count, in: manifest.buckets) else {
            throw CoreMLTextEmbeddingTool.ToolError.predictionFailed(
                "\(tokens.ids.count) tokens exceed the largest bucket")
        }
        let padded = TextEmbeddingPipeline.pad(tokens.ids, to: bucket, padTokenID: manifest.padTokenID)
        let states = try hiddenStates(for: padded, model: model, manifest: manifest)
        let vector = try TextEmbeddingPipeline.meanPool(states, tokenIDs: padded,
                                                        padTokenID: Int32(truncatingIfNeeded: manifest.padTokenID),
                                                        normalize: manifest.normalize)
        guard vector.count == manifest.dimensions else {
            throw CoreMLTextEmbeddingTool.ToolError.unexpectedOutput(
                "\(vector.count)-dimensional vectors, the manifest says \(manifest.dimensions)")
        }
        return TextEmbedding(vector: vector, tokenCount: tokens.ids.count, isTruncated: tokens.isTruncated)
    }

    private static func hiddenStates(for ids: [Int32], model: MLModel, manifest: TextEmbeddingManifest) throws
        -> MLMultiArray {
        let output: MLFeatureProvider
        do {
            let input = try MLDictionaryFeatureProvider(
                dictionary: [manifest.inputName: MLFeatureValue(multiArray: try TextEmbeddingPipeline.makeInputArray(ids))])
            output = try model.prediction(from: input)
        } catch {
            throw CoreMLTextEmbeddingTool.ToolError.predictionFailed(error.localizedDescription)
        }
        guard let states = output.featureValue(for: manifest.outputName)?.multiArrayValue else {
            throw CoreMLTextEmbeddingTool.ToolError.unexpectedOutput("no multi-array output named “\(manifest.outputName)”")
        }
        return states
    }
}
