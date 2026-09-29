import Foundation
import CoreML
import CoreGraphics
import CryptoKit

/// Generic on-device runner for **any Core ML model** the app ships or downloads —
/// Create ML output, models converted with coremltools, etc. It is the bridge that
/// lets a custom model sit next to the system tools. Accepts `.mlmodel` / `.mlpackage`
/// (compiled once and cached under Caches) or an already compiled `.mlmodelc`.
///
/// Thread safety: `MLModel` is not `Sendable`, so the loaded model lives in a private
/// actor that runs on its own serial queue. Copies of a tool share that actor: calls
/// are serialized per tool and blocking Core ML work never occupies the cooperative pool.
public struct CoreMLModelTool: SystemTool {
    public let id: String
    public let displayName: String
    public let summary: String
    public let category = SystemToolCategory.customModel
    /// The model as given: `.mlmodel`, `.mlpackage` or `.mlmodelc`.
    public let modelURL: URL
    public let computeUnits: MLComputeUnits

    private let host: ModelHost

    /// - Parameter compiledModelsDirectory: where compiled bundles are cached;
    ///   `nil` uses `Caches/AuraLocal/CompiledModels`.
    public init(
        modelAt url: URL,
        computeUnits: MLComputeUnits = .all,
        compiledModelsDirectory: URL? = nil,
        displayName: String? = nil,
        summary: String? = nil
    ) {
        let baseName = url.deletingPathExtension().lastPathComponent
        self.id = "coreml.\(baseName)"
        self.displayName = displayName ?? "Core ML model “\(baseName)”"
        self.summary = summary ?? "Run the custom Core ML model “\(baseName)” on-device — no network."
        self.modelURL = url
        self.computeUnits = computeUnits
        self.host = ModelHost(source: url, computeUnits: computeUnits, cacheDirectory: compiledModelsDirectory)
    }

    public enum ToolError: LocalizedError, Equatable {
        case modelNotFound(String)
        case unsupportedModelFormat(String)
        case compilationFailed(String)
        case loadFailed(String)
        case unknownInput(String)
        case incompatibleInput(name: String, expected: String)
        case predictionFailed(String)
        case notAClassifier

        public var errorDescription: String? {
            switch self {
            case .modelNotFound(let path): "No model file at \(path)."
            case .unsupportedModelFormat(let ext):
                "Unsupported model format “.\(ext)”: expected .mlmodel, .mlpackage or .mlmodelc."
            case .compilationFailed(let reason): "The model could not be compiled: \(reason)"
            case .loadFailed(let reason): "The model could not be loaded: \(reason)"
            case .unknownInput(let name): "The model has no input named “\(name)”."
            case .incompatibleInput(let name, let expected): "Input “\(name)” expects \(expected)."
            case .predictionFailed(let reason): "Prediction failed: \(reason)"
            case .notAClassifier: "The model is not a classifier (it declares no class labels)."
            }
        }
    }

    /// A `Sendable` value going into or coming out of the model. Inputs are converted
    /// using the model's own feature description: `.doubles` is reshaped to the input's
    /// multi-array shape and element type, `.image` is scaled to its image constraint.
    public enum FeatureValue: Sendable, Equatable {
        case string(String)
        case int(Int)
        case double(Double)
        /// A flat vector, reshaped to the input's declared shape when the counts match.
        case doubles([Double])
        /// An N-dimensional array, values in row-major order.
        case multiArray(shape: [Int], values: [Double])
        case image(CGImage)
        /// Scores keyed by label: classifier probabilities or a sparse feature dictionary.
        case dictionary([String: Double])
        case strings([String])
        /// An output this runner does not convert (e.g. model state); names its kind.
        case unsupported(String)

        /// Copies an `MLMultiArray` of any element type into a `Sendable` value.
        public static func multiArray(copying array: MLMultiArray) -> FeatureValue {
            let shaped = MLShapedArray<Double>(converting: array)
            return .multiArray(shape: shaped.shape, values: shaped.scalars)
        }

        var labelText: String? {
            switch self {
            case .string(let text): text
            case .int(let number): String(number)
            default: nil
            }
        }

        var scores: [String: Double]? {
            if case .dictionary(let scores) = self { return scores }
            return nil
        }
    }

    public enum FeatureKind: String, Sendable {
        case int64, double, string, image, multiArray, dictionary, sequence, state, invalid
    }

    public struct FeatureSpec: Sendable, Equatable {
        public let name: String
        public let kind: FeatureKind
        public let isOptional: Bool
        /// Multi-array shape, or `[height, width]` for images; empty for other kinds.
        public let shape: [Int]
        /// Multi-array element type (`float32`, …) or image pixel format (`BGRA`, …).
        public let dataType: String?
        /// True when the model also accepts shapes or image sizes other than `shape`.
        public let isShapeFlexible: Bool
    }

    public struct ModelInfo: Sendable, Equatable {
        public let inputs: [FeatureSpec]
        public let outputs: [FeatureSpec]
        /// True when the model declares class labels.
        public let isClassifier: Bool
        /// In the model's own order.
        public let classLabels: [String]
        public let predictedFeatureName: String?
        public let predictedProbabilitiesName: String?
        public let author: String?
        public let shortDescription: String?
        public let version: String?
        public let license: String?
    }

    public struct Classification: Sendable, Equatable {
        public let label: String
        /// Probability per label; empty when the model only outputs the label
        /// (Create ML text classifiers do — use ``TextClassifierTool`` for their scores).
        public let probabilities: [String: Double]

        /// Labels by descending probability.
        public func ranked(limit: Int = .max) -> [(label: String, probability: Double)] {
            probabilities
                .sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
                .prefix(limit)
                .map { (label: $0.key, probability: $0.value) }
        }
    }

    public func availability() async -> SystemToolAvailability {
        do {
            try CompiledModelCache.validate(modelURL)
        } catch {
            return .unavailable(reason: error.localizedDescription)
        }
        if let failure = await host.loadFailure {
            return .unavailable(reason: failure)
        }
        return .available
    }

    /// Inputs, outputs, class labels and metadata. Loads (and if needed compiles) the model.
    public func describe() async throws -> ModelInfo {
        try await host.describe()
    }

    /// Run one prediction. Keys are the model's input names; every output is returned.
    public func predict(_ inputs: [String: FeatureValue]) async throws -> [String: FeatureValue] {
        try await host.predict(inputs)
    }

    /// Top label plus class probabilities, for classifier models.
    public func classify(_ inputs: [String: FeatureValue]) async throws -> Classification {
        try await host.classify(inputs)
    }

    /// The compiled `.mlmodelc`, e.g. to hand the same model to Vision or NaturalLanguage.
    public func compiledModelURL() async throws -> URL {
        try await host.compiledURL()
    }

    /// Release the loaded model; the next call loads it again from the compiled cache.
    public func unload() async {
        await host.unload()
    }
}

extension CoreMLModelTool.FeatureValue: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral,
    ExpressibleByFloatLiteral, ExpressibleByArrayLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .int(value) }
    public init(floatLiteral value: Double) { self = .double(value) }
    public init(arrayLiteral elements: Double...) { self = .doubles(elements) }
}

// MARK: - Model host

private actor ModelHost {
    typealias Value = CoreMLModelTool.FeatureValue

    private let queue = DispatchSerialQueue(label: "AuraLocal.CoreMLModelTool")
    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

    private let source: URL
    private let computeUnits: MLComputeUnits
    private let cacheDirectory: URL?
    private var model: MLModel?
    private var compiled: URL?
    private(set) var loadFailure: String?

    init(source: URL, computeUnits: MLComputeUnits, cacheDirectory: URL?) {
        self.source = source
        self.computeUnits = computeUnits
        self.cacheDirectory = cacheDirectory
    }

    func compiledURL() async throws -> URL {
        if let compiled { return compiled }
        do {
            let url = try await CompiledModelCache.compiledURL(for: source, in: cacheDirectory)
            compiled = url
            return url
        } catch {
            loadFailure = error.localizedDescription
            throw error
        }
    }

    func describe() async throws -> CoreMLModelTool.ModelInfo {
        let model = try await loadedModel()
        return FeatureConversion.makeInfo(model.modelDescription)
    }

    func predict(_ inputs: [String: Value]) async throws -> [String: Value] {
        let model = try await loadedModel()
        return FeatureConversion.values(from: try FeatureConversion.run(model, inputs: inputs))
    }

    func classify(_ inputs: [String: Value]) async throws -> CoreMLModelTool.Classification {
        let model = try await loadedModel()
        let description = model.modelDescription
        // Regressors also set `predictedFeatureName`; only class labels mark a classifier.
        guard description.classLabels?.isEmpty == false else {
            throw CoreMLModelTool.ToolError.notAClassifier
        }
        let outputs = FeatureConversion.values(from: try FeatureConversion.run(model, inputs: inputs))
        return try FeatureConversion.makeClassification(outputs, description: description)
    }

    func unload() {
        model = nil
    }

    private func loadedModel() async throws -> MLModel {
        if let model { return model }
        let url = try await compiledURL()
        // Another call may have finished loading while this one awaited the compile.
        if let model { return model }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        do {
            let loaded = try MLModel(contentsOf: url, configuration: configuration)
            model = loaded
            loadFailure = nil
            return loaded
        } catch {
            loadFailure = error.localizedDescription
            throw CoreMLModelTool.ToolError.loadFailed(error.localizedDescription)
        }
    }
}

// MARK: - Compiled model cache

/// Compiles `.mlmodel` / `.mlpackage` once and keeps the `.mlmodelc` under Caches, keyed
/// by path, size and modification date so an updated model is recompiled.
enum CompiledModelCache {
    static let defaultDirectory: URL = {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return caches.appendingPathComponent("AuraLocal/CompiledModels", isDirectory: true)
    }()

    private static let supportedExtensions: Set<String> = ["mlmodel", "mlpackage", "mlmodelc"]
    private static let stampKeys: Set<URLResourceKey> = [.fileSizeKey, .contentModificationDateKey]

    static func validate(_ url: URL) throws {
        let ext = url.pathExtension.lowercased()
        guard supportedExtensions.contains(ext) else {
            throw CoreMLModelTool.ToolError.unsupportedModelFormat(ext)
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CoreMLModelTool.ToolError.modelNotFound(url.path)
        }
    }

    static func compiledURL(for source: URL, in directory: URL?) async throws -> URL {
        try validate(source)
        if source.pathExtension.lowercased() == "mlmodelc" { return source }

        let files = FileManager.default
        let folder = directory ?? defaultDirectory
        let name = "\(source.deletingPathExtension().lastPathComponent)-\(fingerprint(of: source)).mlmodelc"
        let destination = folder.appendingPathComponent(name, isDirectory: true)
        if files.fileExists(atPath: destination.path) { return destination }

        let temporary: URL
        do {
            temporary = try await MLModel.compileModel(at: source)
        } catch {
            throw CoreMLModelTool.ToolError.compilationFailed(error.localizedDescription)
        }
        do {
            try files.createDirectory(at: folder, withIntermediateDirectories: true)
            try files.moveItem(at: temporary, to: destination)
        } catch {
            // A concurrent compile of the same model may have won the move.
            guard files.fileExists(atPath: destination.path) else {
                let reason = "Could not cache the compiled model: \(error.localizedDescription)"
                throw CoreMLModelTool.ToolError.compilationFailed(reason)
            }
            try? files.removeItem(at: temporary)
        }
        return destination
    }

    static func fingerprint(of url: URL) -> String {
        let nested = FileManager.default
            .enumerator(at: url, includingPropertiesForKeys: Array(stampKeys))?
            .allObjects
            .compactMap { $0 as? URL } ?? []
        let stamps = ([url] + nested).map(stamp).sorted()
        let key = ([url.standardizedFileURL.path] + stamps).joined(separator: "\n")
        return SHA256.hash(data: Data(key.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    private static func stamp(of file: URL) -> String {
        let values = try? file.resourceValues(forKeys: stampKeys)
        let modified = values?.contentModificationDate?.timeIntervalSince1970 ?? 0
        return "\(file.path):\(values?.fileSize ?? 0):\(modified)"
    }
}
