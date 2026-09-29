import Foundation
import CoreML
import NaturalLanguage

/// Runs a **Create ML text classifier** through NaturalLanguage's `NLModel`: text in, label
/// plus per-label probabilities out. Accepts `.mlmodel` (compiled once, cached) or `.mlmodelc`.
/// `NLModel` is not `Sendable`; it lives in an actor on its own queue, shared by copies.
public struct TextClassifierTool: SystemTool {
    public let id: String
    public let displayName: String
    public let summary: String
    public let category = SystemToolCategory.customModel
    public let modelURL: URL

    private let host: TextModelHost

    /// - Parameter compiledModelsDirectory: where the compiled model is cached;
    ///   `nil` uses `Caches/AuraLocal/CompiledModels`.
    public init(
        modelAt url: URL,
        compiledModelsDirectory: URL? = nil,
        displayName: String? = nil,
        summary: String? = nil
    ) {
        let baseName = url.deletingPathExtension().lastPathComponent
        self.id = "coreml.text-classifier.\(baseName)"
        self.displayName = displayName ?? "Text classifier “\(baseName)”"
        self.summary = summary ?? "Classify text with the custom Create ML model “\(baseName)” on-device."
        self.modelURL = url
        self.host = TextModelHost(source: url, cacheDirectory: compiledModelsDirectory)
    }

    public enum ToolError: LocalizedError {
        case modelUnavailable(String)
        case notATextClassifier(String)
        case predictionFailed

        public var errorDescription: String? {
            switch self {
            case .modelUnavailable(let reason): "The text classifier model is unavailable: \(reason)"
            case .notATextClassifier(let reason): "The model is not a text classifier: \(reason)"
            case .predictionFailed:
                "The model returned no label for the text (seen with BERT-based models in the Simulator)."
            }
        }
    }

    public struct Classification: Sendable, Equatable {
        /// Best label; `nil` only for blank input.
        public let label: String?
        /// Probability of each of the top labels (they sum to ≈1 when every label is requested).
        public let hypotheses: [String: Double]

        /// Hypotheses by descending probability.
        public func ranked() -> [(label: String, probability: Double)] {
            hypotheses
                .sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
                .map { (label: $0.key, probability: $0.value) }
        }
    }

    public func availability() async -> SystemToolAvailability {
        if let reason = await host.unavailabilityReason() {
            return .unavailable(reason: reason)
        }
        return .available
    }

    /// Classify `text`. Blank text returns an empty result rather than a guess; a model that
    /// cannot label non-blank text throws ``ToolError/predictionFailed``.
    public func classify(_ text: String, maxHypotheses: Int = 3) async throws -> Classification {
        try await host.classify(text, maxHypotheses: maxHypotheses)
    }

    /// The labels the model can predict, sorted.
    public func labels() async throws -> [String] {
        try await host.labels()
    }
}

private actor TextModelHost {
    private let queue = DispatchSerialQueue(label: "AuraLocal.TextClassifierTool")
    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

    private let source: URL
    private let cacheDirectory: URL?
    private var model: NLModel?
    private var classLabels: [String] = []
    private var lastFailure: ModelLoadFailure?

    init(source: URL, cacheDirectory: URL?) {
        self.source = source
        self.cacheDirectory = cacheDirectory
    }

    func unavailabilityReason() -> String? {
        do {
            try CompiledModelCache.validate(source)
        } catch {
            return error.localizedDescription
        }
        return lastFailure?.reportedReason(for: source)
    }

    func classify(_ text: String, maxHypotheses: Int) async throws -> TextClassifierTool.Classification {
        let model = try await loadedModel()
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return TextClassifierTool.Classification(label: nil, hypotheses: [:])
        }
        // Seen in the Simulator with a BERT model: nil and no hypotheses, without an error.
        guard let label = model.predictedLabel(for: text) else {
            throw TextClassifierTool.ToolError.predictionFailed
        }
        let hypotheses = maxHypotheses > 0
            ? model.predictedLabelHypotheses(for: text, maximumCount: maxHypotheses)
            : [:]
        return TextClassifierTool.Classification(label: label, hypotheses: hypotheses)
    }

    func labels() async throws -> [String] {
        _ = try await loadedModel()
        return classLabels
    }

    private func loadedModel() async throws -> NLModel {
        if let model { return model }
        let compiled: URL
        do {
            compiled = try await CompiledModelCache.compiledURL(for: source, in: cacheDirectory)
        } catch {
            lastFailure = ModelLoadFailure(source: source, error: error)
            throw TextClassifierTool.ToolError.modelUnavailable(error.localizedDescription)
        }
        if let model { return model }
        do {
            let core = try MLModel(contentsOf: compiled)
            let classifier = try NLModel(mlModel: core)
            guard classifier.configuration.type == .classifier else {
                throw TextClassifierTool.ToolError.notATextClassifier("it is a word tagger.")
            }
            classLabels = (core.modelDescription.classLabels ?? []).map { "\($0)" }.sorted()
            model = classifier
            lastFailure = nil
            return classifier
        } catch let error as TextClassifierTool.ToolError {
            lastFailure = ModelLoadFailure(source: source, error: error)
            throw error
        } catch {
            lastFailure = ModelLoadFailure(source: source, error: error)
            throw TextClassifierTool.ToolError.notATextClassifier(error.localizedDescription)
        }
    }
}
