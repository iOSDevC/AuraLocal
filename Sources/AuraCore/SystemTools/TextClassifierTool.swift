import Foundation
import CoreML
import NaturalLanguage

/// Runs a **Create ML text classifier** through NaturalLanguage's `NLModel`: text in,
/// predicted label plus per-label probabilities out (e.g. expense category, intent,
/// ticket priority). Train one on-device with ``TextClassifierTrainer`` or ship one made
/// in the Create ML app. Accepts `.mlmodel` (compiled once, cached) or `.mlmodelc`.
///
/// Thread safety: `NLModel` is not `Sendable`; it lives in a private actor on its own
/// serial queue, shared by copies of the tool.
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

        public var errorDescription: String? {
            switch self {
            case .modelUnavailable(let reason): "The text classifier model is unavailable: \(reason)"
            case .notATextClassifier(let reason): "The model is not a text classifier: \(reason)"
            }
        }
    }

    public struct Classification: Sendable, Equatable {
        /// Best label; `nil` for blank input.
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

    /// Classify `text`. Blank text returns an empty result rather than a guess.
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
    private(set) var loadFailure: String?

    init(source: URL, cacheDirectory: URL?) {
        self.source = source
        self.cacheDirectory = cacheDirectory
    }

    func classify(_ text: String, maxHypotheses: Int) async throws -> TextClassifierTool.Classification {
        let model = try await loadedModel()
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return TextClassifierTool.Classification(label: nil, hypotheses: [:])
        }
        let hypotheses = maxHypotheses > 0
            ? model.predictedLabelHypotheses(for: text, maximumCount: maxHypotheses)
            : [:]
        return TextClassifierTool.Classification(label: model.predictedLabel(for: text), hypotheses: hypotheses)
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
            loadFailure = error.localizedDescription
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
            loadFailure = nil
            return classifier
        } catch let error as TextClassifierTool.ToolError {
            loadFailure = error.localizedDescription
            throw error
        } catch {
            loadFailure = error.localizedDescription
            throw TextClassifierTool.ToolError.notATextClassifier(error.localizedDescription)
        }
    }
}
