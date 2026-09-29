import Foundation
import NaturalLanguage
import TabularData
#if canImport(CreateML)
import CreateML
#endif

/// On-device **training** of a text classifier with Create ML: labelled examples in,
/// a `.mlmodel` out, ready for ``TextClassifierTool`` or ``CoreMLModelTool``. Lets an
/// app learn the user's own categories (expense types, tags, intents) with no server.
///
/// Platforms: Create ML ships in the macOS, iOS and visionOS device SDKs but not in the
/// iOS / visionOS Simulator SDKs, where `availability()` says so and `train` throws.
public struct TextClassifierTrainer: SystemTool {
    public let id = "training.text-classifier"
    public let displayName = "Text classifier training (Create ML)"
    public let summary = "Train a custom text classifier on-device from labelled examples with Create ML and save it as a Core ML model — no server, no upload."
    public let category = SystemToolCategory.training

    public init() {}

    public enum ToolError: LocalizedError {
        case trainingUnavailable
        case notEnoughData(String)
        case invalidValidationFraction(Double)
        case invalidOutputURL(String)
        case invalidCSV(String)
        case trainingFailed(String)

        public var errorDescription: String? {
            switch self {
            case .trainingUnavailable: "Create ML is not available on this platform."
            case .notEnoughData(let reason): "Not enough training data: \(reason)"
            case .invalidValidationFraction(let fraction):
                "The validation fraction must be between 0 and 1, got \(fraction)."
            case .invalidOutputURL(let path):
                "The model must be written to a file URL ending in .mlmodel, got \(path)."
            case .invalidCSV(let reason): "The CSV file could not be read: \(reason)"
            case .trainingFailed(let reason): "Training failed: \(reason)"
            }
        }
    }

    public struct Example: Sendable, Equatable {
        public let text: String
        public let label: String

        public init(text: String, label: String) {
            self.text = text
            self.label = label
        }
    }

    /// Pretrained embeddings for transfer learning; the OS provides them per language.
    public enum Embedding: Sendable, Equatable {
        case staticEmbedding
        case elmoEmbedding
        case bertEmbedding
        /// A custom word-embedding model file.
        case customEmbedding(URL)
    }

    /// The algorithms Create ML's `MLTextClassifier` offers.
    public enum Algorithm: Sendable, Equatable {
        /// Maximum entropy over bag-of-words: fast, needs little data. The default.
        case maxEnt
        /// Conditional random field: also weighs word order.
        case crf
        /// A classifier on top of pretrained embeddings: slower, generalises better.
        case transferLearning(Embedding)
    }

    public enum Validation: Sendable, Equatable {
        /// Create ML decides how many examples to hold out.
        case automatic
        /// Hold out this fraction (0 < fraction < 1) with a fixed seed, for reproducible reports.
        case holdOut(fraction: Double, seed: Int)
        /// Train on every example; the report has no validation accuracy.
        case disabled
    }

    public struct Metadata: Sendable, Equatable {
        public var author: String
        public var shortDescription: String
        public var version: String
        public var license: String?

        public init(
            author: String = "AuraLocal",
            shortDescription: String = "Text classifier trained on-device with AuraLocal.",
            version: String = "1",
            license: String? = nil
        ) {
            self.author = author
            self.shortDescription = shortDescription
            self.version = version
            self.license = license
        }
    }

    public struct Report: Sendable, Equatable {
        public let modelURL: URL
        /// Sorted.
        public let classLabels: [String]
        public let exampleCount: Int
        /// 1 − classification error on the training examples.
        public let trainingAccuracy: Double?
        /// `nil` when validation was disabled or nothing was held out.
        public let validationAccuracy: Double?
    }

    public func availability() async -> SystemToolAvailability {
        #if canImport(CreateML)
        .available
        #else
        .unavailable(reason: "Create ML is not available on this platform (the iOS and visionOS Simulators lack it).")
        #endif
    }

    /// Train on `examples` and write the model to `outputURL` (`.mlmodel`). Runs on a
    /// background queue; blank texts or labels are skipped. At least two labels are needed.
    public func train(
        examples: [Example],
        writingModelTo outputURL: URL,
        algorithm: Algorithm = .maxEnt,
        language: NLLanguage? = nil,
        validation: Validation = .automatic,
        metadata: Metadata = Metadata()
    ) async throws -> Report {
        let textsByLabel = try Self.group(examples)
        guard outputURL.isFileURL, outputURL.pathExtension.lowercased() == "mlmodel" else {
            throw ToolError.invalidOutputURL(outputURL.absoluteString)
        }
        if case .holdOut(let fraction, _) = validation, !(fraction > 0 && fraction < 1) {
            throw ToolError.invalidValidationFraction(fraction)
        }
        #if canImport(CreateML)
        let modelMetadata = MLModelMetadata(
            author: metadata.author, shortDescription: metadata.shortDescription,
            license: metadata.license, version: metadata.version)
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result {
                    // Built here because `ModelParameters` is not Sendable.
                    let parameters = MLTextClassifier.ModelParameters(
                        validation: Self.validationData(validation),
                        algorithm: Self.modelAlgorithm(algorithm),
                        language: language)
                    return try Self.fit(
                        textsByLabel, parameters: parameters, writingTo: outputURL, metadata: modelMetadata)
                })
            }
        }
        #else
        throw ToolError.trainingUnavailable
        #endif
    }

    /// Train from a CSV file with a header row; see ``loadExamples(csvAt:textColumn:labelColumn:)``.
    public func train(
        csvAt csvURL: URL,
        textColumn: String = "text",
        labelColumn: String = "label",
        writingModelTo outputURL: URL,
        algorithm: Algorithm = .maxEnt,
        language: NLLanguage? = nil,
        validation: Validation = .automatic,
        metadata: Metadata = Metadata()
    ) async throws -> Report {
        let examples = try Self.loadExamples(csvAt: csvURL, textColumn: textColumn, labelColumn: labelColumn)
        return try await train(examples: examples, writingModelTo: outputURL, algorithm: algorithm,
                               language: language, validation: validation, metadata: metadata)
    }

    /// Read labelled examples from a CSV file with a header row (RFC 4180 quoting).
    /// Rows with a blank text or label are skipped. Works on every platform.
    public static func loadExamples(
        csvAt url: URL,
        textColumn: String = "text",
        labelColumn: String = "label"
    ) throws -> [Example] {
        let frame: DataFrame
        do {
            frame = try DataFrame(
                contentsOfCSVFile: url,
                columns: [textColumn, labelColumn],
                types: [textColumn: .string, labelColumn: .string])
        } catch {
            throw ToolError.invalidCSV(error.localizedDescription)
        }
        return zip(frame[textColumn, String.self], frame[labelColumn, String.self])
            .compactMap { example((text: $0, label: $1)) }
    }

    private static func example(_ row: (text: String?, label: String?)) -> Example? {
        guard let text = row.text?.trimmingCharacters(in: .whitespacesAndNewlines),
              let label = row.label?.trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty, !label.isEmpty else { return nil }
        return Example(text: text, label: label)
    }

    private static func group(_ examples: [Example]) throws -> [String: [String]] {
        let usable = examples.compactMap { example((text: $0.text, label: $0.label)) }
        let textsByLabel = Dictionary(grouping: usable, by: \.label).mapValues { $0.map(\.text) }
        guard textsByLabel.count >= 2 else {
            let reason = "examples for at least two labels are required, got \(textsByLabel.count)."
            throw ToolError.notEnoughData(reason)
        }
        return textsByLabel
    }

    #if canImport(CreateML)
    private static func fit(
        _ textsByLabel: [String: [String]],
        parameters: MLTextClassifier.ModelParameters,
        writingTo outputURL: URL,
        metadata: MLModelMetadata
    ) throws -> Report {
        let classifier: MLTextClassifier
        do {
            classifier = try MLTextClassifier(trainingData: textsByLabel, parameters: parameters)
        } catch {
            throw ToolError.trainingFailed(error.localizedDescription)
        }
        do {
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try classifier.write(to: outputURL, metadata: metadata)
        } catch {
            throw ToolError.trainingFailed("the model could not be written: \(error.localizedDescription)")
        }
        return Report(
            modelURL: outputURL,
            classLabels: (classifier.model.modelDescription.classLabels ?? []).map { "\($0)" }.sorted(),
            exampleCount: textsByLabel.values.reduce(0) { $0 + $1.count },
            trainingAccuracy: accuracy(classifier.trainingMetrics),
            validationAccuracy: accuracy(classifier.validationMetrics))
    }

    private static func accuracy(_ metrics: MLClassifierMetrics) -> Double? {
        metrics.isValid ? 1 - metrics.classificationError : nil
    }

    // `revision: nil` lets Create ML use the latest revision this OS ships.
    private static func modelAlgorithm(_ algorithm: Algorithm) -> MLTextClassifier.ModelAlgorithmType {
        switch algorithm {
        case .maxEnt: .maxEnt(revision: nil)
        case .crf: .crf(revision: nil)
        case .transferLearning(let embedding): .transferLearning(featureExtractor(embedding), revision: nil)
        }
    }

    private static func featureExtractor(_ embedding: Embedding) -> MLTextClassifier.FeatureExtractorType {
        switch embedding {
        case .staticEmbedding: .staticEmbedding
        case .elmoEmbedding: .elmoEmbedding
        case .bertEmbedding: .bertEmbedding
        case .customEmbedding(let url): .customEmbedding(url)
        }
    }

    private static func validationData(_ validation: Validation) -> MLTextClassifier.ModelParameters.ValidationData {
        switch validation {
        case .automatic: .split(strategy: .automatic)
        case .holdOut(let fraction, let seed): .split(strategy: .fixed(ratio: fraction, seed: seed))
        case .disabled: .none
        }
    }
    #endif
}
