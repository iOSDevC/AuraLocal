import Foundation

// MARK: - Model

/// A supported on-device model available for download and inference.
///
/// Each model maps to a Hugging Face repository (``repoID``) and carries
/// metadata such as ``purpose``, ``displayName``, ``approximateSizeMB``,
/// and ``defaultDocumentPrompt``.
///
/// Models are loaded from a JSON catalog (bundled + remote-updateable).
/// Use the convenience collections ``textModels``, ``visionModels``, and
/// ``specializedModels`` to list available models by category.
public struct Model: Sendable, Identifiable, Codable, CustomStringConvertible {

    // MARK: - Category

    /// JSON-friendly category tag — maps to ``Purpose`` at runtime.
    public enum Category: String, Codable, Sendable {
        case text
        case vision
        case visionSpecialized
    }

    // MARK: - Purpose

    /// The functional category of a model, determining which ``AuraLocal``
    /// factory method should be used to load it.
    public enum Purpose: Sendable {
        /// Language-only generation — load with ``AuraLocal/text(_:onProgress:)``.
        case text
        /// Multimodal image+text — load with ``AuraLocal/vision(_:onProgress:)``.
        case vision
        /// OCR / document extraction — load with ``AuraLocal/specialized(_:onProgress:)``.
        /// When `docTags` is `true` the model outputs DocTags markup (e.g. Granite Docling).
        case visionSpecialized(docTags: Bool)
    }

    // MARK: - Stored properties

    /// Stable identifier (e.g. `"qwen3_1_7b"`). Used as dictionary key and SwiftUI id.
    public let id: String

    /// Hugging Face repository path (e.g. `"mlx-community/Qwen3-1.7B-4bit"`).
    public let repoID: String

    /// A human-readable name for display in UI (e.g. "Qwen3 1.7B").
    public let displayName: String

    /// JSON-serializable category.
    public let category: Category

    /// Whether the model outputs DocTags markup. Only meaningful for ``Category/visionSpecialized``.
    public let docTags: Bool

    /// The weight format of this model.
    public let format: ModelFormat

    /// Approximate download size in megabytes.
    public let approximateSizeMB: Int

    /// Whether this model is an uncensored or abliterated variant.
    public let isUncensored: Bool

    /// The GGUF filename to download (nil for MLX models).
    public let ggufFilename: String?

    /// Default prompt for document/OCR extraction. `nil` for non-specialized models.
    public let defaultDocumentPrompt: String?

    /// Number of transformer layers (used for hardware analysis).
    public let numLayers: Int

    /// Number of KV-cache heads (0 = use flat estimate for MLX models).
    public let kvHeads: Int

    /// Head dimension for KV-cache calculation (0 = use flat estimate).
    public let headDim: Int

    // MARK: - Computed properties

    /// Backward-compatible raw value — returns the HF repository path.
    public var rawValue: String { repoID }

    /// The functional category of this model (with associated value for docTags).
    public var purpose: Purpose {
        switch category {
        case .text:               return .text
        case .vision:             return .vision
        case .visionSpecialized:  return .visionSpecialized(docTags: docTags)
        }
    }

    /// Local cache directory where the model is downloaded.
    /// Mirrors the `<org>/<repo>` folder structure used by mlx-swift.
    public var cacheDirectory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models")
            .appendingPathComponent(repoID)
    }

    /// Returns `true` if the model directory exists on disk.
    public var isDownloaded: Bool {
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: cacheDirectory.path,
            isDirectory: &isDir
        )
        return exists && isDir.boolValue
    }

    /// Whether this model is recommended for macOS only (too large for typical iOS devices).
    public var isMacOSRecommended: Bool {
        approximateSizeMB >= 15_000
    }

    public var description: String { id }

    // MARK: - Hashable / Equatable

    public static func == (lhs: Model, rhs: Model) -> Bool {
        lhs.id == rhs.id
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

// MARK: - Hashable conformance

extension Model: Hashable {}

// MARK: - Static constants (backward compatibility)

public extension Model {

    // MARK: Text (MLX)
    static let qwen3_0_6b   = ModelRegistry.shared.model(id: "qwen3_0_6b")!
    static let qwen3_1_7b   = ModelRegistry.shared.model(id: "qwen3_1_7b")!
    static let qwen3_4b     = ModelRegistry.shared.model(id: "qwen3_4b")!
    static let gemma3_1b    = ModelRegistry.shared.model(id: "gemma3_1b")!
    static let phi3_5_mini  = ModelRegistry.shared.model(id: "phi3_5_mini")!
    static let llama3_2_1b  = ModelRegistry.shared.model(id: "llama3_2_1b")!
    static let llama3_2_3b  = ModelRegistry.shared.model(id: "llama3_2_3b")!

    // MARK: Vision (MLX)
    static let qwen35_0_8b  = ModelRegistry.shared.model(id: "qwen35_0_8b")!
    static let qwen35_2b    = ModelRegistry.shared.model(id: "qwen35_2b")!
    static let smolvlm_500m = ModelRegistry.shared.model(id: "smolvlm_500m")!
    static let smolvlm_2b   = ModelRegistry.shared.model(id: "smolvlm_2b")!

    // MARK: Vision Specialized (MLX)
    static let fastVLM_0_5b_fp16  = ModelRegistry.shared.model(id: "fastVLM_0_5b_fp16")!
    static let fastVLM_1_5b_int8  = ModelRegistry.shared.model(id: "fastVLM_1_5b_int8")!
    static let graniteDocling_258m = ModelRegistry.shared.model(id: "graniteDocling_258m")!
    static let graniteVision_3_3  = ModelRegistry.shared.model(id: "graniteVision_3_3")!

    // MARK: GGUF Standard
    static let llama3_1_8b_gguf   = ModelRegistry.shared.model(id: "llama3_1_8b_gguf")!
    static let qwen2_5_7b_gguf    = ModelRegistry.shared.model(id: "qwen2_5_7b_gguf")!
    static let mistral_7b_gguf    = ModelRegistry.shared.model(id: "mistral_7b_gguf")!
    static let phi3_medium_gguf   = ModelRegistry.shared.model(id: "phi3_medium_gguf")!
    static let gemma2_9b_gguf     = ModelRegistry.shared.model(id: "gemma2_9b_gguf")!

    // MARK: GGUF Large
    static let llama3_1_70b_gguf  = ModelRegistry.shared.model(id: "llama3_1_70b_gguf")!
    static let qwen2_5_32b_gguf   = ModelRegistry.shared.model(id: "qwen2_5_32b_gguf")!

    // MARK: Uncensored (MLX)
    static let josiefied_qwen3_1_7b = ModelRegistry.shared.model(id: "josiefied_qwen3_1_7b")!
    static let josiefied_qwen3_4b   = ModelRegistry.shared.model(id: "josiefied_qwen3_4b")!
    static let josiefied_qwen3_8b   = ModelRegistry.shared.model(id: "josiefied_qwen3_8b")!
    static let dolphin_qwen2_1_5b   = ModelRegistry.shared.model(id: "dolphin_qwen2_1_5b")!

    // MARK: Uncensored (GGUF)
    static let dolphin3_qwen25_1_5b_gguf  = ModelRegistry.shared.model(id: "dolphin3_qwen25_1_5b_gguf")!
    static let dolphin3_qwen25_3b_gguf    = ModelRegistry.shared.model(id: "dolphin3_qwen25_3b_gguf")!
    static let dolphin3_llama31_8b_gguf   = ModelRegistry.shared.model(id: "dolphin3_llama31_8b_gguf")!
    static let llama32_3b_uncensored_gguf = ModelRegistry.shared.model(id: "llama32_3b_uncensored_gguf")!
    static let llama31_8b_abliterated_gguf = ModelRegistry.shared.model(id: "llama31_8b_abliterated_gguf")!

    // MARK: New Models
    static let granite4_tiny    = ModelRegistry.shared.model(id: "granite4_tiny")!
    static let granite4_compact = ModelRegistry.shared.model(id: "granite4_compact")!
    static let cogito_v1_3b     = ModelRegistry.shared.model(id: "cogito_v1_3b")!
    static let gemma4_1b        = ModelRegistry.shared.model(id: "gemma4_1b")!
    static let gemma4_4b        = ModelRegistry.shared.model(id: "gemma4_4b")!
    static let gemma4_12b       = ModelRegistry.shared.model(id: "gemma4_12b")!
    static let lfm2_1_2b        = ModelRegistry.shared.model(id: "lfm2_1_2b")!
    static let lfm25_3_2b       = ModelRegistry.shared.model(id: "lfm25_3_2b")!
    static let lfm25_vl_1_6b    = ModelRegistry.shared.model(id: "lfm25_vl_1_6b")!
    static let ministral_3b     = ModelRegistry.shared.model(id: "ministral_3b")!
    static let qwen35_2b_text   = ModelRegistry.shared.model(id: "qwen35_2b_text")!
    static let smollm3_3b       = ModelRegistry.shared.model(id: "smollm3_3b")!
}

// MARK: - Convenience collections

public extension Model {

    /// All registered models (replaces `CaseIterable.allCases`).
    static var allModels: [Model] {
        ModelRegistry.shared.models
    }

    /// Backward-compatible alias for ``allModels``.
    static var allCases: [Model] {
        allModels
    }

    /// All text-generation models, downloaded first.
    static var textModels: [Model] {
        allModels
            .filter { $0.category == .text }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }

    /// All general-purpose vision models, downloaded first.
    static var visionModels: [Model] {
        allModels
            .filter { $0.category == .vision }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }

    /// All OCR / document-specialized vision models, downloaded first.
    static var specializedModels: [Model] {
        allModels
            .filter { $0.category == .visionSpecialized }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }

    /// All GGUF-format models (require llama.cpp backend), downloaded first.
    static var ggufModels: [Model] {
        allModels
            .filter { $0.format == .gguf }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }

    /// All MLX-format models, downloaded first.
    static var mlxModels: [Model] {
        allModels
            .filter { $0.format == .mlx }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }

    /// Models runnable on the current device, filtered by hardware compatibility.
    static var runnableModels: [Model] {
        let profile = HardwareProfile.current()
        return allModels.filter { model in
            let assessment = HardwareAnalyzer.assess(model, profile: profile)
            return assessment.fitLevel.isRunnable
        }
    }

    /// All uncensored / abliterated models, downloaded first.
    static var uncensoredModels: [Model] {
        allModels
            .filter { $0.isUncensored }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }
}
