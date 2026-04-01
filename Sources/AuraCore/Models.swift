import Foundation

// MARK: - Model

/// A supported on-device model available for download and inference via MLX.
///
/// Each case maps to a Hugging Face repository (the raw value) and carries
/// metadata such as ``purpose``, ``displayName``, ``approximateSizeMB``,
/// and ``defaultDocumentPrompt``.
///
/// Models are grouped into three categories:
/// - **Text** — language-only generation (e.g. ``qwen3_1_7b``).
/// - **Vision** — multimodal image+text (e.g. ``qwen35_0_8b``).
/// - **Vision Specialized** — OCR / document extraction (e.g. ``fastVLM_0_5b_fp16``).
///
/// Use the convenience collections ``textModels``, ``visionModels``, and
/// ``specializedModels`` to list available models by category.
public enum Model: String, CaseIterable, Sendable {

    // MARK: Text
    case qwen3_0_6b   = "mlx-community/Qwen3-0.6B-4bit"
    case qwen3_1_7b   = "mlx-community/Qwen3-1.7B-4bit"
    case qwen3_4b     = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    case gemma3_1b    = "mlx-community/gemma-3-1b-it-4bit"
    case phi3_5_mini  = "mlx-community/Phi-3.5-mini-instruct-4bit"
    case llama3_2_1b  = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    case llama3_2_3b  = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    
    // MARK: Vision
    case qwen35_0_8b  = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
    case qwen35_2b    = "mlx-community/Qwen3.5-2B-4bit"
    case smolvlm_500m = "mlx-community/SmolVLM-500M-Instruct-bf16"
    case smolvlm_2b   = "mlx-community/SmolVLM-Instruct-4bit"
    
    // MARK: Vision · Specialized
    case fastVLM_0_5b_fp16   = "mlx-community/FastVLM-0.5B-bf16"
    case fastVLM_1_5b_int8   = "InsightKeeper/FastVLM-1.5B-MLX-8bit"
    case graniteDocling_258m  = "ibm-granite/granite-docling-258M-mlx"
    case graniteVision_3_3   = "mlx-community/granite-vision-3.2-2b-MLX"

    // MARK: GGUF · Large Models (llama.cpp backend)
    case llama3_1_8b_gguf    = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    case qwen2_5_7b_gguf     = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    case mistral_7b_gguf     = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    case phi3_medium_gguf    = "bartowski/Phi-3-medium-4k-instruct-GGUF"
    case gemma2_9b_gguf      = "bartowski/gemma-2-9b-it-GGUF"

    // MARK: GGUF · Large Models (macOS recommended, iOS streaming only)
    case llama3_1_70b_gguf   = "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF"
    case qwen2_5_32b_gguf    = "Qwen/Qwen2.5-32B-Instruct-GGUF"

    // MARK: Text · Uncensored / Abliterated (MLX)
    /// Josiefied Qwen3 1.7B — abliterated + Josiefied fine-tune. Runs on any iPhone 15+.
    case josiefied_qwen3_1_7b  = "mlx-community/Josiefied-Qwen3-1.7B-abliterated-v1-4bit"
    /// Josiefied Qwen3 4B — abliterated + Josiefied fine-tune. Requires 6 GB RAM.
    case josiefied_qwen3_4b    = "mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit"
    /// Josiefied Qwen3 8B — abliterated + Josiefied fine-tune. macOS / iPad Pro recommended.
    case josiefied_qwen3_8b    = "mlx-community/Josiefied-Qwen3-8B-abliterated-v1-4bit"
    /// Dolphin 2.9.3 Qwen2 1.5B — alignment-scrubbed dataset fine-tune. iOS-friendly.
    case dolphin_qwen2_1_5b    = "mlx-community/dolphin-2.9.3-qwen2-1.5b-4bit"

    // MARK: Text · Uncensored / Abliterated (GGUF)
    /// Dolphin 3.0 Qwen2.5 1.5B — alignment removed at training time. ~1 GB.
    case dolphin3_qwen25_1_5b_gguf  = "bartowski/Dolphin3.0-Qwen2.5-1.5B-GGUF"
    /// Dolphin 3.0 Qwen2.5 3B — alignment removed at training time. ~1.9 GB.
    case dolphin3_qwen25_3b_gguf    = "bartowski/Dolphin3.0-Qwen2.5-3b-GGUF"
    /// Dolphin 3.0 Llama 3.1 8B — alignment removed at training time. Most downloaded Dolphin variant.
    case dolphin3_llama31_8b_gguf   = "cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF"
    /// Llama 3.2 3B Uncensored — fine-tuned on uncensored dataset by chuanli11.
    case llama32_3b_uncensored_gguf = "bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF"
    /// Llama 3.1 8B Abliterated — refusal direction surgically removed (abliteration). Q4_K_M.
    case llama31_8b_abliterated_gguf = "bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF"
    
    // MARK: - Purpose

    /// The functional category of a model, determining which ``AuraLocal``
    /// factory method should be used to load it.
    public enum Purpose {
        /// Language-only generation — load with ``AuraLocal/text(_:onProgress:)``.
        case text
        /// Multimodal image+text — load with ``AuraLocal/vision(_:onProgress:)``.
        case vision
        /// OCR / document extraction — load with ``AuraLocal/specialized(_:onProgress:)``.
        /// When `docTags` is `true` the model outputs DocTags markup (e.g. Granite Docling).
        case visionSpecialized(docTags: Bool)
    }

    /// The functional category of this model.
    public var purpose: Purpose {
        switch self {
        case .qwen3_0_6b, .qwen3_1_7b, .qwen3_4b,
             .gemma3_1b, .phi3_5_mini,
             .llama3_2_1b, .llama3_2_3b,
             .llama3_1_8b_gguf, .qwen2_5_7b_gguf, .mistral_7b_gguf,
             .phi3_medium_gguf, .gemma2_9b_gguf,
             .llama3_1_70b_gguf, .qwen2_5_32b_gguf,
             .josiefied_qwen3_1_7b, .josiefied_qwen3_4b, .josiefied_qwen3_8b,
             .dolphin_qwen2_1_5b,
             .dolphin3_qwen25_1_5b_gguf, .dolphin3_qwen25_3b_gguf,
             .dolphin3_llama31_8b_gguf, .llama32_3b_uncensored_gguf,
             .llama31_8b_abliterated_gguf:
            return .text

        case .qwen35_0_8b, .qwen35_2b,
             .smolvlm_500m, .smolvlm_2b:
            return .vision

        case .fastVLM_0_5b_fp16, .fastVLM_1_5b_int8, .graniteVision_3_3:
            return .visionSpecialized(docTags: false)

        case .graniteDocling_258m:
            return .visionSpecialized(docTags: true)
        }
    }

    // MARK: - Format

    /// The weight format of this model.
    public var format: ModelFormat {
        switch self {
        case .llama3_1_8b_gguf, .qwen2_5_7b_gguf, .mistral_7b_gguf,
             .phi3_medium_gguf, .gemma2_9b_gguf,
             .llama3_1_70b_gguf, .qwen2_5_32b_gguf,
             .dolphin3_qwen25_1_5b_gguf, .dolphin3_qwen25_3b_gguf,
             .dolphin3_llama31_8b_gguf, .llama32_3b_uncensored_gguf,
             .llama31_8b_abliterated_gguf:
            return .gguf
        default:
            return .mlx
        }
    }

    /// The GGUF filename to download for this model (nil for MLX models).
    /// Uses Q4_K_M quantization as the default for best size/quality trade-off.
    public var ggufFilename: String? {
        switch self {
        case .llama3_1_8b_gguf:              return "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        case .qwen2_5_7b_gguf:               return "qwen2.5-7b-instruct-q4_k_m.gguf"
        case .mistral_7b_gguf:               return "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        case .phi3_medium_gguf:              return "Phi-3-medium-4k-instruct-Q4_K_M.gguf"
        case .gemma2_9b_gguf:                return "gemma-2-9b-it-Q4_K_M.gguf"
        case .llama3_1_70b_gguf:             return "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
        case .qwen2_5_32b_gguf:              return "qwen2.5-32b-instruct-q4_k_m.gguf"
        case .dolphin3_qwen25_1_5b_gguf:     return "Dolphin3.0-Qwen2.5-1.5B-Q4_K_M.gguf"
        case .dolphin3_qwen25_3b_gguf:       return "Dolphin3.0-Qwen2.5-3B-Q4_K_M.gguf"
        case .dolphin3_llama31_8b_gguf:      return "Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf"
        case .llama32_3b_uncensored_gguf:    return "Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf"
        case .llama31_8b_abliterated_gguf:   return "Meta-Llama-3.1-8B-Instruct-abliterated-Q4_K_M.gguf"
        default:                             return nil
        }
    }
    
    // MARK: - Metadata
    
    /// A human-readable name for display in UI (e.g. "Qwen3 1.7B").
    public var displayName: String {
        switch self {
            case .qwen3_0_6b:          return "Qwen3 0.6B"
            case .qwen3_1_7b:          return "Qwen3 1.7B"
            case .qwen3_4b:            return "Qwen3 4B"
            case .gemma3_1b:           return "Gemma 3 1B"
            case .phi3_5_mini:         return "Phi-3.5 Mini"
            case .llama3_2_1b:         return "Llama 3.2 1B"
            case .llama3_2_3b:         return "Llama 3.2 3B"
            case .qwen35_0_8b:         return "Qwen3.5 0.8B (default)"
            case .qwen35_2b:           return "Qwen3.5 2B"
            case .smolvlm_500m:        return "SmolVLM 500M"
            case .smolvlm_2b:          return "SmolVLM2 2B"
            case .fastVLM_0_5b_fp16:   return "FastVLM 0.5B FP16"
            case .fastVLM_1_5b_int8:   return "FastVLM 1.5B Int8"
            case .graniteDocling_258m: return "Granite Docling 258M (IBM)"
            case .graniteVision_3_3:   return "Granite Vision 3.2 2B"
            case .llama3_1_8b_gguf:              return "Llama 3.1 8B (GGUF)"
            case .qwen2_5_7b_gguf:               return "Qwen 2.5 7B (GGUF)"
            case .mistral_7b_gguf:               return "Mistral 7B v0.2 (GGUF)"
            case .phi3_medium_gguf:              return "Phi-3 Medium 14B (GGUF)"
            case .gemma2_9b_gguf:                return "Gemma 2 9B (GGUF)"
            case .llama3_1_70b_gguf:             return "Llama 3.1 70B (GGUF)"
            case .qwen2_5_32b_gguf:              return "Qwen 2.5 32B (GGUF)"
            case .josiefied_qwen3_1_7b:          return "Josiefied Qwen3 1.7B"
            case .josiefied_qwen3_4b:            return "Josiefied Qwen3 4B"
            case .josiefied_qwen3_8b:            return "Josiefied Qwen3 8B"
            case .dolphin_qwen2_1_5b:            return "Dolphin 2.9 Qwen2 1.5B"
            case .dolphin3_qwen25_1_5b_gguf:     return "Dolphin 3.0 Qwen2.5 1.5B (GGUF)"
            case .dolphin3_qwen25_3b_gguf:       return "Dolphin 3.0 Qwen2.5 3B (GGUF)"
            case .dolphin3_llama31_8b_gguf:      return "Dolphin 3.0 Llama 3.1 8B (GGUF)"
            case .llama32_3b_uncensored_gguf:    return "Llama 3.2 3B Uncensored (GGUF)"
            case .llama31_8b_abliterated_gguf:   return "Llama 3.1 8B Abliterated (GGUF)"
        }
    }
    
    /// Approximate download size in megabytes. Use this to display storage
    /// requirements before the user downloads a model.
    public var approximateSizeMB: Int {
        switch self {
            case .qwen3_0_6b:          return 400
            case .qwen3_1_7b:          return 1_000
            case .qwen3_4b:            return 2_500
            case .gemma3_1b:           return 700
            case .phi3_5_mini:         return 2_200
            case .llama3_2_1b:         return 700
            case .llama3_2_3b:         return 1_800
            case .qwen35_0_8b:         return 625
            case .qwen35_2b:           return 1_720
            case .smolvlm_500m:        return 1_020
            case .smolvlm_2b:          return 1_460
            case .fastVLM_0_5b_fp16:   return 1_250
            case .fastVLM_1_5b_int8:   return 800
            case .graniteDocling_258m: return 631
            case .graniteVision_3_3:   return 1_200
            case .llama3_1_8b_gguf:              return 4_700
            case .qwen2_5_7b_gguf:               return 4_400
            case .mistral_7b_gguf:               return 4_100
            case .phi3_medium_gguf:              return 8_000
            case .gemma2_9b_gguf:                return 5_500
            case .llama3_1_70b_gguf:             return 40_000
            case .qwen2_5_32b_gguf:              return 18_500
            case .josiefied_qwen3_1_7b:          return 950
            case .josiefied_qwen3_4b:            return 2_300
            case .josiefied_qwen3_8b:            return 4_610
            case .dolphin_qwen2_1_5b:            return 870
            case .dolphin3_qwen25_1_5b_gguf:     return 990
            case .dolphin3_qwen25_3b_gguf:       return 1_930
            case .dolphin3_llama31_8b_gguf:      return 4_920
            case .llama32_3b_uncensored_gguf:    return 2_240
            case .llama31_8b_abliterated_gguf:   return 4_920
        }
    }
    
    /// Default prompt for document/OCR extraction.
    /// Returns `nil` for non-specialized models.
    public var defaultDocumentPrompt: String? {
        switch self {
            case .fastVLM_0_5b_fp16, .fastVLM_1_5b_int8:
                return """
            You are a receipt OCR assistant. Extract all information from this receipt image \
            and return a JSON object with keys: store, date (YYYY-MM-DD), \
            items (array of {name, quantity, price}), subtotal, tax, total, currency. \
            Respond ONLY with valid JSON, no markdown.
            """
            case .graniteDocling_258m:
                return "Convert this page to docling."
            case .graniteVision_3_3:
                return "Describe the image in detail."
            default:
                return nil
        }
    }
    
    // MARK: - Storage
    
    /// Local cache directory where MLX downloads the model.
    /// Mirrors the `<org>/<repo>` folder structure used by mlx-swift.
    public var cacheDirectory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models")
            .appendingPathComponent(rawValue)
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
}

// MARK: - Convenience collections

public extension Model {
    /// All text-generation models, downloaded first.
    static var textModels: [Model] {
        allCases
            .filter { if case .text = $0.purpose { true } else { false } }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }
    /// All general-purpose vision models, downloaded first.
    static var visionModels: [Model] {
        allCases
            .filter { if case .vision = $0.purpose { true } else { false } }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }
    /// All OCR / document-specialized vision models, downloaded first.
    static var specializedModels: [Model] {
        allCases
            .filter { if case .visionSpecialized = $0.purpose { true } else { false } }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }

    /// All GGUF-format models (require llama.cpp backend), downloaded first.
    static var ggufModels: [Model] {
        allCases
            .filter { $0.format == .gguf }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }

    /// All MLX-format models, downloaded first.
    static var mlxModels: [Model] {
        allCases
            .filter { $0.format == .mlx }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }

    /// Models runnable on the current device, filtered by hardware compatibility.
    static var runnableModels: [Model] {
        let profile = HardwareProfile.current()
        return allCases.filter { model in
            let assessment = HardwareAnalyzer.assess(model, profile: profile)
            return assessment.fitLevel.isRunnable
        }
    }

    /// Whether this model is recommended for macOS only (too large for typical iOS devices).
    var isMacOSRecommended: Bool {
        approximateSizeMB >= 15_000  // 15 GB+ models
    }

    /// Whether this model is an uncensored or abliterated variant.
    var isUncensored: Bool {
        switch self {
        case .josiefied_qwen3_1_7b, .josiefied_qwen3_4b, .josiefied_qwen3_8b,
             .dolphin_qwen2_1_5b,
             .dolphin3_qwen25_1_5b_gguf, .dolphin3_qwen25_3b_gguf,
             .dolphin3_llama31_8b_gguf, .llama32_3b_uncensored_gguf,
             .llama31_8b_abliterated_gguf:
            return true
        default:
            return false
        }
    }

    /// All uncensored / abliterated models, downloaded first.
    static var uncensoredModels: [Model] {
        allCases
            .filter { $0.isUncensored }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }
}
