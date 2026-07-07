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

    // MARK: - Domain

    /// Optional domain specialization, orthogonal to ``Category`` (a code model is
    /// still a text model functionally). `nil` means general-purpose.
    public enum Domain: String, Codable, Sendable, CaseIterable, Identifiable {
        case code
        case security
        case finance
        case medicine

        public var id: String { rawValue }

        /// Human-readable section title.
        public var displayName: String {
            switch self {
            case .code:     "Code"
            case .security: "Security"
            case .finance:  "Finance"
            case .medicine: "Medicine"
            }
        }

        /// SF Symbol representing the domain.
        public var systemImage: String {
            switch self {
            case .code:     "chevron.left.forwardslash.chevron.right"
            case .security: "lock.shield"
            case .finance:  "chart.line.uptrend.xyaxis"
            case .medicine: "cross.case"
            }
        }

        /// System prompt that primes a model for this domain when testing it.
        public var testSystemPrompt: String {
            switch self {
            case .code:
                "You are an expert programming assistant. Answer with correct, concise code."
            case .security:
                "You are a mobile application security analyzer. Identify vulnerabilities (with CWE) and give a one-line fix for each. If the code is safe, reply exactly: NO VULNERABILITIES FOUND."
            case .finance:
                "You are a financial analysis assistant. Be precise, concise, and factual."
            case .medicine:
                "You are a clinical information assistant for EDUCATIONAL use only. You are not a doctor and do not give medical advice; recommend consulting a licensed professional."
            }
        }

        /// Curated example prompts to quickly test a model in this domain.
        public var samplePrompts: [String] {
            switch self {
            case .code:
                ["Write a Swift function that reverses a string without using reversed().",
                 "Explain the difference between a struct and a class in Swift, with a short example.",
                 "Refactor to be idiomatic Swift:\nfor i in 0..<arr.count { print(arr[i]) }"]
            case .security:
                ["Find the vulnerability and give a fix:\nlet q = \"SELECT * FROM users WHERE name = '\\(name)'\"",
                 "Is storing a password in UserDefaults secure? What should I use instead?",
                 "Review for issues:\nwebView.configuration.preferences.javaScriptEnabled = true\nwebView.load(URLRequest(url: untrustedURL))"]
            case .finance:
                ["Classify the sentiment (positive/negative/neutral): 'The company missed Q4 earnings and cut its dividend.'",
                 "Explain briefly what a P/E ratio tells an investor.",
                 "Summarize the risk of holding a single stock vs an index fund."]
            case .medicine:
                ["What are common causes of a persistent dry cough? (educational)",
                 "Explain what HbA1c measures and its general reference range.",
                 "What lifestyle changes are generally recommended for high blood pressure?"]
            }
        }
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

    /// Optional domain specialization (code / security / finance / medicine).
    /// `nil` = general-purpose. Absent in JSON decodes as `nil`.
    public let domain: Domain?

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

    /// Explicit download URL for a **user-supplied** model — e.g. a Hugging Face
    /// `…/resolve/<rev>/<file>.gguf` URL the user pasted. `nil` for catalog models, which derive the URL
    /// from `repoID`+`ggufFilename` at the pinned `main` revision. When set, the downloader fetches this
    /// exact URL, honoring any revision / subfolder. Absent in JSON decodes as `nil`. See
    /// ``fromHuggingFaceURL(_:displayName:)``.
    public let downloadURL: URL?

    /// A `file://` path to a GGUF the user **already has on disk** (from ollama, LM Studio, a manual download,
    /// another tool…). When set, the model is loaded IN PLACE — never copied or re-downloaded. `nil` for
    /// catalog + remote-download models. See ``ModelSource`` / ``fromURL(_:displayName:)``.
    public let localFileURL: URL?

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

    /// The on-disk location of this model's GGUF weights: a user's own file loaded in place
    /// (``localFileURL``), otherwise the download destination under ``cacheDirectory``. `nil` for MLX models
    /// or a GGUF with no filename. Both the downloader and the llama.cpp backends resolve the file here.
    public var resolvedFileURL: URL? {
        if let localFileURL { return localFileURL }
        guard let ggufFilename else { return nil }
        return cacheDirectory.appendingPathComponent(ggufFilename)
    }

    /// `true` when the snapshot is fully downloaded on disk.
    /// MLX models check the `.complete` marker at the `AuraHFDownloader` path
    /// (`$Caches/huggingface/hub/models--<sanitized>/snapshots/main/.complete`);
    /// GGUF models check `cacheDirectory` (legacy path).
    public var isDownloaded: Bool {
        switch format {
        case .mlx:
            let sanitized = repoID.replacingOccurrences(of: "/", with: "--")
            let marker = FileManager.default
                .urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appending(path: "huggingface/hub/models--\(sanitized)/snapshots/main/.complete")
            return FileManager.default.fileExists(atPath: marker.path())
        case .gguf:
            // Check the actual .gguf FILE (an in-place local file, or the download destination), not just the
            // cache dir: a partial/interrupted download leaves the dir (and a URLSession temp) but never the
            // destination file (moved into place only on completion), so a dir-only check reports a partial
            // download as ready and then fails at load.
            guard let file = resolvedFileURL else { return false }
            return FileManager.default.fileExists(atPath: file.path)
        }
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

// MARK: - User-supplied models (bring-your-own from Hugging Face)

public extension Model {

    /// Build a **user-supplied** GGUF model from a Hugging Face file URL. The user is responsible for
    /// choosing the model and complying with its license — nothing is bundled or curated.
    ///
    /// Accepts a canonical HF file URL, either form:
    /// - `https://huggingface.co/<owner>/<repo>/resolve/<rev>/[<subdir>/]<file>.gguf`
    /// - `https://huggingface.co/<owner>/<repo>/blob/<rev>/[<subdir>/]<file>.gguf`  (the UI "view file" link)
    ///
    /// Returns `nil` for anything that isn't an https `huggingface.co` URL pointing at a `.gguf` file with a
    /// resolvable `<owner>/<repo>` and revision. The `blob` form is normalised to the `resolve` download URL.
    ///
    /// - Parameters:
    ///   - urlString: the pasted Hugging Face URL.
    ///   - displayName: optional friendly name; defaults to the file name.
    static func fromHuggingFaceURL(_ urlString: String, displayName: String? = nil) -> Model? {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let comps = URLComponents(string: trimmed),
              comps.scheme?.lowercased() == "https",
              let host = comps.host?.lowercased(),
              host == "huggingface.co" || host == "www.huggingface.co"
        else { return nil }

        // /<owner>/<repo>/(resolve|blob)/<rev>/…/<file>.gguf
        let segs = comps.path.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        guard let kindIdx = segs.firstIndex(where: { $0 == "resolve" || $0 == "blob" }),
              kindIdx == 2,                                     // exactly owner/repo before it
              segs.indices.contains(kindIdx + 1),              // a revision follows
              segs.count >= kindIdx + 2,                       // …and at least a filename after the revision
              let filename = segs.last,
              filename.lowercased().hasSuffix(".gguf"),
              filename.count > ".gguf".count
        else { return nil }

        let repoID = "\(segs[0])/\(segs[1])"

        // Canonical download URL — normalise `blob` → `resolve`, preserve revision + any subfolder + query.
        var download = comps
        download.path = "/" + segs.enumerated()
            .map { $0.offset == kindIdx ? "resolve" : $0.element }
            .joined(separator: "/")
        guard let downloadURL = download.url else { return nil }

        let id = "custom__" + repoID.replacingOccurrences(of: "/", with: "--") + "__" + filename
        return custom(id: id, repoID: repoID, displayName: displayName ?? filename,
                      filename: filename, downloadURL: downloadURL, localFileURL: nil)
    }

    /// Auto-detect the source of a pasted string and build a user-supplied GGUF ``Model``. The user owns the
    /// model-license choice — nothing is bundled or curated. Recognizes:
    /// - a `file://` path → an in-place local `.gguf` (never copied or re-downloaded);
    /// - a `huggingface.co` URL → Hugging Face (nice `<owner>/<repo>` extraction);
    /// - any other `https` URL ending in `.gguf` → a direct download (GitHub releases, ModelScope, a CDN, a
    ///   self-hosted server…).
    /// Returns `nil` for anything not a recognizable `.gguf` source.
    static func fromURL(_ string: String, displayName: String? = nil) -> Model? {
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let comps = URLComponents(string: trimmed) else { return nil }
        switch comps.scheme?.lowercased() {
        case "file":
            guard let url = comps.url else { return nil }
            return from(.localFile(url), displayName: displayName)
        case "https":
            if let host = comps.host?.lowercased(), host == "huggingface.co" || host == "www.huggingface.co" {
                return from(.huggingFace(url: trimmed), displayName: displayName)
            }
            guard let url = comps.url else { return nil }
            return from(.directURL(url), displayName: displayName)
        default:
            return nil
        }
    }

    /// Build a user-supplied GGUF ``Model`` from an explicit ``ModelSource`` — the bridge every host app can
    /// call to add a model from any supported source without knowing the download mechanics.
    static func from(_ source: ModelSource, displayName: String? = nil) -> Model? {
        switch source {
        case .huggingFace(let url):
            return fromHuggingFaceURL(url, displayName: displayName)

        case .directURL(let url):
            guard url.scheme?.lowercased() == "https" else { return nil }
            let filename = url.lastPathComponent
            guard filename.lowercased().hasSuffix(".gguf"), filename.count > ".gguf".count else { return nil }
            // Group the cache by host so re-adding the same URL reuses an existing download.
            let repoID = "url/" + (url.host ?? "download").replacingOccurrences(of: ".", with: "_")
            let id = "custom__" + repoID.replacingOccurrences(of: "/", with: "--") + "__" + filename
            return custom(id: id, repoID: repoID, displayName: displayName ?? filename,
                          filename: filename, downloadURL: url, localFileURL: nil)

        case .localFile(let url):
            guard url.isFileURL else { return nil }
            let filename = url.lastPathComponent
            guard filename.lowercased().hasSuffix(".gguf"), filename.count > ".gguf".count else { return nil }
            return custom(id: "local__" + filename, repoID: "local", displayName: displayName ?? filename,
                          filename: filename, downloadURL: nil, localFileURL: url)
        }
    }

    /// Whether this is a user-supplied model (remote download URL or in-place local file) vs a curated catalog entry.
    var isUserSupplied: Bool { downloadURL != nil || localFileURL != nil }

    /// Synthesize a user-supplied GGUF ``Model``, filling the many fixed metadata defaults in one place.
    private static func custom(id: String, repoID: String, displayName: String, filename: String,
                               downloadURL: URL?, localFileURL: URL?) -> Model {
        Model(id: id, repoID: repoID, displayName: displayName, category: .text, domain: nil, docTags: false,
              format: .gguf, approximateSizeMB: 0, isUncensored: false, ggufFilename: filename,
              defaultDocumentPrompt: nil, numLayers: 0, kvHeads: 0, headDim: 0,
              downloadURL: downloadURL, localFileURL: localFileURL)
    }
}

// MARK: - ModelSource

/// Where a user-supplied model comes from — the bridge AuraLocal uses to download/load from any source.
public enum ModelSource: Sendable, Equatable {
    /// A Hugging Face `.gguf` file URL (`resolve`/`blob` form).
    case huggingFace(url: String)
    /// Any direct HTTPS URL to a `.gguf` file (GitHub releases, ModelScope, a CDN, a self-hosted server…).
    case directURL(URL)
    /// A `.gguf` the user already has on disk — loaded in place, never copied or re-downloaded.
    case localFile(URL)
}

// MARK: - Static constants (backward compatibility)

public extension Model {

    // MARK: Text (MLX)
    static let qwen3_0_6b   = ModelRegistry.shared.model(id: "qwen3_0_6b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    // Fall back to a runnable GGUF model when MLX models are filtered out (MLX backend removed).
    static let qwen3_1_7b   = ModelRegistry.shared.model(id: "qwen3_1_7b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let qwen3_4b     = ModelRegistry.shared.model(id: "qwen3_4b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let gemma3_1b    = ModelRegistry.shared.model(id: "gemma3_1b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let phi3_5_mini  = ModelRegistry.shared.model(id: "phi3_5_mini") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let llama3_2_1b  = ModelRegistry.shared.model(id: "llama3_2_1b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let llama3_2_3b  = ModelRegistry.shared.model(id: "llama3_2_3b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    // MARK: Vision (MLX)
    static let qwen35_0_8b  = ModelRegistry.shared.model(id: "qwen35_0_8b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let qwen35_2b    = ModelRegistry.shared.model(id: "qwen35_2b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let smolvlm_500m = ModelRegistry.shared.model(id: "smolvlm_500m") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let smolvlm_2b   = ModelRegistry.shared.model(id: "smolvlm_2b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    // MARK: Vision Specialized (MLX)
    static let fastVLM_0_5b_fp16  = ModelRegistry.shared.model(id: "fastVLM_0_5b_fp16") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let fastVLM_1_5b_int8  = ModelRegistry.shared.model(id: "fastVLM_1_5b_int8") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let graniteDocling_258m = ModelRegistry.shared.model(id: "graniteDocling_258m") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let graniteVision_3_3  = ModelRegistry.shared.model(id: "graniteVision_3_3") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    // MARK: GGUF Standard
    static let llama3_1_8b_gguf   = ModelRegistry.shared.model(id: "llama3_1_8b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let qwen2_5_7b_gguf    = ModelRegistry.shared.model(id: "qwen2_5_7b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let mistral_7b_gguf    = ModelRegistry.shared.model(id: "mistral_7b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let phi3_medium_gguf   = ModelRegistry.shared.model(id: "phi3_medium_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let gemma2_9b_gguf     = ModelRegistry.shared.model(id: "gemma2_9b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    // MARK: GGUF Large
    static let llama3_1_70b_gguf  = ModelRegistry.shared.model(id: "llama3_1_70b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let qwen2_5_32b_gguf   = ModelRegistry.shared.model(id: "qwen2_5_32b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    // MARK: Uncensored (MLX)
    static let josiefied_qwen3_1_7b = ModelRegistry.shared.model(id: "josiefied_qwen3_1_7b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let josiefied_qwen3_4b   = ModelRegistry.shared.model(id: "josiefied_qwen3_4b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let josiefied_qwen3_8b   = ModelRegistry.shared.model(id: "josiefied_qwen3_8b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let dolphin_qwen2_1_5b   = ModelRegistry.shared.model(id: "dolphin_qwen2_1_5b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    // MARK: Uncensored (GGUF)
    static let dolphin3_qwen25_1_5b_gguf  = ModelRegistry.shared.model(id: "dolphin3_qwen25_1_5b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let dolphin3_qwen25_3b_gguf    = ModelRegistry.shared.model(id: "dolphin3_qwen25_3b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let dolphin3_llama31_8b_gguf   = ModelRegistry.shared.model(id: "dolphin3_llama31_8b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let llama32_3b_uncensored_gguf = ModelRegistry.shared.model(id: "llama32_3b_uncensored_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let llama31_8b_abliterated_gguf = ModelRegistry.shared.model(id: "llama31_8b_abliterated_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    // MARK: New Models
    static let granite4_tiny    = ModelRegistry.shared.model(id: "granite4_tiny") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let granite4_compact = ModelRegistry.shared.model(id: "granite4_compact") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let cogito_v1_3b     = ModelRegistry.shared.model(id: "cogito_v1_3b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let gemma4_1b        = ModelRegistry.shared.model(id: "gemma4_1b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let gemma4_4b        = ModelRegistry.shared.model(id: "gemma4_4b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let gemma4_12b       = ModelRegistry.shared.model(id: "gemma4_12b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let lfm2_1_2b        = ModelRegistry.shared.model(id: "lfm2_1_2b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let lfm25_3_2b       = ModelRegistry.shared.model(id: "lfm25_3_2b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let lfm25_vl_1_6b    = ModelRegistry.shared.model(id: "lfm25_vl_1_6b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let ministral_3b     = ModelRegistry.shared.model(id: "ministral_3b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let qwen35_2b_text   = ModelRegistry.shared.model(id: "qwen35_2b_text") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
    static let smollm3_3b       = ModelRegistry.shared.model(id: "smollm3_3b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    // MARK: Domain-specialized (medical / financial)

    /// MedGemma 4B IT 4-bit — Google fine-tuned para razonamiento clínico
    /// multimodal (texto + imagen médica). Licencia HAI-DEF — propagar
    /// use restrictions a downstream users. Vision model: load con
    /// `AuraLocal.vision(.medgemma_4b)` aunque se use text-only.
    static let medgemma_4b = ModelRegistry.shared.model(id: "medgemma_4b") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    /// FinGPT MT (Llama 3 8B base) — multi-task financial assistant:
    /// sentiment, FAQ financiero, headline classification, NER financiero.
    /// GGUF Q4_K_M, ~5 GB. Requiere device con 6 GB+ RAM efectiva.
    static let fingpt_mt_llama3_8b_gguf = ModelRegistry.shared.model(id: "fingpt_mt_llama3_8b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    /// FinGPT Sentiment LFM2 1.2B — lightweight financial sentiment-only.
    /// GGUF Q4_K_M, ~900 MB. Para clasificación de sentiment en news/reports;
    /// NO usar para chat general — está RL-tuneado para single-token outputs.
    static let fingpt_sentiment_lfm2_1_2b_gguf = ModelRegistry.shared.model(id: "fingpt_sentiment_lfm2_1_2b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!

    /// Security SLM Gemma 4 E2B — mobile (iOS/Android) vulnerability triage.
    /// GGUF Q4_K_M, ~3.4 GB. Triage only, not sole authority (misses some criticals).
    static let security_gemma_e2b_gguf = ModelRegistry.shared.model(id: "security_gemma_e2b_gguf") ?? ModelRegistry.shared.models.first(where: { $0.format == .gguf })!
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

    /// All models tagged with `domain`, downloaded first.
    static func models(in domain: Domain) -> [Model] {
        allModels
            .filter { $0.domain == domain }
            .sorted { $0.isDownloaded && !$1.isDownloaded }
    }
}
