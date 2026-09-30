import Foundation

// MARK: - RepoFile

/// One file of a Hugging Face repository, as listed by `/api/models/{id}?blobs=true`.
public struct RepoFile: Sendable, Equatable {
    /// Repo-relative path (may include subfolders).
    public let path: String
    public let sizeBytes: Int64?

    public init(path: String, sizeBytes: Int64?) {
        self.path = path
        self.sizeBytes = sizeBytes
    }

    public var filename: String { (path as NSString).lastPathComponent }
    /// AuraLocal's MLX downloader lists the tree non-recursively, so only these files reach the device.
    public var isTopLevel: Bool { !path.contains("/") }
    public var isSafetensors: Bool { path.lowercased().hasSuffix(".safetensors") }
    public var isGGUF: Bool { path.lowercased().hasSuffix(".gguf") }
    /// A vision projector shipped next to GGUF weights (llama.cpp `mmproj`).
    public var isProjector: Bool { filename.lowercased().contains("mmproj") }
}

// MARK: - HFRepoInfo

/// The parts of `GET /api/models/{id}?blobs=true` the compatibility rules read.
public struct HFRepoInfo: Sendable, Equatable {
    public let repoID: String
    public let files: [RepoFile]
    public let tags: [String]
    public let pipelineTag: String?
    public let libraryName: String?
    /// `nil` when the repo is not gated; otherwise HF's mode (`auto` / `manual`).
    public let gatedMode: String?
    public let isPrivate: Bool
    /// `cardData.license` (the first one when the card lists several).
    public let licenseID: String?
    /// `cardData.license_name`, set for `other` licenses.
    public let licenseName: String?
    public let licenseLink: String?
    /// HF's own GGUF summary, a fallback when the header cannot be read.
    public let ggufArchitecture: String?
    public let ggufContextLength: Int?

    public init(repoID: String, files: [RepoFile], tags: [String] = [], pipelineTag: String? = nil,
                libraryName: String? = nil, gatedMode: String? = nil, isPrivate: Bool = false,
                licenseID: String? = nil, licenseName: String? = nil, licenseLink: String? = nil,
                ggufArchitecture: String? = nil, ggufContextLength: Int? = nil) {
        self.repoID = repoID
        self.files = files
        self.tags = tags
        self.pipelineTag = pipelineTag
        self.libraryName = libraryName
        self.gatedMode = gatedMode
        self.isPrivate = isPrivate
        self.licenseID = licenseID
        self.licenseName = licenseName
        self.licenseLink = licenseLink
        self.ggufArchitecture = ggufArchitecture
        self.ggufContextLength = ggufContextLength
    }

    public var isGated: Bool { gatedMode != nil }

    /// Decode the model-info payload. Pure, so fixtures can drive it.
    public static func parse(_ data: Data) -> HFRepoInfo? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let repoID = (object["id"] as? String) ?? (object["modelId"] as? String)
        else { return nil }
        let siblings = object["siblings"] as? [[String: Any]] ?? []
        let card = object["cardData"] as? [String: Any] ?? [:]
        let gguf = object["gguf"] as? [String: Any] ?? [:]
        return HFRepoInfo(
            repoID: repoID,
            files: siblings.compactMap(file(from:)),
            tags: object["tags"] as? [String] ?? [],
            pipelineTag: object["pipeline_tag"] as? String,
            libraryName: object["library_name"] as? String,
            gatedMode: gatedMode(object["gated"]),
            isPrivate: object["private"] as? Bool ?? false,
            licenseID: (card["license"] as? String) ?? (card["license"] as? [String])?.first,
            licenseName: card["license_name"] as? String,
            licenseLink: card["license_link"] as? String,
            ggufArchitecture: gguf["architecture"] as? String,
            ggufContextLength: gguf["context_length"] as? Int)
    }

    private static func file(from sibling: [String: Any]) -> RepoFile? {
        guard let path = sibling["rfilename"] as? String else { return nil }
        let lfs = sibling["lfs"] as? [String: Any]
        let size = (sibling["size"] as? Int64) ?? (lfs?["size"] as? Int64)
        return RepoFile(path: path, sizeBytes: size)
    }

    /// `gated` is `false`, `"auto"` or `"manual"`.
    private static func gatedMode(_ raw: Any?) -> String? {
        if let flag = raw as? Bool { return flag ? "manual" : nil }
        if let mode = raw as? String, mode.lowercased() != "false" { return mode }
        return nil
    }
}

// MARK: - ModelConfigFacts

/// The `config.json` fields the rules need, flattened (a VLM's `text_config` fills in what the top level lacks).
public struct ModelConfigFacts: Sendable, Equatable {
    /// Top-level `model_type` — the key the MLX factories dispatch on.
    public let modelType: String?
    /// `text_config.model_type` (informative; the factories ignore it).
    public let textModelType: String?
    public let architectures: [String]
    /// MLX-style quantization (`quantization`, or `quantization_config` without a `quant_method`).
    public let isMLXQuantized: Bool
    public let quantizationBits: Int?
    /// `quantization_config.quant_method` (gptq, awq, bitsandbytes, fp8…): a non-MLX quantizer.
    public let foreignQuantMethod: String?
    public let numLayers: Int?
    public let kvHeads: Int?
    public let headDim: Int?
    /// Trained context: `max_position_embeddings`, `n_positions`, `seq_length`…
    public let contextLength: Int?
    /// `type` / `rope_type` of `rope_scaling` (or `rope_parameters`).
    public let ropeType: String?
    public let ropeKeys: [String]
    /// Which key held the rope settings: `rope_scaling` or `rope_parameters`.
    public let ropeSource: String?
    public let hasVisionConfig: Bool
    /// Multi-token-prediction layers (`mtp_num_hidden_layers` / `num_nextn_predict_layers`).
    public let mtpLayers: Int?

    public init(modelType: String?, textModelType: String? = nil, architectures: [String] = [],
                isMLXQuantized: Bool = false, quantizationBits: Int? = nil, foreignQuantMethod: String? = nil,
                numLayers: Int? = nil, kvHeads: Int? = nil, headDim: Int? = nil, contextLength: Int? = nil,
                ropeType: String? = nil, ropeKeys: [String] = [], ropeSource: String? = nil,
                hasVisionConfig: Bool = false,
                mtpLayers: Int? = nil) {
        self.modelType = modelType
        self.textModelType = textModelType
        self.architectures = architectures
        self.isMLXQuantized = isMLXQuantized
        self.quantizationBits = quantizationBits
        self.foreignQuantMethod = foreignQuantMethod
        self.numLayers = numLayers
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.contextLength = contextLength
        self.ropeType = ropeType
        self.ropeKeys = ropeKeys
        self.ropeSource = ropeSource
        self.hasVisionConfig = hasVisionConfig
        self.mtpLayers = mtpLayers
    }

    /// Decode `config.json` (JSON5 accepted, as mlx-swift-lm does). `nil` when it is not a JSON object.
    public static func parse(_ data: Data) -> ModelConfigFacts? {
        guard let root = try? JSONSerialization.jsonObject(with: data, options: [.json5Allowed]) as? [String: Any]
        else { return nil }
        let text = root["text_config"] as? [String: Any] ?? [:]
        func int(_ keys: String...) -> Int? {
            for key in keys {
                if let value = (root[key] as? NSNumber) ?? (text[key] as? NSNumber) { return value.intValue }
            }
            return nil
        }
        let mlxQuant = root["quantization"] as? [String: Any]
        let quantConfig = root["quantization_config"] as? [String: Any]
        let method = quantConfig?["quant_method"] as? String
        let ropeSource = ["rope_scaling", "rope_parameters"].first { root[$0] is [String: Any] || text[$0] is [String: Any] }
        let rope = ropeSource.flatMap { (root[$0] as? [String: Any]) ?? (text[$0] as? [String: Any]) }
        let heads = int("num_attention_heads", "n_head")
        let hidden = int("hidden_size", "n_embd")
        let derivedHeadDim = heads.flatMap { count in hidden.map { count > 0 ? $0 / count : 0 } }
        return ModelConfigFacts(
            modelType: root["model_type"] as? String,
            textModelType: text["model_type"] as? String,
            architectures: root["architectures"] as? [String] ?? [],
            isMLXQuantized: mlxQuant != nil || (quantConfig != nil && method == nil),
            quantizationBits: ((mlxQuant ?? quantConfig)?["bits"] as? NSNumber)?.intValue,
            foreignQuantMethod: method,
            numLayers: int("num_hidden_layers", "n_layer", "num_layers", "n_layers"),
            kvHeads: int("num_key_value_heads") ?? heads,
            headDim: int("head_dim") ?? derivedHeadDim,
            contextLength: int("max_position_embeddings", "n_positions", "seq_length", "max_sequence_length"),
            ropeType: (rope?["type"] as? String) ?? (rope?["rope_type"] as? String),
            ropeKeys: rope.map { $0.keys.sorted() } ?? [],
            ropeSource: ropeSource,
            hasVisionConfig: root["vision_config"] != nil,
            mtpLayers: int("mtp_num_hidden_layers", "num_nextn_predict_layers"))
    }
}

// MARK: - FetchProblem

/// A resource the checker tried to read.
public enum FetchedResource: String, Sendable, CaseIterable {
    case repository = "repository listing"
    case config = "config.json"
    case weightIndex = "model.safetensors.index.json"
    case safetensorsHeader = "safetensors header"
    case ggufHeader = "GGUF header"
}

/// Why a resource could not be read.
public struct FetchProblem: Sendable, Equatable {
    public let subject: FetchedResource
    /// HTTP status, or `nil` for a transport / parse failure.
    public let statusCode: Int?
    public let message: String

    public init(subject: FetchedResource, statusCode: Int?, message: String) {
        self.subject = subject
        self.statusCode = statusCode
        self.message = message
    }

    public var needsToken: Bool { statusCode == 401 || statusCode == 403 }
}

// MARK: - RepoSnapshot

/// Everything fetched about one repository. The rules are pure functions of this, so tests build it inline.
public struct RepoSnapshot: Sendable {
    public let repoID: String
    public var listing: HFRepoInfo?
    public var configuration: ModelConfigFacts?
    /// `weight_map` of `model.safetensors.index.json`: tensor name → file.
    public var weightMap: [String: String]?
    /// Tensor names of a single-file repo, read from the safetensors header.
    public var singleFileHeader: SafetensorsHeader?
    /// Tensor counts of top-level `*.safetensors` files the weight map does not reference.
    public var extraTensorCounts: [String: Int]
    /// Header of one GGUF file (``ggufSamplePath``) — all quants of a repo share it.
    public var ggufMetadata: GGUFHeader?
    public var ggufSamplePath: String?
    public var problems: [FetchProblem]

    public init(repoID: String, listing: HFRepoInfo? = nil, configuration: ModelConfigFacts? = nil,
                weightMap: [String: String]? = nil, singleFileHeader: SafetensorsHeader? = nil,
                extraTensorCounts: [String: Int] = [:], ggufMetadata: GGUFHeader? = nil,
                ggufSamplePath: String? = nil, problems: [FetchProblem] = []) {
        self.repoID = repoID
        self.listing = listing
        self.configuration = configuration
        self.weightMap = weightMap
        self.singleFileHeader = singleFileHeader
        self.extraTensorCounts = extraTensorCounts
        self.ggufMetadata = ggufMetadata
        self.ggufSamplePath = ggufSamplePath
        self.problems = problems
    }

    public func problem(for subject: FetchedResource) -> FetchProblem? {
        problems.first { $0.subject == subject }
    }

    /// Decode `model.safetensors.index.json` into its weight map.
    public static func weightMap(fromIndexJSON data: Data) -> [String: String]? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return object["weight_map"] as? [String: String]
    }
}
