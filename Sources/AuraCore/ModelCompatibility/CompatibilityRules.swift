import Foundation

// MARK: - RuleInput

/// A snapshot plus the facts every rule derives from it, computed once.
struct RuleInput: Sendable {
    let source: RepoSnapshot
    let target: DevicePreset
    let weightFormat: DetectedFormat
    /// Tensor names from the weight map, else from the shipped files' headers; `nil` when neither was read.
    let tensorNames: [String]?
    let quantGroups: [GGUFQuantGroup]

    init(snapshot: RepoSnapshot, target: DevicePreset) {
        source = snapshot
        self.target = target
        weightFormat = DetectedFormat.detect(snapshot)
        tensorNames = snapshot.liveWeightMap.map { Array($0.keys) } ?? snapshot.singleFileHeader?.tensorNames
        quantGroups = GGUFQuantGroup.groups(in: snapshot.listing?.files ?? [])
    }

    var listing: HFRepoInfo? { source.listing }
    var files: [RepoFile] { source.listing?.files ?? [] }
    var settings: ModelConfigFacts? { source.configuration }
    var modelType: String? { settings?.modelType }
    var ggufMetadata: GGUFHeader? { source.ggufMetadata }
    var ggufArchitecture: String? { ggufMetadata?.architecture ?? listing?.ggufArchitecture }
    /// `__metadata__` of the safetensors file mlx-swift-lm hands to `sanitize(weights:metadata:)`.
    var safetensorsMetadata: [String: String]? { source.singleFileHeader?.metadata ?? source.weightMetadata }

    /// Top-level tensor-name prefix → tensor count.
    var tensorPrefixes: [String: Int] {
        (tensorNames ?? []).reduce(into: [:]) { counts, name in
            counts[name.split(separator: ".").first.map(String.init) ?? name, default: 0] += 1
        }
    }

    /// Vision-tower prefixes present in the weights (`vision_tower`, `visual`, …).
    var visionTowers: [String] {
        present(CompatibilityRules.visionTensorPrefixes)
    }

    /// The `prefixes` some tensor name starts with (as `prefix.`), in the given order.
    func present(_ prefixes: [String]) -> [String] {
        let names = tensorNames ?? []
        return prefixes.filter { prefix in names.contains { $0.hasPrefix(prefix + ".") } }
    }

    /// Whether the weights include a vision tower; `nil` when the tensor names are unknown.
    var hasVisionTensors: Bool? {
        tensorNames == nil ? nil : !visionTowers.isEmpty
    }

    /// Text or vision, as AuraLocal would load it: vision only for a registered VLM type with vision weights.
    var inferredCategory: Model.Category? {
        switch weightFormat {
        case .gguf: return .text
        case .mlx:
            guard let modelType else { return nil }
            let isLLM = PinnedRuntimes.mlxLLMModelTypes.contains(modelType)
            let isVLM = PinnedRuntimes.mlxVLMModelTypes.contains(modelType)
            let vision = hasVisionTensors ?? (settings?.hasVisionConfig ?? false)
            if isVLM && (vision || !isLLM) { return .vision }
            return isLLM ? .text : nil
        default: return nil
        }
    }

    /// Bytes AuraLocal would load: the weight-map files, else every top-level safetensors file.
    var safetensorsBytes: Int64? {
        let top = files.filter { $0.isSafetensors && $0.isTopLevel }
        let chosen: [RepoFile]
        if let weightMap = source.liveWeightMap {
            let referenced = Set(weightMap.values)
            chosen = files.filter { referenced.contains($0.path) }
        } else {
            chosen = weightFormat == .imageGeneration ? files.filter(\.isSafetensors) : top
        }
        let sizes = chosen.compactMap(\.sizeBytes)
        return sizes.isEmpty ? nil : sizes.saturatingSum()
    }

    /// Layers, KV heads and head width from the GGUF header or config.json. Values past these limits come from a
    /// corrupt or hostile file and count as unknown, so sizing never multiplies them.
    var layerCount: Int? {
        Self.plausible(weightFormat == .gguf ? ggufMetadata?.blockCount : settings?.numLayers, upTo: 4096)
    }
    var kvHeadCount: Int? {
        Self.plausible(weightFormat == .gguf ? ggufMetadata?.headCountKV : settings?.kvHeads, upTo: 1024)
    }
    var headWidth: Int? {
        Self.plausible(weightFormat == .gguf ? ggufMetadata?.headDimension : settings?.headDim, upTo: 8192)
    }

    static func plausible(_ value: Int?, upTo limit: Int) -> Int? {
        value.flatMap { (1...limit).contains($0) ? $0 : nil }
    }

    var trainedContext: Int? {
        switch weightFormat {
        case .gguf: ggufMetadata?.contextLength ?? listing?.ggufContextLength
        case .imageGeneration: nil
        default: settings?.contextLength
        }
    }
}

// MARK: - CompatibilityRule

/// One check. Rules are pure functions of the fetched snapshot and the target device.
public struct CompatibilityRule: Sendable {
    public let id: String
    public let summary: String
    let evaluate: @Sendable (RuleInput) -> [CompatibilityFinding]

    /// The rule's findings, each stamped with ``id``.
    func findings(for input: RuleInput) -> [CompatibilityFinding] {
        evaluate(input).map { CompatibilityFinding(rule: id, level: $0.level, title: $0.title, detail: $0.detail) }
    }
}

// MARK: - CompatibilityRules

/// The rule set and the tables it reads. Evidence for each table is next to it.
public enum CompatibilityRules {

    public static let all: [CompatibilityRule] = [
        formatRule, generativeRule, mlxConfigRule, mlxModelTypeRule, weightPrefixRule, textOnlyTowerRule,
        extraSafetensorsRule, visionWeightsRule, ropeRule, tokenizerRule, ggufArchitectureRule, ggufNextNRule, ggufShardRule,
        ggufHeaderRule, ggufProjectorRule, imageGenerationRule, licenseRule, contextRule,
    ]

    // MARK: Tables

    /// Pipelines that do not generate text.
    static let nonGenerativePipelines: Set<String> = [
        "fill-mask", "feature-extraction", "sentence-similarity", "zero-shot-image-classification",
        "image-classification", "text-classification", "token-classification", "zero-shot-classification",
        "question-answering", "image-feature-extraction", "object-detection", "image-segmentation",
        "depth-estimation", "audio-classification", "automatic-speech-recognition", "text-to-speech",
        "text-to-audio", "zero-shot-object-detection", "mask-generation", "video-classification",
        "table-question-answering", "tabular-classification", "tabular-regression", "text-ranking",
        "keypoint-detection", "voice-activity-detection",
    ]

    /// transformers heads that classify, embed or fill masks instead of generating.
    static let nonGenerativeArchitectureSuffixes = [
        "ForMaskedLM", "ForSequenceClassification", "ForTokenClassification", "ForQuestionAnswering",
        "ForImageClassification", "ForMultipleChoice", "ForNextSentencePrediction", "ForPreTraining",
        "ForZeroShotImageClassification", "ForAudioClassification", "ForCTC",
    ]

    /// Encoder-only / dual-encoder model types.
    static let encoderModelTypes: Set<String> = [
        "bert", "distilbert", "roberta", "xlm-roberta", "deberta", "deberta-v2", "electra", "albert", "mpnet",
        "camembert", "modernbert", "nomic_bert", "siglip", "siglip2", "clip", "vit", "dinov2",
    ]

    /// Top-level tensor prefixes that hold a vision tower or projector.
    static let visionTensorPrefixes = [
        "vision_tower", "visual", "vision_model", "vision_encoder", "multi_modal_projector", "mm_projector",
        "model.visual", "model.vision_tower", "model.vision_model", "model.vision_encoder",
        "model.multi_modal_projector",
    ]

    /// Audio towers and multimodal embedders (Gemma 3n / Gemma 4 layout).
    static let auxiliaryTowerPrefixes = [
        "audio_tower", "embed_audio", "embed_vision", "model.audio_tower", "model.embed_audio", "model.embed_vision",
    ]

    /// Tags Hugging Face puts on embedding models, whatever their `pipeline_tag` says.
    static let embeddingTags: Set<String> = ["sentence-transformers", "sentence-similarity", "feature-extraction"]

    /// Model types whose pinned `sanitize` accepts only some top-level prefixes.
    static let weightPrefixPolicies: [String: WeightPrefixPolicy] = {
        let qwen35 = WeightPrefixPolicy(
            handledPrefixes: ["language_model", "model", "lm_head", "vision_tower", "mtp"],
            evidence: "mlx-swift-lm 3.31.3: the qwen3_5 LLM sanitize (MLXLLM/Models/Qwen35.swift) drops "
                + "`vision_tower.*` / `model.visual.*`, filters `mtp.` and prefixes the rest with `language_model.`; "
                + "the VLM sanitize (MLXVLM/Models/Qwen35.swift) only renames `model.visual` → `vision_tower`. "
                + "Any other prefix stays unmatched and `update(parameters:verify: [.all])` throws.",
            mlxFormatVisionPrefixes: ["language_model", "vision_tower"],
            mlxFormatEvidence: "mlx-swift-lm 3.31.3: the VLM `sanitize(weights:metadata:)` (MLXVLM/Models/Qwen35.swift) "
                + "returns weights whose safetensors `__metadata__.format` is `mlx` unchanged, and the model's only "
                + "children are `vision_tower` and `language_model`, so any other prefix fails "
                + "`update(parameters:verify: [.all])`.")
        return ["qwen3_5": qwen35, "qwen3_5_moe": qwen35]
    }()

    /// Model types whose pinned configuration also reads `rope_parameters` (the rest read only `rope_scaling`).
    static let ropeParametersModelTypes: Set<String> = [
        "gemma4", "gemma4_text", "lfm2", "lfm2_moe", "mistral3", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "glm_ocr",
    ]

    /// SPDX-style ids with use-based or community terms worth reading before shipping.
    static let restrictedLicenses: Set<String> = [
        "openrail", "openrail++", "creativeml-openrail-m", "bigscience-openrail-m", "bigscience-bloom-rail-1.0",
        "bigcode-openrail-m", "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3", "llama4", "gemma",
        "deepseek", "tongyi-qianwen", "apple-amlr", "cc-by-sa-4.0", "gpl-3.0", "agpl-3.0",
    ]

    // MARK: Format

    static let formatRule = CompatibilityRule(
        id: "format", summary: "MLX safetensors or GGUF; anything else is not loadable"
    ) { input in
        let weightBytes = input.safetensorsBytes.map { " (\(gigabytes($0)) of safetensors)" } ?? ""
        switch input.weightFormat {
        case .mlx:
            var found = [CompatibilityFinding.info("MLX weights",
                "Safetensors with MLX quantization or the `mlx` tag — loaded by mlx-swift-lm \(PinnedRuntimes.mlxSwiftLMVersion).")]
            if input.files.contains(where: { $0.isGGUF && $0.isTopLevel }) {
                found.append(.caveat("Repo also ships GGUF files",
                    "AuraLocal's MLX downloader fetches every top-level file, so the GGUF files download too."))
            }
            return found
        case .gguf:
            return [.info("GGUF weights",
                "Loaded by llama.cpp \(PinnedRuntimes.llamaCppBuild) through LocalLLMClient 0.5.0.")]
        case .imageGeneration:
            return [.info("Image-generation pipeline",
                "Pipeline `\(input.listing?.pipelineTag ?? "diffusers")`: not an LLM. AuraLocal runs FLUX through AuraImageGen (mflux).")]
        case .unconvertedSafetensors:
            let quant = input.settings?.foreignQuantMethod.map { " It is quantized with `\($0)`, which MLX does not read." } ?? ""
            return [.blocker("Not an MLX conversion",
                "Safetensors without MLX quantization or the `mlx` tag\(weightBytes).\(quant) AuraLocal's catalog "
                + "loads MLX conversions and GGUF only; look for an mlx-community or GGUF build of this model.")]
        case .noWeights:
            let kinds = Set(input.files.map { ($0.path as NSString).pathExtension.lowercased() })
                .intersection(["bin", "pt", "pth", "ckpt", "onnx", "h5", "msgpack", "mlmodel", "mlpackage", "tflite"])
                .sorted()
            let shipped = kinds.isEmpty ? "no weight files" : kinds.map { ".\($0)" }.joined(separator: ", ")
            return [.blocker("No MLX or GGUF weights",
                "The repo ships \(shipped). AuraLocal loads MLX safetensors (mlx-swift-lm) and GGUF (llama.cpp).")]
        case .unknown:
            return []
        }
    }

    // MARK: Generative

    static let generativeRule = CompatibilityRule(
        id: "generative", summary: "Pipeline, head and model type must generate text"
    ) { input in
        var reasons: [String] = []
        if let pipeline = input.listing?.pipelineTag, nonGenerativePipelines.contains(pipeline) {
            reasons.append("pipeline `\(pipeline)`")
        }
        let heads = (input.settings?.architectures ?? []).filter { name in
            nonGenerativeArchitectureSuffixes.contains { name.hasSuffix($0) }
        }
        if !heads.isEmpty { reasons.append("architecture \(heads.map { "`\($0)`" }.joined(separator: ", "))") }
        if let type = input.modelType, encoderModelTypes.contains(type) {
            reasons.append("encoder model type `\(type)`")
        }
        let tags = input.listing?.tags ?? []
        let embeddingMarks = embeddingTags.intersection(tags).sorted()
        if !embeddingMarks.isEmpty {
            reasons.append("tags \(embeddingMarks.map { "`\($0)`" }.joined(separator: ", "))")
        }
        let names = [input.source.repoID] + tags.filter { $0.hasPrefix("base_model:") }
        if let named = names.first(where: { $0.lowercased().contains("embedding") }) {
            reasons.append("an embedding model by name (`\(named)`)")
        }
        guard !reasons.isEmpty else { return [] }
        return [.blocker("Not a generative model",
            "It classifies, embeds or fills masks — \(reasons.joined(separator: "; ")). "
            + "AuraLocal's runtimes chat with text generators only.")]
    }
}

// MARK: - CompatibilityRules: MLX

extension CompatibilityRules {

    static let mlxConfigRule = CompatibilityRule(
        id: "mlx.config", summary: "An MLX repo needs a readable config.json with model_type"
    ) { input in
        guard input.weightFormat == .mlx else { return [] }
        if let problem = input.source.problem(for: .config) {
            let hint = problem.needsToken ? " Save a Hugging Face token (Keychain `download.huggingface`) and retry." : ""
            return [.caveat("config.json unreadable", "\(problem.message).\(hint)")]
        }
        guard let config = input.settings else {
            return [.blocker("No config.json", "mlx-swift-lm reads `model_type` from config.json before anything else.")]
        }
        var found: [CompatibilityFinding] = []
        if config.modelType == nil {
            found.append(.blocker("config.json has no model_type",
                "`BaseConfiguration` requires `model_type`; decoding throws without it."))
        }
        if let method = config.foreignQuantMethod {
            found.append(.caveat("Non-MLX quantizer `\(method)`",
                "config.json declares `quant_method: \(method)`; mlx-swift-lm only reads its own affine quantization."))
        }
        if let bits = config.quantizationBits {
            found.append(.info("\(bits)-bit MLX quantization", "From `quantization` in config.json."))
        }
        return found
    }

    static let mlxModelTypeRule = CompatibilityRule(
        id: "mlx.model-type", summary: "model_type must be in the pinned LLM or VLM registry"
    ) { input in
        guard [.mlx, .unconvertedSafetensors, .noWeights].contains(input.weightFormat),
              let type = input.modelType else { return [] }
        let isLLM = PinnedRuntimes.mlxLLMModelTypes.contains(type)
        let isVLM = PinnedRuntimes.mlxVLMModelTypes.contains(type)
        let version = PinnedRuntimes.mlxSwiftLMVersion
        if isLLM || isVLM {
            let routes = [isLLM ? "LLM" : nil, isVLM ? "VLM" : nil].compactMap { $0 }.joined(separator: " + ")
            return [.info("model_type `\(type)` is registered (\(routes))",
                "LLMTypeRegistry / VLMTypeRegistry in mlx-swift-lm \(version).")]
        }
        var detail = "Neither LLMTypeRegistry nor VLMTypeRegistry in mlx-swift-lm \(version) has `\(type)`, so "
            + "`createModel` throws `unsupportedModelType`."
        if let text = input.settings?.textModelType,
           PinnedRuntimes.mlxLLMModelTypes.contains(text) || PinnedRuntimes.mlxVLMModelTypes.contains(text) {
            detail += " `text_config.model_type` `\(text)` is registered, but the factories dispatch on the top-level key."
        }
        return [.blocker("model_type `\(type)` is not supported", detail)]
    }

    static let weightPrefixRule = CompatibilityRule(
        id: "mlx.weight-prefixes", summary: "Tensor prefixes must be ones the pinned sanitize maps"
    ) { input in
        guard input.weightFormat == .mlx, let type = input.modelType,
              let mapping = weightPrefixPolicies[type], input.tensorNames != nil else { return [] }
        let preconverted = input.inferredCategory == .vision
            && input.safetensorsMetadata?["format"]?.lowercased() == "mlx"
        let handled = preconverted ? mapping.mlxFormatVisionPrefixes : mapping.handledPrefixes
        let unhandled = input.tensorPrefixes.filter { !handled.contains($0.key) }
            .sorted { $0.key < $1.key }
        guard !unhandled.isEmpty else { return [] }
        let list = unhandled.map { "`\($0.key).*` (\($0.value) tensors)" }.joined(separator: ", ")
        return [.blocker("Weights under \(unhandled.map { "`\($0.key).`" }.joined(separator: ", ")) will not load",
            "\(list) are not mapped for `\(type)`. \(preconverted ? mapping.mlxFormatEvidence : mapping.evidence)")]
    }

    /// True when the first shard's `__metadata__` could change the prefix verdict of a sharded repo, so the
    /// checker spends a header read on it only then. The device does not affect the category.
    static func verdictDependsOnWeightMetadata(_ snapshot: RepoSnapshot) -> Bool {
        guard snapshot.liveWeightMap != nil, let type = snapshot.configuration?.modelType,
              let mapping = weightPrefixPolicies[type] else { return false }
        let input = RuleInput(snapshot: snapshot, target: .mac32GB)
        guard input.weightFormat == .mlx, input.inferredCategory == .vision else { return false }
        let prefixes = Set(input.tensorPrefixes.keys)
        return prefixes.isSubset(of: mapping.handledPrefixes) && !prefixes.isSubset(of: mapping.mlxFormatVisionPrefixes)
    }

    static let textOnlyTowerRule = CompatibilityRule(
        id: "mlx.towers", summary: "A type registered only as an LLM cannot load vision or audio towers"
    ) { input in
        guard input.weightFormat == .mlx, let type = input.modelType,
              PinnedRuntimes.mlxLLMModelTypes.contains(type), !PinnedRuntimes.mlxVLMModelTypes.contains(type),
              input.tensorNames != nil else { return [] }
        let towers = input.present(visionTensorPrefixes + auxiliaryTowerPrefixes)
        guard !towers.isEmpty else { return [] }
        let names = input.tensorNames ?? []
        let list = towers.map { prefix in
            "`\(prefix).*` (\(names.filter { $0.hasPrefix(prefix + ".") }.count) tensors)"
        }.joined(separator: ", ")
        return [.blocker("Vision or audio towers will not load",
            "\(list): `\(type)` is registered only in LLMTypeRegistry, and its pinned text model has no such modules. "
            + "In mlx-swift-lm \(PinnedRuntimes.mlxSwiftLMVersion) only the gemma4 and qwen3_5 LLM sanitizers drop "
            + "towers (both types load through VLMModelFactory here), so `update(parameters:verify: [.all])` throws "
            + "`unhandledKeys`. Look for a text-only (`-lm-`) conversion.")]
    }

    static let extraSafetensorsRule = CompatibilityRule(
        id: "mlx.extra-safetensors", summary: "Every downloaded *.safetensors must belong to the weight map"
    ) { input in
        guard input.weightFormat == .mlx, let weightMap = input.source.weightMap else { return [] }
        let referenced = Set(weightMap.values)
        if input.source.isWeightMapStale {
            return [.caveat("Stale weight index",
                "model.safetensors.index.json references \(referenced.sorted().prefix(3).joined(separator: ", ")), which "
                + "the repo does not ship. mlx-swift-lm never reads the index (`loadWeights` loads every *.safetensors), "
                + "so the checker used the shipped files' headers instead.")]
        }
        let present = Set(input.files.map(\.path))
        var found: [CompatibilityFinding] = []
        let extras = input.files.filter { $0.isSafetensors && $0.isTopLevel && !referenced.contains($0.path) }
        if !extras.isEmpty {
            let counts = input.source.extraTensorCounts
            let names = extras.map { file in
                "`\(file.path)`" + (counts[file.path].map { " (\($0) tensors)" } ?? "")
            }.joined(separator: ", ")
            found.append(.blocker("Extra safetensors outside the weight map",
                "\(names) \(extras.count == 1 ? "is" : "are") not in model.safetensors.index.json. AuraLocal's downloader "
                + "fetches every top-level file and mlx-swift-lm's `loadWeights` merges every *.safetensors in the "
                + "snapshot, so their tensors reach `update(parameters:verify: [.all])` unhandled and it throws."))
        }
        let missing = referenced.subtracting(present).sorted()
        if !missing.isEmpty {
            found.append(.blocker("Weight map points at missing files",
                "\(missing.prefix(5).joined(separator: ", ")) \(missing.count == 1 ? "is" : "are") referenced but not in the repo."))
        }
        if referenced.contains(where: { $0.contains("/") }) {
            found.append(.blocker("Weights in a subfolder",
                "The weight map references subfolder files; AuraLocal's downloader lists only top-level files."))
        }
        return found
    }

    static let visionWeightsRule = CompatibilityRule(
        id: "mlx.category", summary: "Vision only for a registered VLM type that ships vision weights"
    ) { input in
        guard input.weightFormat == .mlx, let type = input.modelType else { return [] }
        let isLLM = PinnedRuntimes.mlxLLMModelTypes.contains(type)
        let isVLM = PinnedRuntimes.mlxVLMModelTypes.contains(type)
        guard isLLM || isVLM else { return [] }
        switch (input.hasVisionTensors, isVLM, isLLM) {
        case (true?, true, _):
            let towers = input.visionTowers.map { "`\($0)`" }.joined(separator: ", ")
            return [.info("Vision model",
                "`\(type)` is a registered VLM type and the weights hold a vision tower (\(towers)); loads through VLMModelFactory.")]
        case (false?, true, false):
            return [.blocker("VLM type without vision weights",
                "`\(type)` is registered only as a VLM, and the weights hold no vision tower, so verification throws.")]
        case (nil, true, _):
            return [.caveat("Vision weights not verified",
                "Could not list the tensors; the category is inferred from config.json.")]
        case (_, false, _), (false?, true, true):
            return [.info("Text model", "Loads through LLMModelFactory.")]
        }
    }

    static let ropeRule = CompatibilityRule(
        id: "mlx.rope", summary: "rope_scaling must be a type RoPEUtils implements"
    ) { input in
        guard input.weightFormat == .mlx, let config = input.settings, let type = config.modelType,
              let rope = config.ropeType,
              config.ropeSource == "rope_scaling" || ropeParametersModelTypes.contains(type) else { return [] }
        if type == "llama" || type == "mistral" {
            guard config.ropeKeys.contains("factor"), ["linear", "llama3"].contains(rope) else {
                return [.blocker("rope_scaling `\(rope)` is rejected",
                    "LlamaConfiguration (MLXLLM/Models/Llama.swift) requires `factor` and a type of linear / dynamic / "
                    + "llama3, and `initializeRope` then `fatalError`s on `dynamic`.")]
            }
            return []
        }
        guard PinnedRuntimes.mlxRopeModelTypes.contains(type) else { return [] }
        guard PinnedRuntimes.mlxRopeTypes.contains(rope) else {
            return [.blocker("rope_scaling `\(rope)` crashes the app",
                "MLXLMCommon/RoPEUtils.swift `initializeRope` handles \(PinnedRuntimes.mlxRopeTypes.sorted().joined(separator: ", ")) "
                + "and calls `fatalError(\"Unsupported RoPE type\")` for anything else.")]
        }
        if rope == "longrope" {
            let required = ["original_max_position_embeddings", "short_factor", "long_factor"]
            let missing = required.filter { !config.ropeKeys.contains($0) }
            guard missing.isEmpty else {
                return [.blocker("longrope is missing \(missing.joined(separator: ", "))",
                    "`initializeRope` `fatalError`s when a longrope config lacks any of \(required.joined(separator: ", ")).")]
            }
            return [.info("longrope rope_scaling is complete",
                "It carries \(required.joined(separator: ", ")), which RoPEUtils requires.")]
        }
        return []
    }

    static let tokenizerRule = CompatibilityRule(
        id: "mlx.tokenizer", summary: "An MLX repo needs tokenizer files"
    ) { input in
        guard input.weightFormat == .mlx else { return [] }
        let names = Set(input.files.filter(\.isTopLevel).map(\.path))
        guard names.isDisjoint(with: ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]) else { return [] }
        return [.caveat("No tokenizer files",
            "Neither tokenizer.json, tokenizer.model nor tokenizer_config.json is in the repo root.")]
    }
}

// MARK: - CompatibilityRules: GGUF

extension CompatibilityRules {

    static let ggufArchitectureRule = CompatibilityRule(
        id: "gguf.architecture", summary: "general.architecture must be known to llama.cpp b8851"
    ) { input in
        guard input.weightFormat == .gguf, let arch = input.ggufArchitecture else { return [] }
        let build = PinnedRuntimes.llamaCppBuild
        if let later = PinnedRuntimes.laterLlamaCppArchitectures[arch] {
            return [.blocker("Architecture `\(arch)` needs llama.cpp \(later.build)",
                "llama.cpp \(later.build) (\(later.date)) \(later.change); AuraLocal pins \(build) via LocalLLMClient 0.5.0, "
                + "whose loader rejects an unknown architecture.")]
        }
        if arch == "clip" {
            return [.blocker("Only a vision projector",
                "`clip` is an mmproj projector, not a language model.")]
        }
        guard PinnedRuntimes.llamaCppArchitectures.contains(arch) else {
            return [.blocker("Architecture `\(arch)` is unknown to llama.cpp \(build)",
                "It is not in `LLM_ARCH_NAMES` of llama.cpp \(build) (src/llama-arch.cpp), so loading fails.")]
        }
        if let reason = PinnedRuntimes.llamaCppNonChatArchitectures[arch] {
            return [.blocker("`\(arch)` is not a chat model", "It is \(reason).")]
        }
        return [.info("Architecture `\(arch)`", "Listed in llama.cpp \(build) (src/llama-arch.cpp).")]
    }

    static let ggufNextNRule = CompatibilityRule(
        id: "gguf.qwen35-nextn", summary: "qwen35 with an MTP block needs llama.cpp b9180"
    ) { input in
        guard input.weightFormat == .gguf, let header = input.ggufMetadata,
              let arch = header.architecture, ["qwen35", "qwen35moe"].contains(arch),
              let nextn = header.nextnPredictLayers, nextn > 0 else { return [] }
        let milestone = PinnedRuntimes.qwen35NextNFix
        let blocks = header.blockCount.map(String.init) ?? "?"
        return [.blocker("MTP block breaks loading on \(PinnedRuntimes.llamaCppBuild)",
            "`\(arch).block_count` \(blocks) includes \(nextn) NextN layer(s). b8851 marks recurrent layers arithmetically "
            + "(`(i + 1) % full_attention_interval != 0`) over every block, so it demands `ssm_*` tensors for the MTP "
            + "block that do not exist and the load throws. "
            + "Fixed in llama.cpp \(milestone.build) (\(milestone.date), \(milestone.change)).")]
    }

    static let ggufShardRule = CompatibilityRule(
        id: "gguf.shards", summary: "AuraLocal loads single-file GGUFs only"
    ) { input in
        guard input.weightFormat == .gguf else { return [] }
        let split = input.quantGroups.filter(\.isSplit)
        guard !split.isEmpty else { return [] }
        let labels = split.map { $0.label ?? $0.firstPath }.joined(separator: ", ")
        let why = "AuraLocal's GGUF path downloads one file and skips `-0000N-of-0000M` parts (HuggingFaceRepo.isShard)."
        if split.count == input.quantGroups.count {
            return [.blocker("Every quant is split into parts", "\(labels). \(why)")]
        }
        return [.caveat("\(split.count) split quant\(split.count == 1 ? "" : "s") unusable", "\(labels). \(why)")]
    }

    static let ggufHeaderRule = CompatibilityRule(
        id: "gguf.header", summary: "Report how much of the GGUF header was read"
    ) { input in
        guard input.weightFormat == .gguf else { return [] }
        if let problem = input.source.problem(for: .ggufHeader) {
            let fallback = input.listing?.ggufArchitecture.map { " Architecture `\($0)` comes from Hugging Face's own summary." } ?? ""
            return [.caveat("GGUF header unreadable", "\(problem.message).\(fallback)")]
        }
        guard let header = input.ggufMetadata else { return [] }
        let sample = input.source.ggufSamplePath ?? "the sample file"
        guard header.isComplete else {
            return [.info("Header partly read",
                "Read \(header.metadata.count) of \(header.declaredKeyCount) keys from the start of `\(sample)` "
                + "(the rest is tokenizer data); GGUF v\(header.version), \(header.tensorCount) tensors.")]
        }
        return [.info("Header read",
            "\(header.metadata.count) keys from `\(sample)`; GGUF v\(header.version), \(header.tensorCount) tensors.")]
    }

    static let ggufProjectorRule = CompatibilityRule(
        id: "gguf.projector", summary: "mmproj projectors are not used"
    ) { input in
        guard input.weightFormat == .gguf, input.files.contains(where: { $0.isGGUF && $0.isProjector }) else { return [] }
        return [.info("Vision projector ignored",
            "The repo ships an mmproj file; AuraLocal's GGUF path loads the language model only (text in, text out).")]
    }
}

// MARK: - CompatibilityRules: Image generation, license and context

extension CompatibilityRules {

    static let imageGenerationRule = CompatibilityRule(
        id: "imagegen", summary: "Diffusion repos run through AuraImageGen on macOS only"
    ) { input in
        guard input.weightFormat == .imageGeneration else { return [] }
        guard input.target.platform == .macOS else {
            return [.blocker("Image generation is macOS-only",
                "AuraImageGen drives mflux as a subprocess; the iOS build has no engine.")]
        }
        return [.caveat("mflux support not verified",
            "The checker sizes the pipeline (peak ≈ 2.2× the weights) but does not check that mflux knows it; "
            + "mflux runs FLUX.1 schnell / dev and fine-tunes of them.")]
    }

    // MARK: License & context

    static let licenseRule = CompatibilityRule(
        id: "license", summary: "Gating, missing and restrictive licenses"
    ) { input in
        guard let listing = input.listing else { return [] }
        var found: [CompatibilityFinding] = []
        if let mode = listing.gatedMode {
            found.append(.caveat("Gated repository (\(mode))",
                "Accept the terms on huggingface.co and save a token in the Keychain (`download.huggingface`) before downloading."))
        }
        guard let license = listing.licenseID?.lowercased() else {
            found.append(.caveat("No license declared",
                "The model card declares no license; you may not have the right to ship it."))
            return found
        }
        let link = listing.licenseLink.map { " (\(absoluteLicenseLink($0, repoID: listing.repoID)))" } ?? ""
        if license.contains("-nc") || license.contains("noncommercial") || license.contains("non-commercial") {
            found.append(.caveat("Non-commercial license `\(license)`", "Not for commercial apps."))
        } else if license == "other" {
            let name = listing.licenseName.map { "`\($0)`" } ?? "(no name given)"
            found.append(.caveat("Custom license \(name)", "Read its terms before shipping\(link)."))
        } else if restrictedLicenses.contains(license) {
            found.append(.caveat("License `\(license)` has use restrictions",
                "Community / RAIL-style terms; read them before shipping\(link)."))
        } else {
            found.append(.info("License `\(license)`", "From the model card."))
        }
        return found
    }

    static let contextRule = CompatibilityRule(
        id: "context", summary: "Short trained contexts become maxContextLength"
    ) { input in
        guard let context = input.trainedContext, context < CatalogEntry.contextCeiling else { return [] }
        if context < 4096 {
            return [.caveat("Short trained context: \(context) tokens",
                "Prompt plus answer past \(context) tokens degrades or fails; a catalog entry caps it with `maxContextLength`.")]
        }
        return [.info("Trained context \(context) tokens", "A catalog entry caps it with `maxContextLength` \(context).")]
    }

    // MARK: Helpers

    static func gigabytes(_ bytes: Int64) -> String {
        CompatibilityReport.gigabytesText(bytes)
    }

    /// Model cards often give `license_link: LICENSE`; resolve it against the repo so it can be opened.
    static func absoluteLicenseLink(_ link: String, repoID: String) -> String {
        guard URL(string: link)?.scheme == nil else { return link }
        var path = Substring(link)
        while path.hasPrefix("./") { path = path.dropFirst(2) }
        while path.hasPrefix("/") { path = path.dropFirst() }
        return "https://huggingface.co/\(repoID)/blob/main/\(path)"
    }
}

/// Which top-level tensor prefixes a model type's pinned `sanitize` can map.
struct WeightPrefixPolicy: Sendable {
    let handledPrefixes: Set<String>
    let evidence: String
    /// What the VLM route accepts when the weights are already in MLX layout (`__metadata__.format` = `mlx`).
    let mlxFormatVisionPrefixes: Set<String>
    let mlxFormatEvidence: String
}
