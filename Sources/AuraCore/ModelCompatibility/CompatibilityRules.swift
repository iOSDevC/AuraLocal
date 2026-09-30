import Foundation

// MARK: - RuleInput

/// A snapshot plus the facts every rule derives from it, computed once.
struct RuleInput: Sendable {
    let source: RepoSnapshot
    let target: DevicePreset
    let weightFormat: DetectedFormat
    /// Tensor names from the weight map or the single-file header; `nil` when neither was read.
    let tensorNames: [String]?
    let quantGroups: [GGUFQuantGroup]

    init(snapshot: RepoSnapshot, target: DevicePreset) {
        source = snapshot
        self.target = target
        weightFormat = DetectedFormat.detect(snapshot)
        tensorNames = snapshot.weightMap.map { Array($0.keys) } ?? snapshot.singleFileHeader?.tensorNames
        quantGroups = GGUFQuantGroup.groups(in: snapshot.listing?.files ?? [])
    }

    var listing: HFRepoInfo? { source.listing }
    var files: [RepoFile] { source.listing?.files ?? [] }
    var settings: ModelConfigFacts? { source.configuration }
    var modelType: String? { settings?.modelType }
    var ggufMetadata: GGUFHeader? { source.ggufMetadata }
    var ggufArchitecture: String? { ggufMetadata?.architecture ?? listing?.ggufArchitecture }

    /// Top-level tensor-name prefix → tensor count.
    var tensorPrefixes: [String: Int] {
        (tensorNames ?? []).reduce(into: [:]) { counts, name in
            counts[name.split(separator: ".").first.map(String.init) ?? name, default: 0] += 1
        }
    }

    /// Vision-tower prefixes present in the weights (`vision_tower`, `visual`, …).
    var visionTowers: [String] {
        let names = tensorNames ?? []
        return CompatibilityRules.visionTensorPrefixes.filter { prefix in names.contains { $0.hasPrefix(prefix + ".") } }
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
        if let weightMap = source.weightMap {
            let referenced = Set(weightMap.values)
            chosen = files.filter { referenced.contains($0.path) }
        } else {
            chosen = weightFormat == .imageGeneration ? files.filter(\.isSafetensors) : top
        }
        let sizes = chosen.compactMap(\.sizeBytes)
        return sizes.isEmpty ? nil : sizes.reduce(0, +)
    }

    var trainedContext: Int? {
        switch weightFormat {
        case .gguf: ggufMetadata?.contextLength ?? listing?.ggufContextLength
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
}

// MARK: - CompatibilityRules

/// The rule set and the tables it reads. Evidence for each table is next to it.
public enum CompatibilityRules {

    public static let all: [CompatibilityRule] = [
        formatRule, generativeRule, mlxConfigRule, mlxModelTypeRule, weightPrefixRule, extraSafetensorsRule,
        visionWeightsRule, ropeRule, tokenizerRule, ggufArchitectureRule, ggufNextNRule, ggufShardRule,
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

    /// Model types whose pinned `sanitize` accepts only some top-level prefixes.
    static let weightPrefixPolicies: [String: WeightPrefixPolicy] = {
        let qwen35 = WeightPrefixPolicy(
            handledPrefixes: ["language_model", "model", "lm_head", "vision_tower", "mtp"],
            evidence: "mlx-swift-lm 3.31.3: the qwen3_5 LLM sanitize (MLXLLM/Models/Qwen35.swift) drops "
                + "`vision_tower.*` / `model.visual.*`, filters `mtp.` and prefixes the rest with `language_model.`; "
                + "the VLM sanitize (MLXVLM/Models/Qwen35.swift) only renames `model.visual` → `vision_tower`. "
                + "Any other prefix stays unmatched and `update(parameters:verify: [.all])` throws.")
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
            var found = [CompatibilityFinding.info("format", "MLX weights",
                "Safetensors with MLX quantization or the `mlx` tag — loaded by mlx-swift-lm \(PinnedRuntimes.mlxSwiftLMVersion).")]
            if input.files.contains(where: { $0.isGGUF && $0.isTopLevel }) {
                found.append(.caveat("format", "Repo also ships GGUF files",
                    "AuraLocal's MLX downloader fetches every top-level file, so the GGUF files download too."))
            }
            return found
        case .gguf:
            return [.info("format", "GGUF weights",
                "Loaded by llama.cpp \(PinnedRuntimes.llamaCppBuild) through LocalLLMClient 0.5.0.")]
        case .imageGeneration:
            return [.info("format", "Image-generation pipeline",
                "Pipeline `\(input.listing?.pipelineTag ?? "diffusers")`: not an LLM. AuraLocal runs FLUX through AuraImageGen (mflux).")]
        case .unconvertedSafetensors:
            let quant = input.settings?.foreignQuantMethod.map { " It is quantized with `\($0)`, which MLX does not read." } ?? ""
            return [.blocker("format", "Not an MLX conversion",
                "Safetensors without MLX quantization or the `mlx` tag\(weightBytes).\(quant) AuraLocal's catalog "
                + "loads MLX conversions and GGUF only; look for an mlx-community or GGUF build of this model.")]
        case .noWeights:
            let kinds = Set(input.files.map { ($0.path as NSString).pathExtension.lowercased() })
                .intersection(["bin", "pt", "pth", "ckpt", "onnx", "h5", "msgpack", "mlmodel", "mlpackage", "tflite"])
                .sorted()
            let shipped = kinds.isEmpty ? "no weight files" : kinds.map { ".\($0)" }.joined(separator: ", ")
            return [.blocker("format", "No MLX or GGUF weights",
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
        guard !reasons.isEmpty else { return [] }
        return [.blocker("generative", "Not a generative model",
            "It classifies, embeds or fills masks — \(reasons.joined(separator: "; ")). "
            + "AuraLocal's runtimes chat with text generators only.")]
    }

    // MARK: MLX

    static let mlxConfigRule = CompatibilityRule(
        id: "mlx.config", summary: "An MLX repo needs a readable config.json with model_type"
    ) { input in
        guard input.weightFormat == .mlx else { return [] }
        if let problem = input.source.problem(for: .config) {
            let hint = problem.needsToken ? " Save a Hugging Face token (Keychain `download.huggingface`) and retry." : ""
            return [.caveat("mlx.config", "config.json unreadable", "\(problem.message).\(hint)")]
        }
        guard let config = input.settings else {
            return [.blocker("mlx.config", "No config.json", "mlx-swift-lm reads `model_type` from config.json before anything else.")]
        }
        var found: [CompatibilityFinding] = []
        if config.modelType == nil {
            found.append(.blocker("mlx.config", "config.json has no model_type",
                "`BaseConfiguration` requires `model_type`; decoding throws without it."))
        }
        if let method = config.foreignQuantMethod {
            found.append(.caveat("mlx.config", "Non-MLX quantizer `\(method)`",
                "config.json declares `quant_method: \(method)`; mlx-swift-lm only reads its own affine quantization."))
        }
        if let bits = config.quantizationBits {
            found.append(.info("mlx.config", "\(bits)-bit MLX quantization", "From `quantization` in config.json."))
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
            return [.info("mlx.model-type", "model_type `\(type)` is registered (\(routes))",
                "LLMTypeRegistry / VLMTypeRegistry in mlx-swift-lm \(version).")]
        }
        var detail = "Neither LLMTypeRegistry nor VLMTypeRegistry in mlx-swift-lm \(version) has `\(type)`, so "
            + "`createModel` throws `unsupportedModelType`."
        if let text = input.settings?.textModelType,
           PinnedRuntimes.mlxLLMModelTypes.contains(text) || PinnedRuntimes.mlxVLMModelTypes.contains(text) {
            detail += " `text_config.model_type` `\(text)` is registered, but the factories dispatch on the top-level key."
        }
        return [.blocker("mlx.model-type", "model_type `\(type)` is not supported", detail)]
    }

    static let weightPrefixRule = CompatibilityRule(
        id: "mlx.weight-prefixes", summary: "Tensor prefixes must be ones the pinned sanitize maps"
    ) { input in
        guard input.weightFormat == .mlx, let type = input.modelType,
              let mapping = weightPrefixPolicies[type], input.tensorNames != nil else { return [] }
        let unhandled = input.tensorPrefixes.filter { !mapping.handledPrefixes.contains($0.key) }
            .sorted { $0.key < $1.key }
        guard !unhandled.isEmpty else { return [] }
        let list = unhandled.map { "`\($0.key).*` (\($0.value) tensors)" }.joined(separator: ", ")
        return [.blocker("mlx.weight-prefixes", "Weights under \(unhandled.map { "`\($0.key).`" }.joined(separator: ", ")) will not load",
            "\(list) are not mapped for `\(type)`. \(mapping.evidence)")]
    }

    static let extraSafetensorsRule = CompatibilityRule(
        id: "mlx.extra-safetensors", summary: "Every downloaded *.safetensors must belong to the weight map"
    ) { input in
        guard input.weightFormat == .mlx, let weightMap = input.source.weightMap else { return [] }
        let referenced = Set(weightMap.values)
        let present = Set(input.files.map(\.path))
        var found: [CompatibilityFinding] = []
        let extras = input.files.filter { $0.isSafetensors && $0.isTopLevel && !referenced.contains($0.path) }
        if !extras.isEmpty {
            let counts = input.source.extraTensorCounts
            let names = extras.map { file in
                "`\(file.path)`" + (counts[file.path].map { " (\($0) tensors)" } ?? "")
            }.joined(separator: ", ")
            found.append(.blocker("mlx.extra-safetensors", "Extra safetensors outside the weight map",
                "\(names) \(extras.count == 1 ? "is" : "are") not in model.safetensors.index.json. AuraLocal's downloader "
                + "fetches every top-level file and mlx-swift-lm's `loadWeights` merges every *.safetensors in the "
                + "snapshot, so their tensors reach `update(parameters:verify: [.all])` unhandled and it throws."))
        }
        let missing = referenced.subtracting(present).sorted()
        if !missing.isEmpty {
            found.append(.blocker("mlx.extra-safetensors", "Weight map points at missing files",
                "\(missing.prefix(5).joined(separator: ", ")) \(missing.count == 1 ? "is" : "are") referenced but not in the repo."))
        }
        if referenced.contains(where: { $0.contains("/") }) {
            found.append(.blocker("mlx.extra-safetensors", "Weights in a subfolder",
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
            return [.info("mlx.category", "Vision model",
                "`\(type)` is a registered VLM type and the weights hold a vision tower (\(towers)); loads through VLMModelFactory.")]
        case (false?, true, false):
            return [.blocker("mlx.category", "VLM type without vision weights",
                "`\(type)` is registered only as a VLM, and the weights hold no vision tower, so verification throws.")]
        case (nil, true, _):
            return [.caveat("mlx.category", "Vision weights not verified",
                "Could not list the tensors; the category is inferred from config.json.")]
        case (_, false, _), (false?, true, true):
            return [.info("mlx.category", "Text model", "Loads through LLMModelFactory.")]
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
                return [.blocker("mlx.rope", "rope_scaling `\(rope)` is rejected",
                    "LlamaConfiguration (MLXLLM/Models/Llama.swift) requires `factor` and a type of linear / dynamic / "
                    + "llama3, and `initializeRope` then `fatalError`s on `dynamic`.")]
            }
            return []
        }
        guard PinnedRuntimes.mlxRopeModelTypes.contains(type) else { return [] }
        guard PinnedRuntimes.mlxRopeTypes.contains(rope) else {
            return [.blocker("mlx.rope", "rope_scaling `\(rope)` crashes the app",
                "MLXLMCommon/RoPEUtils.swift `initializeRope` handles \(PinnedRuntimes.mlxRopeTypes.sorted().joined(separator: ", ")) "
                + "and calls `fatalError(\"Unsupported RoPE type\")` for anything else.")]
        }
        if rope == "longrope" {
            let required = ["original_max_position_embeddings", "short_factor", "long_factor"]
            let missing = required.filter { !config.ropeKeys.contains($0) }
            guard missing.isEmpty else {
                return [.blocker("mlx.rope", "longrope is missing \(missing.joined(separator: ", "))",
                    "`initializeRope` `fatalError`s when a longrope config lacks any of \(required.joined(separator: ", ")).")]
            }
            return [.info("mlx.rope", "longrope rope_scaling is complete",
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
        return [.caveat("mlx.tokenizer", "No tokenizer files",
            "Neither tokenizer.json, tokenizer.model nor tokenizer_config.json is in the repo root.")]
    }

    // MARK: GGUF

    static let ggufArchitectureRule = CompatibilityRule(
        id: "gguf.architecture", summary: "general.architecture must be known to llama.cpp b8851"
    ) { input in
        guard input.weightFormat == .gguf, let arch = input.ggufArchitecture else { return [] }
        let build = PinnedRuntimes.llamaCppBuild
        if let later = PinnedRuntimes.laterLlamaCppArchitectures[arch] {
            return [.blocker("gguf.architecture", "Architecture `\(arch)` needs llama.cpp \(later.build)",
                "llama.cpp \(later.build) (\(later.date)) \(later.change); AuraLocal pins \(build) via LocalLLMClient 0.5.0, "
                + "whose loader rejects an unknown architecture.")]
        }
        if arch == "clip" {
            return [.blocker("gguf.architecture", "Only a vision projector",
                "`clip` is an mmproj projector, not a language model.")]
        }
        guard PinnedRuntimes.llamaCppArchitectures.contains(arch) else {
            return [.blocker("gguf.architecture", "Architecture `\(arch)` is unknown to llama.cpp \(build)",
                "It is not in `LLM_ARCH_NAMES` of llama.cpp \(build) (src/llama-arch.cpp), so loading fails.")]
        }
        if let reason = PinnedRuntimes.llamaCppNonChatArchitectures[arch] {
            return [.blocker("gguf.architecture", "`\(arch)` is not a chat model", "It is \(reason).")]
        }
        return [.info("gguf.architecture", "Architecture `\(arch)`", "Listed in llama.cpp \(build) (src/llama-arch.cpp).")]
    }

    static let ggufNextNRule = CompatibilityRule(
        id: "gguf.qwen35-nextn", summary: "qwen35 with an MTP block needs llama.cpp b9495"
    ) { input in
        guard input.weightFormat == .gguf, let header = input.ggufMetadata,
              let arch = header.architecture, ["qwen35", "qwen35moe"].contains(arch),
              let nextn = header.nextnPredictLayers, nextn > 0 else { return [] }
        let milestone = PinnedRuntimes.qwen35NextNFix
        let blocks = header.blockCount.map(String.init) ?? "?"
        return [.blocker("gguf.qwen35-nextn", "MTP block breaks loading on \(PinnedRuntimes.llamaCppBuild)",
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
            return [.blocker("gguf.shards", "Every quant is split into parts", "\(labels). \(why)")]
        }
        return [.caveat("gguf.shards", "\(split.count) split quant\(split.count == 1 ? "" : "s") unusable", "\(labels). \(why)")]
    }

    static let ggufHeaderRule = CompatibilityRule(
        id: "gguf.header", summary: "Report how much of the GGUF header was read"
    ) { input in
        guard input.weightFormat == .gguf else { return [] }
        if let problem = input.source.problem(for: .ggufHeader) {
            let fallback = input.listing?.ggufArchitecture.map { " Architecture `\($0)` comes from Hugging Face's own summary." } ?? ""
            return [.caveat("gguf.header", "GGUF header unreadable", "\(problem.message).\(fallback)")]
        }
        guard let header = input.ggufMetadata else { return [] }
        let sample = input.source.ggufSamplePath ?? "the sample file"
        guard header.isComplete else {
            return [.info("gguf.header", "Header partly read",
                "Read \(header.metadata.count) of \(header.declaredKeyCount) keys from the start of `\(sample)` "
                + "(the rest is tokenizer data); GGUF v\(header.version), \(header.tensorCount) tensors.")]
        }
        return [.info("gguf.header", "Header read",
            "\(header.metadata.count) keys from `\(sample)`; GGUF v\(header.version), \(header.tensorCount) tensors.")]
    }

    static let ggufProjectorRule = CompatibilityRule(
        id: "gguf.projector", summary: "mmproj projectors are not used"
    ) { input in
        guard input.weightFormat == .gguf, input.files.contains(where: { $0.isGGUF && $0.isProjector }) else { return [] }
        return [.info("gguf.projector", "Vision projector ignored",
            "The repo ships an mmproj file; AuraLocal's GGUF path loads the language model only (text in, text out).")]
    }

    // MARK: Image generation

    static let imageGenerationRule = CompatibilityRule(
        id: "imagegen", summary: "Diffusion repos run through AuraImageGen on macOS only"
    ) { input in
        guard input.weightFormat == .imageGeneration else { return [] }
        guard input.target.platform == .macOS else {
            return [.blocker("imagegen", "Image generation is macOS-only",
                "AuraImageGen drives mflux as a subprocess; the iOS build has no engine.")]
        }
        return [.caveat("imagegen", "mflux support not verified",
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
            found.append(.caveat("license", "Gated repository (\(mode))",
                "Accept the terms on huggingface.co and save a token in the Keychain (`download.huggingface`) before downloading."))
        }
        guard let license = listing.licenseID?.lowercased() else {
            found.append(.caveat("license", "No license declared",
                "The model card declares no license; you may not have the right to ship it."))
            return found
        }
        let link = listing.licenseLink.map { " (\($0))" } ?? ""
        if license.contains("-nc") || license.contains("noncommercial") || license.contains("non-commercial") {
            found.append(.caveat("license", "Non-commercial license `\(license)`", "Not for commercial apps."))
        } else if license == "other" {
            let name = listing.licenseName.map { "`\($0)`" } ?? "(no name given)"
            found.append(.caveat("license", "Custom license \(name)", "Read its terms before shipping\(link)."))
        } else if restrictedLicenses.contains(license) {
            found.append(.caveat("license", "License `\(license)` has use restrictions",
                "Community / RAIL-style terms; read them before shipping\(link)."))
        } else {
            found.append(.info("license", "License `\(license)`", "From the model card."))
        }
        return found
    }

    static let contextRule = CompatibilityRule(
        id: "context", summary: "Short trained contexts become maxContextLength"
    ) { input in
        guard let context = input.trainedContext, context < CatalogEntry.contextCeiling else { return [] }
        if context < 4096 {
            return [.caveat("context", "Short trained context: \(context) tokens",
                "Prompt plus answer past \(context) tokens degrades or fails; a catalog entry caps it with `maxContextLength`.")]
        }
        return [.info("context", "Trained context \(context) tokens", "A catalog entry caps it with `maxContextLength` \(context).")]
    }

    // MARK: Helpers

    static func gigabytes(_ bytes: Int64) -> String {
        String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
    }
}

/// Which top-level tensor prefixes a model type's pinned `sanitize` can map.
struct WeightPrefixPolicy: Sendable {
    let handledPrefixes: Set<String>
    let evidence: String
}
