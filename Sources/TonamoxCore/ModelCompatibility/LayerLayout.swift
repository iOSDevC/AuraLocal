import Foundation

// MARK: - ConfigValues

/// `config.json` read the way the factories read it: a VLM's `text_config` fills in what the top level lacks.
struct ConfigValues {
    let root: [String: Any]
    let text: [String: Any]

    init(root: [String: Any]) {
        self.root = root
        text = root["text_config"] as? [String: Any] ?? [:]
    }

    /// The top-level value when it has type `T`, else `text_config`'s.
    private func typed<T>(_ key: String, as type: T.Type) -> T? {
        (root[key] as? T) ?? (text[key] as? T)
    }

    func int(_ keys: String...) -> Int? {
        keys.lazy.compactMap { self.typed($0, as: NSNumber.self)?.intValue }.first
    }

    func bool(_ key: String) -> Bool? {
        typed(key, as: NSNumber.self)?.boolValue
    }

    func string(_ key: String) -> String? {
        typed(key, as: String.self)
    }

    func strings(_ key: String) -> [String]? {
        typed(key, as: [String].self)
    }

    func ints(_ key: String) -> [Int]? {
        typed(key, as: [NSNumber].self)?.map(\.intValue)
    }
}

// MARK: - Layer layout

/// The cache one decoder layer gets from the pinned runtime.
enum LayerCacheKind: Equatable {
    case full
    case sliding
    case recurrent
    /// Reuses another layer's cache, or has none (an MLP-only block).
    case uncached
}

extension KVCacheLayout {

    // Evidence for every table: `newCache(parameters:)` / `makeCache(maxKVSize:)` in mlx-swift-lm 3.31.3.

    /// Types whose layers alternate linear and full attention by `(i + 1) % full_attention_interval` (default 4),
    /// whatever `layer_types` says (MLXLLM Qwen35.swift / Qwen3Next.swift, MLXVLM Qwen35.swift).
    static let intervalHybridModelTypes: Set<String> = [
        "qwen3_next", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text",
    ]

    /// Layer `i` is global when `i % sliding_window_pattern == pattern - 1`, else a
    /// `RotatingKVCache(maxSize: sliding_window)`, with the runtime's default pattern. Gemma 3 always
    /// (MLXLLM Gemma3Text.swift, MLXVLM Gemma3.swift); Gemma 4 only without `layer_types` (Gemma4Text.swift).
    static let defaultSlidingPatterns: [String: Int] = [
        "gemma3": 6, "gemma3_text": 6, "gemma4": 5, "gemma4_text": 5,
    ]
    static let gemma3ModelTypes: Set<String> = ["gemma3", "gemma3_text"]

    /// Types that give `sliding_attention` layers a `RotatingKVCache`; every other type caches them in full.
    static let slidingWindowModelTypes: Set<String> = [
        "gemma3", "gemma3_text", "gemma3n", "gemma4", "gemma4_text", "gpt_oss", "olmo3", "mistral3", "afmoe",
    ]

    /// Types whose last `num_kv_shared_layers` layers reuse an earlier layer's cache.
    static let sharedKVModelTypes: Set<String> = ["gemma3n", "gemma4", "gemma4_text"]

    /// LLM types whose `prepare` runs the whole prompt in one pass instead of `LLMModel.prepare`'s chunks
    /// (MLXLLM Gemma3Text.swift, Gemma3nText.swift). The sliding-window VLMs (MLXVLM Gemma3.swift, Gemma4.swift,
    /// Mistral3.swift) do the same; see ``prefilledInOnePass()``.
    static let wholePromptPrefillModelTypes: Set<String> = ["gemma3", "gemma3_text", "gemma3n"]

    /// `GenerateParameters.prefillStepSize`'s default, the chunk `LLMModel.prepare` feeds per step.
    static let defaultPrefillStep = 512

    /// Gemma 4 full-attention layers use `global_head_dim` and, with `attention_k_eq_v`, `num_global_key_value_heads`.
    static let globalWidthModelTypes: Set<String> = ["gemma4", "gemma4_text"]

    /// LFM2: `full_attn_idxs` (else the `full_attention` entries of `layer_types`) get a KV cache, the rest a
    /// short-convolution `MambaCache`.
    static let lfm2ModelTypes: Set<String> = ["lfm2", "lfm2_moe", "lfm2_vl", "lfm2-vl"]

    /// `layer_types` values that name a recurrent layer (linear attention, Mamba, short convolution).
    static let recurrentLayerMarkers = ["linear", "mamba", "conv", "recurrent", "ssm"]

    /// Values past these come from a corrupt or hostile config and make the layout unknown.
    static let maxLayers = 4096
    static let maxWindow = 1 << 24
    static let maxDimension = 1 << 16

    /// The layout the pinned runtime allocates for `config`, or `nil` when the layer count is unknown or implausible.
    static func derive(modelType: String?, config: ConfigValues) -> KVCacheLayout? {
        let type = modelType ?? ""
        let kinds = layerKinds(type: type, config: config)
        guard !kinds.isEmpty, kinds.count <= maxLayers else { return nil }
        let window = config.int("sliding_window").flatMap { (1...maxWindow).contains($0) ? $0 : nil }
        // A sliding layer with no known window is cached in full: overestimate rather than invent a window.
        let resolved = window == nil ? kinds.map { $0 == .sliding ? .full : $0 } : kinds
        let slidingLayers = resolved.filter { $0 == .sliding }.count
        let recurrentLayers = resolved.filter { $0 == .recurrent }.count
        guard let stateBytes = recurrentStateBytes(config, layers: recurrentLayers) else { return nil }
        let widths = globalWidthModelTypes.contains(type) ? globalWidths(config) : (nil, nil)
        let chunked = slidingLayers > 0 && !wholePromptPrefillModelTypes.contains(type)
        return KVCacheLayout(
            fullAttentionLayers: resolved.filter { $0 == .full }.count, slidingLayers: slidingLayers,
            slidingWindow: slidingLayers > 0 ? window ?? 0 : 0, recurrentLayers: recurrentLayers,
            recurrentStateBytes: stateBytes, fullAttentionKVHeads: widths.0, fullAttentionHeadDim: widths.1,
            slidingPrefillStep: chunked ? defaultPrefillStep : nil)
    }

    /// The same caches, for a runtime that prefills the whole prompt in one pass.
    func prefilledInOnePass() -> KVCacheLayout {
        KVCacheLayout(fullAttentionLayers: fullAttentionLayers, slidingLayers: slidingLayers,
                      slidingWindow: slidingWindow, recurrentLayers: recurrentLayers,
                      recurrentStateBytes: recurrentStateBytes, fullAttentionKVHeads: fullAttentionKVHeads,
                      fullAttentionHeadDim: fullAttentionHeadDim, slidingPrefillStep: nil)
    }

    /// One entry per decoder layer.
    static func layerKinds(type: String, config: ConfigValues) -> [LayerCacheKind] {
        if type == "nemotron_h", let pattern = config.string("hybrid_override_pattern") {
            return pattern.prefix(maxLayers + 1).map { $0 == "M" ? .recurrent : $0 == "*" ? .full : .uncached }
        }
        guard let layers = config.int("num_hidden_layers", "n_layer", "num_layers", "n_layers"),
              (1...maxLayers).contains(layers) else { return [] }
        let kinds: [LayerCacheKind]
        if intervalHybridModelTypes.contains(type) {
            kinds = intervalKinds(layers: layers, interval: config.int("full_attention_interval") ?? 4)
        } else if let fallback = defaultSlidingPatterns[type],
                  gemma3ModelTypes.contains(type) || config.strings("layer_types") == nil {
            let pattern = max(config.int("sliding_window_pattern") ?? fallback, 1)
            kinds = (0..<layers).map { $0 % pattern == pattern - 1 ? .full : .sliding }
        } else if type == "exaone4", let pattern = config.string("sliding_window_pattern"), !pattern.isEmpty {
            let marks = Array(pattern)
            kinds = (0..<layers).map { marks[$0 % marks.count] == "L" ? .sliding : .full }
        } else if lfm2ModelTypes.contains(type) {
            let named = config.strings("layer_types")?.enumerated()
                .compactMap { $0.element == "full_attention" ? $0.offset : nil }
            let full = Set(config.ints("full_attn_idxs") ?? named ?? Array(0..<layers))
            kinds = (0..<layers).map { full.contains($0) ? .full : .recurrent }
        } else if let types = config.strings("layer_types") {
            let honorsSliding = slidingWindowModelTypes.contains(type)
            kinds = (0..<layers).map { $0 < types.count ? kind(of: types[$0], honorsSliding: honorsSliding) : .full }
        } else if let interval = config.int("full_attention_interval") {
            kinds = intervalKinds(layers: layers, interval: interval)
        } else {
            kinds = Array(repeating: .full, count: layers)
        }
        guard sharedKVModelTypes.contains(type), let shared = config.int("num_kv_shared_layers"),
              shared > 0, shared < layers else { return kinds }
        return kinds.enumerated().map { $0.offset >= layers - shared ? .uncached : $0.element }
    }

    private static func intervalKinds(layers: Int, interval: Int) -> [LayerCacheKind] {
        let step = max(interval, 1)
        return (0..<layers).map { ($0 + 1) % step == 0 ? .full : .recurrent }
    }

    static func kind(of layerType: String, honorsSliding: Bool) -> LayerCacheKind {
        let name = layerType.lowercased()
        if name.contains("sliding") { return honorsSliding ? .sliding : .full }
        if recurrentLayerMarkers.contains(where: { name.contains($0) }) { return .recurrent }
        return .full
    }

    private static func globalWidths(_ config: ConfigValues) -> (Int?, Int?) {
        let plausible: (Int?) -> Int? = { $0.flatMap { (1...maxDimension).contains($0) ? $0 : nil } }
        let heads = config.bool("attention_k_eq_v") == true ? plausible(config.int("num_global_key_value_heads")) : nil
        // 512 is Gemma4TextConfiguration's default when the key is absent.
        let width = plausible(config.int("global_head_dim") ?? 512)
        return (heads, width == plausible(config.int("head_dim")) ? nil : width)
    }

    /// The fixed state of `layers` recurrent layers at 2 bytes per element, from the first family whose keys the
    /// config has; 0 when none match. `nil` only for implausible values.
    static func recurrentStateBytes(_ config: ConfigValues, layers: Int) -> Int? {
        guard layers > 0 else { return 0 }
        let dims: (String) -> Int? = { config.int($0) }
        var elements: Double = 0
        if let valueHeads = dims("linear_num_value_heads"), let valueDim = dims("linear_value_head_dim"),
           let keyHeads = dims("linear_num_key_heads"), let keyDim = dims("linear_key_head_dim") {
            // Gated DeltaNet: state [Hv, Dv, Dk] + conv state [kernel - 1, 2·Hk·Dk + Hv·Dv].
            let kernel = dims("linear_conv_kernel_dim") ?? 4
            let values: [Int] = [valueHeads, valueDim, keyHeads, keyDim, kernel]
            guard values.allSatisfy({ (1...maxDimension).contains($0) }) else { return nil }
            elements = Double(valueHeads * valueDim * keyDim)
                + Double(kernel - 1) * Double(2 * keyHeads * keyDim + valueHeads * valueDim)
        } else if let heads = dims("mamba_n_heads"), let headDim = dims("mamba_d_head"),
                  let stateDim = dims("mamba_d_state") {
            // Mamba 2: state [heads, headDim, stateDim] + conv state [kernel - 1, heads·headDim + 2·groups·stateDim].
            let kernel = dims("mamba_d_conv") ?? 4
            let groups = dims("mamba_n_groups") ?? 1
            let values: [Int] = [heads, headDim, stateDim, kernel, groups]
            guard values.allSatisfy({ (1...maxDimension).contains($0) }) else { return nil }
            elements = Double(heads * headDim * stateDim)
                + Double(kernel - 1) * Double(heads * headDim + 2 * groups * stateDim)
        } else if let cache = dims("conv_L_cache"), let hidden = dims("hidden_size") {
            // LFM2 short convolution: [L_cache - 1, hidden].
            guard (1...maxDimension).contains(cache), (1...maxDimension).contains(hidden) else { return nil }
            elements = Double(cache - 1) * Double(hidden)
        }
        let bytes = elements * 2 * Double(layers)
        return bytes < Double(Int.max / 2) ? Int(bytes) : nil
    }
}

// MARK: - GGUF layout

/// The KV cache llama.cpp allocates for a GGUF, with the heads and width of the layers ``KVCacheLayout`` sizes at the
/// model's `kvHeads` / `headDim`.
struct GGUFCacheShape: Equatable {
    let layout: KVCacheLayout
    let kvHeads: Int
    let headDim: Int
}

extension KVCacheLayout {

    // Evidence: llama.cpp b8851 llama-model.cpp (LLM_ARCH_GEMMA4 hparams), llama-hparams.cpp (`n_embd_k_gqa(il)`,
    // `has_kv`), llama-kv-cache-iswa.cpp (`swa_full`), llama-context.cpp (`swa_full = true` by default).

    /// llama.cpp's per-layer cache for a header that lists its sliding-window layers
    /// (`attention.sliding_window_pattern`, one flag per layer); `nil` without that list, so every layer is sized
    /// like the widest one, or when the per-layer values are implausible.
    ///
    /// Sliding layers take `attention.key_length_swa` as their width and global layers `attention.key_length`;
    /// `attention.head_count_kv` may list heads per layer. Both kinds are sized for the whole context (`swa_full`).
    /// The last `attention.shared_kv_layers` layers reuse an earlier cache (`n_layer_kv_from_start`).
    static func derive(gguf header: GGUFHeader) -> GGUFCacheShape? {
        guard let architecture = header.architecture,
              let layers = header.blockCount, (1...maxLayers).contains(layers),
              let pattern = perLayer(header.metadata["\(architecture).attention.sliding_window_pattern"], layers),
              let fullWidth = header.headDimension, let fullHeads = header.headCountKV else { return nil }
        let swaWidth = header.architectureInteger("attention.key_length_swa") ?? fullWidth
        let heads = perLayer(header.metadata["\(architecture).attention.head_count_kv"], layers)
            ?? Array(repeating: fullHeads, count: layers)
        let shared = header.architectureInteger("attention.shared_kv_layers")
            .flatMap { (0..<layers).contains($0) ? $0 : nil } ?? 0
        let cached = 0..<(layers - shared)
        let sliding = cached.filter { pattern[$0] != 0 }
        let global = cached.filter { pattern[$0] == 0 }
        guard !sliding.isEmpty,
              let swaHeads = sliding.map({ heads[$0] }).max(),
              (1...maxDimension).contains(swaHeads), (1...maxDimension).contains(swaWidth),
              (1...maxDimension).contains(fullWidth) else { return nil }
        let globalHeads = global.map { heads[$0] }.max() ?? swaHeads
        guard (1...maxDimension).contains(globalHeads) else { return nil }
        // Same maxima on both kinds and nothing shared: the uniform estimate is already exact.
        if shared == 0, globalHeads == swaHeads, fullWidth == swaWidth { return nil }
        let layout = KVCacheLayout(
            fullAttentionLayers: global.count, slidingLayers: sliding.count, slidingWindow: 0,
            fullAttentionKVHeads: global.isEmpty || globalHeads == swaHeads ? nil : globalHeads,
            fullAttentionHeadDim: global.isEmpty || fullWidth == swaWidth ? nil : fullWidth)
        return GGUFCacheShape(layout: layout, kvHeads: swaHeads, headDim: swaWidth)
    }

    /// One integer per layer (flags as 0 / 1), when the array is complete; the parser keeps only its first elements.
    private static func perLayer(_ value: GGUFValue?, _ layers: Int) -> [Int]? {
        guard case .array(let list)? = value, list.count == UInt64(layers), list.elements.count == layers else {
            return nil
        }
        let values = list.elements.compactMap(\.integer)
        return values.count == layers ? values : nil
    }
}

// MARK: - Expert bytes

extension ExpertRouting {

    /// Name fragments of routed-expert tensors: one tensor per expert (`.experts.N.`), stacked MLX experts
    /// (`switch_mlp`), and Granite's stacked `input_linear` / `output_linear`. Shared experts (`shared_expert`,
    /// `shared_experts`) and routers (`.gate.`, `.router.`) match none of them.
    static let expertMarkers = [
        ".experts.", ".switch_mlp.", ".block_sparse_moe.input_linear.", ".block_sparse_moe.output_linear.",
    ]

    /// Tensors a decoded token never reads: vision towers and multi-token-prediction heads.
    static let skippedPrefixes = CompatibilityRules.visionTensorPrefixes + ["mtp", "model.mtp", "language_model.mtp"]

    static func isExpert(_ name: String) -> Bool {
        let dotted = "." + name
        return expertMarkers.contains { dotted.contains($0) }
    }

    static func isSkipped(_ name: String) -> Bool {
        skippedPrefixes.contains { name.hasPrefix($0 + ".") }
    }

    /// Routing measured from exact tensor sizes; `nil` when the counts are implausible or no expert tensor is named.
    static func measure(tensorBytes: [String: Int64], routedExperts: Int, expertsPerToken: Int) -> ExpertRouting? {
        guard routedExperts > 1, routedExperts <= 1 << 16, (1...routedExperts).contains(expertsPerToken) else {
            return nil
        }
        let decoded = tensorBytes.filter { !isSkipped($0.key) }
        let expertBytes = decoded.filter { isExpert($0.key) }.values.reduce(0.0) { $0 + Double($1) }
        let otherBytes = decoded.filter { !isExpert($0.key) }.values.reduce(0.0) { $0 + Double($1) }
        guard expertBytes > 0 else { return nil }
        let active = otherBytes + expertBytes * Double(expertsPerToken) / Double(routedExperts)
        return ExpertRouting(routedExperts: routedExperts, expertsPerToken: expertsPerToken,
                             activeSizeMB: Int((active / 1_048_576).rounded()))
    }
}
