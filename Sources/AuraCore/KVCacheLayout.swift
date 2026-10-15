import Foundation

// MARK: - KVCacheLayout

/// How a model's KV cache grows with the context, per kind of layer, as the pinned runtimes allocate it.
///
/// MLX (mlx-swift-lm): a full-attention layer keeps every token (`KVCacheSimple` / `StandardKVCache`), a
/// sliding-window layer keeps at most ``slidingWindow`` tokens once decoding starts (`RotatingKVCache`), and a
/// linear-attention or state-space layer keeps a fixed-size state (`MambaCache`). During prefill a sliding layer
/// holds more: see ``slidingPrefillStep``.
///
/// llama.cpp (GGUF): `llama_context_default_params()` sets `swa_full`, so a sliding-window layer is sized for the
/// whole context like a global one. Such layers are listed as sliding with ``slidingWindow`` 0 only because their
/// width differs from the global layers'.
///
/// Layers that reuse another layer's cache (Gemma 4's shared-KV layers) are in none of the counts. `models.json`
/// carries it as the optional `kvLayers` object; entries without it size every layer as full attention.
public struct KVCacheLayout: Sendable, Equatable, Codable {
    /// Layers whose cache holds every token of the context.
    public let fullAttentionLayers: Int
    /// Sliding-window layers: see ``slidingWindow`` and ``slidingPrefillStep``.
    public let slidingLayers: Int
    /// Tokens a sliding-window layer keeps once decoding starts; 0 when there are none, or when the runtime sizes
    /// the layer for the whole context (llama.cpp).
    public let slidingWindow: Int
    /// Linear-attention / state-space layers: a fixed state, whatever the context.
    public let recurrentLayers: Int
    /// That fixed state, summed over every recurrent layer, in bytes.
    public let recurrentStateBytes: Int
    /// KV heads of the full-attention layers when they differ from the model's `kvHeads` (Gemma 4); `nil` = same.
    public let fullAttentionKVHeads: Int?
    /// Head width of the full-attention layers when it differs from the model's `headDim` (Gemma 4); `nil` = same.
    public let fullAttentionHeadDim: Int?
    /// Prompt tokens a sliding layer takes per prefill step: before trimming to ``slidingWindow`` it holds up to
    /// `slidingWindow + slidingPrefillStep − 1`. `nil` when the runtime prefills the whole prompt in one pass
    /// (Gemma 3 and the sliding-window VLMs in mlx-swift-lm 3.31.3), so the layer holds the whole prompt until the
    /// first decoded token, and when it is unknown.
    public let slidingPrefillStep: Int?

    public init(fullAttentionLayers: Int, slidingLayers: Int = 0, slidingWindow: Int = 0, recurrentLayers: Int = 0,
                recurrentStateBytes: Int = 0, fullAttentionKVHeads: Int? = nil, fullAttentionHeadDim: Int? = nil,
                slidingPrefillStep: Int? = nil) {
        self.fullAttentionLayers = fullAttentionLayers
        self.slidingLayers = slidingLayers
        self.slidingWindow = slidingWindow
        self.recurrentLayers = recurrentLayers
        self.recurrentStateBytes = recurrentStateBytes
        self.fullAttentionKVHeads = fullAttentionKVHeads
        self.fullAttentionHeadDim = fullAttentionHeadDim
        self.slidingPrefillStep = slidingPrefillStep
    }

    private enum CodingKeys: String, CodingKey {
        case fullAttentionLayers, slidingLayers, slidingWindow, recurrentLayers, recurrentStateBytes
        case fullAttentionKVHeads, fullAttentionHeadDim, slidingPrefillStep
    }

    // One malformed entry must not fail the whole catalog decode, so missing counts read as 0.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        fullAttentionLayers = try container.decodeIfPresent(Int.self, forKey: .fullAttentionLayers) ?? 0
        slidingLayers = try container.decodeIfPresent(Int.self, forKey: .slidingLayers) ?? 0
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 0
        recurrentLayers = try container.decodeIfPresent(Int.self, forKey: .recurrentLayers) ?? 0
        recurrentStateBytes = try container.decodeIfPresent(Int.self, forKey: .recurrentStateBytes) ?? 0
        fullAttentionKVHeads = try container.decodeIfPresent(Int.self, forKey: .fullAttentionKVHeads)
        fullAttentionHeadDim = try container.decodeIfPresent(Int.self, forKey: .fullAttentionHeadDim)
        slidingPrefillStep = try container.decodeIfPresent(Int.self, forKey: .slidingPrefillStep)
    }

    /// Layers that hold any cache.
    public var cachedLayers: Int { fullAttentionLayers + slidingLayers + recurrentLayers }

    /// Peak bytes the caches hold for a `contextLength`-token prompt: K and V at 2 bytes per element (fp16 / bf16),
    /// so `4 × heads × headDim` per token per attention layer, plus the fixed recurrent state. Sliding layers count
    /// at their prefill peak (``slidingPrefillStep``); with no window they count as full attention. Double, because
    /// the inputs can come from untrusted files.
    public func cacheBytes(contextLength: Int, kvHeads: Int, headDim: Int) -> Double {
        bytes(contextLength: contextLength, kvHeads: kvHeads, headDim: headDim, slidingTokens: slidingPeak)
    }

    /// Bytes once decoding has started, after sliding layers are trimmed to their window.
    func decodingCacheBytes(contextLength: Int, kvHeads: Int, headDim: Int) -> Double {
        bytes(contextLength: contextLength, kvHeads: kvHeads, headDim: headDim) { context in
            slidingWindow > 0 ? min(context, Double(slidingWindow)) : context
        }
    }

    private func slidingPeak(_ context: Double) -> Double {
        guard slidingWindow > 0, let step = slidingPrefillStep, step > 0 else { return context }
        return min(context, Double(slidingWindow) + Double(step) - 1)
    }

    private func bytes(contextLength: Int, kvHeads: Int, headDim: Int,
                       slidingTokens: (Double) -> Double) -> Double {
        let context = Double(max(contextLength, 0))
        let fullWidth = 4 * Double(fullAttentionKVHeads ?? kvHeads) * Double(fullAttentionHeadDim ?? headDim)
        let slidingWidth = 4 * Double(kvHeads) * Double(headDim)
        return Double(max(fullAttentionLayers, 0)) * fullWidth * context
            + Double(max(slidingLayers, 0)) * slidingWidth * slidingTokens(context)
            + Double(max(recurrentStateBytes, 0))
    }
}

// MARK: - ExpertRouting

/// Mixture-of-experts routing. Every expert stays resident, so memory fit uses the whole download; a decoded token
/// reads only the non-expert weights and `expertsPerToken / routedExperts` of the expert weights.
///
/// `models.json` carries it as the optional `experts` object.
public struct ExpertRouting: Sendable, Equatable, Codable {
    /// Routed experts per MoE layer (`num_experts`, `num_local_experts`, `n_routed_experts`).
    public let routedExperts: Int
    /// Experts each token is routed to (`num_experts_per_tok`).
    public let expertsPerToken: Int
    /// MiB read per decoded token: every non-expert weight plus `expertsPerToken / routedExperts` of the expert
    /// weights. Shared experts count as non-expert; vision towers and MTP heads are left out.
    public let activeSizeMB: Int

    public init(routedExperts: Int, expertsPerToken: Int, activeSizeMB: Int) {
        self.routedExperts = routedExperts
        self.expertsPerToken = expertsPerToken
        self.activeSizeMB = activeSizeMB
    }

    private enum CodingKeys: String, CodingKey {
        case routedExperts, expertsPerToken, activeSizeMB
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        routedExperts = try container.decodeIfPresent(Int.self, forKey: .routedExperts) ?? 0
        expertsPerToken = try container.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 0
        activeSizeMB = try container.decodeIfPresent(Int.self, forKey: .activeSizeMB) ?? 0
    }
}
