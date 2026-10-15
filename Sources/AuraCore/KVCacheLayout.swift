import Foundation

// MARK: - KVCacheLayout

/// How a model's KV cache grows with the context, per kind of layer, as the pinned mlx-swift-lm allocates it:
/// a full-attention layer keeps every token (`KVCacheSimple` / `StandardKVCache`), a sliding-window layer keeps at
/// most ``slidingWindow`` tokens (`RotatingKVCache`), and a linear-attention or state-space layer keeps a fixed-size
/// state (`MambaCache`). Layers that reuse another layer's cache (Gemma 4's shared-KV layers) are in none of the counts.
///
/// `models.json` carries it as the optional `kvLayers` object; entries without it size every layer as full attention.
public struct KVCacheLayout: Sendable, Equatable, Codable {
    /// Layers whose cache holds every token of the context.
    public let fullAttentionLayers: Int
    /// Layers whose cache holds at most ``slidingWindow`` tokens.
    public let slidingLayers: Int
    /// Tokens a sliding-window layer keeps; 0 when there are none.
    public let slidingWindow: Int
    /// Linear-attention / state-space layers: a fixed state, whatever the context.
    public let recurrentLayers: Int
    /// That fixed state, summed over every recurrent layer, in bytes.
    public let recurrentStateBytes: Int
    /// KV heads of the full-attention layers when they differ from the model's `kvHeads` (Gemma 4); `nil` = same.
    public let fullAttentionKVHeads: Int?
    /// Head width of the full-attention layers when it differs from the model's `headDim` (Gemma 4); `nil` = same.
    public let fullAttentionHeadDim: Int?

    public init(fullAttentionLayers: Int, slidingLayers: Int = 0, slidingWindow: Int = 0, recurrentLayers: Int = 0,
                recurrentStateBytes: Int = 0, fullAttentionKVHeads: Int? = nil, fullAttentionHeadDim: Int? = nil) {
        self.fullAttentionLayers = fullAttentionLayers
        self.slidingLayers = slidingLayers
        self.slidingWindow = slidingWindow
        self.recurrentLayers = recurrentLayers
        self.recurrentStateBytes = recurrentStateBytes
        self.fullAttentionKVHeads = fullAttentionKVHeads
        self.fullAttentionHeadDim = fullAttentionHeadDim
    }

    private enum CodingKeys: String, CodingKey {
        case fullAttentionLayers, slidingLayers, slidingWindow, recurrentLayers, recurrentStateBytes
        case fullAttentionKVHeads, fullAttentionHeadDim
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
    }

    /// Layers that hold any cache.
    public var cachedLayers: Int { fullAttentionLayers + slidingLayers + recurrentLayers }

    /// Bytes the caches hold at `contextLength` tokens: K and V at 2 bytes per element (fp16 / bf16), so
    /// `4 × heads × headDim` per token per attention layer, plus the fixed recurrent state. A sliding layer with no
    /// window counts as full attention. Double, because the inputs can come from untrusted files.
    public func cacheBytes(contextLength: Int, kvHeads: Int, headDim: Int) -> Double {
        let context = Double(max(contextLength, 0))
        let fullWidth = 4 * Double(fullAttentionKVHeads ?? kvHeads) * Double(fullAttentionHeadDim ?? headDim)
        let slidingWidth = 4 * Double(kvHeads) * Double(headDim)
        let kept = slidingWindow > 0 ? min(context, Double(slidingWindow)) : context
        return Double(max(fullAttentionLayers, 0)) * fullWidth * context
            + Double(max(slidingLayers, 0)) * slidingWidth * kept
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
