import Foundation
#if os(iOS) || os(tvOS) || os(watchOS)
import UIKit
#endif

// MARK: - HardwareProfile

/// Snapshot of the device's memory capabilities.
///
/// On Apple Silicon the GPU and CPU share a unified memory pool,
/// so ``totalMemoryGB`` represents the single budget for both inference
/// and the rest of the system.
public struct HardwareProfile: Sendable {

    /// Total physical RAM in gigabytes.
    public let totalMemoryGB: Double

    /// Memory currently available to the process, in gigabytes.
    /// On iOS this uses `os_proc_available_memory()`; on macOS it reads real
    /// reclaimable memory from the kernel (see ``HardwareProfile/current()``).
    public let availableMemoryGB: Double

    /// A human-readable device or chip identifier (e.g. "iPhone", "Apple M2 Pro").
    public let deviceName: String

    /// Unified-memory bandwidth in GB/s for this chip, or `nil` when unknown.
    ///
    /// Decode speed is bandwidth-bound, so this is what turns a RAM-fit answer
    /// into a wall-clock one. Published spec figures; conservative for binned
    /// variants (M3 Max 300 not 400, M4 Max 410 not 546). `nil` on iOS and for
    /// unrecognised chips — better no estimate than a guessed one.
    public let memoryBandwidthGBs: Double?

    public init(
        totalMemoryGB: Double,
        availableMemoryGB: Double,
        deviceName: String,
        memoryBandwidthGBs: Double? = nil
    ) {
        self.totalMemoryGB = totalMemoryGB
        self.availableMemoryGB = availableMemoryGB
        self.deviceName = deviceName
        self.memoryBandwidthGBs = memoryBandwidthGBs
    }

    /// Detects the current device's hardware profile.
    public static func current() -> HardwareProfile {
        let total = Double(ProcessInfo.processInfo.physicalMemory)
        let totalGB = total / 1_073_741_824 // 1 GB

        var availableGB: Double
        #if os(iOS) || os(tvOS) || os(watchOS)
        let proc = os_proc_available_memory()
        availableGB = proc > 0
            ? Double(proc) / 1_073_741_824
            : totalGB * 0.6
        #else
        // Measure, don't guess. The old `totalGB * 0.6` assumed 60 % was free no
        // matter what was running — on a dev machine with Xcode + Simulator
        // (10–14 GB) that overstates the budget ~2x, so models are assessed as
        // fitting when they would thrash. Reading the kernel is right for BOTH a
        // busy dev Mac and an end user's idle one.
        availableGB = macReclaimableMemoryGB() ?? (totalGB * 0.6)
        #endif

        let name: String
        #if os(iOS) || os(tvOS)
        name = UIDevice.current.name
        #elseif os(macOS)
        name = Host.current().localizedName ?? ProcessInfo.processInfo.hostName
        #else
        name = "Apple Device"
        #endif

        return HardwareProfile(
            totalMemoryGB: totalGB,
            availableMemoryGB: availableGB,
            deviceName: name,
            memoryBandwidthGBs: detectMemoryBandwidthGBs()
        )
    }

    /// Published unified-memory bandwidth per Apple Silicon chip (GB/s).
    /// Longest names first — "Apple M1 Pro" must not match the bare "M1" row.
    private static let bandwidthTable: [(chip: String, gbs: Double)] = [
        ("M1 Ultra", 800), ("M1 Max", 400), ("M1 Pro", 200), ("M1", 68),
        ("M2 Ultra", 800), ("M2 Max", 400), ("M2 Pro", 200), ("M2", 100),
        ("M3 Ultra", 800), ("M3 Max", 300), ("M3 Pro", 150), ("M3", 100),
        ("M4 Max", 410), ("M4 Pro", 273), ("M4", 120),
    ]

    /// `nil` on iOS (no reliable public source) and for chips we don't know.
    static func detectMemoryBandwidthGBs() -> Double? {
        #if os(macOS)
        guard let brand = sysctlString("machdep.cpu.brand_string") else { return nil }
        return bandwidthTable.first { brand.contains($0.chip) }?.gbs
        #else
        return nil
        #endif
    }

    #if os(macOS)
    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        if let nul = buffer.firstIndex(of: 0) { buffer.removeSubrange(nul...) }
        return String(decoding: buffer, as: UTF8.self)
    }
    #endif

    #if os(macOS)
    /// Free + reclaimable memory reported by the kernel, in GB.
    ///
    /// `free + inactive + purgeable` is what the system can hand out without
    /// swapping. Returns `nil` if the kernel query fails, so the caller can fall
    /// back to the old heuristic.
    static func macReclaimableMemoryGB() -> Double? {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        let pages = Double(stats.free_count) + Double(stats.inactive_count) + Double(stats.purgeable_count)
        let pageSize = Double(sysconf(_SC_PAGESIZE))   // vm_page_size is a mutable global — not Sendable
        return pages * pageSize / 1_073_741_824
    }
    #endif
}

// MARK: - ModelFitLevel

/// How well a model fits the device's available memory.
///
/// Mirrors llmfit's `FitLevel` adapted for Apple's unified-memory architecture.
/// The ``streamingRequired`` level indicates the model is too large for monolithic
/// loading but can run via layer-streaming (one transformer layer at a time).
public enum ModelFitLevel: Comparable, Sendable {
    /// Recommended memory met with >40 % headroom.
    case excellent
    /// Fits comfortably with 20-40 % headroom.
    case good
    /// Runnable but tight (<20 % headroom).
    case marginal
    /// Too large for full load, but viable with layer-streaming.
    /// Only applicable to GGUF models — MLX models cannot stream layers.
    case streamingRequired
    /// Exceeds available memory even with streaming — not viable on this device.
    case tooLarge

    /// Short label for display in UI.
    public var label: String {
        switch self {
        case .excellent:          "Excellent"
        case .good:               "Good"
        case .marginal:           "Marginal"
        case .streamingRequired:  "Streaming"
        case .tooLarge:           "Too Large"
        }
    }

    /// SF Symbol name for the fit badge.
    public var systemImage: String {
        switch self {
        case .excellent:          "checkmark.seal.fill"
        case .good:               "checkmark.circle.fill"
        case .marginal:           "exclamationmark.triangle.fill"
        case .streamingRequired:  "arrow.down.circle.fill"
        case .tooLarge:           "xmark.octagon.fill"
        }
    }

    /// Whether inference is possible on this device (with any backend).
    public var isRunnable: Bool {
        switch self {
        case .excellent, .good, .marginal, .streamingRequired: return true
        case .tooLarge: return false
        }
    }
}

// MARK: - ModelCompatibility

/// Hardware compatibility assessment for a single ``Model``.
public struct ModelCompatibility: Sendable {

    /// The model being assessed.
    public let model: Model

    /// How well the model fits the device.
    public let fitLevel: ModelFitLevel

    /// Estimated runtime memory footprint in GB (weights + KV cache + overhead).
    public let requiredMemoryGB: Double

    /// Memory available to the process in GB.
    public let availableMemoryGB: Double

    /// Estimated decode speed in tokens/sec, or `nil` when the chip's memory
    /// bandwidth is unknown (iOS, unrecognised chips).
    ///
    /// Fitting in RAM is only half the answer: a model that fits but decodes at
    /// 3 tok/s is unusable, and reporting it as ``ModelFitLevel/excellent`` is
    /// actively misleading. See ``speedLevel``.
    public let estimatedDecodeTokensPerSecond: Double?

    /// Percentage of available memory consumed (0–100+).
    public var utilizationPercent: Double {
        guard availableMemoryGB > 0 else { return 100 }
        return (requiredMemoryGB / availableMemoryGB) * 100
    }

    /// How the model *feels* to use, independent of whether it fits.
    public var speedLevel: SpeedLevel {
        guard let tps = estimatedDecodeTokensPerSecond else { return .unknown }
        switch tps {
        case ..<5:   return .unusable
        case ..<15:  return .slow
        case ..<40:  return .usable
        default:     return .fast
        }
    }

    /// Seconds to produce a `tokens`-long answer, or `nil` if speed is unknown.
    public func estimatedResponseSeconds(tokens: Int = 500) -> Double? {
        guard let tps = estimatedDecodeTokensPerSecond, tps > 0 else { return nil }
        return Double(tokens) / tps
    }

    /// The honest one-liner: RAM **and** wall-clock.
    public var verdict: String {
        guard let tps = estimatedDecodeTokensPerSecond else { return fitLevel.label }
        return String(format: "%@ · ~%.0f tok/s (%@)", fitLevel.label, tps, speedLevel.label)
    }
}

// MARK: - SpeedLevel

/// Decode speed banded into what a user actually experiences.
public enum SpeedLevel: String, Sendable, CaseIterable {
    /// Faster than reading speed.
    case fast
    /// Keeps up with reading; fine for chat.
    case usable
    /// Noticeably slow — minutes per long answer.
    case slow
    /// Fits in RAM, but too slow to interact with.
    case unusable
    /// Bandwidth unknown for this chip; no estimate made.
    case unknown

    public var label: String {
        switch self {
        case .fast:     "fast"
        case .usable:   "usable"
        case .slow:     "slow"
        case .unusable: "too slow"
        case .unknown:  "speed unknown"
        }
    }
}

// MARK: - Model + Runtime Memory Estimate

public extension Model {

    /// Estimated runtime memory in GB for **monolithic** (full-load) inference.
    ///
    /// Formula: model weights + framework overhead + GQA-aware KV cache estimate.
    /// For GGUF models this uses llama.cpp overhead (~200 MB); for MLX it uses ~500 MB.
    var estimatedRuntimeMemoryGB: Double {
        let weightsGB = Double(approximateSizeMB) / 1024.0
        let overheadGB = format == .gguf ? 0.2 : 0.5
        let kvCacheGB = estimatedKVCacheGB(contextLength: 2048)
        return weightsGB + overheadGB + kvCacheGB
    }

    /// Estimated runtime memory in GB for **layer-streaming** inference.
    ///
    /// Only the current layer + prefetched next layer + embedding + KV cache are in memory.
    /// This allows running models that far exceed device RAM.
    var estimatedStreamingMemoryGB: Double {
        let perLayerGB = estimatedPerLayerGB
        let embeddingGB = perLayerGB  // embedding table is roughly one layer's size
        let kvCacheGB = estimatedKVCacheGB(contextLength: 1024)  // reduced context in streaming
        let overheadGB = 0.1  // minimal framework overhead
        // Two layers in memory (current + prefetched) + embedding + KV cache
        return (perLayerGB * 2) + embeddingGB + kvCacheGB + overheadGB
    }

    /// Estimated size of a single transformer layer in GB.
    /// Approximation: total weights / num_layers (assumes ~95% of weights are in layers).
    var estimatedPerLayerGB: Double {
        let weightsGB = Double(approximateSizeMB) / 1024.0
        let layers = Double(max(numLayers, 1))
        return (weightsGB * 0.95) / layers
    }

    /// GQA-aware KV cache estimate in GB.
    ///
    /// Modern models (Llama 3, Mistral, Qwen2) use grouped-query attention
    /// with fewer KV heads than query heads, reducing KV cache by 4-8x.
    /// Formula: `2 * n_layers * n_kv_heads * head_dim * seq_len * bytes / 1 GB`
    ///
    /// For MLX models (kvHeads == 0), returns a flat 0.15 GB estimate.
    func estimatedKVCacheGB(contextLength: Int) -> Double {
        // MLX models use a flat estimate (managed differently at runtime)
        guard kvHeads > 0, headDim > 0 else {
            return 0.15
        }
        // 2 (K+V) * layers * kv_heads * head_dim * seq_len * 2 bytes (fp16)
        let bytes = 2 * numLayers * kvHeads * headDim * contextLength * 2
        return Double(bytes) / 1_073_741_824  // to GB
    }
}

// MARK: - HardwareAnalyzer

/// Analyzes hardware capabilities against the model catalog to determine
/// which models can run on the current device.
///
/// Inspired by [llmfit](https://github.com/AlexsJones/llmfit), simplified
/// for Apple's unified-memory architecture where GPU and CPU share one pool.
///
/// ```swift
/// let results = HardwareAnalyzer.compatibleModels()
/// for result in results where result.fitLevel != .tooLarge {
///     print("\(result.model.displayName): \(result.fitLevel.label)")
/// }
/// ```
// MARK: - ModelKind (runtime memory profile)

/// The runtime memory profile of a downloadable model — its peak RAM relative to the on-disk
/// weight size differs sharply by kind, so the compatibility filter must not treat them alike.
public enum ModelKind: Sendable {
    /// Text LLM (GGUF/MLX): peak ≈ weights + framework overhead + a modest KV cache.
    case llm
    /// Diffusion / image generation (FLUX, SD): peak ≈ **~2×** the weights — the text encoder,
    /// VAE, and activation/latent buffers are co-resident. MEASURED: FLUX schnell 4-bit, 9.2 GB
    /// on disk → ~20 GB peak footprint on an M1 Pro.
    case diffusion

    /// Multiplier from on-disk weight size to estimated peak resident memory.
    public var peakMultiplier: Double {
        switch self {
        case .llm:       1.15
        case .diffusion: 2.2
        }
    }
}

public enum HardwareAnalyzer {

    /// Compatibility of a **raw download** (a HuggingFace search hit) with this device, from its size
    /// alone. Unlike ``assess(_:profile:)`` this has no catalog metadata, so it estimates peak memory
    /// as `weights × kind.peakMultiplier` — the `kind` is what keeps a 6.8 GB FLUX from being called
    /// "excellent" when its real peak is ~15 GB. Streaming applies to LLMs only (diffusion can't stream).
    public static func fitLevel(
        forWeightsBytes bytes: Int,
        kind: ModelKind,
        profile: HardwareProfile = .current()
    ) -> ModelFitLevel {
        let weightsGB = Double(bytes) / 1_073_741_824
        let available = profile.availableMemoryGB
        guard weightsGB > 0, available > 0 else { return .tooLarge }
        let peakGB = weightsGB * kind.peakMultiplier
        let ratio = peakGB / available
        switch ratio {
        case ..<0.6: return .excellent
        case ..<0.8: return .good
        case ..<1.0: return .marginal
        default:
            // LLM weights can be layer-streamed while the page cache holds ≥1/3 of them; diffusion cannot.
            if kind == .llm, weightsGB <= 3.0 * available { return .streamingRequired }
            return .tooLarge
        }
    }

    /// Analyze all models against the current device hardware.
    ///
    /// Results are sorted: best fit first, `.tooLarge` models last.
    /// Within the same fit level, downloaded models appear first,
    /// then sorted by memory utilization (most efficient first).
    public static func compatibleModels(
        from models: [Model] = Model.allModels,
        profile: HardwareProfile = .current()
    ) -> [ModelCompatibility] {
        models
            .map { assess($0, profile: profile) }
            .sorted { lhs, rhs in
                // tooLarge always last
                if lhs.fitLevel != .tooLarge && rhs.fitLevel == .tooLarge { return true }
                if lhs.fitLevel == .tooLarge && rhs.fitLevel != .tooLarge { return false }

                // Downloaded first
                if lhs.model.isDownloaded != rhs.model.isDownloaded {
                    return lhs.model.isDownloaded
                }

                // Better fit first
                if lhs.fitLevel != rhs.fitLevel {
                    return lhs.fitLevel < rhs.fitLevel
                }

                // Lower utilization (more headroom) is better
                return lhs.utilizationPercent < rhs.utilizationPercent
            }
    }

    /// Analyze all models of a specific purpose.
    public static func compatibleModels(
        purpose: Model.Purpose,
        profile: HardwareProfile = .current()
    ) -> [ModelCompatibility] {
        let filtered: [Model]
        switch purpose {
        case .text:
            filtered = Model.textModels
        case .vision:
            filtered = Model.visionModels
        case .visionSpecialized:
            filtered = Model.specializedModels
        }
        return compatibleModels(from: filtered, profile: profile)
    }

    /// KV-aware context window for `model` on this device, right now.
    ///
    /// Backends used to hardcode a platform tier (2048 on iOS, 8192 on macOS),
    /// which collapsed every model's window regardless of what it supported —
    /// a low-KV hybrid model was capped like a KV-heavy dense one. This derives
    /// the window from the model's own GQA-aware KV cost and the memory left
    /// after weights, so each model gets the window it can actually afford.
    ///
    /// Falls back to the platform tier when the catalog lacks KV metadata
    /// (`kvHeads == 0`, e.g. MLX entries, which manage their cache differently).
    ///
    /// - Parameter availableGB: override the live memory reading (for tests).
    public static func recommendedContextWindow(
        for model: Model,
        availableGB: Double? = nil
    ) -> Int {
        #if os(macOS)
        let ceiling  = 32_768
        let fallback = HardwareProfile.current().totalMemoryGB >= 32 ? 8192 : 4096
        #else
        let ceiling  = 8192
        let fallback = HardwareProfile.current().totalMemoryGB >= 8 ? 2048 : 1024
        #endif

        // No KV metadata — can't do the math honestly; keep the old tier.
        guard model.kvHeads > 0, model.numLayers > 0, model.headDim > 0 else { return fallback }

        let available = availableGB
            ?? Double(MemoryBudgetManager.availableMemoryBytes()) / 1_073_741_824
        let weightsGB = Double(model.approximateSizeMB) / 1024.0
        let reserveGB = 0.4                       // framework overhead + safety margin
        let freeForKV = available - weightsGB - reserveGB
        guard freeForKV > 0.05 else { return 1024 }

        // estimatedKVCacheGB is linear in context length — derive the per-token cost.
        let perTokenGB = model.estimatedKVCacheGB(contextLength: 4096) / 4096
        guard perTokenGB > 0 else { return fallback }

        let fits = Int(freeForKV / perTokenGB)
        return max(1024, min((fits / 512) * 512, ceiling))
    }

    /// Assess a single model against the hardware profile.
    ///
    /// For GGUF models that don't fit in RAM, checks whether layer-streaming
    /// is viable before marking as `tooLarge`.
    public static func assess(
        _ model: Model,
        profile: HardwareProfile = .current()
    ) -> ModelCompatibility {
        let required = model.estimatedRuntimeMemoryGB
        let available = profile.availableMemoryGB
        let ratio = available > 0 ? required / available : .infinity

        let fitLevel: ModelFitLevel
        switch ratio {
        case ..<0.6:
            fitLevel = .excellent
        case ..<0.8:
            fitLevel = .good
        case ..<1.0:
            fitLevel = .marginal
        default:
            // Doesn't fit monolithically. Layer streaming (mmap) can still run it — but only
            // inside a usable envelope. Two gates, both must hold.
            if model.format == .gguf {
                let weightsGB = Double(model.approximateSizeMB) / 1024.0
                let workingSetFits = available > 0 && model.estimatedStreamingMemoryGB < available

                // The working-set check is NOT the real limit: with mmap the resident set is a
                // near-constant ~0.5–0.75 GB no matter the model size, so it almost always passes.
                // The real limit is disk thrashing. llama.cpp touches every weight once per token,
                // so any weight not resident in the page cache is re-read from storage each token.
                // When the weights dwarf available RAM the cache holds too little and decode
                // collapses to seconds-per-token — a 70B on a 3 GB phone would fault ~36 GB/token.
                // Require RAM to hold at least ~1/3 of the weights. Heuristic (not measured storage
                // bandwidth), bounded by the codified cases: 8B on 3 GB must stream (1.5×), 70B on
                // 3 GB must not (13×); the streaming backend's documented envelope is ~1.8×.
                let cacheableEnough = available > 0 && weightsGB <= 3.0 * available

                fitLevel = (workingSetFits && cacheableEnough) ? .streamingRequired : .tooLarge
            } else {
                fitLevel = .tooLarge  // MLX models can't stream layers
            }
        }

        return ModelCompatibility(
            model: model,
            fitLevel: fitLevel,
            requiredMemoryGB: required,
            availableMemoryGB: available,
            estimatedDecodeTokensPerSecond: estimatedDecodeTokensPerSecond(
                for: model, profile: profile
            )
        )
    }

    /// Fraction of spec memory bandwidth llama.cpp actually achieves.
    /// Measured on an M1 Pro (200 GB/s spec): 12.1 tok/s for 11.78 GB of weights
    /// ⇒ ~143 GB/s effective. Consistent with published Apple Silicon benchmarks.
    static let bandwidthEfficiency = 0.75

    /// First-order decode-speed estimate.
    ///
    /// Decode is memory-bound, not compute-bound: each token streams the whole
    /// weight set through the memory bus once, so `tok/s ≈ bandwidth / weights`.
    /// Validated against a measured point (M1 Pro, Llama-3.1-8B Q8: predicts
    /// ~12.7, measured 12.12).
    ///
    /// Returns `nil` when bandwidth is unknown rather than inventing a number.
    /// Conservative for MoE models — they stream only the active experts, so the
    /// real speed is higher than this whole-weights estimate.
    static func estimatedDecodeTokensPerSecond(
        for model: Model,
        profile: HardwareProfile
    ) -> Double? {
        guard let bandwidth = profile.memoryBandwidthGBs else { return nil }
        let weightsGB = Double(model.approximateSizeMB) / 1024.0
        guard weightsGB > 0 else { return nil }
        return (bandwidth * bandwidthEfficiency) / weightsGB
    }
}
