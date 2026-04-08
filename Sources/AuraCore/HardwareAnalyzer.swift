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
    /// On iOS this uses `os_proc_available_memory()`; on macOS it
    /// falls back to 60 % of physical RAM.
    public let availableMemoryGB: Double

    /// A human-readable device or chip identifier (e.g. "iPhone", "Apple M2 Pro").
    public let deviceName: String

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
        availableGB = totalGB * 0.6
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
            deviceName: name
        )
    }
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

    /// Percentage of available memory consumed (0–100+).
    public var utilizationPercent: Double {
        guard availableMemoryGB > 0 else { return 100 }
        return (requiredMemoryGB / availableMemoryGB) * 100
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
public enum HardwareAnalyzer {

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
            // Model doesn't fit monolithically — check if streaming is viable
            if model.format == .gguf {
                let streamingRequired = model.estimatedStreamingMemoryGB
                let streamingRatio = available > 0 ? streamingRequired / available : .infinity
                if streamingRatio < 1.0 {
                    fitLevel = .streamingRequired
                } else {
                    fitLevel = .tooLarge
                }
            } else {
                fitLevel = .tooLarge  // MLX models can't stream layers
            }
        }

        return ModelCompatibility(
            model: model,
            fitLevel: fitLevel,
            requiredMemoryGB: required,
            availableMemoryGB: available
        )
    }
}
