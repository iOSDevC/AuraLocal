#if !targetEnvironment(simulator)
import Foundation
#if os(iOS) || os(tvOS) || os(watchOS)
import UIKit
#endif

// MARK: - MemoryBudgetManager

/// Monitors memory usage and enforces jetsam-safe budgets during inference.
///
/// On iOS, the system kills apps that exceed ~1.5 GB of RAM (jetsam).
/// This manager provides:
/// - Real-time available memory queries via `os_proc_available_memory()`
/// - A configurable safety margin (default 200 MB)
/// - Pressure detection for adaptive behavior during layer-streaming
///
/// On macOS, constraints are relaxed since unified memory is typically 16-96 GB.
@MainActor
final class MemoryBudgetManager {

    // MARK: - Configuration

    /// Minimum memory to keep free (bytes). If available memory drops below
    /// this threshold, the manager signals pressure.
    let safetyMarginBytes: Int

    /// Maximum memory the app should use for inference (bytes).
    /// Calculated as: available memory - safety margin.
    private(set) var budgetBytes: Int

    // MARK: - Init

    init(safetyMarginMB: Int = 200) {
        self.safetyMarginBytes = safetyMarginMB * 1024 * 1024
        self.budgetBytes = Self.calculateBudget(safetyMarginBytes: safetyMarginMB * 1024 * 1024)
    }

    // MARK: - Queries

    /// Current available memory in bytes.
    static func availableMemoryBytes() -> Int {
        #if os(iOS) || os(tvOS) || os(watchOS)
        let available = os_proc_available_memory()
        if available > 0 { return Int(available) }
        #endif
        // macOS fallback: 60% of physical RAM
        return Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.6)
    }

    /// Whether the system is under memory pressure.
    /// Returns `true` if available memory is below the safety margin.
    var isUnderPressure: Bool {
        Self.availableMemoryBytes() < safetyMarginBytes
    }

    /// Whether there's enough memory to allocate the given number of bytes.
    func canAllocate(bytes: Int) -> Bool {
        Self.availableMemoryBytes() - bytes > safetyMarginBytes
    }

    /// Recommended context length based on current memory conditions.
    /// Reduces context dynamically when memory is tight.
    func recommendedContextLength(baseContext: Int) -> Int {
        let available = Self.availableMemoryBytes()
        let freeAfterMargin = available - safetyMarginBytes

        if freeAfterMargin > 512 * 1024 * 1024 {
            return baseContext  // Plenty of room
        } else if freeAfterMargin > 256 * 1024 * 1024 {
            return min(baseContext, 1024)  // Moderate pressure
        } else {
            return min(baseContext, 512)   // Tight — minimize context
        }
    }

    // MARK: - Platform-specific budgets

    /// Maximum inference budget for the current platform.
    static func platformBudget() -> InferenceBudget {
        #if os(macOS)
        return macOSBudget()
        #else
        return iOSBudget()
        #endif
    }

    #if os(macOS)
    private static func macOSBudget() -> InferenceBudget {
        let totalGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return InferenceBudget(
            maxModelSizeGB: totalGB * 0.6,         // Use up to 60% for model
            maxContextLength: 8192,                 // Generous context
            maxGPULayers: 999,                      // All layers on GPU
            canStreamLayers: true,
            recommendedQuantization: totalGB >= 32 ? .fp16 : .q4_k_m
        )
    }
    #endif

    private static func iOSBudget() -> InferenceBudget {
        let totalGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824

        if totalGB >= 8 {
            // iPhone 15 Pro, iPad Pro — 8 GB
            return InferenceBudget(
                maxModelSizeGB: 4.5,
                maxContextLength: 2048,
                maxGPULayers: 99,
                canStreamLayers: true,
                recommendedQuantization: .q4_k_m
            )
        } else {
            // iPhone 14, iPhone 15 base — 6 GB
            return InferenceBudget(
                maxModelSizeGB: 2.5,
                maxContextLength: 1024,
                maxGPULayers: 99,
                canStreamLayers: true,
                recommendedQuantization: .q4_k_m
            )
        }
    }

    // MARK: - Helpers

    private static func calculateBudget(safetyMarginBytes: Int) -> Int {
        let available = availableMemoryBytes()
        return max(0, available - safetyMarginBytes)
    }

    /// Refresh the budget (call after significant memory changes).
    func refresh() {
        budgetBytes = Self.calculateBudget(safetyMarginBytes: safetyMarginBytes)
    }
}

// MARK: - InferenceBudget

/// Platform-specific constraints for model inference.
struct InferenceBudget {
    /// Maximum model weight size in GB that can be loaded monolithically.
    let maxModelSizeGB: Double
    /// Maximum context length (number of tokens) for KV cache.
    let maxContextLength: Int
    /// Maximum transformer layers to offload to GPU (Metal).
    let maxGPULayers: Int
    /// Whether layer-streaming is viable on this device.
    let canStreamLayers: Bool
    /// Recommended quantization level for best performance/quality balance.
    let recommendedQuantization: QuantizationLevel
}

// MARK: - QuantizationLevel

enum QuantizationLevel: String, Sendable {
    case fp16 = "FP16"
    case q8_0 = "Q8_0"
    case q4_k_m = "Q4_K_M"
    case q4_0 = "Q4_0"

    var label: String { rawValue }

    /// Approximate compression ratio vs FP16.
    var compressionRatio: Double {
        switch self {
        case .fp16:   return 1.0
        case .q8_0:   return 0.5
        case .q4_k_m: return 0.28
        case .q4_0:   return 0.25
        }
    }
}
#endif
