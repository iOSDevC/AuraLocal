import Foundation
#if os(macOS)
import Metal
#endif

/// The operating system a device preset stands for.
public enum TargetOS: String, Sendable, Codable, CaseIterable {
    case iOS
    case macOS
}

/// A device to judge a model against: this one (measured now) or a class of iPhone / Mac.
///
/// `budgetGB` is the memory an app may fill with weights + KV cache. Only ``thisDevice(profile:)`` and the
/// 32 GB Mac are measured; every other budget is an estimate and says so in ``source``.
public struct DevicePreset: Sendable, Hashable, Identifiable {
    public let id: String
    public let displayName: String
    public let platform: TargetOS
    public let totalMemoryGB: Double
    public let budgetGB: Double
    public let isMeasured: Bool
    /// Where ``budgetGB`` comes from.
    public let source: String
    public let bandwidthGBs: Double?

    public init(id: String, displayName: String, platform: TargetOS, totalMemoryGB: Double, budgetGB: Double,
                isMeasured: Bool, source: String, bandwidthGBs: Double? = nil) {
        self.id = id
        self.displayName = displayName
        self.platform = platform
        self.totalMemoryGB = totalMemoryGB
        self.budgetGB = budgetGB
        self.isMeasured = isMeasured
        self.source = source
        self.bandwidthGBs = bandwidthGBs
    }

    /// The profile ``HardwareAnalyzer`` consumes.
    public var memoryProfile: HardwareProfile {
        HardwareProfile(totalMemoryGB: totalMemoryGB, availableMemoryGB: budgetGB,
                        deviceName: displayName, memoryBandwidthGBs: bandwidthGBs)
    }

    /// Budget with its provenance, e.g. `20.0 GB (measured)` / `≈3.0 GB (estimate)`.
    public var budgetText: String {
        let amount = String(format: "%.1f GB", budgetGB)
        return isMeasured ? "\(amount) (measured)" : "≈\(amount) (estimate)"
    }

    // MARK: Presets

    public static let thisDeviceID = "this-device"

    /// The running device. Without a `profile`, a Mac is judged by its Metal working-set limit (what the GPU may
    /// keep resident, the same budget as ``mac32GB``) and an iPhone by the memory the process can still allocate.
    /// A given `profile` is used as is.
    public static func thisDevice(profile: HardwareProfile? = nil) -> DevicePreset {
        #if os(macOS)
        let platform = TargetOS.macOS
        #else
        let platform = TargetOS.iOS
        #endif
        let measured = profile ?? .current()
        var budget = measured.availableMemoryGB
        var source = "Measured now by HardwareProfile: memory this process can still allocate."
        #if os(macOS)
        // Free + inactive pages swing with other apps and ignore compression; the Metal limit does not.
        if profile == nil, let limit = metalWorkingSetGB {
            budget = limit
            source = "Measured now: Metal recommendedMaxWorkingSetSize, the memory the GPU may keep resident."
        }
        #endif
        return DevicePreset(
            id: thisDeviceID, displayName: "This device (\(measured.deviceName))", platform: platform,
            totalMemoryGB: measured.totalMemoryGB, budgetGB: budget, isMeasured: true, source: source,
            bandwidthGBs: measured.memoryBandwidthGBs)
    }

    #if os(macOS)
    /// Fixed until the next boot or `iogpu.wired_limit_mb` change, so it is read once.
    private static let metalWorkingSetGB: Double? = {
        guard let size = MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize, size > 0 else { return nil }
        return Double(size) / 1_073_741_824
    }()
    #endif

    public static let iPhone4GB = DevicePreset(
        id: "iphone-4gb", displayName: "iPhone, 4 GB class (iPhone 12 / 13)", platform: .iOS,
        totalMemoryGB: 4, budgetGB: 2098.0 / 1024, isMeasured: false,
        source: "Estimate: third-party jetsam measurement, ActiveHard 2098 MB on an iPhone 12.")

    public static let iPhone6GB = DevicePreset(
        id: "iphone-6gb", displayName: "iPhone, 6 GB class (iPhone 12 Pro – 15)", platform: .iOS,
        totalMemoryGB: 6, budgetGB: 3.0, isMeasured: false,
        source: "Estimate: AuraLocal memory guide, ~3 GB with the increased-memory-limit entitlement.")

    public static let iPhone8GB = DevicePreset(
        id: "iphone-8gb", displayName: "iPhone, 8 GB class (iPhone 15 Pro – 17)", platform: .iOS,
        totalMemoryGB: 8, budgetGB: 4.5, isMeasured: false,
        source: "Estimate: AuraLocal memory guide, ~4.5 GB with the increased-memory-limit entitlement.")

    public static let iPhone17Pro = DevicePreset(
        id: "iphone-17-pro", displayName: "iPhone 17 Pro (12 GB)", platform: .iOS,
        totalMemoryGB: 12, budgetGB: 6.4, isMeasured: false,
        source: "Estimate: third-party report, ~6.4 GB only with both memory entitlements "
            + "(increased-memory-limit + extended-virtual-addressing).")

    public static let mac16GB = DevicePreset(
        id: "mac-16gb", displayName: "Mac, 16 GB", platform: .macOS,
        totalMemoryGB: 16, budgetGB: 16.0 * 2 / 3, isMeasured: false,
        source: "Estimate: default Metal working set ≈ 2/3 of RAM on Macs up to 36 GB "
            + "(recommendedMaxWorkingSetSize as reported in llama.cpp load logs).")

    public static let mac32GB = DevicePreset(
        id: "mac-32gb", displayName: "Mac, 32 GB (M1 Pro)", platform: .macOS,
        totalMemoryGB: 32, budgetGB: 20.0, isMeasured: true,
        source: "Measured on one M1 Pro 32 GB whose iogpu.wired_limit_mb is set to 20480 by a local LaunchDaemon; "
            + "Metal recommendedMaxWorkingSetSize follows it (20480 MiB, 2026-09-29). A stock 32 GB Mac's "
            + "default was not measured (≈2/3 of RAM by the 16 GB rule).",
        bandwidthGBs: 200)

    public static let mac64GB = DevicePreset(
        id: "mac-64gb", displayName: "Mac, 64 GB", platform: .macOS,
        totalMemoryGB: 64, budgetGB: 48, isMeasured: false,
        source: "Estimate: default Metal working set ≈ 3/4 of RAM on Macs above 36 GB "
            + "(recommendedMaxWorkingSetSize as reported in llama.cpp load logs).")

    /// Fixed presets, phones first.
    public static let classes: [DevicePreset] = [
        iPhone4GB, iPhone6GB, iPhone8GB, iPhone17Pro, mac16GB, mac32GB, mac64GB,
    ]

    /// This device followed by every fixed preset.
    public static func all(profile: HardwareProfile? = nil) -> [DevicePreset] {
        [thisDevice(profile: profile)] + classes
    }

    /// Look a preset up by ``id`` (`this-device`, `iphone-4gb`, `mac-32gb`, …).
    public static func named(_ id: String, profile: HardwareProfile? = nil) -> DevicePreset? {
        let key = id.lowercased()
        if key == thisDeviceID || key == "this" || key == "current" { return thisDevice(profile: profile) }
        return classes.first { $0.id == key }
    }
}
