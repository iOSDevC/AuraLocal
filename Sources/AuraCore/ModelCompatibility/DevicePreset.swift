import Foundation

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

    /// The running device, from ``HardwareProfile/current()`` (memory the process can take right now).
    public static func thisDevice(profile: HardwareProfile = .current()) -> DevicePreset {
        #if os(macOS)
        let platform = TargetOS.macOS
        #else
        let platform = TargetOS.iOS
        #endif
        return DevicePreset(
            id: thisDeviceID, displayName: "This device (\(profile.deviceName))", platform: platform,
            totalMemoryGB: profile.totalMemoryGB, budgetGB: profile.availableMemoryGB, isMeasured: true,
            source: "Measured now by HardwareProfile.current(): memory this process can still allocate.",
            bandwidthGBs: profile.memoryBandwidthGBs)
    }

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
        source: "Measured on an M1 Pro 32 GB: iogpu.wired_limit_mb = 20480 and Metal "
            + "recommendedMaxWorkingSetSize = 20480 MiB (2026-09-29).",
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
    public static func all(profile: HardwareProfile = .current()) -> [DevicePreset] {
        [thisDevice(profile: profile)] + classes
    }

    /// Look a preset up by ``id`` (`this-device`, `iphone-4gb`, `mac-32gb`, …).
    public static func named(_ id: String, profile: HardwareProfile = .current()) -> DevicePreset? {
        let key = id.lowercased()
        if key == thisDeviceID || key == "this" || key == "current" { return thisDevice(profile: profile) }
        return classes.first { $0.id == key }
    }
}
