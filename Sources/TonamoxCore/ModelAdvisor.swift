import Foundation

/// How well a GGUF file's size fits the machine's RAM. Weights are mmapped, but inference still needs headroom
/// for the KV cache + compute buffers + the OS, so the working-set estimate is ~1.3× the file size.
public enum ModelFit: String, Sendable, Equatable {
    case comfortable   // runs with room to spare
    case tight         // will run but leaves little headroom
    case tooLarge      // likely to thrash / fail on this machine
    case unknown       // size not reported

    /// A short human hint for the picker.
    public var hint: String {
        switch self {
        case .comfortable: "Fits comfortably"
        case .tight: "Tight fit"
        case .tooLarge: "Too large for this Mac"
        case .unknown: ""
        }
    }
}

/// Helps the user pick which quantization to download: estimates each file's fit on this machine and marks a
/// recommended one (the highest-quality quant that still fits comfortably). Pure + testable; the host reads
/// `HardwareProfile.current().totalMemoryGB` for `ramGB`.
public enum ModelAdvisor {

    private static let bytesPerGB = 1_073_741_824.0
    /// Working-set multiplier over the file size (KV cache + compute + OS headroom).
    private static let workingSetFactor = 1.3

    public static func fit(sizeBytes: Int?, ramGB: Double) -> ModelFit {
        guard let sizeBytes, sizeBytes > 0, ramGB > 0 else { return .unknown }
        let needGB = (Double(sizeBytes) / bytesPerGB) * workingSetFactor
        if needGB <= ramGB * 0.6 { return .comfortable }
        if needGB <= ramGB * 0.85 { return .tight }
        return .tooLarge
    }

    /// The recommended file from a size-ascending list: the **largest** quant that still fits comfortably (best
    /// quality without thrashing); else the smallest tight one; else the smallest overall. `nil` for an empty list.
    public static func recommended(from files: [RemoteGGUFFile], ramGB: Double) -> RemoteGGUFFile? {
        guard !files.isEmpty else { return nil }
        let rated = files.map { ($0, fit(sizeBytes: $0.sizeBytes, ramGB: ramGB)) }
        if let bestComfortable = rated.last(where: { $0.1 == .comfortable })?.0 { return bestComfortable }
        if let smallestTight = rated.first(where: { $0.1 == .tight })?.0 { return smallestTight }
        return files.first
    }
}
