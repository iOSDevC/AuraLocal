import Foundation

/// What a compiled Unet's `metadata.json` (~4 KB) says about the model, read without downloading weights.
public struct CoreMLUnetFacts: Sendable, Hashable {
    /// Input name → shape, e.g. `sample` → `[2, 4, 64, 64]`.
    public let inputShapes: [String: [Int]]
    /// `stateSchema` is non-empty (an iOS 18 `MLState` Unet).
    public let isStateful: Bool
    /// e.g. `Mixed (Float16, Palettized (6 bits))` or `Float16`.
    public let storagePrecision: String?
    /// Split-einsum attention, the Neural Engine variant.
    public let usesSplitEinsum: Bool
    /// The highest `IosNN.` op set in the program, e.g. 16.
    public let highestIOSOpSet: Int?
    /// `availability` majors by platform name (`iOS`, `macOS`, …).
    public let minimumOS: [String: Int]
    /// The source model, e.g. `runwayml/stable-diffusion-v1-5`.
    public let version: String?

    init(inputShapes: [String: [Int]], isStateful: Bool = false, storagePrecision: String? = nil,
         usesSplitEinsum: Bool = true, highestIOSOpSet: Int? = nil, minimumOS: [String: Int] = [:],
         version: String? = nil) {
        self.inputShapes = inputShapes
        self.isStateful = isStateful
        self.storagePrecision = storagePrecision
        self.usesSplitEinsum = usesSplitEinsum
        self.highestIOSOpSet = highestIOSOpSet
        self.minimumOS = minimumOS
        self.version = version
    }

    /// Reads the JSON array Core ML writes (or its first element alone). `nil` without an `inputSchema`.
    public static func parse(_ data: Data) -> CoreMLUnetFacts? {
        guard let root = try? JSONSerialization.jsonObject(with: data),
              let entry = (root as? [[String: Any]])?.first ?? (root as? [String: Any]),
              let inputs = entry["inputSchema"] as? [[String: Any]] else { return nil }
        let shapes = inputs.reduce(into: [String: [Int]]()) { result, input in
            guard let name = input["name"] as? String, let shape = Self.shape(input["shape"]) else { return }
            result[name] = shape
        }
        let operations = (entry["mlProgramOperationTypeHistogram"] as? [String: Any])?.keys.map { $0 } ?? []
        let availability = entry["availability"] as? [String: Any] ?? [:]
        return CoreMLUnetFacts(
            inputShapes: shapes,
            isStateful: !((entry["stateSchema"] as? [Any])?.isEmpty ?? true),
            storagePrecision: entry["storagePrecision"] as? String,
            usesSplitEinsum: operations.contains { $0.lowercased().contains("einsum") },
            highestIOSOpSet: operations.compactMap(Self.iOSOpSet).max(),
            minimumOS: availability.compactMapValues { ($0 as? String).flatMap(Self.major) },
            version: entry["version"] as? String)
    }

    /// Output width when the latent is square: 8 × the latent side.
    public var imageSide: Int? {
        guard let sample = inputShapes["sample"], sample.count == 4, sample[2] == sample[3] else { return nil }
        return sample[2] * 8
    }

    /// `encoder_hidden_states[1]`: 768 for SD 1.x, 1024 for SD 2.x, 2048 for SDXL.
    public var textEmbeddingWidth: Int? {
        guard let states = inputShapes["encoder_hidden_states"], states.count >= 2 else { return nil }
        return states[1]
    }

    // MARK: Parsing

    /// `"[2, 4, 64, 64]"` (what Core ML writes) or an integer array.
    static func shape(_ raw: Any?) -> [Int]? {
        if let numbers = raw as? [Int] { return numbers }
        guard let text = raw as? String else { return nil }
        let body = text.trimmingCharacters(in: CharacterSet(charactersIn: "[] "))
        if body.isEmpty { return [] }
        let parts = body.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard parts.allSatisfy({ $0 != nil }) else { return nil }
        return parts.compactMap { $0 }
    }

    /// `Ios16.einsum` → 16.
    static func iOSOpSet(_ operation: String) -> Int? {
        guard operation.hasPrefix("Ios"), let name = operation.split(separator: ".").first else { return nil }
        return Int(name.dropFirst(3))
    }

    /// `"16.0"` → 16.
    static func major(_ version: String) -> Int? {
        version.split(separator: ".").first.flatMap { Int($0) }
    }
}
