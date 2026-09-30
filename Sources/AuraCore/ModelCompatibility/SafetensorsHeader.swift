import Foundation

/// The JSON header at the start of a `.safetensors` file: an 8-byte little-endian length, then JSON mapping
/// tensor names to dtype/shape/offsets plus an optional `__metadata__` string map. Reading it tells which tensors a
/// single-file repo ships without downloading the weights.
public struct SafetensorsHeader: Sendable, Equatable {
    public let tensorNames: [String]
    /// `__metadata__` (e.g. `format: mlx`).
    public let metadata: [String: String]

    public init(tensorNames: [String], metadata: [String: String]) {
        self.tensorNames = tensorNames
        self.metadata = metadata
    }

    /// Header JSON larger than this is treated as corrupt.
    public static let maxHeaderBytes = 32 << 20

    /// Bytes needed to hold the whole header (the 8-byte length plus the JSON), from the file's first 8 bytes.
    public static func requiredByteCount(prefix: Data) -> Int? {
        guard prefix.count >= 8 else { return nil }
        let length = prefix.prefix(8).reversed().reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
        guard length > 0, length <= UInt64(maxHeaderBytes) else { return nil }
        return 8 + Int(length)
    }

    /// Parse the header from the file's leading bytes; `nil` when they are too few or not a safetensors header.
    public static func parse(_ data: Data) -> SafetensorsHeader? {
        guard let needed = requiredByteCount(prefix: data), data.count >= needed else { return nil }
        let json = data.subdata(in: data.startIndex + 8 ..< data.startIndex + needed)
        guard let object = try? JSONSerialization.jsonObject(with: json) as? [String: Any] else { return nil }
        let names = object.keys.filter { $0 != "__metadata__" }.sorted()
        let meta = (object["__metadata__"] as? [String: Any] ?? [:]).compactMapValues { $0 as? String }
        return SafetensorsHeader(tensorNames: names, metadata: meta)
    }
}
