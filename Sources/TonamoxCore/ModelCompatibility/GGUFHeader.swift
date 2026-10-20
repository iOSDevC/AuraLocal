import Foundation

// MARK: - GGUFValueType

/// GGUF metadata value types, numbered as in the GGUF v3 specification.
public enum GGUFValueType: UInt32, Sendable, CaseIterable {
    case uint8 = 0, int8, uint16, int16, uint32, int32, float32, bool, string, array, uint64, int64, float64

    /// Byte width of a fixed-size type; `nil` for strings and arrays.
    var byteWidth: Int? {
        switch self {
        case .uint8, .int8, .bool: 1
        case .uint16, .int16: 2
        case .uint32, .int32, .float32: 4
        case .uint64, .int64, .float64: 8
        case .string, .array: nil
        }
    }
}

// MARK: - GGUFValue

/// One GGUF metadata value.
public enum GGUFValue: Sendable, Equatable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case float32(Float)
    case bool(Bool)
    case string(String)
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)
    case array(GGUFArray)

    /// The value as an `Int`. For an integer array (per-layer head counts) this is its largest element.
    public var integer: Int? {
        switch self {
        case .uint8(let raw): Int(raw)
        case .int8(let raw): Int(raw)
        case .uint16(let raw): Int(raw)
        case .int16(let raw): Int(raw)
        case .uint32(let raw): Int(raw)
        case .int32(let raw): Int(raw)
        case .uint64(let raw): Int(exactly: raw)
        case .int64(let raw): Int(raw)
        case .bool(let raw): raw ? 1 : 0
        case .array(let list): list.elements.compactMap(\.integer).max()
        case .float32, .float64, .string: nil
        }
    }

    /// The value as text, when it is a string.
    public var text: String? {
        if case .string(let raw) = self { return raw }
        return nil
    }
}

/// A GGUF array. Long arrays (tokenizer vocabularies) keep only their first elements, but always their count.
public struct GGUFArray: Sendable, Equatable {
    public let elementType: GGUFValueType
    public let count: UInt64
    /// Up to the first ``GGUFHeaderParser/keptArrayElements`` elements, while the header-wide
    /// ``GGUFHeaderParser/maxKeptArrayValues`` lasts.
    public let elements: [GGUFValue]

    public init(elementType: GGUFValueType, count: UInt64, elements: [GGUFValue]) {
        self.elementType = elementType
        self.count = count
        self.elements = elements
    }
}

// MARK: - GGUFHeader

/// The metadata section of a GGUF file, read from its first bytes (an HTTP Range read is enough).
public struct GGUFHeader: Sendable, Equatable {
    public let version: UInt32
    public let tensorCount: UInt64
    /// Number of key/value pairs the file declares.
    public let declaredKeyCount: UInt64
    public let metadata: [String: GGUFValue]
    /// `false` when the bytes ended before every declared key/value was read.
    public let isComplete: Bool

    public init(version: UInt32, tensorCount: UInt64, declaredKeyCount: UInt64,
                metadata: [String: GGUFValue], isComplete: Bool) {
        self.version = version
        self.tensorCount = tensorCount
        self.declaredKeyCount = declaredKeyCount
        self.metadata = metadata
        self.isComplete = isComplete
    }

    /// `general.architecture` — the llama.cpp architecture name (e.g. `qwen35`).
    public var architecture: String? { metadata["general.architecture"]?.text }
    /// `general.name`.
    public var modelName: String? { metadata["general.name"]?.text }
    public var blockCount: Int? { architectureInteger("block_count") }
    public var contextLength: Int? { architectureInteger("context_length") }
    public var embeddingLength: Int? { architectureInteger("embedding_length") }
    public var headCount: Int? { architectureInteger("attention.head_count") }
    public var headCountKV: Int? { architectureInteger("attention.head_count_kv") }
    public var keyLength: Int? { architectureInteger("attention.key_length") }
    /// Multi-token-prediction (NextN) layers appended after the regular blocks.
    public var nextnPredictLayers: Int? { architectureInteger("nextn_predict_layers") }

    /// Per-head dimension: `attention.key_length`, else `embedding_length / head_count`.
    public var headDimension: Int? {
        if let keyLength { return keyLength }
        guard let embeddingLength, let headCount, headCount > 0 else { return nil }
        return embeddingLength / headCount
    }

    /// `<architecture>.<suffix>` as an integer.
    public func architectureInteger(_ suffix: String) -> Int? {
        guard let architecture else { return nil }
        return metadata["\(architecture).\(suffix)"]?.integer
    }
}

// MARK: - GGUFHeaderParser

/// Pure-Swift parser for GGUF v2/v3 metadata (little-endian). Stops cleanly at the end of the supplied bytes
/// and reports ``GGUFHeader/isComplete`` `false` instead of failing, so a partial download is enough.
public enum GGUFHeaderParser {

    public enum ParseError: LocalizedError, Equatable {
        case notGGUF
        case tooShort
        case unsupportedVersion(UInt32)
        case invalidValueType(UInt32)
        case nestingTooDeep

        public var errorDescription: String? {
            switch self {
            case .notGGUF: "Not a GGUF file (missing the GGUF magic)."
            case .tooShort: "Too few bytes to hold a GGUF header."
            case .unsupportedVersion(let version): "Unsupported GGUF version \(version)."
            case .invalidValueType(let raw): "Invalid GGUF value type \(raw)."
            case .nestingTooDeep: "GGUF arrays nested too deeply."
            }
        }
    }

    /// Elements kept per array; the rest are skipped.
    public static let keptArrayElements = 64
    /// Array elements kept across the whole header, so nested arrays in a crafted file cannot multiply memory.
    public static let maxKeptArrayValues = 4096
    private static let maxNesting = 3
    private static let magic = Array("GGUF".utf8)

    public static func parse(_ data: Data) throws -> GGUFHeader {
        guard data.count >= magic.count else { throw ParseError.tooShort }
        guard data.prefix(magic.count).elementsEqual(magic) else { throw ParseError.notGGUF }

        var reader = ByteReader(data)
        let version: UInt32
        let tensorCount: UInt64
        let keyCount: UInt64
        do {
            try reader.skip(UInt64(magic.count))
            version = try reader.integer(UInt32.self)
            // v1 used 32-bit counts; a byte-swapped version means a big-endian file.
            guard version == 2 || version == 3 else { throw ParseError.unsupportedVersion(version) }
            tensorCount = try reader.integer(UInt64.self)
            keyCount = try reader.integer(UInt64.self)
        } catch is Truncated {
            throw ParseError.tooShort
        }

        var metadata: [String: GGUFValue] = [:]
        var complete = true
        var budget = maxKeptArrayValues
        var index: UInt64 = 0
        while index < keyCount {
            do {
                let entry = try readEntry(from: &reader, budget: &budget)
                metadata[entry.key] = entry.value
            } catch is Truncated {
                complete = false
                break
            }
            index += 1
        }
        return GGUFHeader(version: version, tensorCount: tensorCount, declaredKeyCount: keyCount,
                          metadata: metadata, isComplete: complete)
    }

    private static func readEntry(from reader: inout ByteReader,
                                  budget: inout Int) throws -> (key: String, value: GGUFValue) {
        let key = try reader.string()
        let type = try valueType(reader.integer(UInt32.self))
        return (key, try readValue(of: type, from: &reader, depth: 0, budget: &budget))
    }

    private static func valueType(_ raw: UInt32) throws -> GGUFValueType {
        guard let type = GGUFValueType(rawValue: raw) else { throw ParseError.invalidValueType(raw) }
        return type
    }

    private static func readValue(of type: GGUFValueType, from reader: inout ByteReader, depth: Int,
                                  budget: inout Int) throws -> GGUFValue {
        switch type {
        case .uint8: return try .uint8(reader.integer(UInt8.self))
        case .int8: return try .int8(reader.integer(Int8.self))
        case .uint16: return try .uint16(reader.integer(UInt16.self))
        case .int16: return try .int16(reader.integer(Int16.self))
        case .uint32: return try .uint32(reader.integer(UInt32.self))
        case .int32: return try .int32(reader.integer(Int32.self))
        case .float32: return try .float32(Float(bitPattern: reader.integer(UInt32.self)))
        case .bool: return try .bool(reader.integer(UInt8.self) != 0)
        case .string: return try .string(reader.string())
        case .uint64: return try .uint64(reader.integer(UInt64.self))
        case .int64: return try .int64(reader.integer(Int64.self))
        case .float64: return try .float64(Double(bitPattern: reader.integer(UInt64.self)))
        case .array: return try .array(readArray(from: &reader, depth: depth + 1, budget: &budget))
        }
    }

    private static func readArray(from reader: inout ByteReader, depth: Int, budget: inout Int) throws -> GGUFArray {
        guard depth <= maxNesting else { throw ParseError.nestingTooDeep }
        let elementType = try valueType(reader.integer(UInt32.self))
        let count = try reader.integer(UInt64.self)
        let keep = min(count, UInt64(keptArrayElements), UInt64(max(budget, 0)))
        budget -= Int(keep)
        var kept: [GGUFValue] = []
        kept.reserveCapacity(Int(keep))
        var index: UInt64 = 0
        while index < keep {
            kept.append(try readValue(of: elementType, from: &reader, depth: depth, budget: &budget))
            index += 1
        }
        try skipValues(of: elementType, count: count - keep, in: &reader, depth: depth)
        return GGUFArray(elementType: elementType, count: count, elements: kept)
    }

    private static func skipValues(of type: GGUFValueType, count: UInt64,
                                   in reader: inout ByteReader, depth: Int) throws {
        if let width = type.byteWidth {
            let (total, overflow) = count.multipliedReportingOverflow(by: UInt64(width))
            guard !overflow else { throw Truncated() }
            try reader.skip(total)
            return
        }
        var index: UInt64 = 0
        while index < count {
            try skipValue(of: type, in: &reader, depth: depth)
            index += 1
        }
    }

    private static func skipValue(of type: GGUFValueType, in reader: inout ByteReader, depth: Int) throws {
        switch type {
        case .string: try reader.skip(reader.integer(UInt64.self))
        case .array:
            var none = 0
            _ = try readArray(from: &reader, depth: depth + 1, budget: &none)
        default: try reader.skip(UInt64(type.byteWidth ?? 0))
        }
    }
}

// MARK: - Byte reader

/// Thrown when the buffer ends mid-value; the parser turns it into an incomplete header.
private struct Truncated: Error {}

private struct ByteReader {
    private let bytes: [UInt8]
    private var offset = 0

    init(_ data: Data) {
        bytes = [UInt8](data)
    }

    private var remaining: Int { bytes.count - offset }

    mutating func integer<T: FixedWidthInteger>(_: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        guard remaining >= size else { throw Truncated() }
        let start = offset
        let raw = bytes.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: start, as: T.self) }
        offset += size
        return T(littleEndian: raw)
    }

    mutating func string() throws -> String {
        let length = try integer(UInt64.self)
        guard length <= UInt64(remaining) else { throw Truncated() }
        let end = offset + Int(length)
        let decoded = String(decoding: bytes[offset..<end], as: UTF8.self)
        offset = end
        return decoded
    }

    mutating func skip(_ count: UInt64) throws {
        guard count <= UInt64(remaining) else { throw Truncated() }
        offset += Int(count)
    }
}
