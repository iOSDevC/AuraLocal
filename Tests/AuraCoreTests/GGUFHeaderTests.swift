import XCTest
@testable import AuraCore

final class GGUFHeaderTests: XCTestCase {

    // MARK: - Scalars, strings, arrays

    func testParsesEveryScalarTypeAndStrings() throws {
        var file = GGUFBytes(tensorCount: 7)
        file.key("u8", .uint8) { $0.int(UInt8(200)) }
        file.key("i8", .int8) { $0.int(Int8(-5)) }
        file.key("u16", .uint16) { $0.int(UInt16(65_000)) }
        file.key("i16", .int16) { $0.int(Int16(-30_000)) }
        file.key("u32", .uint32) { $0.int(UInt32(4_000_000_000)) }
        file.key("i32", .int32) { $0.int(Int32(-2_000_000_000)) }
        file.key("f32", .float32) { $0.int(Float(1.5).bitPattern) }
        file.key("flag", .bool) { $0.int(UInt8(1)) }
        file.key("name", .string) { $0.string("Qwen ✓") }
        file.key("u64", .uint64) { $0.int(UInt64.max) }
        file.key("i64", .int64) { $0.int(Int64.min) }
        file.key("f64", .float64) { $0.int(Double(-0.25).bitPattern) }

        let header = try GGUFHeaderParser.parse(file.encoded)

        XCTAssertEqual(header.version, 3)
        XCTAssertEqual(header.tensorCount, 7)
        XCTAssertEqual(header.declaredKeyCount, 12)
        XCTAssertTrue(header.isComplete)
        XCTAssertEqual(header.metadata["u8"], .uint8(200))
        XCTAssertEqual(header.metadata["i8"], .int8(-5))
        XCTAssertEqual(header.metadata["u16"], .uint16(65_000))
        XCTAssertEqual(header.metadata["i16"], .int16(-30_000))
        XCTAssertEqual(header.metadata["u32"], .uint32(4_000_000_000))
        XCTAssertEqual(header.metadata["i32"], .int32(-2_000_000_000))
        XCTAssertEqual(header.metadata["f32"], .float32(1.5))
        XCTAssertEqual(header.metadata["flag"], .bool(true))
        XCTAssertEqual(header.metadata["name"]?.text, "Qwen ✓")
        XCTAssertEqual(header.metadata["u64"], .uint64(.max))
        XCTAssertNil(header.metadata["u64"]?.integer, "UInt64.max does not fit an Int")
        XCTAssertEqual(header.metadata["i64"], .int64(.min))
        XCTAssertEqual(header.metadata["f64"], .float64(-0.25))
    }

    func testParsesArraysKeepingOnlyTheFirstElementsOfLongOnes() throws {
        var file = GGUFBytes()
        file.key("short", .array) { $0.array(of: .int32, [1, 2, 3].map(GGUFBytes.int32Value)) }
        file.key("words", .array) { $0.array(of: .string, ["a", "bc"].map(GGUFBytes.stringValue)) }
        file.key("long", .array) { $0.array(of: .uint32, (0..<1000).map(GGUFBytes.uint32Value)) }
        file.key("vocab", .array) { $0.array(of: .string, (0..<500).map { GGUFBytes.stringValue("tok\($0)") }) }
        file.key("nested", .array) { writer in
            writer.array(of: .array, [
                { $0.array(of: .uint8, [GGUFBytes.uint8Value(7)]) },
                { $0.array(of: .uint8, []) },
            ])
        }
        file.key("after", .string) { $0.string("still read") }

        let header = try GGUFHeaderParser.parse(file.encoded)

        XCTAssertTrue(header.isComplete)
        guard case .array(let short)? = header.metadata["short"] else { return XCTFail("short is not an array") }
        XCTAssertEqual(short.elementType, .int32)
        XCTAssertEqual(short.elements, [.int32(1), .int32(2), .int32(3)])
        guard case .array(let words)? = header.metadata["words"] else { return XCTFail("words is not an array") }
        XCTAssertEqual(words.elements.compactMap(\.text), ["a", "bc"])
        guard case .array(let long)? = header.metadata["long"] else { return XCTFail("long is not an array") }
        XCTAssertEqual(long.count, 1000)
        XCTAssertEqual(long.elements.count, GGUFHeaderParser.keptArrayElements)
        guard case .array(let vocab)? = header.metadata["vocab"] else { return XCTFail("vocab is not an array") }
        XCTAssertEqual(vocab.count, 500)
        XCTAssertEqual(vocab.elements.last?.text, "tok\(GGUFHeaderParser.keptArrayElements - 1)")
        guard case .array(let nested)? = header.metadata["nested"] else { return XCTFail("nested is not an array") }
        XCTAssertEqual(nested.count, 2)
        XCTAssertEqual(header.metadata["after"]?.text, "still read")
    }

    // MARK: - Architecture accessors

    func testArchitectureAccessorsReadPrefixedKeys() throws {
        let header = try GGUFHeaderParser.parse(GGUFBytes.qwen35(nextn: 1).encoded)
        XCTAssertEqual(header.architecture, "qwen35")
        XCTAssertEqual(header.modelName, "Swift 1.5")
        XCTAssertEqual(header.blockCount, 65)
        XCTAssertEqual(header.contextLength, 262_144)
        XCTAssertEqual(header.embeddingLength, 5120)
        XCTAssertEqual(header.headCount, 24)
        XCTAssertEqual(header.headCountKV, 4)
        XCTAssertEqual(header.keyLength, 256)
        XCTAssertEqual(header.headDimension, 256)
        XCTAssertEqual(header.nextnPredictLayers, 1)
    }

    func testPerLayerHeadCountArrayUsesItsLargestValue() throws {
        var file = GGUFBytes()
        file.key("general.architecture", .string) { $0.string("jamba") }
        file.key("jamba.attention.head_count_kv", .array) { $0.array(of: .int32, [0, 8, 0].map(GGUFBytes.int32Value)) }
        file.key("jamba.embedding_length", .uint32) { $0.int(UInt32(4096)) }
        file.key("jamba.attention.head_count", .uint32) { $0.int(UInt32(32)) }
        let header = try GGUFHeaderParser.parse(file.encoded)
        XCTAssertEqual(header.headCountKV, 8)
        XCTAssertEqual(header.headDimension, 128, "no key_length → embedding_length / head_count")
    }

    // MARK: - Truncation and bad input

    func testTruncatedHeaderIsIncompleteInsteadOfThrowing() throws {
        let full = GGUFBytes.qwen35(nextn: 1, vocabulary: 2000).encoded
        let cut = full.prefix(full.count - 3000)

        let header = try GGUFHeaderParser.parse(Data(cut))

        XCTAssertFalse(header.isComplete)
        XCTAssertLessThan(header.metadata.count, Int(header.declaredKeyCount))
        XCTAssertEqual(header.architecture, "qwen35", "keys before the cut survive")
        XCTAssertEqual(header.nextnPredictLayers, 1)
        XCTAssertNil(header.metadata["tokenizer.ggml.tokens"])
    }

    func testCutInsideAKeyNameIsIncomplete() throws {
        let full = GGUFBytes.qwen35(nextn: 0).encoded
        for length in [24, 30, 60, full.count - 1] {
            let header = try GGUFHeaderParser.parse(full.prefix(length))
            XCTAssertFalse(header.isComplete, "cut at \(length)")
        }
    }

    func testRejectsNonGGUFOldVersionsAndStubs() {
        XCTAssertThrowsError(try GGUFHeaderParser.parse(Data("GGML0000".utf8))) {
            XCTAssertEqual($0 as? GGUFHeaderParser.ParseError, .notGGUF)
        }
        XCTAssertThrowsError(try GGUFHeaderParser.parse(Data("GG".utf8))) {
            XCTAssertEqual($0 as? GGUFHeaderParser.ParseError, .tooShort)
        }
        XCTAssertThrowsError(try GGUFHeaderParser.parse(GGUFBytes(version: 1).encoded)) {
            XCTAssertEqual($0 as? GGUFHeaderParser.ParseError, .unsupportedVersion(1))
        }
        XCTAssertThrowsError(try GGUFHeaderParser.parse(GGUFBytes().encoded.prefix(20))) {
            XCTAssertEqual($0 as? GGUFHeaderParser.ParseError, .tooShort)
        }
    }

    func testUnknownValueTypeThrows() {
        var file = GGUFBytes()
        file.rawKey("weird", type: 99) { $0.int(UInt32(1)) }
        XCTAssertThrowsError(try GGUFHeaderParser.parse(file.encoded)) {
            XCTAssertEqual($0 as? GGUFHeaderParser.ParseError, .invalidValueType(99))
        }
    }

    // MARK: - Safetensors header

    func testSafetensorsHeaderListsTensorsAndMetadata() throws {
        let data = SafetensorsBytes.make(["model.layers.0.weight", "lm_head.weight"], metadata: ["format": "mlx"])
        XCTAssertEqual(SafetensorsHeader.requiredByteCount(prefix: data), data.count)
        let header = try XCTUnwrap(SafetensorsHeader.parse(data))
        XCTAssertEqual(header.tensorNames, ["lm_head.weight", "model.layers.0.weight"])
        XCTAssertEqual(header.metadata, ["format": "mlx"])
        XCTAssertNil(SafetensorsHeader.parse(data.prefix(data.count - 1)), "a partial header is not parsed")
        XCTAssertNil(SafetensorsHeader.requiredByteCount(prefix: Data([1, 2])))
    }
}

// MARK: - Byte builders

/// Writes a little-endian GGUF header. Values are closures so arrays can nest.
struct GGUFBytes {
    typealias Value = (inout GGUFBytes) -> Void

    private(set) var body = Data()
    private var keyCount: UInt64 = 0
    private let version: UInt32
    private let tensorCount: UInt64

    init(version: UInt32 = 3, tensorCount: UInt64 = 0) {
        self.version = version
        self.tensorCount = tensorCount
    }

    var encoded: Data {
        var out = Data("GGUF".utf8)
        out.append(Self.littleEndian(version))
        out.append(Self.littleEndian(tensorCount))
        out.append(Self.littleEndian(keyCount))
        return out + body
    }

    mutating func key(_ name: String, _ type: GGUFValueType, _ value: Value) {
        rawKey(name, type: type.rawValue, value)
    }

    mutating func rawKey(_ name: String, type: UInt32, _ value: Value) {
        string(name)
        int(type)
        value(&self)
        keyCount += 1
    }

    mutating func int<T: FixedWidthInteger>(_ value: T) {
        body.append(Self.littleEndian(value))
    }

    mutating func string(_ text: String) {
        int(UInt64(text.utf8.count))
        body.append(contentsOf: Array(text.utf8))
    }

    mutating func array(of type: GGUFValueType, _ values: [Value]) {
        int(type.rawValue)
        int(UInt64(values.count))
        values.forEach { $0(&self) }
    }

    static func int32Value(_ value: Int) -> Value { { $0.int(Int32(value)) } }
    static func uint32Value(_ value: Int) -> Value { { $0.int(UInt32(value)) } }
    static func uint8Value(_ value: Int) -> Value { { $0.int(UInt8(value)) } }
    static func stringValue(_ text: String) -> Value { { $0.string(text) } }

    static func littleEndian<T: FixedWidthInteger>(_ value: T) -> Data {
        withUnsafeBytes(of: value.littleEndian) { Data($0) }
    }

    /// A header shaped like the Swift-1.5 27B GGUF: qwen35 keys first, then a tokenizer vocabulary.
    static func qwen35(nextn: UInt32, vocabulary: Int = 0) -> GGUFBytes {
        var file = GGUFBytes(tensorCount: 866)
        file.key("general.architecture", .string) { $0.string("qwen35") }
        file.key("general.name", .string) { $0.string("Swift 1.5") }
        file.key("qwen35.block_count", .uint32) { $0.int(UInt32(65)) }
        file.key("qwen35.context_length", .uint32) { $0.int(UInt32(262_144)) }
        file.key("qwen35.embedding_length", .uint32) { $0.int(UInt32(5120)) }
        file.key("qwen35.attention.head_count", .uint32) { $0.int(UInt32(24)) }
        file.key("qwen35.attention.head_count_kv", .uint32) { $0.int(UInt32(4)) }
        file.key("qwen35.attention.key_length", .uint32) { $0.int(UInt32(256)) }
        file.key("qwen35.nextn_predict_layers", .uint32) { $0.int(nextn) }
        file.key("tokenizer.ggml.tokens", .array) {
            $0.array(of: .string, (0..<vocabulary).map { stringValue("token-\($0)") })
        }
        return file
    }
}

enum SafetensorsBytes {
    static func make(_ names: [String], metadata: [String: String] = [:]) -> Data {
        var object: [String: Any] = Dictionary(uniqueKeysWithValues: names.map { name in
            (name, ["dtype": "F16", "shape": [1], "data_offsets": [0, 2]] as [String: Any])
        })
        if !metadata.isEmpty { object["__metadata__"] = metadata }
        let json = (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
        return GGUFBytes.littleEndian(UInt64(json.count)) + json
    }
}
