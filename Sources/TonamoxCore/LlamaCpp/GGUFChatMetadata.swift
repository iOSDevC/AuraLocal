import Foundation

// MARK: - GGUFChatMetadata

/// The chat template and special-token texts a GGUF carries, resolved the way llama.cpp resolves them, so the
/// GGUF backends can render a conversation with real roles themselves.
struct GGUFChatMetadata: Sendable, Equatable {
    /// `tokenizer.chat_template`; `nil` when the file has none or it is empty.
    let template: String?
    /// The variables LocalLLMClient hands its template renderer (`bos_token`, `eos_token`, …), with the same values.
    let specialTokens: [String: String]
    /// Texts of the tokens llama.cpp's tokenizer splits on before anything else (control, user-defined and unknown
    /// types), in vocabulary order; empty when the file has no token types.
    var controlTokenTexts: [String] = []

    enum ReadError: LocalizedError, Equatable {
        case notGGUF
        case truncated
        case unsupportedVersion(UInt32)
        case invalidValueType(UInt32)

        var errorDescription: String? {
            switch self {
            case .notGGUF: "Not a GGUF file (missing the GGUF magic)."
            case .truncated: "The GGUF metadata ends early."
            case .unsupportedVersion(let version): "Unsupported GGUF version \(version)."
            case .invalidValueType(let raw): "Invalid GGUF value type \(raw)."
            }
        }
    }

    /// Maps the file rather than reading it: only the metadata pages are touched.
    static func read(from url: URL) throws -> GGUFChatMetadata {
        try parse(Data(contentsOf: url, options: .alwaysMapped))
    }

    static func parse(_ data: Data) throws -> GGUFChatMetadata {
        try data.withUnsafeBytes { raw in
            var cursor = MetadataCursor(raw)
            return try readMetadata(&cursor)
        }
    }

    // MARK: Special tokens

    /// llama.cpp's per-tokenizer defaults (`llama_vocab::impl::load`), before the GGUF overrides them.
    struct TokenIDs: Equatable {
        var bos: Int
        var eos: Int
        var sep: Int
        var pad: Int

        static func defaults(tokenizerModel: String?, preTokenizer: String?) -> TokenIDs {
            var ids: TokenIDs = switch tokenizerModel {
            case "llama": TokenIDs(bos: 1, eos: 2, sep: -1, pad: -1)
            case "bert": TokenIDs(bos: 101, eos: -1, sep: 102, pad: 0)
            case "gpt2": TokenIDs(bos: 11, eos: 11, sep: -1, pad: -1)
            case "t5": TokenIDs(bos: -1, eos: 1, sep: -1, pad: 0)
            case "plamo2": TokenIDs(bos: 1, eos: 2, sep: -1, pad: 3)
            default: TokenIDs(bos: -1, eos: -1, sep: -1, pad: -1)
            }
            if tokenizerModel == "gpt2", preTokenizer == "glm4" || preTokenizer == "chatglm-bpe" {
                ids.bos = -1
            }
            return ids
        }
    }

    /// Mirrors LocalLLMClient's `StandardLlamaDecoder.extractSpecialTokens`: an unset id reads token 0, and
    /// `unk_token` is always token 0.
    static func specialTokens(ids: TokenIDs, text: (Int) -> String) -> [String: String] {
        [
            "bos_token": text(max(0, ids.bos)),
            "eos_token": text(max(0, ids.eos)),
            "unk_token": text(0),
            "sep_token": text(max(0, ids.sep)),
            "pad_token": text(max(0, ids.pad)),
            "cls_token": text(max(0, ids.bos)),
            "mask_token": "",
        ]
    }

    // MARK: Parsing

    private static func readMetadata(_ cursor: inout MetadataCursor) throws -> GGUFChatMetadata {
        guard cursor.remaining >= 4, cursor.bytes(4).elementsEqual("GGUF".utf8) else { throw ReadError.notGGUF }
        let version = try cursor.integer(UInt32.self)
        guard version == 2 || version == 3 else { throw ReadError.unsupportedVersion(version) }
        _ = try cursor.integer(UInt64.self)   // tensor count
        let keyCount = try cursor.integer(UInt64.self)

        var strings: [String: String] = [:]
        var integers: [String: Int] = [:]
        var tokenList: (offset: Int, count: Int)?
        var tokenTypes: (offset: Int, count: Int, type: GGUFValueType)?
        var index: UInt64 = 0
        while index < keyCount {
            let key = try cursor.string()
            let type = try valueType(cursor.integer(UInt32.self))
            switch (key, type) {
            case ("tokenizer.ggml.tokens", .array):
                tokenList = try cursor.skipStringArray()
            case ("tokenizer.ggml.token_type", .array):
                tokenTypes = try cursor.skipIntegerArray()
            case (_, .string):
                strings[key] = try cursor.string()
            case (_, .array):
                try cursor.skipValue(of: .array)
            default:
                integers[key] = try cursor.integer(of: type)
            }
            index += 1
        }

        var ids = TokenIDs.defaults(tokenizerModel: strings["tokenizer.ggml.model"],
                                    preTokenizer: strings["tokenizer.ggml.pre"])
        // llama.cpp keeps the default when the file names an id outside the vocabulary.
        let vocabularySize = tokenList?.count ?? 0
        func override(_ key: String) -> Int? {
            integers["tokenizer.ggml.\(key)_token_id"].flatMap { (0..<vocabularySize).contains($0) ? $0 : nil }
        }
        ids.bos = override("bos") ?? ids.bos
        ids.eos = override("eos") ?? ids.eos
        ids.sep = override("seperator") ?? ids.sep   // llama.cpp's spelling of the key
        ids.pad = override("padding") ?? ids.pad

        let texts = try tokenTexts(at: [0, ids.bos, ids.eos, ids.sep, ids.pad], in: tokenList, cursor: cursor)
        let template = strings["tokenizer.chat_template"].flatMap { $0.isEmpty ? nil : $0 }
        return GGUFChatMetadata(template: template, specialTokens: specialTokens(ids: ids) { texts[$0] ?? "" },
                                controlTokenTexts: try controlTexts(in: tokenList, types: tokenTypes, cursor: cursor))
    }

    /// llama.cpp's `LLAMA_TOKEN_TYPE_UNKNOWN`, `_CONTROL` and `_USER_DEFINED`: the types its tokenizer matches as
    /// whole tokens before splitting the rest of the text.
    static let controlTypes: Set<Int> = [2, 3, 4]

    private static func controlTexts(in list: (offset: Int, count: Int)?,
                                     types: (offset: Int, count: Int, type: GGUFValueType)?,
                                     cursor: MetadataCursor) throws -> [String] {
        guard let list, let types, types.count == list.count, let width = types.type.byteWidth else { return [] }
        var reader = cursor
        reader.offset = list.offset
        var kinds = cursor
        var texts: [String] = []
        for id in 0..<list.count {
            kinds.offset = types.offset + id * width
            if controlTypes.contains(try kinds.integer(of: types.type)) {
                let text = try reader.string()
                if !text.isEmpty { texts.append(text) }
            } else {
                try reader.skipValue(of: .string)
            }
        }
        return texts
    }

    /// Walks the vocabulary only as far as the highest id needed.
    private static func tokenTexts(at wanted: [Int], in list: (offset: Int, count: Int)?,
                                   cursor: MetadataCursor) throws -> [Int: String] {
        guard let list else { return [:] }
        let targets = Set(wanted.map { max(0, $0) }.filter { $0 < list.count })
        guard let last = targets.max() else { return [:] }
        var reader = cursor
        reader.offset = list.offset
        var texts: [Int: String] = [:]
        for id in 0...last {
            if targets.contains(id) {
                texts[id] = try reader.string()
            } else {
                try reader.skipValue(of: .string)
            }
        }
        return texts
    }

    fileprivate static func valueType(_ raw: UInt32) throws -> GGUFValueType {
        guard let type = GGUFValueType(rawValue: raw) else { throw ReadError.invalidValueType(raw) }
        return type
    }
}

// MARK: - MetadataCursor

/// Reads in place over the mapped file. `GGUFHeaderParser` copies its input and keeps only the first 64 array
/// elements, which misses special tokens deep in a 150k vocabulary.
private struct MetadataCursor {
    static let integerTypes: Set<GGUFValueType> = [.uint8, .int8, .uint16, .int16, .uint32, .int32, .uint64, .int64]

    let buffer: UnsafeRawBufferPointer
    var offset = 0

    init(_ buffer: UnsafeRawBufferPointer) {
        self.buffer = buffer
    }

    var remaining: Int { buffer.count - offset }

    mutating func bytes(_ count: Int) -> UnsafeRawBufferPointer.SubSequence {
        defer { offset += count }
        return buffer[offset..<(offset + count)]
    }

    mutating func integer<T: FixedWidthInteger>(_: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        guard remaining >= size else { throw GGUFChatMetadata.ReadError.truncated }
        let value = buffer.loadUnaligned(fromByteOffset: offset, as: T.self)
        offset += size
        return T(littleEndian: value)
    }

    /// Reads a scalar as `Int`; a float reads as 0, since no token id is a float.
    mutating func integer(of type: GGUFValueType) throws -> Int {
        switch type {
        case .uint8: return Int(try integer(UInt8.self))
        case .int8: return Int(try integer(Int8.self))
        case .uint16: return Int(try integer(UInt16.self))
        case .int16: return Int(try integer(Int16.self))
        case .uint32: return Int(try integer(UInt32.self))
        case .int32: return Int(try integer(Int32.self))
        case .uint64: return Int(truncatingIfNeeded: try integer(UInt64.self))
        case .int64: return Int(try integer(Int64.self))
        case .bool: return Int(try integer(UInt8.self))
        case .float32, .float64, .string, .array:
            try skipValue(of: type)
            return 0
        }
    }

    mutating func string() throws -> String {
        let length = try integer(UInt64.self)
        guard length <= UInt64(remaining) else { throw GGUFChatMetadata.ReadError.truncated }
        return String(decoding: bytes(Int(length)), as: UTF8.self)
    }

    mutating func skipValue(of type: GGUFValueType, depth: Int = 0) throws {
        switch type {
        case .string:
            try skip(integer(UInt64.self))
        case .array:
            guard depth < 3 else { throw GGUFChatMetadata.ReadError.truncated }
            let elementType = try GGUFChatMetadata.valueType(integer(UInt32.self))
            try skipValues(of: elementType, count: integer(UInt64.self), depth: depth + 1)
        default:
            try skip(UInt64(type.byteWidth ?? 0))
        }
    }

    /// Skips an array, returning where its elements start and how many there are when they are strings.
    mutating func skipStringArray() throws -> (offset: Int, count: Int)? {
        let elementType = try GGUFChatMetadata.valueType(integer(UInt32.self))
        let count = try integer(UInt64.self)
        let start = offset
        try skipValues(of: elementType, count: count)
        guard elementType == .string, let elements = Int(exactly: count) else { return nil }
        return (start, elements)
    }

    /// Skips an array, returning where its elements start, how many there are and their type when they are integers.
    mutating func skipIntegerArray() throws -> (offset: Int, count: Int, type: GGUFValueType)? {
        let elementType = try GGUFChatMetadata.valueType(integer(UInt32.self))
        let count = try integer(UInt64.self)
        let start = offset
        try skipValues(of: elementType, count: count)
        guard Self.integerTypes.contains(elementType), let elements = Int(exactly: count) else { return nil }
        return (start, elements, elementType)
    }

    mutating func skipValues(of type: GGUFValueType, count: UInt64, depth: Int = 0) throws {
        if let width = type.byteWidth {
            let (total, overflow) = count.multipliedReportingOverflow(by: UInt64(width))
            guard !overflow else { throw GGUFChatMetadata.ReadError.truncated }
            try skip(total)
            return
        }
        var index: UInt64 = 0
        while index < count {
            try skipValue(of: type, depth: depth)
            index += 1
        }
    }

    mutating func skip(_ count: UInt64) throws {
        guard count <= UInt64(remaining) else { throw GGUFChatMetadata.ReadError.truncated }
        offset += Int(count)
    }
}
