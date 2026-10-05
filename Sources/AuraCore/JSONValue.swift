import Foundation

/// A parsed JSON value, used by ``OutputSchemaValidator``. The parser is strict RFC 8259: no trailing commas,
/// comments, `NaN` or duplicate member names, so text it accepts reads the same in any conforming parser.
/// Numbers stay exact (``JSONNumber``).
enum JSONValue: Equatable, Sendable {
    case null
    case bool(Bool)
    case number(JSONNumber)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    /// `nil` when `text` is not exactly one JSON value (surrounding whitespace allowed).
    static func parse(_ text: String) -> JSONValue? {
        JSONParser.parse(text)
    }

    /// The JSON Schema type name, `integer` for whole numbers.
    var typeName: String {
        switch self {
            case .null: return "null"
            case .bool: return "boolean"
            case .number(let number): return number.isInteger ? "integer" : "number"
            case .string: return "string"
            case .array: return "array"
            case .object: return "object"
        }
    }

    /// Compact JSON with sorted keys, for messages.
    var serialized: String {
        switch self {
            case .null: return "null"
            case .bool(let flag): return flag ? "true" : "false"
            case .number(let number): return number.description
            case .string(let text): return Self.quote(text)
            case .array(let elements): return "[" + elements.map(\.serialized).joined(separator: ",") + "]"
            case .object(let members):
                return "{" + members.sorted { $0.key < $1.key }
                    .map { Self.quote($0.key) + ":" + $0.value.serialized }
                    .joined(separator: ",") + "}"
        }
    }

    static func quote(_ text: String) -> String {
        var quoted = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
                case "\"": quoted += "\\\""
                case "\\": quoted += "\\\\"
                case "\n": quoted += "\\n"
                case "\r": quoted += "\\r"
                case "\t": quoted += "\\t"
                case _ where scalar.value < 0x20: quoted += escaped(scalar)
                default: quoted.unicodeScalars.append(scalar)
            }
        }
        return quoted + "\""
    }

    private static func escaped(_ scalar: Unicode.Scalar) -> String {
        let hex = String(scalar.value, radix: 16)
        return "\\u" + String(repeating: "0", count: 4 - hex.count) + hex
    }
}

// MARK: - JSONNumber

/// A JSON number kept as an exact decimal, `digits × 10^exponent`, so integers above 2^53 and exponents beyond
/// `Double` (`1e400`) compare correctly. `digits` has no leading or trailing zeros and is empty for zero, which
/// makes equal values equal whatever their spelling (`1`, `1.0`, `10e-1`).
struct JSONNumber: Hashable, Sendable, Comparable, CustomStringConvertible {
    let isNegative: Bool
    /// ASCII digits.
    let digits: [UInt8]
    let exponent: Int

    init(isNegative: Bool, digits: some Collection<UInt8>, exponent: Int) {
        let significant = digits.drop { $0 == ASCII.zero }
        let trailingZeros = significant.reversed().prefix { $0 == ASCII.zero }.count
        self.digits = Array(significant.dropLast(trailingZeros))
        self.isNegative = isNegative && !self.digits.isEmpty
        self.exponent = self.digits.isEmpty ? 0 : exponent + trailingZeros
    }

    var isInteger: Bool { exponent >= 0 }

    /// The value as an `Int` when it is a whole number of at most 18 digits.
    var intValue: Int? {
        guard isInteger, digits.count + exponent <= 18 else { return nil }
        let magnitude = digits.reduce(0) { $0 * 10 + Int($1 - ASCII.zero) } * power(of: exponent)
        return isNegative ? -magnitude : magnitude
    }

    private func power(of exponent: Int) -> Int {
        (0..<exponent).reduce(1) { accumulated, _ in accumulated * 10 }
    }

    static func < (lhs: JSONNumber, rhs: JSONNumber) -> Bool {
        if lhs.isNegative != rhs.isNegative { return lhs.isNegative }
        return lhs.isNegative ? magnitudeLess(rhs, lhs) : magnitudeLess(lhs, rhs)
    }

    private static func magnitudeLess(_ lhs: JSONNumber, _ rhs: JSONNumber) -> Bool {
        if lhs.digits.isEmpty || rhs.digits.isEmpty { return lhs.digits.isEmpty && !rhs.digits.isEmpty }
        let lhsMagnitude = lhs.digits.count + lhs.exponent
        let rhsMagnitude = rhs.digits.count + rhs.exponent
        guard lhsMagnitude == rhsMagnitude else { return lhsMagnitude < rhsMagnitude }
        return lhs.digits.lexicographicallyPrecedes(rhs.digits)
    }

    /// Plain notation up to 21 integer digits or 5 zeros after the point, scientific otherwise.
    var description: String {
        guard let first = digits.first else { return "0" }
        let text = String(decoding: digits, as: UTF8.self)
        let point = digits.count + exponent
        let body: String
        if exponent >= 0 && point <= 21 {
            body = text + String(repeating: "0", count: exponent)
        } else if point > 0 && point < digits.count {
            body = String(text.prefix(point)) + "." + String(text.dropFirst(point))
        } else if point <= 0 && point > -6 {
            body = "0." + String(repeating: "0", count: -point) + text
        } else {
            let rest = String(text.dropFirst())
            body = String(Unicode.Scalar(first)) + (rest.isEmpty ? "" : "." + rest) + "e\(point - 1)"
        }
        return isNegative ? "-" + body : body
    }
}

// MARK: - Parser

private enum ASCII {
    static let zero = UInt8(ascii: "0")
    static let nine = UInt8(ascii: "9")
    static let quote = UInt8(ascii: "\"")
    static let backslash = UInt8(ascii: "\\")
    static let minus = UInt8(ascii: "-")
    static let plus = UInt8(ascii: "+")
    static let dot = UInt8(ascii: ".")
    static let comma = UInt8(ascii: ",")
    static let colon = UInt8(ascii: ":")
    static let openBrace = UInt8(ascii: "{")
    static let closeBrace = UInt8(ascii: "}")
    static let openBracket = UInt8(ascii: "[")
    static let closeBracket = UInt8(ascii: "]")
    static let whitespace: Set<UInt8> = [0x20, 0x09, 0x0A, 0x0D]

    static func isDigit(_ byte: UInt8) -> Bool { byte >= zero && byte <= nine }
}

/// Recursive descent over UTF-8 bytes. Every failure returns `nil`; nesting is capped so hostile model output
/// cannot overflow the stack.
private struct JSONParser {
    private static let maxDepth = 512
    /// Exponents are clamped here: far beyond any real value, yet the digit arithmetic cannot overflow.
    private static let exponentLimit = 1_000_000_000

    private let bytes: [UInt8]
    private var index = 0
    private var depth = 0

    private init(bytes: [UInt8]) { self.bytes = bytes }

    static func parse(_ text: String) -> JSONValue? {
        var parser = JSONParser(bytes: Array(text.utf8))
        guard let value = parser.value() else { return nil }
        parser.skipWhitespace()
        return parser.index == parser.bytes.count ? value : nil
    }

    private var current: UInt8? { index < bytes.count ? bytes[index] : nil }

    private mutating func skipWhitespace() {
        while let byte = current, ASCII.whitespace.contains(byte) { index += 1 }
    }

    private mutating func consume(_ byte: UInt8) -> Bool {
        guard current == byte else { return false }
        index += 1
        return true
    }

    private mutating func value() -> JSONValue? {
        skipWhitespace()
        switch current {
            case ASCII.openBrace, ASCII.openBracket:
                guard depth < Self.maxDepth else { return nil }
                depth += 1
                defer { depth -= 1 }
                return current == ASCII.openBrace ? object() : array()
            case ASCII.quote: return string().map(JSONValue.string)
            case UInt8(ascii: "t"): return literal("true", .bool(true))
            case UInt8(ascii: "f"): return literal("false", .bool(false))
            case UInt8(ascii: "n"): return literal("null", .null)
            default: return number().map(JSONValue.number)
        }
    }

    private mutating func literal(_ word: StaticString, _ result: JSONValue) -> JSONValue? {
        let length = word.utf8CodeUnitCount
        guard bytes.count - index >= length,
              word.withUTF8Buffer({ bytes[index..<index + length].elementsEqual($0) }) else { return nil }
        index += length
        return result
    }

    private mutating func object() -> JSONValue? {
        index += 1
        var members: [String: JSONValue] = [:]
        skipWhitespace()
        if consume(ASCII.closeBrace) { return .object(members) }
        repeat {
            skipWhitespace()
            guard current == ASCII.quote, let name = string(), members[name] == nil else { return nil }
            skipWhitespace()
            guard consume(ASCII.colon), let member = value() else { return nil }
            members[name] = member
            skipWhitespace()
        } while consume(ASCII.comma)
        return consume(ASCII.closeBrace) ? .object(members) : nil
    }

    private mutating func array() -> JSONValue? {
        index += 1
        var elements: [JSONValue] = []
        skipWhitespace()
        if consume(ASCII.closeBracket) { return .array(elements) }
        repeat {
            guard let element = value() else { return nil }
            elements.append(element)
            skipWhitespace()
        } while consume(ASCII.comma)
        return consume(ASCII.closeBracket) ? .array(elements) : nil
    }

    // MARK: Strings

    private mutating func string() -> String? {
        index += 1
        var utf8: [UInt8] = []
        while let byte = current, byte != ASCII.quote {
            index += 1
            if byte == ASCII.backslash {
                guard let scalar = escape() else { return nil }
                UTF8.encode(scalar) { utf8.append($0) }
            } else if byte < 0x20 {
                return nil
            } else {
                utf8.append(byte)
            }
        }
        guard consume(ASCII.quote) else { return nil }
        return String(decoding: utf8, as: UTF8.self)
    }

    private mutating func escape() -> Unicode.Scalar? {
        guard let byte = current else { return nil }
        index += 1
        switch byte {
            case ASCII.quote, ASCII.backslash, UInt8(ascii: "/"): return Unicode.Scalar(byte)
            case UInt8(ascii: "b"): return "\u{08}"
            case UInt8(ascii: "f"): return "\u{0C}"
            case UInt8(ascii: "n"): return "\n"
            case UInt8(ascii: "r"): return "\r"
            case UInt8(ascii: "t"): return "\t"
            case UInt8(ascii: "u"): return unicodeEscape()
            default: return nil
        }
    }

    /// `\uXXXX`, joining a UTF-16 surrogate pair; a lone surrogate is rejected.
    private mutating func unicodeEscape() -> Unicode.Scalar? {
        guard let unit = hexUnit() else { return nil }
        guard (0xD800...0xDBFF).contains(unit) else { return Unicode.Scalar(unit) }
        guard consume(ASCII.backslash), consume(UInt8(ascii: "u")), let low = hexUnit(),
              (0xDC00...0xDFFF).contains(low) else { return nil }
        return Unicode.Scalar(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00))
    }

    private mutating func hexUnit() -> UInt32? {
        guard bytes.count - index >= 4 else { return nil }
        var unit: UInt32 = 0
        for byte in bytes[index..<index + 4] {
            guard let nibble = Self.hexValue(byte) else { return nil }
            unit = unit << 4 | nibble
        }
        index += 4
        return unit
    }

    private static func hexValue(_ byte: UInt8) -> UInt32? {
        switch byte {
            case ASCII.zero...ASCII.nine: return UInt32(byte - ASCII.zero)
            case UInt8(ascii: "a")...UInt8(ascii: "f"): return UInt32(byte - UInt8(ascii: "a") + 10)
            case UInt8(ascii: "A")...UInt8(ascii: "F"): return UInt32(byte - UInt8(ascii: "A") + 10)
            default: return nil
        }
    }

    // MARK: Numbers

    private mutating func number() -> JSONNumber? {
        let isNegative = consume(ASCII.minus)
        let integerPart = digitRun()
        guard let leading = integerPart.first, leading != ASCII.zero || integerPart.count == 1 else { return nil }
        var fraction: ArraySlice<UInt8> = []
        if consume(ASCII.dot) {
            fraction = digitRun()
            guard !fraction.isEmpty else { return nil }
        }
        var exponent = 0
        if consume(UInt8(ascii: "e")) || consume(UInt8(ascii: "E")) {
            guard let written = exponentValue() else { return nil }
            exponent = written
        }
        return JSONNumber(isNegative: isNegative, digits: integerPart + fraction, exponent: exponent - fraction.count)
    }

    private mutating func digitRun() -> ArraySlice<UInt8> {
        let start = index
        while let byte = current, ASCII.isDigit(byte) { index += 1 }
        return bytes[start..<index]
    }

    private mutating func exponentValue() -> Int? {
        let isNegative = consume(ASCII.minus)
        if !isNegative { _ = consume(ASCII.plus) }
        let run = digitRun()
        guard !run.isEmpty else { return nil }
        let magnitude = run.reduce(0) { min($0 * 10 + Int($1 - ASCII.zero), Self.exponentLimit) }
        return isNegative ? -magnitude : magnitude
    }
}
