import Foundation

/// A parsed JSON value, used by ``OutputSchemaValidator``. Parsing goes through `JSONDecoder`, which keeps
/// booleans and numbers apart (unlike `JSONSerialization`'s `NSNumber`).
enum JSONValue: Equatable, Sendable, Decodable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let flag = try? container.decode(Bool.self) {
            self = .bool(flag)
        } else if let number = try? container.decode(Double.self) {
            self = .number(number)
        } else if let text = try? container.decode(String.self) {
            self = .string(text)
        } else if let elements = try? container.decode([JSONValue].self) {
            self = .array(elements)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    /// `nil` when `text` is not exactly one JSON value (surrounding whitespace allowed).
    static func parse(_ text: String) -> JSONValue? {
        try? decoder.decode(JSONValue.self, from: Data(text.utf8))
    }

    private static let decoder: JSONDecoder = { JSONDecoder() }()

    /// The JSON Schema type name, `integer` for whole numbers.
    var typeName: String {
        switch self {
            case .null: return "null"
            case .bool: return "boolean"
            case .number(let number): return number.rounded(.towardZero) == number ? "integer" : "number"
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
            case .number(let number): return Self.format(number)
            case .string(let text): return Self.quote(text)
            case .array(let elements): return "[" + elements.map(\.serialized).joined(separator: ",") + "]"
            case .object(let members):
                return "{" + members.sorted { $0.key < $1.key }
                    .map { Self.quote($0.key) + ":" + $0.value.serialized }
                    .joined(separator: ",") + "}"
        }
    }

    static func format(_ number: Double) -> String {
        if number.rounded(.towardZero) == number, abs(number) < 1e15 { return String(Int64(number)) }
        return "\(number)"
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
