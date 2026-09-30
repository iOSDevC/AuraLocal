import Foundation

// MARK: - OutputSchemaError

/// Why an ``OutputSchema`` was rejected, or why a reply failed it.
public enum OutputSchemaError: Error, LocalizedError, Sendable, Equatable {
    /// The schema is unusable: not JSON (a GBNF grammar, for instance), not a schema object or boolean, or a
    /// keyword with a malformed value. The string names the offending location.
    case invalidSchema(String)
    /// The schema uses keywords the validator does not enforce, as JSON pointers into the schema
    /// (e.g. `/properties/email/format`). Rejected so no constraint is silently ignored.
    case unsupportedKeywords([String])
    /// The reply broke the schema: the raw model output and what was wrong with it. From ``AuraSession`` this
    /// is the last attempt, after every repair was spent.
    case violations(output: String, violations: [OutputSchemaViolation])

    public var errorDescription: String? {
        switch self {
            case .invalidSchema(let detail):
                return "Invalid output schema: \(detail)"
            case .unsupportedKeywords(let pointers):
                return "Output schema uses unsupported keywords: \(pointers.joined(separator: ", "))"
            case .violations(_, let found):
                return "Reply does not conform to the output schema: "
                    + found.map(\.description).joined(separator: "; ")
        }
    }
}

// MARK: - OutputSchemaViolation

/// One way a JSON value breaks a schema.
public struct OutputSchemaViolation: Sendable, Equatable, CustomStringConvertible {
    /// RFC 6901 JSON pointer to the offending value; `""` is the whole document.
    public let path: String
    public let message: String

    public init(path: String, message: String) {
        self.path = path
        self.message = message
    }

    public var description: String { "\(path.isEmpty ? "(root)" : path): \(message)" }
}

// MARK: - OutputSchemaValidator

/// Compiles a JSON Schema once and checks JSON values against it. This is validation, not constrained decoding:
/// the model can still produce anything, and ``AuraSession`` re-prompts when it does.
///
/// Supported keywords: `type` (a name or an array of names), `properties`, `required`, `additionalProperties`
/// (boolean or schema), `items` (a single schema), `enum`, `const`, `minItems`, `maxItems`, `minLength`,
/// `maxLength`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` (numbers), `anyOf`, `oneOf`, `allOf`.
/// `description`, `title`, `$schema`, `examples`, `default` and `$comment` are accepted and ignored. Any other
/// keyword (`$ref`, `pattern`, `format`, `patternProperties`, `if`/`then`/`else`, …) fails compilation.
public struct OutputSchemaValidator: Sendable {
    /// The schema text as given, for prompting the model.
    let source: String
    private let root: SchemaNode

    public init(_ schema: OutputSchema) throws(OutputSchemaError) {
        switch schema {
            case .json(let text):
                source = text
                root = try SchemaCompiler.compile(text)
        }
    }

    /// The violations of `json`, a JSON text. Empty means it conforms; text that is not JSON yields one violation.
    public func validate(_ json: String) -> [OutputSchemaViolation] {
        guard let value = JSONValue.parse(json) else {
            return [OutputSchemaViolation(path: "", message: "not valid JSON")]
        }
        return Self.violations(of: value, against: root, at: "")
    }

    /// Returns the text of the first JSON value in raw model output that conforms (candidates in
    /// ``extractJSON(from:)`` order); otherwise throws ``OutputSchemaError/violations(output:violations:)`` with the
    /// violations of the first value found.
    public func conformingJSON(in output: String) throws(OutputSchemaError) -> String {
        try check(output).get()
    }

    func check(_ output: String) -> Result<String, OutputSchemaError> {
        var firstFound: [OutputSchemaViolation]?
        for candidate in Self.candidates(in: output) {
            let found = Self.violations(of: candidate.value, against: root, at: "")
            if found.isEmpty { return .success(candidate.text) }
            if firstFound == nil { firstFound = found }
        }
        let found = firstFound ?? [OutputSchemaViolation(path: "", message: "no JSON value found in the reply")]
        return .failure(.violations(output: output, violations: found))
    }

    /// The first JSON value in raw model output. Candidates, in order: the whole trimmed text; then, with
    /// `<think>…</think>` reasoning dropped, the whole text again, each ```` ```json ```` (or bare ```` ``` ````)
    /// fenced block, and each top-level balanced `{…}` or `[…]`. A candidate counts only when it parses as strict
    /// JSON; brace matching skips string literals. `nil` when none is found.
    public static func extractJSON(from output: String) -> String? {
        candidates(in: output).first?.text
    }
}

// MARK: - Extraction

extension OutputSchemaValidator {
    private struct Candidate {
        let text: String
        let value: JSONValue
    }

    private static func candidates(in output: String) -> [Candidate] {
        if let whole = candidate(output[...]) { return [whole] }
        let answer = withoutReasoning(output)
        if let whole = candidate(answer[...]) { return [whole] }
        return fencedBlocks(in: answer).compactMap(candidate) + balancedValues(in: answer)
    }

    private static func candidate(_ text: Substring) -> Candidate? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return JSONValue.parse(trimmed).map { Candidate(text: trimmed, value: $0) }
    }

    /// Drops `<think>…</think>` spans, a reasoning prefix closed by a lone `</think>` (chat templates that open
    /// the block in the prompt), and an unterminated `<think>` with everything after it.
    static func withoutReasoning(_ text: String) -> String {
        var rest = text[...]
        if let close = rest.range(of: "</think>"),
           rest.range(of: "<think>").map({ $0.lowerBound > close.lowerBound }) ?? true {
            rest = rest[close.upperBound...]
        }
        var answer = ""
        while let open = rest.range(of: "<think>") {
            answer += rest[..<open.lowerBound]
            guard let close = rest[open.upperBound...].range(of: "</think>") else { return answer }
            rest = rest[close.upperBound...]
        }
        return answer + rest
    }

    private static func fencedBlocks(in text: String) -> [Substring] {
        var blocks: [Substring] = []
        var cursor = text.startIndex
        while let open = text.range(of: "```", range: cursor..<text.endIndex),
              let lineEnd = text[open.upperBound...].firstIndex(where: \.isNewline),
              let close = text.range(of: "```", range: lineEnd..<text.endIndex) {
            let info = text[open.upperBound..<lineEnd].trimmingCharacters(in: .whitespaces).lowercased()
            if info.isEmpty || info == "json" { blocks.append(text[lineEnd..<close.lowerBound]) }
            cursor = close.upperBound
        }
        return blocks
    }

    // Byte-level scan is safe: every delimiter is ASCII and UTF-8 continuation bytes never are.
    private static func balancedValues(in text: String) -> [Candidate] {
        let bytes = Array(text.utf8)
        var found: [Candidate] = []
        var start = 0
        while let open = bytes[start...].firstIndex(where: { $0 == ASCII.openBrace || $0 == ASCII.openBracket }) {
            start = open + 1
            guard let end = balancedEnd(in: bytes, from: open),
                  let value = candidate(decodedSlice(bytes[open...end])) else { continue }
            found.append(value)
            start = end + 1                            // values nested in an accepted one are not candidates
        }
        return found
    }

    private static func decodedSlice(_ slice: ArraySlice<UInt8>) -> Substring {
        String(decoding: slice, as: UTF8.self)[...]
    }

    private static func balancedEnd(in bytes: [UInt8], from open: Int) -> Int? {
        var closers: [UInt8] = []
        var inString = false
        var escaped = false
        for index in open..<bytes.count {
            let byte = bytes[index]
            if inString {
                if escaped { escaped = false }
                else if byte == ASCII.backslash { escaped = true }
                else if byte == ASCII.quote { inString = false }
                continue
            }
            switch byte {
                case ASCII.quote: inString = true
                case ASCII.openBrace: closers.append(ASCII.closeBrace)
                case ASCII.openBracket: closers.append(ASCII.closeBracket)
                case ASCII.closeBrace, ASCII.closeBracket:
                    guard closers.popLast() == byte else { return nil }
                    if closers.isEmpty { return index }
                default: break
            }
        }
        return nil
    }

    private enum ASCII {
        static let quote = UInt8(ascii: "\"")
        static let backslash = UInt8(ascii: "\\")
        static let openBrace = UInt8(ascii: "{")
        static let closeBrace = UInt8(ascii: "}")
        static let openBracket = UInt8(ascii: "[")
        static let closeBracket = UInt8(ascii: "]")
    }
}

// MARK: - Validation

extension OutputSchemaValidator {
    private static func violation(_ pointer: String, _ message: String) -> OutputSchemaViolation {
        OutputSchemaViolation(path: pointer, message: message)
    }

    private static func violations(of value: JSONValue, against node: SchemaNode, at pointer: String) -> [OutputSchemaViolation] {
        switch node {
            case .constant(true):
                return []
            case .constant(false):
                return [violation(pointer, "no value is allowed here")]
            case .rules(let rules):
                return violations(of: value, rules: rules, at: pointer)
        }
    }

    private static func violations(of value: JSONValue, rules: SchemaRules, at pointer: String) -> [OutputSchemaViolation] {
        var found: [OutputSchemaViolation] = []
        if let types = rules.types, !types.contains(where: { $0.admits(value) }) {
            found.append(violation(pointer, "expected \(types.map(\.rawValue).joined(separator: " or ")), got \(value.typeName)"))
        }
        if let allowed = rules.allowedValues, !allowed.contains(value) {
            found.append(violation(pointer, "must be one of \(allowed.map(\.serialized).joined(separator: ", "))"))
        }
        if let expected = rules.constValue, expected != value {
            found.append(violation(pointer, "must equal \(expected.serialized)"))
        }
        switch value {
            case .string(let text):
                found += lengthViolations(text.unicodeScalars.count, rules: rules, at: pointer)
            case .number(let number):
                found += rangeViolations(number, rules: rules, at: pointer)
            case .array(let elements):
                found += arrayViolations(elements, rules: rules, at: pointer)
            case .object(let members):
                found += objectViolations(members, rules: rules, at: pointer)
            case .null, .bool:
                break
        }
        found += combinatorViolations(value, rules: rules, at: pointer)
        return found
    }

    private static func lengthViolations(_ length: Int, rules: SchemaRules, at pointer: String) -> [OutputSchemaViolation] {
        var found: [OutputSchemaViolation] = []
        if let min = rules.minLength, length < min {
            found.append(violation(pointer, "must be at least \(min) characters, got \(length)"))
        }
        if let max = rules.maxLength, length > max {
            found.append(violation(pointer, "must be at most \(max) characters, got \(length)"))
        }
        return found
    }

    private static func rangeViolations(_ number: JSONNumber, rules: SchemaRules, at pointer: String) -> [OutputSchemaViolation] {
        var found: [OutputSchemaViolation] = []
        if let bound = rules.minimum, number < bound {
            found.append(violation(pointer, "must be >= \(bound), got \(number)"))
        }
        if let bound = rules.maximum, number > bound {
            found.append(violation(pointer, "must be <= \(bound), got \(number)"))
        }
        if let bound = rules.exclusiveMinimum, number <= bound {
            found.append(violation(pointer, "must be > \(bound), got \(number)"))
        }
        if let bound = rules.exclusiveMaximum, number >= bound {
            found.append(violation(pointer, "must be < \(bound), got \(number)"))
        }
        return found
    }

    private static func arrayViolations(_ elements: [JSONValue], rules: SchemaRules, at pointer: String) -> [OutputSchemaViolation] {
        var found: [OutputSchemaViolation] = []
        if let min = rules.minItems, elements.count < min {
            found.append(violation(pointer, "must have at least \(min) items, got \(elements.count)"))
        }
        if let max = rules.maxItems, elements.count > max {
            found.append(violation(pointer, "must have at most \(max) items, got \(elements.count)"))
        }
        if let itemNode = rules.items {
            for (index, element) in elements.enumerated() {
                found += violations(of: element, against: itemNode, at: "\(pointer)/\(index)")
            }
        }
        return found
    }

    private static func objectViolations(_ members: [String: JSONValue], rules: SchemaRules,
                                         at pointer: String) -> [OutputSchemaViolation] {
        var found: [OutputSchemaViolation] = []
        for name in rules.required where members[name] == nil {
            found.append(violation(pointer, "missing required property \(JSONValue.quote(name))"))
        }
        for (name, member) in members.sorted(by: { $0.key < $1.key }) {
            let memberPointer = "\(pointer)/\(JSONPointer.escape(name))"
            if let propertyNode = rules.properties[name] {
                found += violations(of: member, against: propertyNode, at: memberPointer)
            } else if case .constant(false) = rules.additionalProperties {
                found.append(violation(memberPointer, "property not allowed"))
            } else if let extraNode = rules.additionalProperties {
                found += violations(of: member, against: extraNode, at: memberPointer)
            }
        }
        return found
    }

    private static func combinatorViolations(_ value: JSONValue, rules: SchemaRules, at pointer: String) -> [OutputSchemaViolation] {
        var found: [OutputSchemaViolation] = []
        if let branches = rules.allOf {
            found += branches.flatMap { violations(of: value, against: $0, at: pointer) }
        }
        if let branches = rules.anyOf {
            let outcomes = branches.map { violations(of: value, against: $0, at: pointer) }
            if !outcomes.contains(where: \.isEmpty) {
                found.append(violation(pointer, "must match at least one anyOf schema (\(summary(of: outcomes)))"))
            }
        }
        if let branches = rules.oneOf {
            let outcomes = branches.map { violations(of: value, against: $0, at: pointer) }
            let matched = outcomes.indices.filter { outcomes[$0].isEmpty }
            if matched.isEmpty {
                found.append(violation(pointer, "must match exactly one oneOf schema, matched none (\(summary(of: outcomes)))"))
            } else if matched.count > 1 {
                let which = matched.map { "#\($0)" }.joined(separator: ", ")
                found.append(violation(pointer, "must match exactly one oneOf schema, matched \(matched.count) (\(which))"))
            }
        }
        return found
    }

    /// The first violation of each failed branch, so a repair prompt says why every alternative missed.
    private static func summary(of outcomes: [[OutputSchemaViolation]]) -> String {
        outcomes.enumerated()
            .map { "#\($0.offset): \($0.element.first?.description ?? "ok")" }
            .joined(separator: "; ")
    }
}

// MARK: - Compiled schema

private indirect enum SchemaNode: Sendable {
    /// `true` accepts any value, `false` none.
    case constant(Bool)
    case rules(SchemaRules)
}

private struct SchemaRules: Sendable {
    var types: [SchemaType]?
    var properties: [String: SchemaNode] = [:]
    var required: [String] = []
    var additionalProperties: SchemaNode?
    var items: SchemaNode?
    var allowedValues: [JSONValue]?
    var constValue: JSONValue?
    var minItems: Int?
    var maxItems: Int?
    var minLength: Int?
    var maxLength: Int?
    var minimum: JSONNumber?
    var maximum: JSONNumber?
    var exclusiveMinimum: JSONNumber?
    var exclusiveMaximum: JSONNumber?
    var allOf: [SchemaNode]?
    var anyOf: [SchemaNode]?
    var oneOf: [SchemaNode]?
}

private enum SchemaType: String, Sendable {
    case null, boolean, object, array, number, integer, string

    func admits(_ value: JSONValue) -> Bool {
        switch (self, value) {
            case (.null, .null), (.boolean, .bool), (.object, .object), (.array, .array),
                 (.number, .number), (.string, .string):
                return true
            case (.integer, .number(let number)):
                return number.isInteger
            default:
                return false
        }
    }
}

private enum JSONPointer {
    static func escape(_ token: String) -> String {
        token.replacingOccurrences(of: "~", with: "~0").replacingOccurrences(of: "/", with: "~1")
    }
}

// MARK: - Compilation

private struct SchemaCompiler {
    private static let annotations: Set<String> = ["description", "title", "$schema", "examples", "default", "$comment"]
    private static let keywords: Set<String> = [
        "type", "properties", "required", "additionalProperties", "items", "enum", "const",
        "minItems", "maxItems", "minLength", "maxLength",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "anyOf", "oneOf", "allOf",
    ]

    private var unsupported: [String] = []

    static func compile(_ text: String) throws(OutputSchemaError) -> SchemaNode {
        guard let document = JSONValue.parse(text) else {
            let hint = text.contains("::=") ? " (looks like a GBNF grammar; only JSON Schema is supported)" : ""
            throw .invalidSchema("the schema is not valid JSON\(hint)")
        }
        var compiler = SchemaCompiler()
        let root = try compiler.node(document, at: "")
        guard compiler.unsupported.isEmpty else { throw .unsupportedKeywords(compiler.unsupported) }
        return root
    }

    private mutating func node(_ value: JSONValue, at pointer: String) throws(OutputSchemaError) -> SchemaNode {
        switch value {
            case .bool(let accepts):
                return .constant(accepts)
            case .object(let members):
                return .rules(try rules(members, at: pointer))
            default:
                throw .invalidSchema("\(Self.location(pointer)) must be an object or a boolean")
        }
    }

    private mutating func rules(_ members: [String: JSONValue], at pointer: String) throws(OutputSchemaError) -> SchemaRules {
        var compiled = SchemaRules()
        for key in members.keys.sorted() where !Self.annotations.contains(key) {
            let keyPointer = "\(pointer)/\(JSONPointer.escape(key))"
            guard Self.keywords.contains(key), let member = members[key] else {
                unsupported.append(keyPointer)
                continue
            }
            try apply(key, member, to: &compiled, at: keyPointer)
        }
        return compiled
    }

    private mutating func apply(_ key: String, _ value: JSONValue, to compiled: inout SchemaRules,
                                at pointer: String) throws(OutputSchemaError) {
        switch key {
            case "type": compiled.types = try Self.types(value, at: pointer)
            case "properties": compiled.properties = try subschemaMap(value, at: pointer)
            case "required": compiled.required = try Self.names(value, at: pointer)
            case "additionalProperties": compiled.additionalProperties = try node(value, at: pointer)
            case "items": compiled.items = try itemsNode(value, at: pointer)
            case "enum": compiled.allowedValues = try Self.allowedValues(value, at: pointer)
            case "const": compiled.constValue = value
            case "allOf", "anyOf", "oneOf": try applyCombinator(key, value, to: &compiled, at: pointer)
            default: try applyBound(key, value, to: &compiled, at: pointer)
        }
    }

    private mutating func applyCombinator(_ key: String, _ value: JSONValue, to compiled: inout SchemaRules,
                                          at pointer: String) throws(OutputSchemaError) {
        let branches = try subschemaList(value, at: pointer)
        switch key {
            case "allOf": compiled.allOf = branches
            case "anyOf": compiled.anyOf = branches
            default: compiled.oneOf = branches
        }
    }

    /// The count and numeric bound keywords, split from ``apply(_:_:to:at:)`` to keep each switch small.
    private mutating func applyBound(_ key: String, _ value: JSONValue, to compiled: inout SchemaRules,
                                     at pointer: String) throws(OutputSchemaError) {
        switch key {
            case "minItems": compiled.minItems = try Self.count(value, at: pointer)
            case "maxItems": compiled.maxItems = try Self.count(value, at: pointer)
            case "minLength": compiled.minLength = try Self.count(value, at: pointer)
            case "maxLength": compiled.maxLength = try Self.count(value, at: pointer)
            case "minimum": compiled.minimum = try Self.number(value, at: pointer)
            case "maximum": compiled.maximum = try Self.number(value, at: pointer)
            case "exclusiveMinimum": compiled.exclusiveMinimum = try Self.number(value, at: pointer)
            case "exclusiveMaximum": compiled.exclusiveMaximum = try Self.number(value, at: pointer)
            default: unsupported.append(pointer)
        }
    }

    private mutating func itemsNode(_ value: JSONValue, at pointer: String) throws(OutputSchemaError) -> SchemaNode {
        if case .array = value {
            throw .invalidSchema("\(pointer) must be a single schema; the tuple form is not supported")
        }
        return try node(value, at: pointer)
    }

    private static func allowedValues(_ value: JSONValue, at pointer: String) throws(OutputSchemaError) -> [JSONValue] {
        guard case .array(let allowed) = value, !allowed.isEmpty else {
            throw .invalidSchema("\(pointer) must be a non-empty array")
        }
        return allowed
    }

    private mutating func subschemaMap(_ value: JSONValue, at pointer: String) throws(OutputSchemaError) -> [String: SchemaNode] {
        guard case .object(let members) = value else { throw .invalidSchema("\(pointer) must be an object") }
        var compiled: [String: SchemaNode] = [:]
        for (name, member) in members.sorted(by: { $0.key < $1.key }) {
            compiled[name] = try node(member, at: "\(pointer)/\(JSONPointer.escape(name))")
        }
        return compiled
    }

    private mutating func subschemaList(_ value: JSONValue, at pointer: String) throws(OutputSchemaError) -> [SchemaNode] {
        guard case .array(let members) = value, !members.isEmpty else {
            throw .invalidSchema("\(pointer) must be a non-empty array of schemas")
        }
        var compiled: [SchemaNode] = []
        for (index, member) in members.enumerated() {
            compiled.append(try node(member, at: "\(pointer)/\(index)"))
        }
        return compiled
    }

    private static func types(_ value: JSONValue, at pointer: String) throws(OutputSchemaError) -> [SchemaType] {
        let names: [JSONValue]
        switch value {
            case .string: names = [value]
            case .array(let list) where !list.isEmpty: names = list
            default: throw .invalidSchema("\(pointer) must be a type name or a non-empty array of them")
        }
        var compiled: [SchemaType] = []
        for name in names {
            guard let known = schemaType(name) else {
                throw .invalidSchema("\(pointer) has an unknown type \(name.serialized)")
            }
            compiled.append(known)
        }
        return compiled
    }

    private static func schemaType(_ name: JSONValue) -> SchemaType? {
        guard case .string(let raw) = name else { return nil }
        return SchemaType(rawValue: raw)
    }

    private static func names(_ value: JSONValue, at pointer: String) throws(OutputSchemaError) -> [String] {
        guard case .array(let list) = value else { throw .invalidSchema("\(pointer) must be an array of strings") }
        var compiled: [String] = []
        for item in list {
            guard case .string(let name) = item else { throw .invalidSchema("\(pointer) must be an array of strings") }
            compiled.append(name)
        }
        return compiled
    }

    private static func count(_ value: JSONValue, at pointer: String) throws(OutputSchemaError) -> Int {
        guard case .number(let number) = value, let count = number.intValue, (0...Int(Int32.max)).contains(count) else {
            throw .invalidSchema("\(pointer) must be a non-negative integer")
        }
        return count
    }

    private static func number(_ value: JSONValue, at pointer: String) throws(OutputSchemaError) -> JSONNumber {
        guard case .number(let number) = value else { throw .invalidSchema("\(pointer) must be a number") }
        return number
    }

    private static func location(_ pointer: String) -> String { pointer.isEmpty ? "the schema" : pointer }
}
