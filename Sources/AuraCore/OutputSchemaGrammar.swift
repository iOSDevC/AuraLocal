import Foundation

// MARK: - OutputSchemaGrammar

/// A llama.cpp GBNF grammar for a compiled schema: JSON syntax narrowed by what GBNF can express, the rest left to
/// validation. A free-form object can repeat a member name, which validation rejects.
enum OutputSchemaGrammar {
    static let rootRule = "root"
    /// Larger item and length bounds are not expressed; validation covers them.
    static let maxBoundedRepetition = 64
    /// More properties than this, none of them required, fall back to `jobject`.
    static let maxOptionalChain = 16
    /// Past this many rules, or ``maxGrammarBytes``, the grammar admits any JSON value.
    static let maxRules = 512
    static let maxGrammarBytes = 65_536

    /// JSON itself. Digits are unbounded so a long number is not cut short; `\u` escapes exclude surrogates, so each
    /// `jchar` is one Unicode scalar. llama.cpp rule names cannot contain `_`.
    static let baseRules = #"""
        ws ::= | " " | "\n" [ \t]{0,20}
        jchar ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" ([0-9a-cA-CefEF] [0-9a-fA-F]{3} | [dD] [0-7] [0-9a-fA-F]{2}))
        jstring ::= "\"" jchar* "\"" ws
        jint ::= "0" | [1-9] [0-9]*
        jfrac ::= "." [0-9]+
        jexp ::= [eE] [-+]? [0-9]+
        jinteger ::= "-"? jint ws
        jnonneg-integer ::= jint ws
        jpos-integer ::= [1-9] [0-9]* ws
        jnumber ::= "-"? jint jfrac? jexp? ws
        jnonneg-number ::= jint jfrac? jexp? ws
        jboolean ::= ("true" | "false") ws
        jnull ::= "null" ws
        jvalue ::= jobject | jarray | jstring | jnumber | jboolean | jnull
        jobject ::= "{" ws ( jstring ":" ws jvalue ( "," ws jstring ":" ws jvalue )* )? "}" ws
        jarray ::= "[" ws ( jvalue ( "," ws jvalue )* )? "]" ws
        """#

    /// The grammar for `root`, whose start rule is ``rootRule``; `nil` when the schema detectably admits no value.
    static func gbnf(for root: SchemaNode) -> String? {
        var builder = SchemaGrammarBuilder()
        guard let start = builder.reference(for: root) else { return nil }
        let text = document(start: start, definitions: builder.definitions)
        guard builder.definitions.count <= maxRules, text.utf8.count <= maxGrammarBytes else {
            return document(start: "jvalue", definitions: [])
        }
        return text
    }

    private static func document(start: String, definitions: [String]) -> String {
        (["\(rootRule) ::= \(start)"] + definitions + [baseRules]).joined(separator: "\n")
    }

    /// A GBNF string literal that matches `text` exactly.
    static func literal(_ text: String) -> String {
        var quoted = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
                case "\\": quoted += #"\\"#
                case "\"": quoted += #"\""#
                case "\n": quoted += #"\n"#
                case "\r": quoted += #"\r"#
                case "\t": quoted += #"\t"#
                case _ where scalar.value < 0x20 || scalar.value == 0x7F: quoted += hexEscape(scalar)
                default: quoted.unicodeScalars.append(scalar)
            }
        }
        return quoted + "\""
    }

    private static func hexEscape(_ scalar: Unicode.Scalar) -> String {
        let hex = String(scalar.value, radix: 16, uppercase: true)
        return #"\x"# + String(repeating: "0", count: 2 - hex.count) + hex
    }

    /// The GBNF suffix that repeats a symbol `minimum` to `maximum` times; a nil maximum is unbounded.
    static func repetition(_ minimum: Int, _ maximum: Int?) -> String {
        switch (minimum, maximum) {
            case (0, nil): return "*"
            case (0, 1): return "?"
            case (_, nil): return "{\(minimum),}"
            case (_, let upper?) where upper == minimum: return "{\(minimum)}"
            case (_, let upper?): return "{\(minimum),\(upper)}"
        }
    }
}

// MARK: - SchemaGrammarBuilder

/// Emits one rule per distinct body, so equal subschemas share a rule.
struct SchemaGrammarBuilder {
    /// `sN ::= body`, in creation order.
    private(set) var definitions: [String] = []
    /// Rule body to rule name.
    private var ruleNames: [String: String] = [:]

    private static let one = JSONNumber(isNegative: false, digits: [UInt8(ascii: "1")], exponent: 0)

    /// A rule name that matches `node`; `nil` when no value can.
    mutating func reference(for node: SchemaNode) -> String? {
        switch node {
            case .constant(let accepts): return accepts ? "jvalue" : nil
            case .rules(let rules): return reference(for: rules)
        }
    }

    private mutating func reference(for rules: SchemaRules) -> String? {
        if let expected = rules.constValue { return constantReference(expected, rules: rules) }
        if let allowed = rules.allowedValues { return enumReference(allowed, rules: rules) }
        if let branches = Self.alternativeBranches(of: rules) {
            return alternation(branches.compactMap { reference(for: $0) })
        }
        let kinds = rules.types ?? Self.inferredTypes(of: rules)
        guard !kinds.isEmpty else { return "jvalue" }
        return alternation(kinds.compactMap { typeReference($0, rules: rules) })
    }

    private mutating func define(_ body: String) -> String {
        if let name = ruleNames[body] { return name }
        let name = "s\(definitions.count + 1)"
        definitions.append("\(name) ::= \(body)")
        ruleNames[body] = name
        return name
    }

    private mutating func alternation(_ refs: [String]) -> String? {
        let distinct = Self.uniqued(refs)
        guard let first = distinct.first else { return nil }
        return distinct.count == 1 ? first : define(distinct.joined(separator: " | "))
    }

    // MARK: Values

    private mutating func constantReference(_ expected: JSONValue, rules: SchemaRules) -> String? {
        guard Self.typesAdmit(expected, rules.types),
              rules.allowedValues?.contains(expected) ?? true else { return nil }
        return define(Self.valueLiteral(expected))
    }

    private mutating func enumReference(_ allowed: [JSONValue], rules: SchemaRules) -> String? {
        let literals = Self.uniqued(allowed.filter { Self.typesAdmit($0, rules.types) }.map(Self.valueLiteral))
        guard !literals.isEmpty else { return nil }
        return define(literals.joined(separator: " | "))
    }

    private static func valueLiteral(_ value: JSONValue) -> String {
        OutputSchemaGrammar.literal(value.serialized) + " ws"
    }

    private static func typesAdmit(_ value: JSONValue, _ kinds: [SchemaType]?) -> Bool {
        kinds?.contains { $0.admits(value) } ?? true
    }

    // MARK: Combinators

    /// Only a node that is nothing but an `anyOf`, a `oneOf` or a one-branch `allOf` becomes alternatives; any other
    /// combinator is left to validation.
    private static func alternativeBranches(of rules: SchemaRules) -> [SchemaNode]? {
        guard rules.types == nil, inferredTypes(of: rules).isEmpty else { return nil }
        switch (rules.anyOf, rules.oneOf, rules.allOf) {
            case (let branches?, nil, nil), (nil, let branches?, nil): return branches
            case (nil, nil, let branches?) where branches.count == 1: return branches
            default: return nil
        }
    }

    private static func combinatorBranches(of rules: SchemaRules) -> [SchemaNode] {
        (rules.allOf ?? []) + (rules.anyOf ?? []) + (rules.oneOf ?? [])
    }

    /// Combinators left to validation may name members this node does not declare; listing only the declared ones
    /// would then fail every reply, so such an object admits any members instead.
    private static func branchesNameUndeclaredMembers(_ rules: SchemaRules) -> Bool {
        if case .constant(false) = rules.additionalProperties { return false }
        let declared = Set(rules.properties.keys).union(rules.required)
        return combinatorBranches(of: rules).contains { !memberNames(in: $0).isSubset(of: declared) }
    }

    private static func memberNames(in node: SchemaNode) -> Set<String> {
        guard case .rules(let rules) = node else { return [] }
        let own = Set(rules.properties.keys).union(rules.required)
        return combinatorBranches(of: rules).reduce(into: own) { $0.formUnion(memberNames(in: $1)) }
    }

    /// Whether a combinator left to validation, at any depth, constrains the subschema `location` finds in a branch.
    private static func branches(of rules: SchemaRules, reach location: (SchemaRules) -> Bool) -> Bool {
        combinatorBranches(of: rules).contains { branch in
            guard case .rules(let inner) = branch else { return false }
            return location(inner) || branches(of: inner, reach: location)
        }
    }

    private static func hasValueSchema(_ rules: SchemaRules) -> Bool {
        guard case .rules = rules.additionalProperties else { return false }
        return true
    }

    /// A subschema a combinator also constrains admits any value, since the combinator may require members it does not
    /// declare. Its rules are built on a copy and dropped, so only satisfiability is kept.
    private mutating func reference(for node: SchemaNode, constrainedElsewhere: Bool) -> String? {
        guard constrainedElsewhere else { return reference(for: node) }
        var scratch = self
        return scratch.reference(for: node).map { _ in "jvalue" }
    }

    // MARK: Types

    private static func inferredTypes(of rules: SchemaRules) -> [SchemaType] {
        let numericBounds = [rules.minimum, rules.maximum, rules.exclusiveMinimum, rules.exclusiveMaximum]
        let evidence: [(kind: SchemaType, present: Bool)] = [
            (.object, !rules.properties.isEmpty || !rules.required.isEmpty || rules.additionalProperties != nil),
            (.array, rules.items != nil || rules.minItems != nil || rules.maxItems != nil),
            (.string, rules.minLength != nil || rules.maxLength != nil),
            (.number, numericBounds.contains { $0 != nil }),
        ]
        return evidence.filter(\.present).map(\.kind)
    }

    private mutating func typeReference(_ kind: SchemaType, rules: SchemaRules) -> String? {
        switch kind {
            case .null: return "jnull"
            case .boolean: return "jboolean"
            case .integer: return Self.integerRule(rules)
            case .number: return Self.numberRule(rules)
            case .string: return stringReference(rules)
            case .array: return arrayReference(rules)
            case .object: return objectReference(rules)
        }
    }

    /// Zero counts as non-negative: ``JSONNumber`` normalises `-0`.
    private static func integerRule(_ rules: SchemaRules) -> String {
        let atLeastOne = rules.minimum.map { !($0 < one) } ?? false
        if isNonNegative(rules.exclusiveMinimum) || atLeastOne { return "jpos-integer" }
        return isNonNegative(rules.minimum) ? "jnonneg-integer" : "jinteger"
    }

    private static func numberRule(_ rules: SchemaRules) -> String {
        isNonNegative(rules.minimum) || isNonNegative(rules.exclusiveMinimum) ? "jnonneg-number" : "jnumber"
    }

    private static func isNonNegative(_ bound: JSONNumber?) -> Bool {
        bound.map { !$0.isNegative } ?? false
    }

    private mutating func stringReference(_ rules: SchemaRules) -> String? {
        guard let length = Self.cappedBounds(rules.minLength, rules.maxLength) else { return nil }
        if length.min == 0 && length.max == nil { return "jstring" }
        let characters = "jchar" + OutputSchemaGrammar.repetition(length.min, length.max)
        return define(#""\"" "# + characters + #" "\"" ws"#)
    }

    private mutating func arrayReference(_ rules: SchemaRules) -> String? {
        guard let count = Self.cappedBounds(rules.minItems, rules.maxItems) else { return nil }
        var itemRule: String? = "jvalue"
        if let items = rules.items {
            itemRule = reference(for: items, constrainedElsewhere: Self.branches(of: rules) { $0.items != nil })
        }
        guard let item = itemRule, count.max != 0 else {
            return (rules.minItems ?? 0) > 0 ? nil : define(#""[" ws "]" ws"#)
        }
        if item == "jvalue" && count.min == 0 && count.max == nil { return "jarray" }
        // The repeated group holds a single rule name, so llama.cpp never multiplies nested repetitions.
        let lower = max(count.min - 1, 0)
        let upper = count.max.map { $0 - 1 }
        let tail = #" ( "," ws "# + item + " )" + OutputSchemaGrammar.repetition(lower, upper)
        let more = lower == 0 && upper == 0 ? "" : tail
        return count.min == 0
            ? define(#""[" ws ( "# + item + more + #" )? "]" ws"#)
            : define(#""[" ws "# + item + more + #" "]" ws"#)
    }

    /// Bounds above ``OutputSchemaGrammar/maxBoundedRepetition`` are dropped; nil when the minimum exceeds the maximum.
    private static func cappedBounds(_ lower: Int?, _ upper: Int?) -> (min: Int, max: Int?)? {
        let cap = OutputSchemaGrammar.maxBoundedRepetition
        let minimum = lower ?? 0
        if let upper, minimum > upper { return nil }
        return (minimum > cap ? 0 : minimum, upper.flatMap { $0 > cap ? nil : $0 })
    }

    // MARK: Objects

    // Narrower than the schema but satisfiable: only declared members, since extra ones would risk duplicate keys.
    private mutating func objectReference(_ rules: SchemaRules) -> String? {
        if rules.properties.isEmpty && rules.required.isEmpty { return mapReference(rules) }
        if Self.branchesNameUndeclaredMembers(rules) { return "jobject" }
        let requiredKeys = Self.uniqued(rules.required)
        let optionalKeys = rules.properties.keys.filter { !requiredKeys.contains($0) }.sorted()
        let requiredMembers = members(requiredKeys, of: rules)
        guard requiredMembers.count == requiredKeys.count else { return nil }
        let optionalMembers = members(optionalKeys, of: rules)
        if requiredMembers.isEmpty && optionalMembers.count > OutputSchemaGrammar.maxOptionalChain { return "jobject" }
        let requiredPairs = requiredMembers.map { define(Self.pair($0.key, $0.value)) }
        let optionalPairs = optionalMembers.map { define(Self.pair($0.key, $0.value)) }
        guard let first = requiredPairs.first else { return optionalChain(optionalPairs) }
        let rest = requiredPairs.dropFirst().map { #""," ws "# + $0 } + optionalPairs.map(Self.optionalMember)
        return define(([#""{" ws"#, first] + rest + [#""}" ws"#]).joined(separator: " "))
    }

    /// Each key with its value rule; a key whose value admits nothing is left out.
    private mutating func members(_ keys: [String], of rules: SchemaRules) -> [(key: String, value: String)] {
        keys.compactMap { key in valueReference(for: key, in: rules).map { (key: key, value: $0) } }
    }

    private mutating func valueReference(for key: String, in rules: SchemaRules) -> String? {
        let subschema = rules.properties[key] ?? rules.additionalProperties ?? .constant(true)
        let reached = Self.branches(of: rules) { $0.properties[key] != nil || Self.hasValueSchema($0) }
        return reference(for: subschema, constrainedElsewhere: reached)
    }

    private static func pair(_ key: String, _ value: String) -> String {
        OutputSchemaGrammar.literal(JSONValue.quote(key)) + #" ws ":" ws "# + value
    }

    private static func optionalMember(_ pair: String) -> String {
        #"( "," ws "# + pair + " )?"
    }

    private static func optionalTail(_ pairs: ArraySlice<String>) -> String {
        pairs.map(optionalMember).joined(separator: " ")
    }

    /// Any in-order subset of optional members: `( O1 T1 | O2 T2 | … | On )?`, where Ti allows each later member.
    private mutating func optionalChain(_ pairs: [String]) -> String {
        guard !pairs.isEmpty else { return define(#""{" ws "}" ws"#) }
        let tails = pairs.indices.dropLast().map { define(Self.optionalTail(pairs[($0 + 1)...])) }
        let alternatives = pairs.indices.map { $0 < tails.count ? pairs[$0] + " " + tails[$0] : pairs[$0] }
        return define(#""{" ws ( "# + alternatives.joined(separator: " | ") + #" )? "}" ws"#)
    }

    /// An object with no declared members: any names, each value admitted by `additionalProperties`.
    private mutating func mapReference(_ rules: SchemaRules) -> String {
        guard let extra = rules.additionalProperties else { return "jobject" }
        let reached = Self.branches(of: rules) { !$0.properties.isEmpty || Self.hasValueSchema($0) }
        guard let value = reference(for: extra, constrainedElsewhere: reached) else { return define(#""{" ws "}" ws"#) }
        if value == "jvalue" { return "jobject" }
        let member = #"jstring ":" ws "# + value
        return define(#""{" ws ( "# + member + #" ( "," ws "# + member + #" )* )? "}" ws"#)
    }

    private static func uniqued(_ items: [String]) -> [String] {
        var seen: Set<String> = []
        return items.filter { seen.insert($0).inserted }
    }
}
