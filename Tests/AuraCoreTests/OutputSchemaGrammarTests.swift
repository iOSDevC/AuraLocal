import XCTest
@testable import AuraCore

final class OutputSchemaGrammarTests: XCTestCase {

    private static let personSchema = #"""
        {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0}},
         "required":["name","age"],"additionalProperties":false}
        """#

    // MARK: Helpers

    private func grammar(_ schema: String) throws -> String? {
        try OutputSchemaValidator(.json(schema)).grammar
    }

    /// The generated rules, root line first, without the base rules.
    private func rules(_ schema: String, file: StaticString = #filePath, line: UInt = #line) throws -> String {
        let text = try XCTUnwrap(grammar(schema), "no grammar for \(schema)", file: file, line: line)
        let suffix = "\n" + OutputSchemaGrammar.baseRules
        XCTAssertTrue(text.hasSuffix(suffix), "base rules missing", file: file, line: line)
        return String(text.dropLast(suffix.count))
    }

    private func rootReference(_ schema: String, file: StaticString = #filePath, line: UInt = #line) throws -> String {
        let generated = try rules(schema, file: file, line: line)
        let root = try XCTUnwrap(generated.split(separator: "\n").first, file: file, line: line)
        XCTAssertTrue(root.hasPrefix("root ::= "), file: file, line: line)
        return String(root.dropFirst("root ::= ".count))
    }

    private func numbered(_ count: Int, _ make: (Int) -> String) -> String {
        (1...count).map(make).joined(separator: ",")
    }

    // MARK: Goldens

    func testPersonSchemaGolden() throws {
        XCTAssertEqual(try grammar(Self.personSchema), #"""
            root ::= s3
            s1 ::= "\"name\"" ws ":" ws jstring
            s2 ::= "\"age\"" ws ":" ws jnonneg-integer
            s3 ::= "{" ws s1 "," ws s2 "}" ws
            """# + "\n" + OutputSchemaGrammar.baseRules)
    }

    func testOptionalOnlyObjectGolden() throws {
        XCTAssertEqual(try rules(#"{"properties":{"a":{"type":"string"},"b":{"type":"integer"}}}"#), #"""
            root ::= s4
            s1 ::= "\"a\"" ws ":" ws jstring
            s2 ::= "\"b\"" ws ":" ws jinteger
            s3 ::= ( "," ws s2 )?
            s4 ::= "{" ws ( s1 s3 | s2 )? "}" ws
            """#)
    }

    func testRequiredOrderThenOptionalByName() throws {
        let schema = #"""
            {"type":"object","required":["m","z","m"],
             "properties":{"z":{"type":"string"},"a":{"type":"boolean"},"m":{"type":"null"},"b":{"type":"string"},
                           "never":false}}
            """#
        XCTAssertEqual(try rules(schema), #"""
            root ::= s5
            s1 ::= "\"m\"" ws ":" ws jnull
            s2 ::= "\"z\"" ws ":" ws jstring
            s3 ::= "\"a\"" ws ":" ws jboolean
            s4 ::= "\"b\"" ws ":" ws jstring
            s5 ::= "{" ws s1 "," ws s2 ( "," ws s3 )? ( "," ws s4 )? "}" ws
            """#)

        XCTAssertEqual(try rules(#"{"required":["id"],"additionalProperties":{"type":"integer"}}"#), #"""
            root ::= s2
            s1 ::= "\"id\"" ws ":" ws jinteger
            s2 ::= "{" ws s1 "}" ws
            """#)
    }

    func testEnumAndConstLiteralsAreEscaped() throws {
        let schema = #"{"enum":["say \"hi\"","back\\slash","ñandú",1,null,{"b":[true],"a":1}]}"#
        XCTAssertEqual(try rules(schema), #"""
            root ::= s1
            s1 ::= "\"say \\\"hi\\\"\"" ws | "\"back\\\\slash\"" ws | "\"ñandú\"" ws | "1" ws | "null" ws | "{\"a\":1,\"b\":[true]}" ws
            """#)
        XCTAssertEqual(try rules(#"{"const":"a\u0001b"}"#), #"""
            root ::= s1
            s1 ::= "\"a\\u0001b\"" ws
            """#)

        XCTAssertEqual(OutputSchemaGrammar.literal("a\u{01}b\u{7F}c"), #""a\x01b\x7Fc""#)
        XCTAssertEqual(OutputSchemaGrammar.literal("tab\tline\nreturn\r"), #""tab\tline\nreturn\r""#)
        XCTAssertEqual(OutputSchemaGrammar.literal("\"\\é"), #""\"\\é""#)
    }

    func testEnumFilteredByType() throws {
        XCTAssertEqual(try rules(#"{"type":"string","enum":["a",1,"b",null,"a"]}"#), #"""
            root ::= s1
            s1 ::= "\"a\"" ws | "\"b\"" ws
            """#)
        XCTAssertEqual(try rules(#"{"type":"integer","enum":[1.5,2,2.0]}"#), #"""
            root ::= s1
            s1 ::= "2" ws
            """#)
        XCTAssertNil(try grammar(#"{"type":"string","enum":[1,2]}"#))
    }

    func testTypeArraysAlternate() throws {
        XCTAssertEqual(try rules(#"{"type":["string","null"]}"#), #"""
            root ::= s1
            s1 ::= jstring | jnull
            """#)
        XCTAssertEqual(try rootReference(#"{"type":["boolean","boolean"]}"#), "jboolean")
    }

    func testLowerBoundsPickUnsignedRules() throws {
        let cases: [(schema: String, rule: String)] = [
            (#"{"type":"integer"}"#, "jinteger"),
            (#"{"type":"integer","minimum":0}"#, "jnonneg-integer"),
            (#"{"type":"integer","minimum":-0}"#, "jnonneg-integer"),
            (#"{"type":"integer","minimum":0.5}"#, "jnonneg-integer"),
            (#"{"type":"integer","minimum":-3}"#, "jinteger"),
            (#"{"type":"integer","exclusiveMinimum":0}"#, "jpos-integer"),
            (#"{"type":"integer","exclusiveMinimum":-0.5}"#, "jinteger"),
            (#"{"type":"integer","minimum":1}"#, "jpos-integer"),
            (#"{"type":"integer","minimum":1e3}"#, "jpos-integer"),
            (#"{"type":"integer","maximum":-1}"#, "jinteger"),
            (#"{"type":"number"}"#, "jnumber"),
            (#"{"type":"number","minimum":0}"#, "jnonneg-number"),
            (#"{"type":"number","exclusiveMinimum":2.5}"#, "jnonneg-number"),
            (#"{"type":"number","exclusiveMinimum":-1}"#, "jnumber"),
            (#"{"minimum":0}"#, "jnonneg-number"),
        ]
        for (schema, rule) in cases {
            XCTAssertEqual(try rootReference(schema), rule, schema)
        }
    }

    func testStringBoundsAndCap() throws {
        XCTAssertEqual(try rules(#"{"type":"string","minLength":2,"maxLength":5}"#), #"""
            root ::= s1
            s1 ::= "\"" jchar{2,5} "\"" ws
            """#)
        let cases: [(schema: String, body: String)] = [
            (#"{"type":"string","minLength":3}"#, #""\"" jchar{3,} "\"" ws"#),
            (#"{"type":"string","maxLength":1}"#, #""\"" jchar? "\"" ws"#),
            (#"{"type":"string","minLength":4,"maxLength":4}"#, #""\"" jchar{4} "\"" ws"#),
            (#"{"type":"string","maxLength":0}"#, #""\"" jchar{0} "\"" ws"#),
            (#"{"type":"string","minLength":2,"maxLength":100}"#, #""\"" jchar{2,} "\"" ws"#),
            (#"{"maxLength":64}"#, #""\"" jchar{0,64} "\"" ws"#),
        ]
        for (schema, body) in cases {
            XCTAssertEqual(try rules(schema), "root ::= s1\ns1 ::= " + body, schema)
        }
        XCTAssertEqual(try rootReference(#"{"type":"string","maxLength":100}"#), "jstring")
        XCTAssertEqual(try rootReference(#"{"type":"string","minLength":100}"#), "jstring")
        XCTAssertEqual(try rootReference(#"{"type":"string","minLength":100,"maxLength":120}"#), "jstring")
    }

    func testArrayBounds() throws {
        let cases: [(schema: String, body: String)] = [
            (#"{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":3}"#,
             #""[" ws jstring ( "," ws jstring ){0,2} "]" ws"#),
            (#"{"type":"array","items":{"type":"integer"},"maxItems":1}"#, #""[" ws ( jinteger )? "]" ws"#),
            (#"{"type":"array","items":{"type":"integer"},"maxItems":2}"#,
             #""[" ws ( jinteger ( "," ws jinteger )? )? "]" ws"#),
            (#"{"type":"array","items":{"type":"integer"}}"#, #""[" ws ( jinteger ( "," ws jinteger )* )? "]" ws"#),
            (#"{"type":"array","items":{"type":"integer"},"minItems":1}"#,
             #""[" ws jinteger ( "," ws jinteger )* "]" ws"#),
            (#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":1}"#, #""[" ws jinteger "]" ws"#),
            (#"{"type":"array","items":{"type":"integer"},"minItems":3,"maxItems":3}"#,
             #""[" ws jinteger ( "," ws jinteger ){2} "]" ws"#),
            (#"{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":100}"#,
             #""[" ws jinteger ( "," ws jinteger ){1,} "]" ws"#),
            (#"{"type":"array","items":{"type":"integer"},"minItems":100}"#,
             #""[" ws ( jinteger ( "," ws jinteger )* )? "]" ws"#),
            (#"{"type":"array","maxItems":0}"#, #""[" ws "]" ws"#),
            (#"{"minItems":1}"#, #""[" ws jvalue ( "," ws jvalue )* "]" ws"#),
        ]
        for (schema, body) in cases {
            XCTAssertEqual(try rules(schema), "root ::= s1\ns1 ::= " + body, schema)
        }
        XCTAssertEqual(try rootReference(#"{"type":"array"}"#), "jarray")
        XCTAssertEqual(try rootReference(#"{"type":"array","items":true,"maxItems":200}"#), "jarray")
    }

    func testItemsFalse() throws {
        let empty = #""[" ws "]" ws"#
        XCTAssertEqual(try rules(#"{"type":"array","items":false}"#), "root ::= s1\ns1 ::= " + empty)
        XCTAssertEqual(try rules(#"{"type":"array","items":{"type":"string","minLength":3,"maxLength":1}}"#),
                       "root ::= s1\ns1 ::= " + empty)
        XCTAssertNil(try grammar(#"{"type":"array","items":false,"minItems":1}"#))
        XCTAssertNil(try grammar(#"{"type":"array","items":false,"minItems":100}"#))
    }

    func testUnsatisfiableSchemasHaveNoGrammar() throws {
        let schemas = [
            "false",
            #"{"type":"object","properties":{"a":false},"required":["a"]}"#,
            #"{"type":"string","minLength":5,"maxLength":2}"#,
            #"{"type":"string","minLength":100,"maxLength":70}"#,
            #"{"type":"array","minItems":3,"maxItems":1}"#,
            #"{"const":"x","enum":["y"]}"#,
            #"{"type":"string","const":1}"#,
            #"{"type":"object","required":["name"],"additionalProperties":false}"#,
            #"{"anyOf":[false,{"type":"string","enum":[1]}]}"#,
            #"{"type":"object","properties":{"inner":{"type":"array","items":false,"minItems":1}},"required":["inner"]}"#,
        ]
        for schema in schemas {
            XCTAssertNil(try grammar(schema), schema)
        }
        XCTAssertNotNil(try grammar(#"{"type":["string","object"],"required":["x"],"additionalProperties":false}"#),
                        "one satisfiable type is enough")
    }

    func testCombinators() throws {
        XCTAssertEqual(try rules(#"{"anyOf":[{"type":"string"},{"type":"integer"},{"type":"string"},false]}"#), #"""
            root ::= s1
            s1 ::= jstring | jinteger
            """#)
        XCTAssertEqual(try rules(#"{"oneOf":[{"type":"string"},{"type":"null"}]}"#), #"""
            root ::= s1
            s1 ::= jstring | jnull
            """#)
        XCTAssertEqual(try rootReference(#"{"anyOf":[false,{"type":"boolean"}]}"#), "jboolean")
        XCTAssertEqual(try rules(#"{"allOf":[{"type":"string","minLength":2}]}"#), #"""
            root ::= s1
            s1 ::= "\"" jchar{2,} "\"" ws
            """#)
        XCTAssertEqual(try rootReference(#"{"allOf":[{"type":"string"},{"maxLength":3}]}"#), "jvalue")
        XCTAssertEqual(try rootReference(#"{"type":"string","allOf":[{"minLength":2},{"maxLength":3}]}"#), "jstring")
        XCTAssertEqual(try rootReference(#"{"anyOf":[{"type":"string"}],"oneOf":[{"type":"integer"}]}"#), "jvalue")
        XCTAssertEqual(try rootReference(#"{"anyOf":[{"type":"string"}],"allOf":[{"type":"string"}]}"#), "jvalue")
        XCTAssertEqual(try rootReference(#"{"type":"integer","anyOf":[{"minimum":0},{"maximum":-5}]}"#), "jinteger")
    }

    func testCombinatorsNamingUndeclaredMembersAdmitAnyMembers() throws {
        let union = #"""
            {"type":"object","properties":{"kind":{"enum":["a","b"]}},"required":["kind"],
             "oneOf":[{"properties":{"a":{"type":"string"}},"required":["a"]},{"required":["b"]}]}
            """#
        XCTAssertEqual(try rootReference(union), "jobject")

        let declared = #"""
            {"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},
             "anyOf":[{"required":["a"]},{"required":["b"]}]}
            """#
        XCTAssertNotEqual(try rootReference(declared), "jobject")

        let closed = #"""
            {"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false,
             "anyOf":[{"required":["z"]}]}
            """#
        XCTAssertNotEqual(try rootReference(closed), "jobject", "closed objects cannot gain members anyway")
    }

    func testAdditionalPropertiesSchemaOnly() throws {
        XCTAssertEqual(try rules(#"{"type":"object","additionalProperties":{"type":"integer"}}"#), #"""
            root ::= s1
            s1 ::= "{" ws ( jstring ":" ws jinteger ( "," ws jstring ":" ws jinteger )* )? "}" ws
            """#)
        XCTAssertEqual(try rules(#"{"type":"object","additionalProperties":false}"#), #"""
            root ::= s1
            s1 ::= "{" ws "}" ws
            """#)
        XCTAssertEqual(try rules(#"{"additionalProperties":{"type":"string","minLength":2,"maxLength":1}}"#), #"""
            root ::= s1
            s1 ::= "{" ws "}" ws
            """#)
        XCTAssertEqual(try rootReference(#"{"type":"object","additionalProperties":true}"#), "jobject")
        XCTAssertEqual(try rootReference(#"{"type":"object"}"#), "jobject")
    }

    func testManyOptionalPropertiesFallBackToJobject() throws {
        func optionalOnly(_ count: Int) -> String {
            #"{"type":"object","properties":{"# + numbered(count) { #""p\#($0)":{"type":"string"}"# } + "}}"
        }
        XCTAssertEqual(try rootReference(optionalOnly(OutputSchemaGrammar.maxOptionalChain + 1)), "jobject")

        let chained = try rules(optionalOnly(OutputSchemaGrammar.maxOptionalChain))
        let count = OutputSchemaGrammar.maxOptionalChain
        XCTAssertEqual(chained.split(separator: "\n").count, 1 + count + (count - 1) + 1)   // root, pairs, tails, object

        let withRequired = #"{"type":"object","required":["p1"],"properties":{"#
            + numbered(20) { #""p\#($0)":{"type":"string"}"# } + "}}"
        XCTAssertNotEqual(try rootReference(withRequired), "jobject")
    }

    func testOversizedGrammarFallsBackToAnyValue() throws {
        let anyValue = "root ::= jvalue\n" + OutputSchemaGrammar.baseRules
        let manyRules = #"{"type":"object","properties":{"#
            + numbered(OutputSchemaGrammar.maxRules) { #""p\#($0)":{"const":\#($0)}"# } + #"},"required":["p1"]}"#
        XCTAssertEqual(try grammar(manyRules), anyValue)

        let longText = String(repeating: "x", count: 1_000)
        let manyBytes = #"{"enum":["# + numbered(70) { #""\#($0)\#(longText)""# } + "]}"
        XCTAssertEqual(try grammar(manyBytes), anyValue)

        let justFits = #"{"enum":["# + numbered(50) { #""\#($0)\#(longText)""# } + "]}"
        XCTAssertNotEqual(try grammar(justFits), anyValue)
    }

    func testSharedSubschemasShareARule() throws {
        let schema = #"""
            {"type":"object","required":["a","b","c"],
             "properties":{"a":{"type":"string","minLength":2},"b":{"type":"string","minLength":2,"title":"B"},
                           "c":{"type":"array","items":{"type":"string","minLength":2}}}}
            """#
        XCTAssertEqual(try rules(schema), #"""
            root ::= s6
            s1 ::= "\"" jchar{2,} "\"" ws
            s2 ::= "[" ws ( s1 ( "," ws s1 )* )? "]" ws
            s3 ::= "\"a\"" ws ":" ws s1
            s4 ::= "\"b\"" ws ":" ws s1
            s5 ::= "\"c\"" ws ":" ws s2
            s6 ::= "{" ws s3 "," ws s4 "," ws s5 "}" ws
            """#)
    }

    // MARK: Well-formedness

    private static let corpus = [
        personSchema,
        #"{"properties":{"a":{"type":"string"},"b":{"type":"integer"}}}"#,
        #"{"enum":["red","green","blue"]}"#,
        #"{"type":["string","null"],"maxLength":10}"#,
        #"{"anyOf":[{"type":"string","minLength":1},{"type":"array","items":{"type":"number"},"maxItems":4}]}"#,
        #"{}"#,
        "true",
        #"{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer","exclusiveMinimum":0}},"required":["id"]}}"#,
        #"""
        {"type":"object","required":["order"],"properties":{"order":{"type":"object","required":["lines","total"],
         "properties":{"lines":{"type":"array","minItems":1,"items":{"type":"object","required":["sku","qty"],
         "properties":{"sku":{"type":"string","minLength":3,"maxLength":12},"qty":{"type":"integer","minimum":1},
         "note":{"type":["string","null"]}}}},"total":{"type":"number","minimum":0},"currency":{"const":"EUR"}}}}}
        """#,
        #"{"type":"object","additionalProperties":{"type":"boolean"}}"#,
        #"{"oneOf":[{"const":"a\"b\\c\u0001"},{"enum":[1,2.5,-3e2,null,true]}]}"#,
    ]

    func testEveryGrammarIsWellFormed() throws {
        for schema in Self.corpus {
            lint(try XCTUnwrap(grammar(schema), schema), schema)
        }
    }

    func testLintFindsUndefinedRulesAndLargeBounds() {
        let scan = GBNFScan(body: #""a\"b" ( other-rule [x\]y]{65} | s2 )? jchar{0,64}"#)
        XCTAssertEqual(scan.references, ["other-rule", "s2", "jchar"])
        XCTAssertEqual(scan.repetitionBounds, [65, 0, 64])
        XCTAssertTrue(scan.isBalanced)
        XCTAssertFalse(GBNFScan(body: #"( "a""#).isBalanced)
        XCTAssertNil(GBNFLine("bad_name ::= x"[...]))
        XCTAssertNotNil(GBNFLine("good-name2 ::= x"[...]))
    }

    private func lint(_ text: String, _ schema: String) {
        XCTAssertTrue(text.hasSuffix(OutputSchemaGrammar.baseRules), schema)
        XCTAssertEqual(text.components(separatedBy: OutputSchemaGrammar.baseRules).count, 2, "base rules once: \(schema)")
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        let parsed = lines.compactMap(GBNFLine.init)
        XCTAssertEqual(parsed.count, lines.count, "every line is `name ::= body`: \(schema)")
        let names = parsed.map(\.name)
        XCTAssertEqual(Set(names).count, names.count, "rules defined once: \(schema)")
        XCTAssertEqual(names.filter { $0 == "root" }.count, 1, schema)
        let defined = Set(names)
        let scans = parsed.map { GBNFScan(body: $0.body) }
        let undefined = Set(scans.flatMap(\.references)).subtracting(defined)
        XCTAssertEqual(undefined, [], "undefined rules: \(schema)")
        let bounds = scans.flatMap(\.repetitionBounds)
        XCTAssertTrue(bounds.allSatisfy { $0 <= OutputSchemaGrammar.maxBoundedRepetition }, "\(bounds): \(schema)")
        XCTAssertTrue(scans.allSatisfy(\.isBalanced), "unbalanced literal, class or group: \(schema)")
    }
}

/// One `name ::= body` line; nil when the name is not a llama.cpp rule name.
private struct GBNFLine {
    let name: String
    let body: String

    init?(_ line: Substring) {
        guard let separator = line.range(of: " ::= ") else { return nil }
        let candidate = line[..<separator.lowerBound]
        guard !candidate.isEmpty, candidate.allSatisfy(GBNFScan.isWordCharacter) else { return nil }
        name = String(candidate)
        body = String(line[separator.upperBound...])
    }
}

/// Rule references and repetition counts in a body, skipping literals and character classes.
private struct GBNFScan {
    private(set) var references: [String] = []
    private(set) var repetitionBounds: [Int] = []
    private var mode = ScanMode.plain
    private var escaped = false
    private var word = ""
    private var depth = 0

    var isBalanced: Bool { depth == 0 && mode == .plain }

    static func isWordCharacter(_ character: Character) -> Bool {
        character.isASCII && (character.isLetter || character.isNumber || character == "-")
    }

    init(body: String) {
        for character in body { step(character) }
        flushReference()
    }

    private mutating func step(_ character: Character) {
        switch mode {
            case .literal, .characterClass:
                let closer: Character = mode == .literal ? "\"" : "]"
                if escaped { escaped = false }
                else if character == "\\" { escaped = true }
                else if character == closer { mode = .plain }
            case .braces:
                if character == "}" || character == "," { flushBound() } else { word.append(character) }
                if character == "}" { mode = .plain }
            case .plain:
                if Self.isWordCharacter(character) { return word.append(character) }
                flushReference()
                enterMode(after: character)
        }
    }

    private mutating func enterMode(after character: Character) {
        switch character {
            case "\"": mode = .literal
            case "[": mode = .characterClass
            case "{": mode = .braces
            case "(": depth += 1
            case ")": depth -= 1
            default: break
        }
    }

    private mutating func flushReference() {
        if !word.isEmpty { references.append(word) }
        word = ""
    }

    private mutating func flushBound() {
        if let bound = Int(word) { repetitionBounds.append(bound) }
        word = ""
    }

    private enum ScanMode {
        case plain, literal, characterClass, braces
    }
}
