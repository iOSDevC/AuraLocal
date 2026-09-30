import XCTest
@testable import AuraCore

final class OutputSchemaValidatorTests: XCTestCase {

    private func compile(_ schema: String) throws -> OutputSchemaValidator {
        try OutputSchemaValidator(.json(schema))
    }

    private func compileError(_ schema: String) -> OutputSchemaError? {
        do throws(OutputSchemaError) {
            _ = try OutputSchemaValidator(.json(schema))
            return nil
        } catch {
            return error
        }
    }

    private func unsupportedPointers(_ schema: String) -> [String]? {
        guard case .unsupportedKeywords(let pointers) = compileError(schema) else { return nil }
        return pointers
    }

    private func isInvalidSchema(_ schema: String) -> Bool {
        if case .invalidSchema = compileError(schema) { return true }
        return false
    }

    private func paths(_ found: [OutputSchemaViolation]) -> [String] { found.map(\.path) }

    // MARK: Keywords

    func testTypeAcceptsSingleNamesAndArrays() throws {
        let text = try compile(#"{"type":"string"}"#)
        XCTAssertEqual(text.validate(#""a""#), [])
        XCTAssertEqual(text.validate("1"), [OutputSchemaViolation(path: "", message: "expected string, got integer")])

        let nullable = try compile(#"{"type":["string","null"]}"#)
        XCTAssertEqual(nullable.validate("null"), [])
        XCTAssertEqual(nullable.validate("1.5").first?.message, "expected string or null, got number")

        let integer = try compile(#"{"type":"integer"}"#)
        XCTAssertEqual(integer.validate("3"), [])
        XCTAssertEqual(integer.validate("3.0"), [])
        XCTAssertEqual(paths(integer.validate("3.5")), [""])

        let number = try compile(#"{"type":"number"}"#)
        XCTAssertEqual(number.validate("3"), [])
        XCTAssertEqual(number.validate("true").first?.message, "expected number, got boolean")

        let boolean = try compile(#"{"type":"boolean"}"#)
        XCTAssertEqual(boolean.validate("false"), [])
        XCTAssertEqual(paths(boolean.validate("1")), [""])
        XCTAssertEqual(try compile(#"{"type":"object"}"#).validate("[]").first?.message, "expected object, got array")
    }

    func testPropertiesRequiredAndAdditionalProperties() throws {
        let person = try compile(#"""
            {"type":"object",
             "properties":{"name":{"type":"string"},"age":{"type":"integer"}},
             "required":["name","age"],
             "additionalProperties":false}
            """#)
        XCTAssertEqual(person.validate(#"{"name":"Ana","age":30}"#), [])

        let found = person.validate(#"{"name":1,"extra":true}"#)
        XCTAssertEqual(found, [
            OutputSchemaViolation(path: "", message: #"missing required property "age""#),
            OutputSchemaViolation(path: "/extra", message: "property not allowed"),
            OutputSchemaViolation(path: "/name", message: "expected string, got integer"),
        ])

        let numbers = try compile(#"{"properties":{"id":{"type":"string"}},"additionalProperties":{"type":"number"}}"#)
        XCTAssertEqual(numbers.validate(#"{"id":"x","a":1}"#), [])
        XCTAssertEqual(paths(numbers.validate(#"{"id":"x","a":1,"b":"no"}"#)), ["/b"])

        XCTAssertEqual(try compile(#"{"properties":{"a":{"type":"string"}}}"#).validate(#"{"z":1}"#), [])
    }

    func testItemsAndArrayBounds() throws {
        let list = try compile(#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}"#)
        XCTAssertEqual(list.validate("[1,2]"), [])
        XCTAssertEqual(list.validate("[]").first?.message, "must have at least 1 items, got 0")
        XCTAssertEqual(list.validate("[1,2,3]").first?.message, "must have at most 2 items, got 3")
        XCTAssertEqual(paths(list.validate(#"[1,"a"]"#)), ["/1"])
    }

    func testEnumAndConst() throws {
        let color = try compile(#"{"enum":["red","green",null,1]}"#)
        XCTAssertEqual(color.validate(#""red""#), [])
        XCTAssertEqual(color.validate("null"), [])
        XCTAssertEqual(color.validate("1.0"), [])
        XCTAssertEqual(color.validate("true").first?.message, #"must be one of "red", "green", null, 1"#)

        let fixed = try compile(#"{"const":{"a":[1,2]}}"#)
        XCTAssertEqual(fixed.validate(#"{"a":[1,2]}"#), [])
        XCTAssertEqual(fixed.validate(#"{"a":[2,1]}"#).first?.message, #"must equal {"a":[1,2]}"#)
    }

    func testStringLengthCountsUnicodeScalarsAndOnlyAppliesToStrings() throws {
        let short = try compile(#"{"minLength":2,"maxLength":3}"#)
        XCTAssertEqual(short.validate(#""ab""#), [])
        XCTAssertEqual(short.validate(#""é🙂""#), [])
        XCTAssertEqual(short.validate(#""a""#).first?.message, "must be at least 2 characters, got 1")
        XCTAssertEqual(short.validate(#""abcd""#).first?.message, "must be at most 3 characters, got 4")
        XCTAssertEqual(short.validate("5"), [])
    }

    func testNumericBounds() throws {
        let inclusive = try compile(#"{"minimum":1,"maximum":10}"#)
        XCTAssertEqual(inclusive.validate("1"), [])
        XCTAssertEqual(inclusive.validate("10"), [])
        XCTAssertEqual(inclusive.validate("0").first?.message, "must be >= 1, got 0")
        XCTAssertEqual(inclusive.validate("10.5").first?.message, "must be <= 10, got 10.5")

        let exclusive = try compile(#"{"exclusiveMinimum":0,"exclusiveMaximum":1}"#)
        XCTAssertEqual(exclusive.validate("0.5"), [])
        XCTAssertEqual(exclusive.validate("0").first?.message, "must be > 0, got 0")
        XCTAssertEqual(exclusive.validate("1").first?.message, "must be < 1, got 1")
        XCTAssertEqual(exclusive.validate(#""text""#), [])
    }

    func testCombinators() throws {
        let either = try compile(#"{"anyOf":[{"type":"string"},{"type":"integer"}]}"#)
        XCTAssertEqual(either.validate(#""a""#), [])
        XCTAssertEqual(either.validate("1"), [])
        let anyMiss = try XCTUnwrap(either.validate("true").first)
        XCTAssertEqual(anyMiss.path, "")
        XCTAssertTrue(anyMiss.message.hasPrefix("must match at least one anyOf schema"))
        XCTAssertTrue(anyMiss.message.contains("#1: (root): expected integer, got boolean"))

        let exactlyOne = try compile(#"{"oneOf":[{"type":"integer"},{"minimum":0}]}"#)
        XCTAssertEqual(exactlyOne.validate("-1"), [])
        XCTAssertEqual(exactlyOne.validate("2.5"), [])
        XCTAssertEqual(exactlyOne.validate("5").first?.message, "must match exactly one oneOf schema, matched 2 (#0, #1)")
        XCTAssertTrue(try XCTUnwrap(exactlyOne.validate("-2.5").first).message.contains("matched none"))

        let both = try compile(#"{"allOf":[{"type":"integer"},{"minimum":3}]}"#)
        XCTAssertEqual(both.validate("4"), [])
        XCTAssertEqual(both.validate("2"), [OutputSchemaViolation(path: "", message: "must be >= 3, got 2")])
    }

    func testBooleanSchemasAndIgnoredAnnotations() throws {
        XCTAssertEqual(try compile("true").validate(#"{"anything":[1]}"#), [])
        XCTAssertEqual(try compile("false").validate("1").first?.message, "no value is allowed here")
        XCTAssertEqual(paths(try compile(#"{"properties":{"x":false}}"#).validate(#"{"x":1}"#)), ["/x"])

        let annotated = try compile(#"""
            {"$schema":"https://json-schema.org/draft/2020-12/schema","title":"T","description":"d",
             "examples":[1],"default":1,"$comment":"c","type":"integer"}
            """#)
        XCTAssertEqual(annotated.validate("2"), [])
        XCTAssertEqual(paths(annotated.validate(#""x""#)), [""])
    }

    func testViolationPathsNestAndEscapePointerTokens() throws {
        let nested = try compile(#"{"properties":{"a/b~c":{"items":{"type":"string"}}}}"#)
        XCTAssertEqual(paths(nested.validate(#"{"a/b~c":["x",1]}"#)), ["/a~1b~0c/1"])
    }

    func testValidateReportsTextThatIsNotJSON() throws {
        XCTAssertEqual(try compile("{}").validate("nope"), [OutputSchemaViolation(path: "", message: "not valid JSON")])
    }

    // MARK: Rejected schemas

    func testUnsupportedKeywordsAreListedAsPointers() {
        let schema = #"""
            {"$ref":"#/$defs/a",
             "anyOf":[{"if":{"type":"string"}}],
             "items":{"pattern":"^a"},
             "properties":{"email":{"type":"string","format":"email"},"pattern":{"type":"string"},
                           "a/b":{"patternProperties":{}}}}
            """#
        XCTAssertEqual(compileError(schema), .unsupportedKeywords([
            "/$ref", "/anyOf/0/if", "/items/pattern", "/properties/a~1b/patternProperties", "/properties/email/format",
        ]))
    }

    func testEveryKeywordOutsideTheSubsetIsRejected() {
        let rejected = ["pattern", "format", "patternProperties", "if", "then", "else", "not", "$defs", "$id",
                        "prefixItems", "uniqueItems", "multipleOf", "minProperties", "dependentRequired", "contains"]
        for keyword in rejected {
            XCTAssertEqual(unsupportedPointers(#"{"\#(keyword)":{}}"#), ["/\(keyword)"], keyword)
        }
    }

    func testGBNFAndNonSchemaDocumentsAreRejected() {
        guard case .invalidSchema(let detail) = compileError(#"root ::= "yes" | "no""#) else {
            return XCTFail("a GBNF grammar must not compile")
        }
        XCTAssertTrue(detail.contains("GBNF"))
        XCTAssertTrue(isInvalidSchema("not json"))
        XCTAssertTrue(isInvalidSchema("42"))
        XCTAssertTrue(isInvalidSchema("[]"))
        XCTAssertTrue(isInvalidSchema(#"{"properties":{"a":"string"}}"#))
    }

    func testMalformedKeywordValuesAreRejected() {
        let malformed = [
            #"{"type":"date"}"#, #"{"type":[]}"#, #"{"required":"name"}"#, #"{"required":[1]}"#,
            #"{"minLength":-1}"#, #"{"minItems":1.5}"#, #"{"maxLength":"3"}"#, #"{"items":[{}]}"#,
            #"{"enum":[]}"#, #"{"enum":"a"}"#, #"{"anyOf":[]}"#, #"{"oneOf":{}}"#, #"{"exclusiveMinimum":true}"#,
            #"{"properties":[]}"#, #"{"additionalProperties":"no"}"#,
        ]
        for schema in malformed {
            XCTAssertTrue(isInvalidSchema(schema), schema)
        }
    }

    // MARK: Extraction

    func testExtractsBareValues() {
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: "  {\"a\":1}\n"), #"{"a":1}"#)
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: "42"), "42")
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: #""hi""#), #""hi""#)
    }

    func testExtractsFencedBlocks() {
        let fenced = "Here you go:\n```json\n{\"a\": 1}\n```\nAnything else?"
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: fenced), #"{"a": 1}"#)
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: "```JSON\r\n[1,2]\r\n```"), "[1,2]")
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: "```\n\"plain\"\n```"), #""plain""#)
        let otherLanguage = "```swift\nlet x = 1\n```\nThen {\"a\":1}"
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: otherLanguage), #"{"a":1}"#)
    }

    func testExtractsFirstBalancedValueSkippingStrings() {
        let braces = #"The answer is {"text": "use } and { freely", "n": 2}. Done."#
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: braces), #"{"text": "use } and { freely", "n": 2}"#)
        let escaped = #"Quote: {"q": "say \"}\" now"} ok"#
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: escaped), #"{"q": "say \"}\" now"}"#)
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: #"Set {x} then {"ok": true}"#), #"{"ok": true}"#)
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: "Result: [1, 2, 3] as asked"), "[1, 2, 3]")
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: #"{"a":1} {"b":2}"#), #"{"a":1}"#)
        XCTAssertEqual(OutputSchemaValidator.extractJSON(from: #"Résumé: {"nombre": "José 🙂"}!"#), #"{"nombre": "José 🙂"}"#)
    }

    func testExtractionFindsNothingInProseOrBrokenJSON() {
        XCTAssertNil(OutputSchemaValidator.extractJSON(from: "I cannot help with that."))
        XCTAssertNil(OutputSchemaValidator.extractJSON(from: #"{"a": 1"#))
        XCTAssertNil(OutputSchemaValidator.extractJSON(from: #"{"a": [1}"#))
        XCTAssertNil(OutputSchemaValidator.extractJSON(from: ""))
    }

    func testConformingJSONExtractsThenValidates() throws {
        let person = try compile(#"{"type":"object","required":["name"]}"#)
        XCTAssertEqual(try person.conformingJSON(in: "Sure: {\"name\":\"Ana\"}"), #"{"name":"Ana"}"#)

        XCTAssertThrowsError(try person.conformingJSON(in: "No idea.")) { error in
            XCTAssertEqual(error as? OutputSchemaError, .violations(
                output: "No idea.",
                violations: [OutputSchemaViolation(path: "", message: "no JSON value found in the reply")]))
        }
        XCTAssertThrowsError(try person.conformingJSON(in: "{}")) { error in
            guard case .violations(let output, let found) = error as? OutputSchemaError else {
                return XCTFail("expected violations, got \(error)")
            }
            XCTAssertEqual(output, "{}")
            XCTAssertEqual(found.map(\.description), [#"(root): missing required property "name""#])
        }
    }
}
