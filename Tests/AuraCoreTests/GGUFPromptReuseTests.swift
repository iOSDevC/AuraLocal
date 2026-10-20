import XCTest
@testable import AuraCore

final class GGUFPromptReuseTests: XCTestCase {

    private static let chatML = GGUFPromptReuse(boundaries: ["<|im_start|>", "<|im_end|>", "<|endoftext|>"])
    private static let resident = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!"

    func testAPromptThatGoesOnAtASpecialTokenDecodesOnlyTheRest() {
        let rest = "<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n"
        XCTAssertEqual(Self.chatML.suffix(of: Self.resident + rest, after: Self.resident), rest)
    }

    func testAPromptThatDivergesOrDoesNotGoOnMisses() {
        let reuse = Self.chatML
        XCTAssertNil(reuse.suffix(of: Self.resident.replacingOccurrences(of: "Hi", with: "Ho") + "<|im_end|>",
                                  after: Self.resident), "an earlier turn changed")
        XCTAssertNil(reuse.suffix(of: Self.resident, after: Self.resident), "nothing new to decode")
        XCTAssertNil(reuse.suffix(of: String(Self.resident.dropLast()), after: Self.resident), "a shorter prompt")
    }

    func testTheRestMustStartWithASpecialToken() {
        let reuse = Self.chatML
        XCTAssertNil(reuse.suffix(of: Self.resident + " And more.<|im_end|>", after: Self.resident),
                     "text right after the reply could merge with its last token")
        XCTAssertNil(reuse.suffix(of: Self.resident + "\n<|im_end|>", after: Self.resident))
        XCTAssertNil(reuse.suffix(of: Self.resident + "<|im_", after: Self.resident), "half a special token")
    }

    func testTheComparisonIsByteForByte() {
        let composed = "<|im_start|>user\nJos\u{E9}"
        let decomposed = "<|im_start|>user\nJose\u{301}<|im_end|>"
        XCTAssertTrue(decomposed.hasPrefix(composed), "String comparison treats both spellings as equal")
        XCTAssertNil(Self.chatML.suffix(of: decomposed, after: composed), "their tokens differ")
    }

    func testWithoutAResidentOrBoundariesNothingIsReused() {
        let prompt = Self.resident + "<|im_end|>"
        XCTAssertNil(Self.chatML.suffix(of: prompt, after: ""))
        XCTAssertNil(GGUFPromptReuse(boundaries: []).suffix(of: prompt, after: Self.resident))
        XCTAssertEqual(GGUFPromptReuse(boundaries: ["", "<|im_end|>"]).boundaries, ["<|im_end|>"])
    }

    // MARK: - Special-token texts from the GGUF

    func testControlTokenTextsComeFromTheTokenTypes() throws {
        // llama.cpp token types: 1 normal, 2 unknown, 3 control, 4 user-defined, 6 byte.
        let data = GGUFFixture.make(strings: [("tokenizer.ggml.model", "gpt2")], ids: [],
                                    tokens: ["<unk>", "a", "<|im_end|>", "<tool_call>", "<0x0A>", ""],
                                    extras: [("tokenizer.ggml.token_type", .int32s([2, 1, 3, 4, 6, 3]))])
        XCTAssertEqual(try GGUFChatMetadata.parse(data).controlTokenTexts, ["<unk>", "<|im_end|>", "<tool_call>"])
    }

    func testWithoutMatchingTokenTypesThereAreNoControlTokenTexts() throws {
        let tokens = ["<unk>", "<|im_end|>"]
        let none = GGUFFixture.make(strings: [], ids: [], tokens: tokens)
        XCTAssertEqual(try GGUFChatMetadata.parse(none).controlTokenTexts, [])
        let short = GGUFFixture.make(strings: [], ids: [], tokens: tokens,
                                     extras: [("tokenizer.ggml.token_type", .int32s([3]))])
        XCTAssertEqual(try GGUFChatMetadata.parse(short).controlTokenTexts, [], "one type per token or none")
    }
}
