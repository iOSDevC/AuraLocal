import XCTest
@testable import TonamoxImageGen

final class CoreMLUnetFactsTests: XCTestCase {
    private typealias Fixtures = CoreMLDiffusionFixtures

    func testStringShapesFromARecordedUnet() throws {
        let facts = try Fixtures.unet(Fixtures.sd21Split)

        XCTAssertEqual(facts.inputShapes["sample"], [2, 4, 64, 64])
        XCTAssertEqual(facts.inputShapes["timestep"], [2])
        XCTAssertEqual(facts.inputShapes["encoder_hidden_states"], [2, 1024, 1, 77])
        XCTAssertEqual(facts.imageSide, 512)
        XCTAssertEqual(facts.textEmbeddingWidth, 1024)
        XCTAssertEqual(facts.version, "stabilityai/stable-diffusion-2-1-base")
    }

    func testIntegerArrayShapes() throws {
        let json = #"[{"inputSchema": [{"name": "sample", "shape": [2, 4, 48, 48]},"#
            + #"{"name": "encoder_hidden_states", "shape": [2, 768, 1, 77]}]}]"#
        let facts = try XCTUnwrap(CoreMLUnetFacts.parse(Data(json.utf8)))

        XCTAssertEqual(facts.imageSide, 384)
        XCTAssertEqual(facts.textEmbeddingWidth, 768)
        XCTAssertFalse(facts.usesSplitEinsum, "no histogram, no einsum")
        XCTAssertNil(facts.highestIOSOpSet)
    }

    func testSplitEinsumAndOriginalAttention() throws {
        XCTAssertTrue(try Fixtures.unet(Fixtures.sd21Split).usesSplitEinsum)
        XCTAssertFalse(try Fixtures.unet(Fixtures.sd21Original).usesSplitEinsum)
        XCTAssertFalse(try Fixtures.unet(Fixtures.sd15Original).usesSplitEinsum)
    }

    func testStoragePrecision() throws {
        XCTAssertEqual(try Fixtures.unet(Fixtures.sd21Split).storagePrecision, "Mixed (Float16, Palettized (6 bits))")
        XCTAssertEqual(try Fixtures.unet(Fixtures.sd21Fp16Chunk).storagePrecision, "Float16")
    }

    func testStatefulness() throws {
        XCTAssertTrue(try Fixtures.unet(Fixtures.stateful).isStateful)
        XCTAssertFalse(try Fixtures.unet(Fixtures.turboBatchOne).isStateful, "an empty stateSchema is not stateful")
        XCTAssertFalse(try Fixtures.unet(Fixtures.sd21Split).isStateful)
    }

    func testAvailabilityAndOpSet() throws {
        let apple = try Fixtures.unet(Fixtures.sd21Split)
        XCTAssertEqual(apple.minimumOS["iOS"], 16)
        XCTAssertEqual(apple.minimumOS["macOS"], 13)
        XCTAssertEqual(apple.highestIOSOpSet, 16)

        let clover = try Fixtures.unet(Fixtures.stateful)
        XCTAssertEqual(clover.minimumOS["iOS"], 18)
        XCTAssertEqual(clover.highestIOSOpSet, 18)
    }

    func testGarbageIsNotParsed() {
        let texts = ["not json", "[]", "[{}]", #"{"a": 1}"#, #"[{"inputSchema": 3}]"#]
        let parsed = texts.map { CoreMLUnetFacts.parse(Data($0.utf8)) }
        XCTAssertTrue(parsed.allSatisfy { $0 == nil }, "\(parsed)")
    }
}
