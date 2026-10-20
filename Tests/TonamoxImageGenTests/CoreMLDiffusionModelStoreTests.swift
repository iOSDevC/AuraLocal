import XCTest
@testable import TonamoxImageGen

final class CoreMLDiffusionModelStoreTests: XCTestCase {
    private var suiteName = ""
    private var defaults: UserDefaults!
    private var records: CoreMLDiffusionModelStore!

    override func setUpWithError() throws {
        suiteName = "CoreMLDiffusionModelStoreTests-\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        records = CoreMLDiffusionModelStore(defaults: defaults)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suiteName)
        super.tearDown()
    }

    private static func model(_ name: String, folder: String = "compiled",
                              revision: String = "a") -> CoreMLDiffusionModel {
        CoreMLDiffusionModel(
            id: "hf:o/\(name)/\(folder)", displayName: name, repoID: "o/\(name)", revision: revision, folder: folder,
            components: ["Unet.mlmodelc/"], approximateDownloadBytes: 900_000_000, imageSide: 512,
            licenseName: "mit", licenseURL: URL(fileURLWithPath: "/"), preferredComputeUnits: .gpu,
            recommendedSteps: 4, recommendedGuidance: 1)
    }

    func testRememberAndForget() {
        let model = Self.model("dreamshaper")
        records.remember(model)
        XCTAssertEqual(records.model(id: model.id), model)
        XCTAssertEqual(records.all.map(\.id), [CoreMLDiffusionModel.stableDiffusion21Base.id, model.id])

        records.forget(model)
        XCTAssertNil(records.model(id: model.id))
        XCTAssertEqual(records.all, [.stableDiffusion21Base])
    }

    func testTheBuiltInIsListedFirstAndNeverStored() {
        records.remember(.stableDiffusion21Base)
        XCTAssertNil(defaults.dictionary(forKey: CoreMLDiffusionModelStore.defaultsKey))
        XCTAssertEqual(records.model(id: CoreMLDiffusionModel.stableDiffusion21Base.id), .stableDiffusion21Base)

        records.remember(Self.model("zeta"))
        records.remember(Self.model("alpha"))
        XCTAssertEqual(records.all.map(\.displayName),
                       [CoreMLDiffusionModel.stableDiffusion21Base.displayName, "alpha", "zeta"])

        records.forget(.stableDiffusion21Base)
        XCTAssertEqual(records.all.first, .stableDiffusion21Base)
    }

    func testANewerRevisionOfTheSameFolderReplacesTheRecord() {
        records.remember(Self.model("rv", revision: "old"))
        records.remember(Self.model("rv", revision: "new"))

        XCTAssertEqual(records.all.count, 2)
        XCTAssertEqual(records.model(id: "hf:o/rv/compiled")?.revision, "new")
    }

    func testConcurrentRememberDropsNothing() {
        let records = records!
        DispatchQueue.concurrentPerform(iterations: 40) { index in
            records.remember(Self.model("m\(index)"))
        }
        XCTAssertEqual(records.all.count, 41)
    }

    func testUndecodableRecordsAreSkipped() {
        defaults.set(["hf:o/broken/compiled": Data("{}".utf8)], forKey: CoreMLDiffusionModelStore.defaultsKey)
        XCTAssertEqual(records.all, [.stableDiffusion21Base])
        XCTAssertNil(records.model(id: "hf:o/broken/compiled"))
    }
}
