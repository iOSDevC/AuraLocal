import XCTest
import TonamoxCore
@testable import TonamoxImageGen

final class CoreMLDiffusionModelTests: XCTestCase {
    private var directories: [URL] = []

    override func tearDown() {
        directories.forEach { try? FileManager.default.removeItem(at: $0) }
        super.tearDown()
    }

    func testPresetIsPinned() {
        let model = CoreMLDiffusionModel.stableDiffusion21Base
        XCTAssertEqual(model.id, "sd-2.1-base-palettized")
        XCTAssertEqual(model.repoID, "apple/coreml-stable-diffusion-2-1-base-palettized")
        XCTAssertEqual(model.revision, "dca5f2fda9c5143c71aded72bbb4ce39f3c2cc9c")
        XCTAssertEqual(model.folder, "split_einsum_v2/compiled")
        XCTAssertEqual(model.components,
                       ["TextEncoder.mlmodelc/", "Unet.mlmodelc/", "VAEDecoder.mlmodelc/", "vocab.json", "merges.txt"])
        XCTAssertFalse(model.components.contains { $0.hasPrefix("VAEEncoder") })
        XCTAssertEqual(model.imageSide, 512)
        XCTAssertEqual(model.approximateDownloadBytes, 1_073_000_000)
        XCTAssertEqual(model.licenseName, "CreativeML Open RAIL++-M")
        XCTAssertEqual(model.licenseURL.host, "huggingface.co")
        XCTAssertEqual(model.preferredComputeUnits, .neuralEngine)
        XCTAssertEqual(model.recommendedSteps, 20)
        XCTAssertEqual(model.recommendedGuidance, 7.5)
        XCTAssertTrue(model.isBuiltIn)
    }

    // MARK: Adopted from Hugging Face

    private func variant(_ fixture: String = CoreMLDiffusionFixtures.sd15Original,
                         folder: CoreMLDiffusionFolder = CoreMLDiffusionFixtures.folder("original/compiled",
                                                                                         bytes: 886_784_300),
                         listing: HFRepoInfo = CoreMLDiffusionFixtures.listing(
                            "apple/coreml-stable-diffusion-v1-5-palettized", tags: ["coreml", "sd-turbo"],
                            license: "creativeml-openrail-m",
                            revision: CoreMLDiffusionFixtures.sd15Revision)) throws -> CoreMLDiffusionVariant {
        CoreMLDiffusionRules.evaluate(folder, unet: try CoreMLDiffusionFixtures.unet(fixture), listing: listing,
                                      on: .iPhone17Pro, osMajor: 26)
    }

    func testInitFromAVariantMapsEveryField() throws {
        let source = try variant()
        XCTAssertEqual(source.verdict, .runnableWithCaveats)

        let model = try CoreMLDiffusionModel(variant: source)

        XCTAssertEqual(model.id, "hf:apple/coreml-stable-diffusion-v1-5-palettized/original/compiled")
        XCTAssertEqual(model.displayName, "coreml-stable-diffusion-v1-5-palettized")
        XCTAssertEqual(model.repoID, "apple/coreml-stable-diffusion-v1-5-palettized")
        XCTAssertEqual(model.revision, CoreMLDiffusionFixtures.sd15Revision)
        XCTAssertEqual(model.folder, "original/compiled")
        XCTAssertEqual(model.components, source.folder.components)
        XCTAssertEqual(model.approximateDownloadBytes, 886_784_300)
        XCTAssertEqual(model.imageSide, 512)
        XCTAssertEqual(model.licenseName, "creativeml-openrail-m")
        XCTAssertEqual(model.licenseURL.absoluteString,
                       "https://huggingface.co/apple/coreml-stable-diffusion-v1-5-palettized")
        XCTAssertEqual(model.preferredComputeUnits, .gpu)
        XCTAssertEqual(model.recommendedSteps, 4, "the sd-turbo tag marks it distilled")
        XCTAssertEqual(model.recommendedGuidance, 1)
        XCTAssertFalse(model.isBuiltIn)
    }

    func testInitThrowsUnlessTheVariantRunsWithKnownSizeAndSide() throws {
        let blocked = try variant(CoreMLDiffusionFixtures.turboBatchOne)
        let unknown = CoreMLDiffusionRules.evaluate(CoreMLDiffusionFixtures.folder(), unet: nil,
                                                    listing: CoreMLDiffusionFixtures.listing(), on: .iPhone17Pro,
                                                    osMajor: 26)
        let runs = try variant()
        let sideless = CoreMLDiffusionVariant(
            repoID: runs.repoID, revision: runs.revision, folder: runs.folder, unet: nil, licenseName: "mit",
            licenseURL: runs.licenseURL, findings: [], verdict: .runnable, fit: .excellent,
            preferredComputeUnits: .neuralEngine, recommendedSteps: 20, recommendedGuidance: 7.5)
        let unsized = CoreMLDiffusionVariant(
            repoID: runs.repoID, revision: runs.revision, folder: CoreMLDiffusionFixtures.folder(bytes: nil),
            unet: runs.unet, licenseName: "mit", licenseURL: runs.licenseURL, findings: [], verdict: .runnable,
            fit: .excellent, preferredComputeUnits: .neuralEngine, recommendedSteps: 20, recommendedGuidance: 7.5)
        XCTAssertEqual(blocked.verdict, .notRunnable)
        XCTAssertEqual(unknown.verdict, .unknown)

        let sources = [blocked, unknown, sideless, unsized]
        let outcomes = sources.map { source in Result { try CoreMLDiffusionModel(variant: source) } }
        for (source, outcome) in zip(sources, outcomes) {
            guard case .failure(let error) = outcome,
                  case .notRunnable(let repoID, let folder, let reasons)? = error as? CoreMLDiffusionModelError else {
                XCTFail("\(source.id): \(outcome)")
                continue
            }
            XCTAssertEqual(repoID, source.repoID)
            XCTAssertEqual(folder, source.folder.path)
            XCTAssertFalse(reasons.isEmpty)
        }
    }

    func testTheBuiltInModelsOwnFolderAdoptsAsTheBuiltInModel() throws {
        let own = try variant(CoreMLDiffusionFixtures.sd21Split, folder: CoreMLDiffusionFixtures.folder(),
                              listing: CoreMLDiffusionFixtures.listing())
        let sibling = try variant(CoreMLDiffusionFixtures.sd21Original,
                                  folder: CoreMLDiffusionFixtures.folder("original/compiled"),
                                  listing: CoreMLDiffusionFixtures.listing())
        XCTAssertTrue(own.verdict.isRunnable)
        XCTAssertTrue(sibling.verdict.isRunnable)

        XCTAssertEqual(try CoreMLDiffusionModel(variant: own), .stableDiffusion21Base)
        XCTAssertFalse(try CoreMLDiffusionModel(variant: sibling).isBuiltIn)
    }

    func testCodableRoundTrip() throws {
        let adopted = try CoreMLDiffusionModel(variant: try variant())
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        for model in [adopted, .stableDiffusion21Base] {
            XCTAssertEqual(try decoder.decode(CoreMLDiffusionModel.self, from: try encoder.encode(model)), model)
        }
    }

    func testDeleteRemovesOnlyItsFolder() throws {
        // The public delete targets the real Caches root, so the repo id is unique and removed afterwards.
        let repoID = "auralocal-tests/delete-\(UUID().uuidString.lowercased())"
        let adopted = try CoreMLDiffusionModel(variant: try variant(listing: CoreMLDiffusionFixtures.listing(
            repoID, license: "mit", revision: CoreMLDiffusionFixtures.sd15Revision)))
        let removed = ModelDownloads.localFolder(repoID: repoID, folder: adopted.folder)
        let sibling = ModelDownloads.localFolder(repoID: repoID, folder: "split_einsum_v2/compiled")
        let repoFolder = removed.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        defer { try? FileManager.default.removeItem(at: repoFolder) }
        try FileManager.default.createDirectory(at: removed, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: sibling, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: removed.appending(path: "vocab.json"))
        try Data("x".utf8).write(to: sibling.appending(path: "vocab.json"))
        XCTAssertTrue(repoFolder.lastPathComponent.hasPrefix("models--auralocal-tests--delete-"))

        try adopted.delete()

        XCTAssertFalse(FileManager.default.fileExists(atPath: removed.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: sibling.appending(path: "vocab.json").path))
    }

    func testResourcesLiveInTheRepoSnapshotFolder() {
        let path = CoreMLDiffusionModel.stableDiffusion21Base.resourcesDirectory.standardizedFileURL.path
        let expected = "huggingface/hub/models--apple--coreml-stable-diffusion-2-1-base-palettized"
            + "/snapshots/main/split_einsum_v2/compiled"
        XCTAssertTrue(path.hasSuffix(expected), path)
    }

    func testDownloadedNeedsTheMarkerAndEveryComponent() throws {
        let model = CoreMLDiffusionModel.stableDiffusion21Base
        let complete = try fixture()
        XCTAssertTrue(model.isDownloaded(at: complete))

        XCTAssertFalse(model.isDownloaded(at: try fixture(marker: false)))
        for component in model.components {
            XCTAssertFalse(model.isDownloaded(at: try fixture(omitting: component)), component)
        }
    }

    func testAFolderFinishedAtAnotherRevisionIsNotDownloaded() throws {
        let model = CoreMLDiffusionModel.stableDiffusion21Base
        XCTAssertFalse(model.isDownloaded(at: try fixture(markerRevision: "0000000")))
        XCTAssertFalse(model.isDownloaded(at: try fixture(markerRevision: "")))
    }

    private func fixture(omitting: String? = nil, marker: Bool = true, markerRevision: String? = nil) throws -> URL {
        let directory = try DiffusionFixtures.downloadedResources(omitting: omitting, marker: marker,
                                                                  markerRevision: markerRevision)
        directories.append(directory)
        return directory
    }
}
