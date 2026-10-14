import XCTest
@testable import AuraImageGen

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
