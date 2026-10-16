import XCTest
import AuraCore
@testable import AuraImageGen

final class CoreMLDiffusionLayoutTests: XCTestCase {
    private typealias Fixtures = CoreMLDiffusionFixtures

    func testAppleTwoFolderRepoGivesTwoFoldersSortedByPath() {
        let files = Fixtures.compiledFiles(in: "split_einsum_v2/compiled")
            + Fixtures.compiledFiles(in: "original/compiled")
            + [RepoFile(path: "README.md", sizeBytes: 100), RepoFile(path: "original_compiled.zip", sizeBytes: 9)]

        let folders = CoreMLDiffusionLayout.folders(in: files)

        XCTAssertEqual(folders.map(\.path), ["original/compiled", "split_einsum_v2/compiled"])
        XCTAssertEqual(folders[1].components,
                       ["TextEncoder.mlmodelc/", "Unet.mlmodelc/", "VAEDecoder.mlmodelc/", "vocab.json", "merges.txt"])
        XCTAssertEqual(folders[1].unetMetadataPath, "split_einsum_v2/compiled/Unet.mlmodelc/metadata.json")
        XCTAssertFalse(folders[1].isChunked)
        XCTAssertFalse(folders[1].isRepoRoot)
        XCTAssertNil(CoreMLDiffusionLayout.unsupportedReason(in: files), "a zip next to a folder does not matter")
    }

    func testChunksArePreferredOverTheWholeUnet() throws {
        let files = Fixtures.compiledFiles(in: "split_einsum/compiled",
                                           unet: ["Unet.mlmodelc", "UnetChunk1.mlmodelc", "UnetChunk2.mlmodelc"])

        let folder = try XCTUnwrap(CoreMLDiffusionLayout.folders(in: files).first)

        XCTAssertTrue(folder.isChunked)
        XCTAssertEqual(folder.components, ["TextEncoder.mlmodelc/", "UnetChunk1.mlmodelc/", "UnetChunk2.mlmodelc/",
                                           "VAEDecoder.mlmodelc/", "vocab.json", "merges.txt"])
        XCTAssertEqual(folder.unetMetadataPath, "split_einsum/compiled/UnetChunk1.mlmodelc/metadata.json")
        XCTAssertEqual(folder.unetBytes, 2 * 4 * 10, "both chunks, not the Unet")
        XCTAssertEqual(folder.downloadBytes, 4 * 4 * 10 + 2 * 10)
    }

    func testOptionalModelsAreNotDownloadedAndTheSafetyCheckerIsFlagged() throws {
        let files = Fixtures.compiledFiles(in: "Resources",
                                           extras: ["VAEEncoder.mlmodelc", "SafetyChecker.mlmodelc",
                                                    "ControlledUnet.mlmodelc"])
            + [RepoFile(path: "Resources/controlnet/Canny.mlmodelc/model.mil", sizeBytes: 10)]

        let folder = try XCTUnwrap(CoreMLDiffusionLayout.folders(in: files).first)

        XCTAssertTrue(folder.hasSafetyChecker)
        XCTAssertFalse(folder.components.contains { $0.hasPrefix("VAEEncoder") || $0.hasPrefix("SafetyChecker")
            || $0.hasPrefix("ControlledUnet") || $0.hasPrefix("controlnet") })
        XCTAssertEqual(folder.downloadBytes, 3 * 4 * 10 + 2 * 10)
    }

    func testANestedUnetIsNotAFolder() {
        let nested = Fixtures.compiledFiles(in: "split_einsum/Resources").map { file in
            RepoFile(path: file.path.replacingOccurrences(of: "Unet.mlmodelc/", with: "Unet.mlmodelc/Unet.mlmodelc/"),
                     sizeBytes: file.sizeBytes)
        }
        XCTAssertTrue(CoreMLDiffusionLayout.folders(in: nested).isEmpty)
    }

    func testTheRepoRootIsListedAndMarked() throws {
        let folder = try XCTUnwrap(CoreMLDiffusionLayout.folders(in: Fixtures.compiledFiles(in: "")).first)

        XCTAssertEqual(folder.path, "")
        XCTAssertTrue(folder.isRepoRoot)
        XCTAssertEqual(folder.unetMetadataPath, "Unet.mlmodelc/metadata.json")
    }

    func testEveryRequiredPieceIsChecked() {
        let complete = Fixtures.compiledFiles(in: "compiled")
        for missing in ["compiled/vocab.json", "compiled/merges.txt", "compiled/VAEDecoder.mlmodelc/model.mil",
                        "compiled/TextEncoder.mlmodelc/coremldata.bin", "compiled/Unet.mlmodelc/weights/weight.bin"] {
            XCTAssertTrue(CoreMLDiffusionLayout.folders(in: complete.filter { $0.path != missing }).isEmpty, missing)
        }
        let oneChunk = Fixtures.compiledFiles(in: "compiled", unet: ["UnetChunk1.mlmodelc"])
        XCTAssertTrue(CoreMLDiffusionLayout.folders(in: oneChunk).isEmpty, "a single chunk is not a Unet")
    }

    func testTextEncoder2OnlyIsMarkedSDXL() throws {
        let files = Fixtures.compiledFiles(in: "compiled", encoders: ["TextEncoder2.mlmodelc"])
        let folder = try XCTUnwrap(CoreMLDiffusionLayout.folders(in: files).first)

        XCTAssertTrue(folder.hasOnlyTextEncoder2)
        XCTAssertEqual(folder.components.first, "TextEncoder2.mlmodelc/")
    }

    func testSizesAreSummedOrUnknown() throws {
        let sized = Fixtures.compiledFiles(in: "compiled", size: 7)
        let folder = try XCTUnwrap(CoreMLDiffusionLayout.folders(in: sized).first)
        XCTAssertEqual(folder.downloadBytes, 3 * 4 * 7 + 2 * 7)
        XCTAssertEqual(folder.unetBytes, 4 * 7)

        let unsized = try XCTUnwrap(CoreMLDiffusionLayout.folders(in: Fixtures.compiledFiles(in: "compiled",
                                                                                             size: nil)).first)
        XCTAssertNil(unsized.downloadBytes)
        XCTAssertNil(unsized.unetBytes)
    }

    func testTheMostSpecificReasonIsGivenWhenNoFolderQualifies() {
        let cases: [(path: String, reason: String)] = [
            ("models/dreamshaper_split-einsum.zip", "Core ML archives (zip) are not supported yet"),
            ("Unet.mlpackage/Manifest.json", "Uncompiled Core ML packages are not supported yet"),
            ("unet/coreml_model.mlmodelc/model.mil", "Diffusers-style Core ML layout is not supported"),
            ("FLUX.aimodel/weights.bin", "Core AI models need iOS 27; AuraImageGen uses Core ML"),
            ("model.safetensors", CoreMLDiffusionLayout.genericReason),
        ]
        let reasons = cases.map { CoreMLDiffusionLayout.unsupportedReason(in: [RepoFile(path: $0.path, sizeBytes: 1)]) }
        XCTAssertEqual(reasons, cases.map { $0.reason as String? })
    }

    func testTheRecordedAppleListingHasTwoFoldersWithExactSizes() throws {
        let listing = try XCTUnwrap(HFRepoInfo.parse(try Fixtures.data(Fixtures.sd15Listing)))
        let folders = CoreMLDiffusionLayout.folders(in: listing.files)

        XCTAssertEqual(folders.map(\.path), ["original/compiled", "split_einsum_v2/compiled"])
        XCTAssertEqual(folders.map(\.downloadBytes), [886_784_300, 888_856_623])
        XCTAssertEqual(folders.map(\.hasSafetyChecker), [true, true])
        XCTAssertEqual(listing.revision, Fixtures.sd15Revision)
    }
}
