import XCTest
import AuraCore
@testable import AuraImageGen

final class CoreMLDiffusionRulesTests: XCTestCase {
    private typealias Fixtures = CoreMLDiffusionFixtures

    private func evaluate(_ fixture: String?, folder: CoreMLDiffusionFolder = Fixtures.folder(),
                          listing: HFRepoInfo = Fixtures.listing(), on target: DevicePreset = .iPhone17Pro,
                          osMajor: Int = 26) throws -> CoreMLDiffusionVariant {
        let unet = try fixture.map { try Fixtures.unet($0) }
        return CoreMLDiffusionRules.evaluate(folder, unet: unet, listing: listing, on: target, osMajor: osMajor)
    }

    private func rules(_ variant: CoreMLDiffusionVariant, _ level: FindingSeverity) -> [String] {
        variant.findings.filter { $0.level == level }.map(\.rule)
    }

    // MARK: Runs

    func testSD21SixBitSplitEinsumRunsOnTheNeuralEngine() throws {
        let variant = try evaluate(Fixtures.sd21Split)

        XCTAssertEqual(variant.verdict, .runnable, "\(variant.findings)")
        XCTAssertEqual(variant.preferredComputeUnits, .neuralEngine)
        XCTAssertEqual(variant.unet?.imageSide, 512)
        XCTAssertEqual(variant.fit, .excellent)
        XCTAssertEqual(variant.recommendedSteps, 20)
        XCTAssertEqual(variant.recommendedGuidance, 7.5)
        XCTAssertEqual(variant.revision, "dca5f2fda9c5143c71aded72bbb4ce39f3c2cc9c")
        XCTAssertEqual(variant.id, "apple/coreml-stable-diffusion-2-1-base-palettized/split_einsum_v2/compiled")
        XCTAssertTrue(variant.findings.contains { $0.rule == "FIT" && $0.level == .info })
    }

    func testOriginalAttentionRunsOnTheGPUWithACaveat() throws {
        let variant = try evaluate(Fixtures.sd15Original, folder: Fixtures.folder("original/compiled"),
                                   listing: Fixtures.listing("apple/coreml-stable-diffusion-v1-5-palettized",
                                                             license: "creativeml-openrail-m"))

        XCTAssertEqual(variant.verdict, .runnableWithCaveats)
        XCTAssertEqual(variant.preferredComputeUnits, .gpu)
        XCTAssertEqual(rules(variant, .caveat), ["R10"])
    }

    func testFP16ChunkedSD21IsTightOnA17ProAndTooLargeOnA6GBPhone() throws {
        // 2.52 GB is the fp16 SD 2.x folder size from the survey.
        let folder = Fixtures.folder("split_einsum/compiled", bytes: 2_520_000_000, unetBytes: 1_730_000_000,
                                     chunked: true)
        let pro = try evaluate(Fixtures.sd21Fp16Chunk, folder: folder, on: .iPhone17Pro)
        let older = try evaluate(Fixtures.sd21Fp16Chunk, folder: folder, on: .iPhone6GB)

        XCTAssertEqual(pro.fit, .marginal)
        XCTAssertEqual(pro.verdict, .runnableWithCaveats)
        XCTAssertEqual(rules(pro, .caveat), ["FIT"])
        XCTAssertEqual(older.fit, .tooLarge)
        XCTAssertEqual(older.verdict, .notRunnable)
        XCTAssertEqual(rules(older, .blocker), ["FIT"])
    }

    // MARK: Won't run

    func testBatchOneTurboIsBlocked() throws {
        let variant = try evaluate(Fixtures.turboBatchOne,
                                   listing: Fixtures.listing("jivenchy/sd-turbo-coreml-384-6bit"))
        XCTAssertEqual(variant.verdict, .notRunnable)
        XCTAssertTrue(variant.findings.contains { $0.rule == "R3" && $0.title.contains("classifier-free guidance") })
    }

    func testInpaintingUnetIsBlocked() throws {
        let variant = try evaluate(Fixtures.inpainting)
        XCTAssertEqual(variant.verdict, .notRunnable)
        XCTAssertTrue(variant.findings.contains { $0.rule == "R3" && $0.title.contains("Inpainting") })
    }

    func testStatefulUnetIsBlocked() throws {
        let variant = try evaluate(Fixtures.stateful)
        XCTAssertEqual(variant.verdict, .notRunnable)
        XCTAssertTrue(rules(variant, .blocker).contains("R6"))
        XCTAssertFalse(rules(variant, .blocker).contains("R8"), "iOS 26 runs iOS 18 op sets")

        let onOlderOS = try evaluate(Fixtures.stateful, osMajor: 17)
        XCTAssertTrue(onOlderOS.findings.contains { $0.rule == "R8" && $0.title == "Needs iOS 18" })
    }

    func testSDXLIsBlockedWithOrWithoutMicroConditioning() throws {
        let apple = try evaluate(Fixtures.sdxlIOS, folder: Fixtures.folder(chunked: true))
        let withoutTimeIDs = try evaluate(Fixtures.sdxlWithoutTimeIDs)

        XCTAssertEqual(rules(apple, .blocker), ["R5"], "time_ids and text_embeds are SDXL's, not R7")
        XCTAssertEqual(apple.verdict, .notRunnable)
        XCTAssertTrue(rules(withoutTimeIDs, .blocker).contains("R5"))
        XCTAssertEqual(withoutTimeIDs.verdict, .notRunnable)

        let encoderOnly = try evaluate(Fixtures.sd21Split, folder: Fixtures.folder(onlyTextEncoder2: true))
        XCTAssertTrue(rules(encoderOnly, .blocker).contains("R5"))
    }

    func testVPredictionIsBlocked() throws {
        let side768 = CoreMLUnetFacts(inputShapes: ["sample": [2, 4, 96, 96], "timestep": [2],
                                                    "encoder_hidden_states": [2, 1024, 1, 77]],
                                      version: "stabilityai/stable-diffusion-2-1")
        let variant = CoreMLDiffusionRules.evaluate(Fixtures.folder(), unet: side768, listing: Fixtures.listing(),
                                                    on: .iPhone17Pro, osMajor: 26)
        XCTAssertTrue(rules(variant, .blocker).contains("R12"))

        let base768 = CoreMLUnetFacts(inputShapes: side768.inputShapes, version: "someone/sd-2-base-768")
        let base = CoreMLDiffusionRules.evaluate(Fixtures.folder(), unet: base768, listing: Fixtures.listing(),
                                                 on: .iPhone17Pro, osMajor: 26)
        XCTAssertFalse(rules(base, .blocker).contains("R12"))

        let named = try evaluate(Fixtures.sd21Split, listing: Fixtures.listing("o/sd21-vpred-coreml"))
        XCTAssertTrue(rules(named, .blocker).contains("R12"))
    }

    func testUnchunkedFP16UnetOverOneGiBIsBlockedOnIPhoneOnly() throws {
        let folder = Fixtures.folder("split_einsum/compiled", bytes: 2_520_000_000, unetBytes: 1_730_000_000)
        let phone = try evaluate(Fixtures.sd21Fp16Chunk, folder: folder, on: .iPhone17Pro)
        let mac = try evaluate(Fixtures.sd21Fp16Chunk, folder: folder, on: .mac32GB, osMajor: 26)

        XCTAssertTrue(rules(phone, .blocker).contains("R11"))
        XCTAssertEqual(phone.verdict, .notRunnable)
        XCTAssertTrue(rules(mac, .info).contains("R11"))
        XCTAssertFalse(rules(mac, .blocker).contains("R11"))
    }

    func testUnexpectedInputsAreBlocked() throws {
        let controlled = CoreMLUnetFacts(inputShapes: ["sample": [2, 4, 64, 64], "timestep": [2],
                                                       "encoder_hidden_states": [2, 768, 1, 77],
                                                       "adapter_res_samples_00": [2, 320, 64, 64]])
        let variant = CoreMLDiffusionRules.evaluate(Fixtures.folder(), unet: controlled, listing: Fixtures.listing(),
                                                    on: .iPhone17Pro, osMajor: 26)
        XCTAssertEqual(rules(variant, .blocker), ["R7"])

        let foreign = CoreMLUnetFacts(inputShapes: ["latents": [1, 4, 64, 64]])
        let other = CoreMLDiffusionRules.evaluate(Fixtures.folder(), unet: foreign, listing: Fixtures.listing(),
                                                  on: .iPhone17Pro, osMajor: 26)
        XCTAssertEqual(rules(other, .blocker), ["R2"])
    }

    // MARK: Listing rules

    func testGatedRepoIsBlockedWithoutAProbe() throws {
        let variant = try evaluate(nil, listing: Fixtures.listing(gated: "manual"))
        XCTAssertEqual(variant.verdict, .notRunnable)
        XCTAssertTrue(rules(variant, .blocker).contains("R17"))
        XCTAssertFalse(variant.findings.contains { $0.rule == "R1" }, "the blocker already decides")
    }

    func testRepoRootIsBlocked() throws {
        let variant = try evaluate(Fixtures.sd21Split, folder: Fixtures.folder(""))
        XCTAssertEqual(rules(variant, .blocker), ["R16"])
    }

    func testRestrictiveOrMissingLicensesAreCaveats() throws {
        for license in ["other", "cc-by-nc-4.0", "stabilityai-ai-community", nil] as [String?] {
            let variant = try evaluate(Fixtures.sd21Split, listing: Fixtures.listing(license: license))
            XCTAssertEqual(rules(variant, .caveat), ["R18"], license ?? "nil")
            XCTAssertEqual(variant.verdict, .runnableWithCaveats)
        }
        for license in ["openrail++", "creativeml-openrail-m", "apache-2.0", "bigscience-openrail-m"] {
            XCTAssertTrue(rules(try evaluate(Fixtures.sd21Split, listing: Fixtures.listing(license: license)),
                                .caveat).isEmpty, license)
        }
    }

    func testTheSafetyCheckerIsReportedAsNotDownloaded() throws {
        let variant = try evaluate(Fixtures.sd21Split, folder: Fixtures.folder(safetyChecker: true))
        XCTAssertTrue(rules(variant, .info).contains("R19"))
        XCTAssertEqual(variant.verdict, .runnable)
    }

    // MARK: Names and defaults

    func testLatentConsistencyAndDistilledNamesSetDefaults() throws {
        let lcm = try evaluate(Fixtures.sd21Split,
                               listing: Fixtures.listing("LocalMuseAI/coreml-cyberrealistic-final-lcm-6bit"))
        XCTAssertEqual(rules(lcm, .caveat), ["R13"])
        XCTAssertEqual(lcm.recommendedSteps, 8)
        XCTAssertEqual(lcm.recommendedGuidance, 1)

        let turbo = try evaluate(Fixtures.sd21Split, listing: Fixtures.listing("CommuniDev/sd-turbo-coreml"))
        XCTAssertTrue(rules(turbo, .info).contains("R14"))
        XCTAssertEqual(turbo.recommendedSteps, 4)
        XCTAssertEqual(turbo.recommendedGuidance, 1)
        XCTAssertEqual(turbo.verdict, .runnable)

        let tagged = try evaluate(Fixtures.sd21Split, listing: Fixtures.listing("o/fast", tags: ["dmd2"]))
        XCTAssertEqual(tagged.recommendedSteps, 4)

        let realistic = try evaluate(Fixtures.sd21Split, listing: Fixtures.listing("o/hyperrealistic-v2"))
        XCTAssertEqual(realistic.recommendedSteps, 20, "a word that starts with hyper is not Hyper-SD")
    }

    // MARK: Unknown

    func testNoProbeIsUnknownNeverRuns() throws {
        let variant = try evaluate(nil)
        XCTAssertEqual(variant.verdict, .unknown)
        XCTAssertTrue(variant.findings.contains { $0.rule == "R1" })

        let unsized = try evaluate(Fixtures.sd21Split, folder: Fixtures.folder(bytes: nil))
        XCTAssertEqual(unsized.verdict, .unknown)
        XCTAssertNil(unsized.fit)

        let unpinned = try evaluate(Fixtures.sd21Split, listing: Fixtures.listing(revision: nil))
        XCTAssertEqual(unpinned.verdict, .unknown)
    }

    func testFindingsListBlockersFirst() throws {
        let variant = try evaluate(Fixtures.turboBatchOne, folder: Fixtures.folder(safetyChecker: true),
                                   listing: Fixtures.listing(license: "other"))
        let levels = variant.findings.map(\.level)
        XCTAssertEqual(levels, levels.sorted())
    }

    // MARK: Recommendation

    func testRecommendedPrefersRunsThenNeuralEngineThenSmaller() throws {
        let listing = Fixtures.listing(license: "creativeml-openrail-m")
        let blocked = try evaluate(Fixtures.turboBatchOne, folder: Fixtures.folder("a"), listing: listing)
        let gpu = try evaluate(Fixtures.sd21Original, folder: Fixtures.folder("b", bytes: 500_000_000),
                               listing: listing)
        let large = try evaluate(Fixtures.sd21Split, folder: Fixtures.folder("c", bytes: 1_100_000_000),
                                 listing: listing)
        let small = try evaluate(Fixtures.sd21Split, folder: Fixtures.folder("d", bytes: 900_000_000),
                                 listing: listing)
        let report = { (variants: [CoreMLDiffusionVariant]) in
            CoreMLDiffusionRepoReport(repoID: "o/r", revision: "x", variants: variants, unsupportedReason: nil)
        }

        XCTAssertEqual(report([blocked, gpu, large, small]).recommended?.folder.path, "d")
        XCTAssertEqual(report([blocked, gpu]).recommended?.folder.path, "b")

        let caveatANE = try evaluate(Fixtures.sd21Split, folder: Fixtures.folder("e", bytes: 1_200_000_000),
                                     listing: Fixtures.listing(license: "other"))
        XCTAssertEqual(report([gpu, caveatANE]).recommended?.folder.path, "e", "the Neural Engine wins a tie")
        XCTAssertNil(report([]).recommended)
    }
}
