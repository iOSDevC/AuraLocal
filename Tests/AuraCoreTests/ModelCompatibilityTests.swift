import XCTest
#if os(macOS)
import Metal
#endif
@testable import AuraCore

final class ModelCompatibilityTests: XCTestCase {

    private let mac = DevicePreset.mac32GB

    // MARK: - MLX weight layout

    func testQwen35VisualPrefixBlocksLoading() {
        let snapshot = Fixture.mlx(
            "ukisai/Swift-1.5-4bit-MLX", config: Fixture.qwen35Config,
            weightMap: Fixture.weightMap(["language_model": 4, "visual": 3, "mtp": 2]))

        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)

        XCTAssertEqual(report.status, .notRunnable)
        let blocker = report.blockers.first { $0.rule == "mlx.weight-prefixes" }
        XCTAssertNotNil(blocker)
        XCTAssertTrue(blocker?.detail.contains("`visual.*` (3 tensors)") == true)
        XCTAssertFalse(blocker?.detail.contains("mtp.*") == true, "mtp is filtered by the pinned sanitize")
        XCTAssertNil(report.suggestedEntry)
    }

    func testQwen35LanguageModelOnlyRunsAsText() throws {
        let snapshot = Fixture.mlx(
            "ukisai/Swift-1.5-3bit-MLX-TextOnly", config: Fixture.qwen35Config,
            weightMap: Fixture.weightMap(["language_model": 5]), sizes: [11_771_374_457])

        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)

        XCTAssertEqual(report.status, .runnable)
        XCTAssertEqual(report.modelCategory, .text)
        let fit = try XCTUnwrap(report.weightsFit)
        XCTAssertEqual(fit.budgetGB, 20.0)
        XCTAssertTrue(fit.rating <= .good)
        XCTAssertEqual(report.suggestedEntry?.numLayers, 64)
    }

    func testRegisteredVLMWithVisionTowerIsVision() {
        let snapshot = Fixture.mlx(
            "mlx-community/Qwen3.5-27B-4bit", config: Fixture.qwen35Config,
            weightMap: Fixture.weightMap(["language_model": 4, "vision_tower": 2]))

        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)

        XCTAssertEqual(report.status, .runnable)
        XCTAssertEqual(report.modelCategory, .vision)
        XCTAssertEqual(report.suggestedEntry?.modelCategory, .vision)
    }

    func testExtraSafetensorsOutsideTheWeightMapBlock() {
        var snapshot = Fixture.mlx(
            "Edge0/Edge0-35B-A3B-preview", config: Fixture.config(type: "qwen3_5_moe"),
            weightMap: Fixture.weightMap(["language_model": 3]),
            extraFiles: ["lora_edge0_35b.safetensors", "prerouter_edge0_35b.safetensors"])
        snapshot.extraTensorCounts = ["lora_edge0_35b.safetensors": 620, "prerouter_edge0_35b.safetensors": 99]

        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)

        XCTAssertEqual(report.status, .notRunnable)
        let blocker = report.blockers.first { $0.rule == "mlx.extra-safetensors" }
        XCTAssertTrue(blocker?.detail.contains("`lora_edge0_35b.safetensors` (620 tensors)") == true)
        XCTAssertTrue(blocker?.detail.contains("prerouter_edge0_35b.safetensors") == true)
    }

    func testStaleWeightIndexFallsBackToTheShippedHeader() {
        let names = (0..<3).map { "language_model.model.layers.\($0).weight" }
            + (0..<2).map { "vision_tower.blocks.\($0).weight" } + ["multi_modal_projector.linear.weight"]
        var snapshot = Fixture.mlx("mlx-community/gemma-3-4b-it-qat-4bit", config: Fixture.config(type: "gemma3"),
                                   weightMap: nil, sizes: [2_993_000_000])
        snapshot.weightMap = Fixture.weightMap(["language_model": 4])
        snapshot.singleFileHeader = SafetensorsHeader(tensorNames: names, metadata: ["format": "mlx"])

        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)

        XCTAssertTrue(snapshot.isWeightMapStale)
        XCTAssertEqual(report.status, .runnableWithCaveats)
        XCTAssertEqual(report.modelCategory, .vision)
        XCTAssertEqual(report.caveats.first { $0.rule == "mlx.extra-safetensors" }?.title, "Stale weight index")
        XCTAssertEqual(report.weightsBytes, 2_993_000_000, "sized from the shipped file, not the index")
    }

    func testWeightMapWithSomeShardsMissingStillBlocks() {
        var snapshot = Fixture.mlx("o/half-uploaded", config: Fixture.config(type: "llama"),
                                   weightMap: Fixture.weightMap(["model": 4]))
        let shipped = snapshot.listing?.files.filter { $0.path != "model-00002-of-00002.safetensors" } ?? []
        snapshot.listing = HFRepoInfo(repoID: "o/half-uploaded", files: shipped, tags: ["mlx"], licenseID: "mit")

        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)

        XCTAssertFalse(snapshot.isWeightMapStale)
        XCTAssertTrue(report.blockers.contains { $0.title == "Weight map points at missing files" })
    }

    func testTextOnlyTypeWithVisionAndAudioTowersBlocks() {
        let snapshot = Fixture.mlx(
            "mlx-community/gemma-3n-E4B-it-bf16", config: Fixture.config(type: "gemma3n", quantized: false),
            weightMap: Fixture.weightMap(["model.language_model": 4, "model.vision_tower": 2, "model.audio_tower": 2,
                                          "model.embed_audio": 1]))

        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)

        XCTAssertEqual(report.status, .notRunnable)
        let blocker = report.blockers.first { $0.rule == "mlx.towers" }
        XCTAssertTrue(blocker?.detail.contains("`model.audio_tower.*` (2 tensors)") == true, blocker?.detail ?? "")
        XCTAssertTrue(blocker?.detail.contains("`model.vision_tower.*`") == true)
    }

    func testTextOnlyTypeWithOnlyLanguageModelWeightsRuns() {
        let snapshot = Fixture.mlx("Oscilla/gemma-3n-E4B-it-mlx-4Bit", config: Fixture.config(type: "gemma3n"),
                                   weightMap: Fixture.weightMap(["model.language_model": 4]))

        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)

        XCTAssertEqual(report.status, .runnable)
        XCTAssertEqual(report.modelCategory, .text)
    }

    func testQwen35VisionWeightsInMLXFormatLoadOnlyUnderTheModuleNames() {
        func report(_ names: [String]) -> CompatibilityReport {
            var snapshot = Fixture.mlx("o/qwen35-vl", config: Fixture.qwen35Config, weightMap: nil)
            snapshot.singleFileHeader = SafetensorsHeader(tensorNames: names, metadata: ["format": "mlx"])
            return CompatibilityEvaluator.evaluate(snapshot, on: mac)
        }

        let hfLayout = report(["model.language_model.layers.0.weight", "model.visual.blocks.0.weight", "lm_head.weight"])
        let mlxLayout = report(["language_model.model.layers.0.weight", "vision_tower.blocks.0.weight"])

        let blocker = hfLayout.blockers.first { $0.rule == "mlx.weight-prefixes" }
        XCTAssertTrue(blocker?.detail.contains("`lm_head.*`") == true, blocker?.detail ?? "no blocker")
        XCTAssertTrue(blocker?.detail.contains("`model.*`") == true)
        XCTAssertEqual(mlxLayout.status, .runnable)
        XCTAssertEqual(mlxLayout.modelCategory, .vision)
    }

    func testUnregisteredModelTypeBlocks() {
        let snapshot = Fixture.mlx("mlx-community/MiniCPM3-4B-4bit", config: Fixture.config(type: "minicpm3"),
                                   weightMap: nil)
        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)
        XCTAssertEqual(report.status, .notRunnable)
        XCTAssertEqual(report.blockers.first?.rule, "mlx.model-type")
    }

    func testTextConfigTypeDoesNotRescueAnUnregisteredTopLevelType() {
        let config = Fixture.config(type: "qwen4_exp", extra: #", "text_config": {"model_type": "qwen3_5_text"}"#)
        let report = CompatibilityEvaluator.evaluate(Fixture.mlx("o/r", config: config, weightMap: nil), on: mac)
        let blocker = report.blockers.first { $0.rule == "mlx.model-type" }
        XCTAssertTrue(blocker?.detail.contains("dispatch on the top-level key") == true)
    }

    func testLongropeNeedsItsFields() {
        let complete = Fixture.config(type: "minicpm", extra: #"""
            , "rope_scaling": {"rope_type": "longrope", "original_max_position_embeddings": 65536,
                               "short_factor": [1.0], "long_factor": [1.0]}
            """#)
        let missing = Fixture.config(type: "minicpm", extra: #", "rope_scaling": {"rope_type": "longrope", "long_factor": [1.0]}"#)

        XCTAssertEqual(CompatibilityEvaluator.evaluate(Fixture.mlx("o/ok", config: complete, weightMap: nil), on: mac).status,
                       .runnable)
        let broken = CompatibilityEvaluator.evaluate(Fixture.mlx("o/bad", config: missing, weightMap: nil), on: mac)
        XCTAssertEqual(broken.blockers.first?.rule, "mlx.rope")
    }

    func testUnknownRopeTypeWouldCrash() {
        let config = Fixture.config(type: "granite", extra: #", "rope_scaling": {"type": "dynamic", "factor": 2.0}"#)
        let report = CompatibilityEvaluator.evaluate(Fixture.mlx("o/r", config: config, weightMap: nil), on: mac)
        XCTAssertEqual(report.blockers.first?.rule, "mlx.rope")
    }

    // MARK: - Not generative / not converted

    func testFillMaskEncoderIsNotGenerative() {
        let snapshot = RepoSnapshot(
            repoID: "medicalai/ClinicalBERT",
            listing: HFRepoInfo(repoID: "medicalai/ClinicalBERT",
                                files: [RepoFile(path: "config.json", sizeBytes: 466),
                                        RepoFile(path: "pytorch_model.bin", sizeBytes: 267_000_000)],
                                tags: ["transformers", "pytorch", "distilbert", "fill-mask"],
                                pipelineTag: "fill-mask", libraryName: "transformers"),
            configuration: Fixture.facts(#"{"model_type": "distilbert", "architectures": ["DistilBertForMaskedLM"]}"#))

        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)

        XCTAssertEqual(report.weightFormat, .noWeights)
        XCTAssertEqual(report.status, .notRunnable)
        let generative = report.blockers.first { $0.rule == "generative" }
        XCTAssertTrue(generative?.detail.contains("fill-mask") == true)
        XCTAssertTrue(generative?.detail.contains("DistilBertForMaskedLM") == true)
        XCTAssertTrue(report.findings.contains { $0.rule == "license" && $0.title == "No license declared" })
    }

    func testEmbeddingModelTaggedAsTextGenerationIsNotGenerative() {
        var snapshot = Fixture.mlx("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ", config: Fixture.config(type: "qwen3"),
                                   weightMap: nil)
        snapshot.listing = HFRepoInfo(
            repoID: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ", files: snapshot.listing?.files ?? [],
            tags: ["mlx", "qwen3", "text-generation", "sentence-transformers", "sentence-similarity",
                   "feature-extraction", "base_model:Qwen/Qwen3-Embedding-0.6B"],
            pipelineTag: "text-generation", libraryName: "mlx", licenseID: "apache-2.0")
        let byName = Fixture.gguf(architecture: "qwen3", files: [("Qwen3-Embedding-0.6B-Q8_0.gguf", 640_000_000)])

        let tagged = CompatibilityEvaluator.evaluate(snapshot, on: mac)
        let named = CompatibilityEvaluator.evaluate(
            RepoSnapshot(repoID: "Qwen/Qwen3-Embedding-0.6B-GGUF", listing: HFRepoInfo(
                repoID: "Qwen/Qwen3-Embedding-0.6B-GGUF", files: byName.listing?.files ?? [], tags: ["gguf"],
                licenseID: "apache-2.0", ggufArchitecture: "qwen3")), on: mac)

        XCTAssertEqual(tagged.status, .notRunnable)
        XCTAssertTrue(tagged.blockers.first { $0.rule == "generative" }?.detail.contains("`sentence-transformers`") == true)
        XCTAssertNil(tagged.suggestedEntry)
        XCTAssertEqual(named.blockers.first?.rule, "generative")
    }

    func testPlainTransformersSafetensorsAreNotAnMLXConversion() {
        let snapshot = RepoSnapshot(
            repoID: "Qwen/Qwen3.8-Flash-Next",
            listing: HFRepoInfo(repoID: "Qwen/Qwen3.8-Flash-Next",
                                files: [RepoFile(path: "config.json", sizeBytes: 4745),
                                        RepoFile(path: "model-00001-of-00002.safetensors", sizeBytes: 180_000_000_000),
                                        RepoFile(path: "model-00002-of-00002.safetensors", sizeBytes: 180_000_000_000)],
                                tags: ["transformers", "safetensors"], pipelineTag: "image-text-to-text",
                                libraryName: "transformers", licenseID: "other", licenseName: "qwen-community-1.0"),
            configuration: Fixture.config(type: "qwen4_exp", quantized: false))

        let report = CompatibilityEvaluator.evaluate(snapshot, on: .mac64GB)

        XCTAssertEqual(report.weightFormat, .unconvertedSafetensors)
        XCTAssertEqual(report.status, .notRunnable)
        XCTAssertEqual(Set(report.blockers.map(\.rule)), ["format", "mlx.model-type"])
        XCTAssertTrue(report.findings.contains { $0.title == "Custom license `qwen-community-1.0`" })
    }

    // MARK: - GGUF

    func testQwen35WithNextNNeedsANewerLlamaCpp() throws {
        let header = try GGUFHeaderParser.parse(GGUFBytes.qwen35(nextn: 1).encoded)
        let report = CompatibilityEvaluator.evaluate(Fixture.gguf(header: header), on: mac)

        XCTAssertEqual(report.status, .notRunnable)
        let blocker = try XCTUnwrap(report.blockers.first { $0.rule == "gguf.qwen35-nextn" })
        XCTAssertTrue(blocker.detail.contains("b9495"))
        XCTAssertEqual(report.quantFits.count, 2)
    }

    func testQwen35WithoutNextNRunsAndFillsTheEntryFromTheHeader() throws {
        let header = try GGUFHeaderParser.parse(GGUFBytes.qwen35(nextn: 0).encoded)
        let report = CompatibilityEvaluator.evaluate(Fixture.gguf(header: header), on: mac)

        XCTAssertTrue(report.status.isRunnable)
        let entry = try XCTUnwrap(report.suggestedEntry)
        XCTAssertEqual(entry.weightFormat, .gguf)
        XCTAssertEqual(entry.ggufFilename, "m-Q8_0.gguf", "the largest quant that fits comfortably")
        XCTAssertEqual(entry.numLayers, 65)
        XCTAssertEqual(entry.kvHeads, 4)
        XCTAssertEqual(entry.headDim, 256)
        XCTAssertNil(entry.maxContextLength, "262144 needs no cap")
    }

    func testLaterArchitectureBlocks() {
        let report = CompatibilityEvaluator.evaluate(Fixture.gguf(architecture: "qwen4exp"), on: mac)
        let blocker = report.blockers.first { $0.rule == "gguf.architecture" }
        XCTAssertTrue(blocker?.title.contains("b10660") == true)
    }

    func testEmbeddingArchitectureIsNotAChatModel() {
        let report = CompatibilityEvaluator.evaluate(Fixture.gguf(architecture: "nomic-bert"), on: mac)
        XCTAssertEqual(report.blockers.first?.rule, "gguf.architecture")
    }

    func testSplitQuantsAreACaveatUntilEveryQuantIsSplit() {
        let mixed = Fixture.gguf(architecture: "qwen2", files: [
            ("q-q2_k.gguf", 3_000_000_000),
            ("q-q4_k_m-00001-of-00002.gguf", 3_900_000_000), ("q-q4_k_m-00002-of-00002.gguf", 700_000_000),
        ])
        let allSplit = Fixture.gguf(architecture: "qwen2", files: [
            ("q-q4_k_m-00001-of-00002.gguf", 3_900_000_000), ("q-q4_k_m-00002-of-00002.gguf", 700_000_000),
        ])

        let mixedReport = CompatibilityEvaluator.evaluate(mixed, on: mac)
        let splitReport = CompatibilityEvaluator.evaluate(allSplit, on: mac)

        XCTAssertEqual(mixedReport.status, .runnableWithCaveats)
        XCTAssertTrue(mixedReport.caveats.contains { $0.rule == "gguf.shards" })
        XCTAssertEqual(mixedReport.suggestedEntry?.ggufFilename, "q-q2_k.gguf")
        let split = mixedReport.quantFits.first { !$0.isUsable }
        XCTAssertEqual(split?.option.paths.count, 2)
        XCTAssertEqual(split?.option.totalBytes, 4_600_000_000)
        XCTAssertEqual(split?.option.label, "Q4_K_M")
        XCTAssertEqual(splitReport.status, .notRunnable)
        XCTAssertTrue(splitReport.blockers.contains { $0.rule == "gguf.shards" })
    }

    // MARK: - Fit per device

    func testFitFollowsTheDevicePreset() {
        let snapshot = Fixture.mlx("o/big", config: Fixture.qwen35Config,
                                   weightMap: Fixture.weightMap(["language_model": 2]), sizes: [11_771_374_457])

        let phone = CompatibilityEvaluator.evaluate(snapshot, on: .iPhone6GB)
        let desktop = CompatibilityEvaluator.evaluate(snapshot, on: .mac32GB)

        XCTAssertEqual(phone.status, .notRunnable)
        XCTAssertEqual(phone.weightsFit?.rating, .tooLarge)
        XCTAssertTrue(phone.blockers.contains { $0.rule == "fit" })
        XCTAssertEqual(desktop.status, .runnable)
        XCTAssertEqual(desktop.weightsFit?.budgetGB, 20.0)
    }

    func testExtremeArchitectureValuesAndSizesDoNotTrap() {
        let huge: [GGUFValue] = [.uint32(.max), .uint64(.max), .int64(.max), .int64(.min), .int32(-1), .uint32(0)]
        let headers = huge.map { value in
            GGUFHeader(version: 3, tensorCount: 1, declaredKeyCount: 5, metadata: [
                "general.architecture": .string("llama"), "llama.block_count": value,
                "llama.attention.head_count_kv": value, "llama.attention.key_length": value,
                "llama.context_length": value,
            ], isComplete: true)
        }
        let parts: [(String, Int64)] = [("m-Q4_K_M-00001-of-00002.gguf", 6_000_000_000_000_000_000),
                                        ("m-Q4_K_M-00002-of-00002.gguf", 6_000_000_000_000_000_000),
                                        ("m-Q2_K.gguf", 9_000_000_000_000_000_000)]
        let configs = ["1e6", "1e30", "-3", "9223372036854775807"].map { number in
            Fixture.facts(#"{"model_type": "llama", "quantization": {"bits": 4}, "num_hidden_layers": \#(number), "#
                + #""num_key_value_heads": \#(number), "head_dim": \#(number), "max_position_embeddings": \#(number)}"#)
        }
        let snapshots = headers.map { Fixture.gguf(header: $0, files: parts) }
            + configs.map { Fixture.mlx("o/huge", config: $0, weightMap: nil, sizes: [.max, .max]) }

        for snapshot in snapshots {
            for device in DevicePreset.classes {
                let report = CompatibilityEvaluator.evaluate(snapshot, on: device)
                XCTAssertNil(report.overview.layers, "\(snapshot.repoID) on \(device.id)")
                XCTAssertNotEqual(report.status, .runnable)
            }
        }
    }

    func testMacPresetsCarryTheirProvenance() {
        XCTAssertFalse(DevicePreset.iPhone6GB.isMeasured)
        XCTAssertTrue(DevicePreset.mac32GB.isMeasured)
        XCTAssertTrue(DevicePreset.mac32GB.source.contains("LaunchDaemon"))
        let given = HardwareProfile(totalMemoryGB: 16, availableMemoryGB: 7.5, deviceName: "Test Mac")
        XCTAssertEqual(DevicePreset.thisDevice(profile: given).budgetGB, 7.5, "a given profile is used as is")
        #if os(macOS)
        if MTLCreateSystemDefaultDevice() != nil {
            XCTAssertTrue(DevicePreset.thisDevice().source.contains("recommendedMaxWorkingSetSize"))
        }
        #endif
    }

    func testImagePipelinesRunOnlyOnMacThroughAuraImageGen() {
        let listing = HFRepoInfo(
            repoID: "o/flux-mflux-4bit",
            files: [RepoFile(path: "transformer/0.safetensors", sizeBytes: 6_000_000_000),
                    RepoFile(path: "text_encoder_2/0.safetensors", sizeBytes: 3_000_000_000)],
            tags: ["mlx", "text-to-image"], pipelineTag: "text-to-image", licenseID: "apache-2.0")
        let snapshot = RepoSnapshot(repoID: "o/flux-mflux-4bit", listing: listing)

        let onMac = CompatibilityEvaluator.evaluate(snapshot, on: .mac64GB)
        let onPhone = CompatibilityEvaluator.evaluate(snapshot, on: .iPhone17Pro)

        XCTAssertEqual(onMac.weightFormat, .imageGeneration)
        XCTAssertEqual(onMac.status, .runnableWithCaveats)
        XCTAssertEqual(onMac.weightsBytes, 9_000_000_000, "every safetensors of the pipeline counts")
        XCTAssertNil(onMac.suggestedEntry)
        XCTAssertTrue(onPhone.blockers.contains { $0.rule == "imagegen" })
    }

    func testEveryPresetIsReachableByID() {
        for preset in DevicePreset.classes {
            XCTAssertEqual(DevicePreset.named(preset.id)?.id, preset.id)
        }
        XCTAssertEqual(DevicePreset.named("this-device")?.id, DevicePreset.thisDeviceID)
        XCTAssertNil(DevicePreset.named("toaster"))
    }

    // MARK: - Gating and licenses

    func testGatedRepoAndRestrictiveLicensesAreCaveats() {
        func findings(license: String?, name: String? = nil, gated: String? = nil) -> [CompatibilityFinding] {
            let listing = HFRepoInfo(repoID: "o/r", files: [], gatedMode: gated, licenseID: license, licenseName: name)
            return CompatibilityEvaluator.evaluate(RepoSnapshot(repoID: "o/r", listing: listing), on: mac)
                .findings.filter { $0.rule == "license" }
        }

        XCTAssertEqual(findings(license: "apache-2.0").map(\.level), [.info])
        XCTAssertEqual(findings(license: nil).first?.title, "No license declared")
        XCTAssertEqual(findings(license: "cc-by-nc-4.0").first?.title, "Non-commercial license `cc-by-nc-4.0`")
        XCTAssertEqual(findings(license: "other", name: "health-ai-developer-foundations").first?.title,
                       "Custom license `health-ai-developer-foundations`")
        let gated = findings(license: "gemma", gated: "manual")
        XCTAssertEqual(gated.map(\.level), [.caveat, .caveat])
        XCTAssertEqual(gated.first?.title, "Gated repository (manual)")
    }

    func testRelativeLicenseLinkResolvesAgainstTheRepo() {
        let listing = HFRepoInfo(repoID: "Qwen/Qwen3.8-Flash-Next", files: [], licenseID: "other",
                                 licenseName: "qwen-community-1.0", licenseLink: "LICENSE")
        let report = CompatibilityEvaluator.evaluate(RepoSnapshot(repoID: listing.repoID, listing: listing), on: mac)

        XCTAssertTrue(report.caveats.contains {
            $0.detail.contains("(https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)")
        })
        XCTAssertEqual(CompatibilityRules.absoluteLicenseLink("https://example.com/terms", repoID: "o/r"),
                       "https://example.com/terms")
        XCTAssertEqual(CompatibilityRules.absoluteLicenseLink("./docs/LICENSE.md", repoID: "o/r"),
                       "https://huggingface.co/o/r/blob/main/docs/LICENSE.md")
    }

    func testEveryRuleFindingCarriesItsRuleID() {
        let ids = Set(CompatibilityRules.all.map(\.id)).union(["fit", "entry", "repository"])
        let snapshots = [
            Fixture.mlx("o/a", config: Fixture.qwen35Config, weightMap: Fixture.weightMap(["visual": 1, "language_model": 1])),
            Fixture.gguf(architecture: "qwen4exp"),
            RepoSnapshot(repoID: "o/missing"),
        ]
        for snapshot in snapshots {
            for finding in CompatibilityEvaluator.evaluate(snapshot, on: mac).findings {
                XCTAssertTrue(ids.contains(finding.rule), "\(finding.rule): \(finding.title)")
            }
        }
    }

    // MARK: - Catalog entry

    func testMLXEntryMatchesTheCatalogFormatAndDecodesAsAModel() throws {
        let snapshot = Fixture.mlx("mlx-community/MiniCPM5-1B-4bit",
                                   config: Fixture.config(type: "llama", extra: #", "max_position_embeddings": 4096"#),
                                   weightMap: nil, sizes: [608_026_621])

        let entry = try XCTUnwrap(CompatibilityEvaluator.evaluate(snapshot, on: .iPhone4GB).suggestedEntry)

        XCTAssertEqual(entry.jsonText(), """
        {
          "id": "minicpm5_1b_4bit",
          "repoID": "mlx-community/MiniCPM5-1B-4bit",
          "displayName": "MiniCPM5 1B 4bit (MLX 4-bit)",
          "category": "text",
          "docTags": false,
          "format": "mlx",
          "approximateSizeMB": 580,
          "isUncensored": false,
          "ggufFilename": null,
          "defaultDocumentPrompt": null,
          "numLayers": 24,
          "kvHeads": 0,
          "headDim": 0,
          "maxContextLength": 4096
        }
        """)
        let model = try XCTUnwrap(entry.decodedModel())
        XCTAssertEqual(model.repoID, "mlx-community/MiniCPM5-1B-4bit")
        XCTAssertEqual(model.format, .mlx)
        XCTAssertEqual(model.maxContextLength, 4096)
        XCTAssertTrue(entry.jsonText(indent: 4).hasPrefix("    {\n      \"id\""))
    }

    func testEntryNamesDropFormatSuffixesAndEscapeQuotes() {
        XCTAssertEqual(CatalogEntry.identifier(repoID: "Qwen/Qwen2.5-7B-Instruct-GGUF", quant: "Q4_K_M"),
                       "qwen2_5_7b_instruct_q4_k_m_gguf")
        XCTAssertEqual(CatalogEntry.displayName(repoID: "Qwen/Qwen2.5-7B-Instruct-GGUF", format: .gguf, quant: "Q4_K_M", bits: nil),
                       "Qwen2.5 7B Instruct (GGUF Q4_K_M)")
        XCTAssertTrue(CatalogEntry.looksUncensored(repoID: "x/Llama-3-8B-abliterated", tags: []))
        let entry = CatalogEntry(id: "a", repoID: "o/r", displayName: "Quote \" and \\ slash", modelCategory: .text,
                                 weightFormat: .mlx, approximateSizeMB: 1, isUncensored: false, ggufFilename: nil,
                                 numLayers: 1, kvHeads: 0, headDim: 0, maxContextLength: nil)
        XCTAssertEqual(entry.decodedModel()?.displayName, "Quote \" and \\ slash")
    }

    // MARK: - Unreadable repos

    func testMissingListingIsUnknown() {
        let snapshot = RepoSnapshot(repoID: "o/missing", problems: [
            FetchProblem(subject: .repository, statusCode: 404, message: "repository listing: not found (HTTP 404)"),
        ])
        let report = CompatibilityEvaluator.evaluate(snapshot, on: mac)
        XCTAssertEqual(report.status, .unknown)
        XCTAssertEqual(report.headline, "Repository not found")
    }

    func testRepoIDAcceptsURLsAndRejectsExtraSegmentsOrDotDot() {
        XCTAssertEqual(ModelCompatibilityChecker.repoID(from: "https://huggingface.co/o/r/tree/main"), "o/r")
        XCTAssertEqual(ModelCompatibilityChecker.repoID(from: " o/r.v2 "), "o/r.v2")
        XCTAssertNil(ModelCompatibilityChecker.repoID(from: "o/r/extra"))
        XCTAssertNil(ModelCompatibilityChecker.repoID(from: "o/../r"))
        XCTAssertNil(ModelCompatibilityChecker.repoID(from: "o/r?x=1"))
    }

    func testSearchURLCarriesTheFormatTag() throws {
        let url = try XCTUnwrap(HuggingFaceSearch.searchURL(query: "qwen", limit: 5, sort: .downloads, tag: "gguf"))
        XCTAssertTrue(url.absoluteString.contains("filter=gguf"))
        XCTAssertFalse(try XCTUnwrap(HuggingFaceSearch.searchURL(query: "qwen", limit: 5, sort: .downloads))
            .absoluteString.contains("filter="))
    }

    // MARK: - Pinned tables vs the checkout

    func testPinnedMLXRegistriesMatchTheCheckout() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let candidates = [".build/checkouts/mlx-swift-lm", "build/dd/SourcePackages/checkouts/mlx-swift-lm"]
            .map { root.appendingPathComponent($0) }
        guard let checkout = candidates.first(where: {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("Libraries/MLXLLM/LLMModelFactory.swift").path)
        }) else {
            throw XCTSkip("No mlx-swift-lm checkout; run `swift build` first.")
        }

        let resolved = try String(contentsOf: root.appendingPathComponent("Package.resolved"), encoding: .utf8)
        XCTAssertTrue(resolved.contains("\"version\" : \"\(PinnedRuntimes.mlxSwiftLMVersion)\""),
                      "Package.resolved no longer pins mlx-swift-lm \(PinnedRuntimes.mlxSwiftLMVersion)")
        XCTAssertEqual(try registryKeys(checkout.appendingPathComponent("Libraries/MLXLLM/LLMModelFactory.swift")),
                       PinnedRuntimes.mlxLLMModelTypes)
        XCTAssertEqual(try registryKeys(checkout.appendingPathComponent("Libraries/MLXVLM/VLMModelFactory.swift")),
                       PinnedRuntimes.mlxVLMModelTypes)
    }

    func testLaterArchitecturesAreNotInThePinnedList() {
        XCTAssertTrue(PinnedRuntimes.laterLlamaCppArchitectures.keys.allSatisfy {
            !PinnedRuntimes.llamaCppArchitectures.contains($0)
        })
        XCTAssertTrue(Set(PinnedRuntimes.llamaCppNonChatArchitectures.keys).isSubset(of: PinnedRuntimes.llamaCppArchitectures))
        let registered = PinnedRuntimes.mlxLLMModelTypes.union(PinnedRuntimes.mlxVLMModelTypes)
        XCTAssertTrue(PinnedRuntimes.mlxRopeModelTypes.isSubset(of: registered))
    }

    private func registryKeys(_ url: URL) throws -> Set<String> {
        let source = try String(contentsOf: url, encoding: .utf8)
        let start = try XCTUnwrap(source.range(of: "creators: ["))
        let tail = source[start.upperBound...]
        let end = try XCTUnwrap(tail.range(of: "])"))
        let pattern = try Regex(#""([^"]+)":\s*create\("#)
        return Set(tail[..<end.lowerBound].matches(of: pattern).compactMap { $0.output[1].substring.map(String.init) })
    }
}

// MARK: - Fixtures

enum Fixture {
    static let qwen35Config = facts(#"""
        {"model_type": "qwen3_5", "quantization": {"bits": 3, "group_size": 64},
         "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 64, "num_key_value_heads": 4,
                         "head_dim": 256, "max_position_embeddings": 262144,
                         "rope_parameters": {"type": "default", "mrope_section": [11, 11, 10]}}}
        """#)

    static func config(type: String, quantized: Bool = true, extra: String = "") -> ModelConfigFacts? {
        let quant = quantized ? #", "quantization": {"bits": 4, "group_size": 64}"# : ""
        return facts(#"{"model_type": "\#(type)", "num_hidden_layers": 24, "num_key_value_heads": 2, "head_dim": 128\#(quant)\#(extra)}"#)
    }

    static func facts(_ json: String) -> ModelConfigFacts? {
        ModelConfigFacts.parse(Data(json.utf8))
    }

    /// `prefix → tensor count`, spread over two shards.
    static func weightMap(_ prefixes: [String: Int]) -> [String: String] {
        let names = prefixes.flatMap { prefix, count in (0..<count).map { "\(prefix).layers.\($0).weight" } }
        return Dictionary(uniqueKeysWithValues: names.enumerated().map { index, name in
            (name, index.isMultiple(of: 2) ? "model-00001-of-00002.safetensors" : "model-00002-of-00002.safetensors")
        })
    }

    static let singleTensor = SafetensorsHeader(tensorNames: ["model.layers.0.weight"], metadata: [:])

    static func mlx(_ repoID: String, config: ModelConfigFacts?, weightMap: [String: String]?,
                    sizes: [Int64] = [2_000_000_000, 1_000_000_000], extraFiles: [String] = []) -> RepoSnapshot {
        let weightFiles = weightMap.map { Array(Set($0.values)).sorted() } ?? ["model.safetensors"]
        let sized = zip(weightFiles, sizes + Array(repeating: 1_000_000, count: weightFiles.count))
            .map { RepoFile(path: $0.0, sizeBytes: $0.1) }
        let others = (["config.json", "tokenizer.json"] + extraFiles).map { RepoFile(path: $0, sizeBytes: 40_000_000) }
        let listing = HFRepoInfo(repoID: repoID, files: sized + others, tags: ["mlx", "safetensors"],
                                 pipelineTag: "text-generation", libraryName: "mlx", licenseID: "apache-2.0")
        return RepoSnapshot(repoID: repoID, listing: listing, configuration: config, weightMap: weightMap,
                            singleFileHeader: weightMap == nil ? singleTensor : nil)
    }

    static func gguf(header: GGUFHeader? = nil, architecture: String? = nil,
                     files: [(String, Int64)] = [("m-Q4_K_M.gguf", 5_000_000_000), ("m-Q8_0.gguf", 8_500_000_000)]) -> RepoSnapshot {
        let listing = HFRepoInfo(repoID: "o/m-GGUF", files: files.map { RepoFile(path: $0.0, sizeBytes: $0.1) },
                                 tags: ["gguf"], pipelineTag: "text-generation", licenseID: "apache-2.0",
                                 ggufArchitecture: architecture)
        return RepoSnapshot(repoID: "o/m-GGUF", listing: listing, ggufMetadata: header, ggufSamplePath: files.first?.0)
    }
}

// MARK: - Checker over a stubbed Hugging Face

final class ModelCompatibilityCheckerTests: XCTestCase {

    private let stubs = StubHub.shared

    override func setUp() {
        super.setUp()
        stubs.reset()
    }

    private var stubbedChecker: ModelCompatibilityChecker {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        return ModelCompatibilityChecker(transport: URLSession(configuration: configuration),
                                         authorizer: NoDownloadAuth(), ggufHeaderBytes: 16_384)
    }

    func testReadsListingConfigIndexAndCountsExtraTensors() async throws {
        let repo = "stub/edge-mlx"
        stubs.serve("api/models/\(repo)?blobs=true", StubHub.listing(repo, files: [
            ("config.json", 900), ("tokenizer.json", 9_000_000), ("model.safetensors.index.json", 4000),
            ("model-00001-of-00001.safetensors", 4_000_000_000), ("lora.safetensors", 40_000_000),
        ], tags: ["mlx"]))
        stubs.serve("\(repo)/resolve/main/config.json",
                  Data(#"{"model_type": "qwen3_5_moe", "quantization": {"bits": 4}, "num_hidden_layers": 40}"#.utf8))
        stubs.serve("\(repo)/resolve/main/model.safetensors.index.json", Data(#"""
            {"metadata": {}, "weight_map": {"language_model.model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}
            """#.utf8))
        stubs.serve("\(repo)/resolve/main/lora.safetensors",
                  SafetensorsBytes.make(["lora.a", "lora.b", "router.w"]) + Data(count: 300_000))

        let report = await stubbedChecker.check(repo, on: .mac32GB)

        XCTAssertEqual(report.status, .notRunnable)
        let blocker = try XCTUnwrap(report.blockers.first { $0.rule == "mlx.extra-safetensors" })
        XCTAssertTrue(blocker.detail.contains("`lora.safetensors` (3 tensors)"), blocker.detail)
        XCTAssertEqual(stubs.requests(for: "lora.safetensors").first?.value(forHTTPHeaderField: "Range"), "bytes=0-262143")
        XCTAssertTrue(stubs.requests(for: "model-00001-of-00001.safetensors").isEmpty, "weights are never fetched")
    }

    func testReadsAGGUFHeaderThroughARangeRequest() async throws {
        let repo = "stub/qwen35-gguf"
        stubs.serve("api/models/\(repo)?blobs=true", StubHub.listing(repo, files: [
            ("m-Q4_K_M.gguf", 5_000_000_000), ("m-Q8_0.gguf", 8_500_000_000), ("mmproj-m-F16.gguf", 900_000_000),
        ], tags: ["gguf"]))
        stubs.serve("\(repo)/resolve/main/m-Q4_K_M.gguf", GGUFBytes.qwen35(nextn: 1, vocabulary: 3000).encoded)

        let report = await stubbedChecker.check(repo, on: .mac32GB)

        XCTAssertEqual(stubs.requests(for: "m-Q4_K_M.gguf").first?.value(forHTTPHeaderField: "Range"), "bytes=0-16383")
        XCTAssertTrue(stubs.requests(for: "mmproj").isEmpty)
        XCTAssertEqual(report.status, .notRunnable)
        XCTAssertNotNil(report.blockers.first { $0.rule == "gguf.qwen35-nextn" })
        XCTAssertTrue(report.findings.contains { $0.title == "Header partly read" })
        XCTAssertTrue(report.findings.contains { $0.title == "Vision projector ignored" })
        XCTAssertEqual(report.overview.layers, 65)
    }

    func testStaleIndexReadsTheShippedFileHeader() async throws {
        let repo = "stub/gemma3-qat"
        stubs.serve("api/models/\(repo)?blobs=true", StubHub.listing(repo, files: [
            ("config.json", 900), ("tokenizer.json", 9_000_000), ("model.safetensors.index.json", 90_000),
            ("model.safetensors", 2_993_000_000),
        ], tags: ["mlx"]))
        stubs.serve("\(repo)/resolve/main/config.json", Data(#"{"model_type": "gemma3", "quantization": {"bits": 4}}"#.utf8))
        stubs.serve("\(repo)/resolve/main/model.safetensors.index.json", Data(#"""
            {"weight_map": {"language_model.model.embed_tokens.weight": "model-00001-of-00002.safetensors"}}
            """#.utf8))
        stubs.serve("\(repo)/resolve/main/model.safetensors",
                    SafetensorsBytes.make(["language_model.model.embed_tokens.weight", "vision_tower.patch.weight"],
                                          metadata: ["format": "mlx"]) + Data(count: 1000))

        let report = await stubbedChecker.check(repo, on: .mac32GB)

        XCTAssertEqual(report.status, .runnableWithCaveats, report.findings.map(\.title).description)
        XCTAssertEqual(report.modelCategory, .vision)
        XCTAssertFalse(stubs.requests(for: "model.safetensors").filter { $0.url?.lastPathComponent == "model.safetensors" }.isEmpty)
    }

    func testShardedQwen35ReadsOnlyTheFirstShardHeaderForItsMetadata() async throws {
        let repo = "stub/qwen35-vl-shards"
        stubs.serve("api/models/\(repo)?blobs=true", StubHub.listing(repo, files: [
            ("config.json", 900), ("tokenizer.json", 9_000_000), ("model.safetensors.index.json", 4000),
            ("model-00001-of-00002.safetensors", 4_000_000_000), ("model-00002-of-00002.safetensors", 4_000_000_000),
        ], tags: ["mlx"]))
        stubs.serve("\(repo)/resolve/main/config.json", Data(#"""
            {"model_type": "qwen3_5", "quantization": {"bits": 4}, "vision_config": {}}
            """#.utf8))
        stubs.serve("\(repo)/resolve/main/model.safetensors.index.json", Data(#"""
            {"weight_map": {"model.language_model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                            "model.visual.patch_embed.weight": "model-00002-of-00002.safetensors"}}
            """#.utf8))
        stubs.serve("\(repo)/resolve/main/model-00001-of-00002.safetensors",
                    SafetensorsBytes.make(["model.language_model.embed_tokens.weight"], metadata: ["format": "mlx"]))

        let report = await stubbedChecker.check(repo, on: .mac32GB)

        XCTAssertEqual(stubs.requests(for: "model-00001-of-00002.safetensors").count, 1)
        XCTAssertTrue(stubs.requests(for: "model-00002-of-00002.safetensors").isEmpty)
        XCTAssertTrue(report.blockers.contains { $0.rule == "mlx.weight-prefixes" }, report.findings.map(\.title).description)
    }

    func testHTTPFailuresBecomeFindings() async {
        let missing = await stubbedChecker.check("stub/does-not-exist", on: .mac32GB)
        XCTAssertEqual(missing.status, .unknown)
        XCTAssertEqual(missing.headline, "Repository not found")

        let repo = "stub/gated-mlx"
        stubs.serve("api/models/\(repo)?blobs=true", StubHub.listing(repo, files: [
            ("config.json", 900), ("model.safetensors", 1_000_000_000),
        ], tags: ["mlx"], gated: "manual"))
        stubs.serve("\(repo)/resolve/main/config.json", Data(), status: 401)
        stubs.serve("\(repo)/resolve/main/model.safetensors", Data(), status: 401)

        let gated = await stubbedChecker.check(repo, on: .mac32GB)

        XCTAssertEqual(gated.status, .unknown)
        XCTAssertTrue(gated.caveats.contains { $0.rule == "mlx.config" && $0.detail.contains("token") })
        XCTAssertTrue(gated.caveats.contains { $0.title == "Gated repository (manual)" })
    }
}

/// Canned Hugging Face responses keyed by URL; honours `Range: bytes=0-N`.
final class StubHub: @unchecked Sendable {
    static let shared = StubHub()

    private let lock = NSLock()
    private var routes: [String: (status: Int, body: Data)] = [:]
    private var received: [URLRequest] = []

    func serve(_ path: String, _ body: Data, status: Int = 200) {
        lock.withLock { routes["https://huggingface.co/" + path] = (status, body) }
    }

    func respond(to request: URLRequest) -> (status: Int, body: Data) {
        lock.withLock {
            received.append(request)
            return routes[request.url?.absoluteString ?? ""] ?? (404, Data())
        }
    }

    func reset() {
        lock.withLock {
            routes = [:]
            received = []
        }
    }

    func requests(for fragment: String) -> [URLRequest] {
        lock.withLock { received.filter { $0.url?.absoluteString.contains(fragment) == true } }
    }

    static func listing(_ repo: String, files: [(String, Int64)], tags: [String], gated: String? = nil) -> Data {
        let object: [String: Any] = [
            "id": repo, "tags": tags, "pipeline_tag": "text-generation", "gated": gated ?? false,
            "cardData": ["license": "apache-2.0"],
            "siblings": files.map { ["rfilename": $0.0, "size": $0.1] as [String: Any] },
        ]
        return (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
    }
}

final class StubURLProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        guard let url = request.url else { return }
        let reply = StubHub.shared.respond(to: request)
        var body = reply.body
        var status = reply.status
        if status == 200, let end = Self.rangeEnd(request.value(forHTTPHeaderField: "Range")) {
            body = body.prefix(end + 1)
            status = 206
        }
        let headers = ["Content-Length": "\(body.count)"]
        if let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers) {
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        }
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    private static func rangeEnd(_ header: String?) -> Int? {
        guard let header, header.hasPrefix("bytes=0-") else { return nil }
        return Int(header.dropFirst("bytes=0-".count))
    }
}
