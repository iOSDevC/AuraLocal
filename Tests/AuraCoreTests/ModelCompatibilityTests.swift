import XCTest
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

        XCTAssertEqual(phone.status, .notRunnable)
        XCTAssertEqual(phone.weightsFit?.rating, .tooLarge)
        XCTAssertTrue(phone.blockers.contains { $0.rule == "fit" })
        XCTAssertFalse(DevicePreset.iPhone6GB.isMeasured)
        XCTAssertTrue(DevicePreset.mac32GB.isMeasured)
    }

    func testEveryPresetIsReachableByID() {
        for preset in DevicePreset.classes {
            XCTAssertEqual(DevicePreset.named(preset.id)?.id, preset.id)
        }
        XCTAssertEqual(DevicePreset.named("this-device")?.id, DevicePreset.thisDeviceID)
        XCTAssertNil(DevicePreset.named("toaster"))
    }

    // MARK: - Gating and licenses

    func testGatingAndLicenseFindings() {
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

    func testEntryNamesAndEscaping() {
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

    func testRepoIDNormalization() {
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
