import XCTest
@testable import AuraCore

final class KVLayoutTests: XCTestCase {

    private let mib: Int64 = 1_048_576

    // MARK: - Layer types from config.json

    func testQwen35TwoBCountsSixFullAttentionLayersOfTwentyFour() throws {
        let facts = try XCTUnwrap(LayoutFixture.facts(LayoutFixture.qwen35TwoB))
        let layout = try XCTUnwrap(facts.kvLayers)

        XCTAssertEqual(facts.numLayers, 24)
        XCTAssertEqual(layout.fullAttentionLayers, 6)
        XCTAssertEqual(layout.recurrentLayers, 18)
        XCTAssertEqual(layout.slidingLayers, 0)
        // Gated DeltaNet per layer: 16·128·128 state + 3·(2·16·128 + 16·128) conv, 2 bytes each.
        XCTAssertEqual(layout.recurrentStateBytes, 18 * 2 * (16 * 128 * 128 + 3 * (2 * 16 * 128 + 16 * 128)))
    }

    func testQwen35KVAtEightKCountsOnlyTheFullAttentionLayers() throws {
        let layout = try XCTUnwrap(LayoutFixture.facts(LayoutFixture.qwen35TwoB)?.kvLayers)
        let hybrid = LayoutFixture.model(layers: 24, kvHeads: 2, headDim: 256, kvLayers: layout)
        let uniform = LayoutFixture.model(layers: 24, kvHeads: 2, headDim: 256, kvLayers: nil)

        let expected = (6.0 * 4 * 2 * 256 * 8192 + Double(layout.recurrentStateBytes)) / 1_073_741_824
        XCTAssertEqual(hybrid.estimatedKVCacheGB(contextLength: 8192), expected, accuracy: 1e-9)
        XCTAssertEqual(uniform.estimatedKVCacheGB(contextLength: 8192), 24.0 * 4 * 2 * 256 * 8192 / 1_073_741_824,
                       accuracy: 1e-9)
    }

    func testQwen35IntervalDefaultsToFourWithoutLayerTypes() throws {
        let layout = try XCTUnwrap(LayoutFixture.facts(#"{"model_type": "qwen3_next", "num_hidden_layers": 48}"#)?.kvLayers)
        XCTAssertEqual(layout.fullAttentionLayers, 12)
        XCTAssertEqual(layout.recurrentLayers, 36)
        XCTAssertEqual(layout.recurrentStateBytes, 0, "no linear_* keys, no state to size")
    }

    func testGemma3SlidingLayersStopAtTheirWindowOnceDecodingStarts() throws {
        let facts = try XCTUnwrap(LayoutFixture.facts(LayoutFixture.gemma3TwelveB))
        let layout = try XCTUnwrap(facts.kvLayers)
        XCTAssertEqual(layout.fullAttentionLayers, 8)
        XCTAssertEqual(layout.slidingLayers, 40)
        XCTAssertEqual(layout.slidingWindow, 1024)
        XCTAssertNil(layout.slidingPrefillStep, "Gemma3Text.prepare runs the whole prompt in one pass")

        let perFullToken = 8.0 * 4 * 8 * 256
        let growth = layout.decodingCacheBytes(contextLength: 8192, kvHeads: 8, headDim: 256)
            - layout.decodingCacheBytes(contextLength: 4096, kvHeads: 8, headDim: 256)
        XCTAssertEqual(growth, perFullToken * 4096, accuracy: 1e-6, "past the window only the 8 global layers grow")
    }

    func testGemma3PeaksAtTheWholePromptOnEveryLayer() throws {
        let layout = try XCTUnwrap(LayoutFixture.facts(LayoutFixture.gemma3TwelveB)?.kvLayers)
        let gemma = LayoutFixture.model(layers: 48, kvHeads: 8, headDim: 256, kvLayers: layout)
        let uniform = LayoutFixture.model(layers: 48, kvHeads: 8, headDim: 256, kvLayers: nil)

        XCTAssertEqual(gemma.estimatedKVCacheGB(contextLength: 8192), uniform.estimatedKVCacheGB(contextLength: 8192),
                       accuracy: 1e-9)
    }

    func testChunkedPrefillHoldsTheWindowPlusOneStep() {
        let layout = KVCacheLayout(fullAttentionLayers: 0, slidingLayers: 1, slidingWindow: 1024, slidingPrefillStep: 512)
        XCTAssertEqual(layout.cacheBytes(contextLength: 8192, kvHeads: 1, headDim: 1), 4 * 1535)
        XCTAssertEqual(layout.cacheBytes(contextLength: 1000, kvHeads: 1, headDim: 1), 4 * 1000)
        XCTAssertEqual(layout.decodingCacheBytes(contextLength: 8192, kvHeads: 1, headDim: 1), 4 * 1024)
    }

    func testVisionEntriesPrefillTheWholePrompt() throws {
        let layout = KVCacheLayout(fullAttentionLayers: 2, slidingLayers: 8, slidingWindow: 512, slidingPrefillStep: 512)
        XCTAssertNil(layout.prefilledInOnePass().slidingPrefillStep)
        XCTAssertEqual(layout.prefilledInOnePass().slidingWindow, 512)
    }

    func testGemma3VisionConfigWithoutHeadsUsesTheRuntimeDefaults() throws {
        let facts = try XCTUnwrap(LayoutFixture.facts(#"""
            {"model_type": "gemma3", "text_config": {"model_type": "gemma3_text", "num_hidden_layers": 34,
             "hidden_size": 2560, "sliding_window": 1024}}
            """#))
        XCTAssertEqual(facts.kvHeads, 4)
        XCTAssertEqual(facts.headDim, 256)
        XCTAssertEqual(facts.kvLayers?.fullAttentionLayers, 5)
        XCTAssertEqual(facts.kvLayers?.slidingLayers, 29)
    }

    func testGemma4SharedLayersHoldNoCacheAndGlobalLayersUseTheirOwnWidth() throws {
        let layout = try XCTUnwrap(LayoutFixture.facts(#"""
            {"model_type": "gemma4", "text_config": {"model_type": "gemma4_text", "num_hidden_layers": 10,
             "num_key_value_heads": 2, "num_global_key_value_heads": 1, "attention_k_eq_v": true, "head_dim": 256,
             "global_head_dim": 512, "num_kv_shared_layers": 4, "sliding_window": 512,
             "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
                             "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
                             "sliding_attention", "full_attention"]}}
            """#)?.kvLayers)

        XCTAssertEqual(layout, KVCacheLayout(fullAttentionLayers: 1, slidingLayers: 5, slidingWindow: 512,
                                             fullAttentionKVHeads: 1, fullAttentionHeadDim: 512, slidingPrefillStep: 512))
        XCTAssertEqual(layout.cacheBytes(contextLength: 2048, kvHeads: 2, headDim: 256),
                       4.0 * 1 * 512 * 2048 + 5 * 4.0 * 2 * 256 * (512 + 511))
        XCTAssertEqual(layout.decodingCacheBytes(contextLength: 2048, kvHeads: 2, headDim: 256),
                       4.0 * 1 * 512 * 2048 + 5 * 4.0 * 2 * 256 * 512)
    }

    func testGemma4WithoutLayerTypesFollowsTheRuntimePatternOfFive() throws {
        let layout = try XCTUnwrap(LayoutFixture.facts(#"""
            {"model_type": "gemma4_text", "num_hidden_layers": 10, "num_key_value_heads": 2, "head_dim": 256,
             "global_head_dim": 256, "sliding_window": 512}
            """#)?.kvLayers)
        XCTAssertEqual(layout, KVCacheLayout(fullAttentionLayers: 2, slidingLayers: 8, slidingWindow: 512,
                                             slidingPrefillStep: 512))
    }

    func testSlidingLayerTypesCountAsFullWhereTheRuntimeIgnoresThem() throws {
        let gptOSS = try XCTUnwrap(LayoutFixture.facts(#"""
            {"model_type": "gpt_oss", "num_hidden_layers": 4, "num_key_value_heads": 8, "head_dim": 64,
             "sliding_window": 128,
             "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]}
            """#)?.kvLayers)
        let qwen2 = try XCTUnwrap(LayoutFixture.facts(#"""
            {"model_type": "qwen2", "num_hidden_layers": 4, "num_key_value_heads": 2, "head_dim": 128,
             "sliding_window": 4096,
             "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "full_attention"]}
            """#)?.kvLayers)

        XCTAssertEqual(gptOSS, KVCacheLayout(fullAttentionLayers: 2, slidingLayers: 2, slidingWindow: 128,
                                             slidingPrefillStep: 512))
        XCTAssertEqual(qwen2, KVCacheLayout(fullAttentionLayers: 4))
    }

    func testLFM2ConvolutionLayersKeepAFixedState() throws {
        let layout = try XCTUnwrap(LayoutFixture.facts(#"""
            {"model_type": "lfm2", "num_hidden_layers": 16, "num_key_value_heads": 8, "num_attention_heads": 32,
             "hidden_size": 2048, "conv_L_cache": 3, "full_attn_idxs": [2, 5, 8, 10, 12, 14]}
            """#)?.kvLayers)
        XCTAssertEqual(layout, KVCacheLayout(fullAttentionLayers: 6, recurrentLayers: 10,
                                             recurrentStateBytes: 10 * 2 * 2048 * 2))
    }

    func testGraniteHybridMambaLayersKeepAFixedState() throws {
        let layout = try XCTUnwrap(LayoutFixture.facts(#"""
            {"model_type": "granitemoehybrid", "num_hidden_layers": 4, "num_key_value_heads": 8, "head_dim": 64,
             "layer_types": ["mamba", "mamba", "attention", "mamba"], "mamba_n_heads": 4, "mamba_d_head": 8,
             "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_n_groups": 1}
            """#)?.kvLayers)
        // Per layer: 4·8·16 state + 3·(4·8 + 2·16) conv.
        XCTAssertEqual(layout, KVCacheLayout(fullAttentionLayers: 1, recurrentLayers: 3,
                                             recurrentStateBytes: 3 * 2 * (4 * 8 * 16 + 3 * (32 + 32))))
    }

    func testImplausibleConfigValuesLeaveTheLayoutUnknownOrFull() throws {
        XCTAssertNil(LayoutFixture.facts(#"{"model_type": "llama", "num_hidden_layers": 100000000}"#)?.kvLayers)
        XCTAssertNil(LayoutFixture.facts(#"""
            {"model_type": "qwen3_5", "num_hidden_layers": 8, "linear_num_value_heads": 9999999,
             "linear_value_head_dim": 128, "linear_num_key_heads": 16, "linear_key_head_dim": 128}
            """#)?.kvLayers)
        let hugeWindow = try XCTUnwrap(LayoutFixture.facts(
            #"{"model_type": "gemma3_text", "num_hidden_layers": 12, "sliding_window": 99999999999}"#)?.kvLayers)
        XCTAssertEqual(hugeWindow, KVCacheLayout(fullAttentionLayers: 12), "no plausible window: cache in full")
    }

    // MARK: - GGUF per-layer cache

    func testGemma4GGUFSizesGlobalAndSlidingLayersAtTheirOwnWidths() throws {
        let shape = try XCTUnwrap(KVCacheLayout.derive(gguf: LayoutFixture.gemma4TwelveBHeader()))

        XCTAssertEqual(shape.kvHeads, 8)
        XCTAssertEqual(shape.headDim, 256)
        XCTAssertEqual(shape.layout, KVCacheLayout(fullAttentionLayers: 8, slidingLayers: 40, slidingWindow: 0,
                                                   fullAttentionKVHeads: 1, fullAttentionHeadDim: 512))
        // llama.cpp b8851 with swa_full: 40 × 2 × 8 × 256 × 2 B + 8 × 2 × 1 × 512 × 2 B per token on every layer.
        XCTAssertEqual(shape.layout.cacheBytes(contextLength: 2048, kvHeads: 8, headDim: 256), 344_064 * 2048)
    }

    func testGGUFSharedKVLayersHoldNoCache() throws {
        let shape = try XCTUnwrap(KVCacheLayout.derive(gguf: LayoutFixture.gemma4E2BHeader()))

        XCTAssertEqual(shape.layout.cachedLayers, 15, "the last 20 of 35 layers reuse an earlier cache")
        XCTAssertEqual(shape.layout.fullAttentionLayers, 3)
        XCTAssertEqual(shape.kvHeads, 1)
        XCTAssertEqual(shape.layout.cacheBytes(contextLength: 1, kvHeads: 1, headDim: 256), 3 * 4 * 512 + 12 * 4 * 256)
    }

    func testGGUFWithoutAPerLayerPatternStaysUniform() {
        var metadata = LayoutFixture.gemma4TwelveBHeader().metadata
        metadata["gemma4.attention.sliding_window_pattern"] = nil
        XCTAssertNil(KVCacheLayout.derive(gguf: LayoutFixture.header(metadata)))

        var sameWidths = LayoutFixture.gemma4TwelveBHeader().metadata
        sameWidths["gemma4.attention.key_length_swa"] = .uint32(512)
        sameWidths["gemma4.attention.head_count_kv"] = .uint32(8)
        XCTAssertNil(KVCacheLayout.derive(gguf: LayoutFixture.header(sameWidths)),
                     "one width and one head count on every layer is the uniform estimate already")
    }

    func testGemma4GGUFReportUsesTheSlidingWidthAndExplainsTheSplit() throws {
        let snapshot = Fixture.gguf(header: LayoutFixture.gemma4TwelveBHeader(), architecture: "gemma4")
        let report = CompatibilityEvaluator.evaluate(snapshot, on: .mac32GB)
        let entry = try XCTUnwrap(report.suggestedEntry)
        let finding = try XCTUnwrap(report.findings.first { $0.rule == "kv" })

        XCTAssertEqual(report.overview.headDim, 256)
        XCTAssertEqual(entry.headDim, 256)
        XCTAssertEqual(entry.kvLayers?.fullAttentionHeadDim, 512)
        XCTAssertEqual(entry.decodedModel()?.kvLayers, entry.kvLayers)
        XCTAssertEqual(finding.title, "KV cache: 8 global and 40 sliding-window layers")
        XCTAssertTrue(finding.detail.contains("with every layer at 8 × 512"), finding.detail)
    }

    func testGemma3ReportSaysTheSlidingLayersHoldThePromptUntilDecoding() throws {
        let snapshot = Fixture.mlx("o/gemma3", config: LayoutFixture.facts(LayoutFixture.gemma3TwelveB), weightMap: nil)
        let finding = try XCTUnwrap(CompatibilityEvaluator.evaluate(snapshot, on: .mac32GB).findings.first { $0.rule == "kv" })

        XCTAssertEqual(finding.title, "KV cache grows on 8 of 48 layers once decoding starts")
        XCTAssertTrue(finding.detail.contains("3072 MB at the prefill peak, 832 MB once decoding starts"), finding.detail)
    }

    // MARK: - Context window

    func testHybridModelEarnsALargerWindowThanTheSameModelSizedUniformly() throws {
        let layout = KVCacheLayout(fullAttentionLayers: 8, slidingLayers: 40, slidingWindow: 1024, slidingPrefillStep: 512)
        let hybrid = LayoutFixture.model(sizeMB: 7657, layers: 48, kvHeads: 8, headDim: 256, kvLayers: layout)
        let uniform = LayoutFixture.model(sizeMB: 7657, layers: 48, kvHeads: 8, headDim: 256, kvLayers: nil)

        let wide = HardwareAnalyzer.recommendedContextWindow(for: hybrid, availableGB: 10)
        let narrow = HardwareAnalyzer.recommendedContextWindow(for: uniform, availableGB: 10)
        XCTAssertGreaterThan(wide, narrow)
    }

    func testContextWindowIsTheLargestFittingMultipleOf512() throws {
        let layout = try XCTUnwrap(LayoutFixture.facts(LayoutFixture.gemma3TwelveB)?.kvLayers)
        let model = LayoutFixture.model(sizeMB: 7657, layers: 48, kvHeads: 8, headDim: 256, kvLayers: layout)
        let available = 8.5
        let free = available - 7657.0 / 1024 - 0.4

        let window = HardwareAnalyzer.recommendedContextWindow(for: model, availableGB: available)

        XCTAssertEqual(window % 512, 0)
        XCTAssertLessThanOrEqual(model.estimatedKVCacheGB(contextLength: window), free)
        XCTAssertGreaterThan(model.estimatedKVCacheGB(contextLength: window + 512), free)
    }

    func testOnlySlidingLayersReachThePlatformCeiling() {
        let model = LayoutFixture.model(sizeMB: 1000, layers: 20, kvHeads: 8, headDim: 256,
                                        kvLayers: KVCacheLayout(fullAttentionLayers: 0, slidingLayers: 20,
                                                                slidingWindow: 1024, slidingPrefillStep: 512))
        #if os(macOS)
        XCTAssertEqual(HardwareAnalyzer.recommendedContextWindow(for: model, availableGB: 4), 32_768)
        #else
        XCTAssertEqual(HardwareAnalyzer.recommendedContextWindow(for: model, availableGB: 4), 8192)
        #endif
    }

    // MARK: - Safetensors sizes and experts

    func testSafetensorsHeaderKeepsTheBytesOfEveryTensor() throws {
        let data = LayoutFixture.safetensors(["a.weight": 4096, "b.scales": 100])
        let header = try XCTUnwrap(SafetensorsHeader.parse(data))
        XCTAssertEqual(header.tensorBytes, ["a.weight": 4096, "b.scales": 100])

        let inverted = Data(#"{"x": {"dtype": "F16", "shape": [1], "data_offsets": [10, 5]}}"#.utf8)
        let parsed = try XCTUnwrap(SafetensorsHeader.parse(GGUFBytes.littleEndian(UInt64(inverted.count)) + inverted))
        XCTAssertEqual(parsed.tensorNames, ["x"])
        XCTAssertTrue(parsed.tensorBytes.isEmpty)
    }

    func testExpertTensorsAreTheRoutedOnes() {
        XCTAssertTrue(ExpertRouting.isExpert("language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"))
        XCTAssertTrue(ExpertRouting.isExpert("model.layers.3.block_sparse_moe.experts.7.w1.weight"))
        XCTAssertTrue(ExpertRouting.isExpert("model.layers.0.mlp.experts.gate_up_proj"))
        XCTAssertTrue(ExpertRouting.isExpert("model.layers.0.block_sparse_moe.input_linear.weight"))
        XCTAssertFalse(ExpertRouting.isExpert("language_model.model.layers.0.mlp.shared_expert.up_proj.weight"))
        XCTAssertFalse(ExpertRouting.isExpert("model.layers.0.mlp.shared_experts.down_proj.weight"))
        XCTAssertFalse(ExpertRouting.isExpert("model.layers.0.block_sparse_moe.gate.weight"))
        XCTAssertFalse(ExpertRouting.isExpert("model.layers.0.block_sparse_moe.router.layer.weight"))
    }

    func testActiveBytesAreNonExpertPlusTheRoutedShareOfExperts() throws {
        let bytes: [String: Int64] = [
            "language_model.model.embed_tokens.weight": 1000 * mib,
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": 25_600 * mib,
            "language_model.model.layers.0.mlp.shared_expert.up_proj.weight": 300 * mib,
            "language_model.model.layers.0.mlp.gate.weight": 100 * mib,
            "vision_tower.blocks.0.weight": 5000 * mib,
            "mtp.layers.0.weight": 7000 * mib,
        ]
        let routing = try XCTUnwrap(ExpertRouting.measure(tensorBytes: bytes, routedExperts: 256, expertsPerToken: 8))
        XCTAssertEqual(routing, ExpertRouting(routedExperts: 256, expertsPerToken: 8, activeSizeMB: 1400 + 800))

        XCTAssertNil(ExpertRouting.measure(tensorBytes: bytes, routedExperts: 8, expertsPerToken: 9))
        XCTAssertNil(ExpertRouting.measure(tensorBytes: ["model.embed.weight": mib], routedExperts: 8, expertsPerToken: 2))
    }

    func testOrnithMoEIsNoLongerSlowOnTheThirtyTwoGigabyteMac() throws {
        var snapshot = LayoutFixture.ornithSnapshot()
        let report = CompatibilityEvaluator.evaluate(snapshot, on: .mac32GB)
        let fit = try XCTUnwrap(report.weightsFit)
        let range = try XCTUnwrap(fit.tokensPerSecondRange)
        let entry = try XCTUnwrap(report.suggestedEntry)
        let model = try XCTUnwrap(entry.decodedModel())

        XCTAssertEqual(entry.experts, ExpertRouting(routedExperts: 256, expertsPerToken: 8, activeSizeMB: 1865))
        XCTAssertEqual(entry.kvLayers?.fullAttentionLayers, 10)
        XCTAssertEqual(range.upperBound,
                       1 / ((1865.0 / 1024) / (200 * HardwareAnalyzer.bandwidthEfficiency) + HardwareAnalyzer.perTokenOverhead),
                       accuracy: 1e-9)
        XCTAssertEqual(range.lowerBound, range.upperBound * HardwareAnalyzer.uncalibratedMoEFloor, accuracy: 1e-9)
        XCTAssertEqual(fit.tokensPerSecond, range.lowerBound)
        let assessed = HardwareAnalyzer.assess(model, profile: DevicePreset.mac32GB.memoryProfile)
        XCTAssertFalse([.slow, .unusable].contains(assessed.speedLevel), assessed.verdict)
        XCTAssertTrue(assessed.verdict.contains("uncalibrated MoE"), assessed.verdict)
        XCTAssertTrue(report.findings.contains { $0.rule == "moe" })

        snapshot.tensorBytes = nil
        let unsized = CompatibilityEvaluator.evaluate(snapshot, on: .mac32GB)
        let dense = try XCTUnwrap(unsized.suggestedEntry?.decodedModel())
        XCTAssertNil(unsized.weightsFit?.tokensPerSecondRange)
        XCTAssertEqual(HardwareAnalyzer.assess(dense, profile: DevicePreset.mac32GB.memoryProfile).speedLevel, .slow,
                       "without tensor sizes the whole download is charged per token")
    }

    func testDenseModelsKeepASingleDecodeFigure() {
        let result = HardwareAnalyzer.assess(.llama3_1_8b_gguf, profile: DevicePreset.mac32GB.memoryProfile)
        XCTAssertNotNil(result.estimatedDecodeTokensPerSecond)
        XCTAssertNil(result.estimatedDecodeTokensPerSecondRange)
        XCTAssertFalse(result.verdict.contains("MoE"))
    }

    // MARK: - Catalog

    func testHybridEntryWritesAndDecodesItsLayoutAndExperts() throws {
        let report = CompatibilityEvaluator.evaluate(LayoutFixture.ornithSnapshot(), on: .mac32GB)
        let entry = try XCTUnwrap(report.suggestedEntry)
        let text = entry.jsonText()

        XCTAssertTrue(text.contains(#""kvHeads": 2,"#), text)
        XCTAssertTrue(text.contains(#""headDim": 256,"#), text)
        XCTAssertTrue(text.contains(#""kvLayers": {"fullAttentionLayers": 10, "slidingLayers": 0, "slidingWindow": 0, "recurrentLayers": 30, "#), text)
        XCTAssertTrue(text.contains(#""experts": {"routedExperts": 256, "expertsPerToken": 8, "activeSizeMB": 1865}"#), text)
        let model = try XCTUnwrap(entry.decodedModel())
        XCTAssertEqual(model.kvLayers, entry.kvLayers)
        XCTAssertEqual(model.experts, entry.experts)
    }

    func testEntriesWithoutTheNewKeysOrWithPartialOnesStillDecode() throws {
        let common = #""repoID":"a/b","category":"text","docTags":false,"format":"mlx","approximateSizeMB":100,"#
            + #""isUncensored":false,"ggufFilename":null,"defaultDocumentPrompt":null,"numLayers":24"#
        let legacy = #"{"id":"x","displayName":"X","# + common + #","kvHeads":0,"headDim":0}"#
        let partial = #"{"id":"y","displayName":"Y","# + common
            + #","kvHeads":2,"headDim":256,"kvLayers":{"fullAttentionLayers":6},"experts":{}}"#

        let old = try JSONDecoder().decode(Model.self, from: Data(legacy.utf8))
        let new = try JSONDecoder().decode(Model.self, from: Data(partial.utf8))

        XCTAssertNil(old.kvLayers)
        XCTAssertNil(old.experts)
        XCTAssertEqual(old.estimatedKVCacheGB(contextLength: 2048), 0.15, "no KV metadata keeps the flat estimate")
        XCTAssertEqual(new.kvLayers, KVCacheLayout(fullAttentionLayers: 6))
        XCTAssertEqual(new.experts?.activeSizeMB, 0)
        XCTAssertNotNil(HardwareAnalyzer.assess(new, profile: DevicePreset.mac32GB.memoryProfile).estimatedDecodeTokensPerSecond,
                        "an empty experts object falls back to the whole download")
    }

    func testBundledCatalogStillLoads() {
        XCTAssertFalse(ModelRegistry.shared.models.isEmpty)
    }

    func testUniformDenseEntryOmitsTheLayout() throws {
        let snapshot = Fixture.mlx("o/dense", config: Fixture.config(type: "llama"), weightMap: nil, sizes: [600_000_000])
        let entry = try XCTUnwrap(CompatibilityEvaluator.evaluate(snapshot, on: .mac32GB).suggestedEntry)
        XCTAssertNil(entry.kvLayers)
        XCTAssertFalse(entry.jsonText().contains("kvLayers"))
    }
}

// MARK: - Checker over a stubbed Hugging Face

final class ExpertHeaderCheckerTests: XCTestCase {

    private let stubs = StubHub.shared

    override func setUp() {
        super.setUp()
        stubs.reset()
    }

    func testMixtureOfExpertsReadsEveryShardHeader() async throws {
        let repo = "stub/moe-mlx"
        let mib: Int64 = 1_048_576
        stubs.serve("api/models/\(repo)?blobs=true", StubHub.listing(repo, files: [
            ("config.json", 900), ("tokenizer.json", 9_000_000), ("model.safetensors.index.json", 4000),
            ("model-00001-of-00002.safetensors", 1000 * mib), ("model-00002-of-00002.safetensors", 25_600 * mib),
        ], tags: ["mlx"]))
        stubs.serve("\(repo)/resolve/main/config.json", Data(#"""
            {"model_type": "qwen3_moe", "quantization": {"bits": 4}, "num_hidden_layers": 2, "num_key_value_heads": 4,
             "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8}
            """#.utf8))
        stubs.serve("\(repo)/resolve/main/model.safetensors.index.json", Data(#"""
            {"weight_map": {"model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                            "model.layers.0.mlp.switch_mlp.up_proj.weight": "model-00002-of-00002.safetensors"}}
            """#.utf8))
        stubs.serve("\(repo)/resolve/main/model-00001-of-00002.safetensors",
                    LayoutFixture.safetensors(["model.embed_tokens.weight": 1000 * mib]) + Data(count: 1000))
        stubs.serve("\(repo)/resolve/main/model-00002-of-00002.safetensors",
                    LayoutFixture.safetensors(["model.layers.0.mlp.switch_mlp.up_proj.weight": 25_600 * mib]) + Data(count: 1000))

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        let checker = ModelCompatibilityChecker(transport: URLSession(configuration: configuration),
                                                authorizer: NoDownloadAuth(), ggufHeaderBytes: 16_384)
        let report = await checker.check(repo, on: .mac32GB)

        XCTAssertEqual(report.overview.experts, ExpertRouting(routedExperts: 128, expertsPerToken: 8, activeSizeMB: 1000 + 1600))
        XCTAssertEqual(stubs.requests(for: "model-00002-of-00002.safetensors").first?.value(forHTTPHeaderField: "Range"),
                       "bytes=0-262143")
        XCTAssertNotNil(report.weightsFit?.tokensPerSecondRange)
    }
}

// MARK: - Fixtures

enum LayoutFixture {
    static let qwen35TwoB: String = {
        let types = (0..<24).map { ($0 + 1) % 4 == 0 ? "\"full_attention\"" : "\"linear_attention\"" }
        return """
            {"model_type": "qwen3_5", "quantization": {"group_size": 64, "bits": 4},
             "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 24, "num_key_value_heads": 2,
                             "num_attention_heads": 8, "head_dim": 256, "hidden_size": 2048, "full_attention_interval": 4,
                             "layer_types": [\(types.joined(separator: ", "))], "linear_conv_kernel_dim": 4,
                             "linear_key_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 16,
                             "linear_value_head_dim": 128, "max_position_embeddings": 262144}}
            """
    }()

    static let gemma3TwelveB = #"""
        {"model_type": "gemma3", "text_config": {"model_type": "gemma3_text", "num_hidden_layers": 48,
         "num_key_value_heads": 8, "num_attention_heads": 16, "head_dim": 256, "hidden_size": 3840,
         "sliding_window": 1024, "sliding_window_pattern": 6}}
        """#

    static let ornithConfig = #"""
        {"model_type": "qwen3_5_moe", "quantization": {"group_size": 64, "bits": 4}, "num_hidden_layers": 40,
         "num_key_value_heads": 2, "num_attention_heads": 16, "head_dim": 256, "hidden_size": 2048,
         "full_attention_interval": 4, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
         "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_value_head_dim": 128,
         "num_experts": 256, "num_experts_per_tok": 8, "max_position_embeddings": 262144}
        """#

    static func facts(_ json: String) -> ModelConfigFacts? {
        ModelConfigFacts.parse(Data(json.utf8))
    }

    static func header(_ metadata: [String: GGUFValue]) -> GGUFHeader {
        GGUFHeader(version: 3, tensorCount: 0, declaredKeyCount: UInt64(metadata.count), metadata: metadata,
                   isComplete: true)
    }

    static func flags(_ values: [Bool]) -> GGUFValue {
        .array(GGUFArray(elementType: .bool, count: UInt64(values.count), elements: values.map(GGUFValue.bool)))
    }

    /// The keys of gemma4-coding-Q4_K_M.gguf that size its cache: every sixth of 48 blocks global, with one KV head.
    static func gemma4TwelveBHeader() -> GGUFHeader {
        let heads = (0..<48).map { GGUFValue.uint32($0 % 6 == 5 ? 1 : 8) }
        return header([
            "general.architecture": .string("gemma4"),
            "gemma4.block_count": .uint32(48),
            "gemma4.attention.head_count": .uint32(16),
            "gemma4.attention.head_count_kv": .array(GGUFArray(elementType: .uint32, count: 48, elements: heads)),
            "gemma4.attention.key_length": .uint32(512),
            "gemma4.attention.key_length_swa": .uint32(256),
            "gemma4.attention.sliding_window": .uint32(1024),
            "gemma4.attention.sliding_window_pattern": flags((0..<48).map { $0 % 6 != 5 }),
        ])
    }

    /// security-gemma-4-e2b-it.Q4_K_M.gguf: 35 blocks, every fifth global, one KV head, the last 20 shared.
    static func gemma4E2BHeader() -> GGUFHeader {
        header([
            "general.architecture": .string("gemma4"),
            "gemma4.block_count": .uint32(35),
            "gemma4.attention.head_count": .uint32(8),
            "gemma4.attention.head_count_kv": .uint32(1),
            "gemma4.attention.key_length": .uint32(512),
            "gemma4.attention.key_length_swa": .uint32(256),
            "gemma4.attention.sliding_window": .uint32(512),
            "gemma4.attention.shared_kv_layers": .uint32(20),
            "gemma4.attention.sliding_window_pattern": flags((0..<35).map { $0 % 5 != 4 }),
        ])
    }

    static func model(sizeMB: Int = 1000, layers: Int, kvHeads: Int, headDim: Int, kvLayers: KVCacheLayout?) -> Model {
        Model(id: "layout", repoID: "o/layout", displayName: "layout", category: .text, domain: nil, docTags: false,
              format: .mlx, approximateSizeMB: sizeMB, isUncensored: false, license: nil, ggufFilename: nil,
              defaultDocumentPrompt: nil, numLayers: layers, kvHeads: kvHeads, headDim: headDim, kvLayers: kvLayers,
              experts: nil, maxContextLength: nil, downloadURL: nil, localFileURL: nil)
    }

    /// Ornith 1.5 35B-A3B 4-bit as measured from its four shard headers: 1325 MiB non-expert, 17 280 MiB experts.
    static func ornithSnapshot() -> RepoSnapshot {
        let mib: Int64 = 1_048_576
        var snapshot = Fixture.mlx("ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit", config: facts(ornithConfig),
                                   weightMap: Fixture.weightMap(["language_model": 2]),
                                   sizes: [1325 * mib, 17_280 * mib])
        snapshot.tensorBytes = [
            "language_model.model.embed_tokens.weight": 1325 * mib,
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": 17_280 * mib,
        ]
        return snapshot
    }

    static func safetensors(_ sizes: [String: Int64]) -> Data {
        var offset: Int64 = 0
        let entries = sizes.sorted { $0.key < $1.key }.map { name, size -> (String, Any) in
            defer { offset += size }
            return (name, ["dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]] as [String: Any])
        }
        let json = (try? JSONSerialization.data(withJSONObject: Dictionary(uniqueKeysWithValues: entries))) ?? Data()
        return GGUFBytes.littleEndian(UInt64(json.count)) + json
    }
}
