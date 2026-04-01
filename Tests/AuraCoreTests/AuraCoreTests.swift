import XCTest
@testable import AuraCore

// MARK: - Model Tests

final class ModelTests: XCTestCase {

    func testAllModelsHaveDisplayName() {
        for model in Model.allCases {
            XCTAssertFalse(model.displayName.isEmpty, "\(model) has empty displayName")
        }
    }

    func testAllModelsHavePositiveSize() {
        for model in Model.allCases {
            XCTAssertGreaterThan(model.approximateSizeMB, 0, "\(model) has zero size")
        }
    }

    func testModelFormatConsistency() {
        // MLX models should NOT have ggufFilename
        for model in Model.mlxModels {
            XCTAssertEqual(model.format, .mlx)
            XCTAssertNil(model.ggufFilename, "\(model) is MLX but has ggufFilename")
        }

        // GGUF models MUST have ggufFilename
        for model in Model.ggufModels {
            XCTAssertEqual(model.format, .gguf)
            XCTAssertNotNil(model.ggufFilename, "\(model) is GGUF but has no ggufFilename")
        }
    }

    func testGGUFModelsAreTextOnly() {
        // All current GGUF models should be text purpose
        for model in Model.ggufModels {
            if case .text = model.purpose {
                // OK
            } else {
                XCTFail("\(model) is GGUF but not text purpose")
            }
        }
    }

    func testModelCollections() {
        XCTAssertFalse(Model.textModels.isEmpty)
        XCTAssertFalse(Model.visionModels.isEmpty)
        XCTAssertFalse(Model.specializedModels.isEmpty)
        XCTAssertFalse(Model.ggufModels.isEmpty)
        XCTAssertFalse(Model.mlxModels.isEmpty)

        // No overlap between MLX and GGUF
        let mlxSet = Set(Model.mlxModels)
        let ggufSet = Set(Model.ggufModels)
        XCTAssertTrue(mlxSet.isDisjoint(with: ggufSet))

        // Union should cover all cases
        XCTAssertEqual(mlxSet.count + ggufSet.count, Model.allCases.count)
    }

    func testMacOSRecommendedModels() {
        let macOnly = Model.allCases.filter { $0.isMacOSRecommended }
        // 70B and 32B should be macOS recommended
        XCTAssertTrue(macOnly.contains(.llama3_1_70b_gguf))
        XCTAssertTrue(macOnly.contains(.qwen2_5_32b_gguf))
        // 7B should NOT be macOS-only
        XCTAssertFalse(macOnly.contains(.llama3_1_8b_gguf))
    }
}

// MARK: - HardwareAnalyzer Tests

final class HardwareAnalyzerTests: XCTestCase {

    func testAssessSmallModelFitsEasily() {
        // Simulate a device with 8 GB RAM, 5 GB available
        let profile = HardwareProfile(
            totalMemoryGB: 8.0,
            availableMemoryGB: 5.0,
            deviceName: "Test Device"
        )

        let result = HardwareAnalyzer.assess(.qwen3_0_6b, profile: profile)
        XCTAssertEqual(result.fitLevel, .excellent, "0.6B model should fit easily on 8 GB device")
    }

    func testAssessLargeModelNeedsStreaming() {
        // Simulate a 6 GB iPhone with 3 GB available
        let profile = HardwareProfile(
            totalMemoryGB: 6.0,
            availableMemoryGB: 3.0,
            deviceName: "iPhone 14"
        )

        let result = HardwareAnalyzer.assess(.llama3_1_8b_gguf, profile: profile)
        // 8B model needs ~5 GB for full load, only 3 GB available → streaming required
        XCTAssertEqual(result.fitLevel, .streamingRequired,
                       "8B model on 6GB device should require streaming")
    }

    func testAssessHugeModelTooLarge() {
        // 70B model should be too large even for streaming on a 6 GB device
        let profile = HardwareProfile(
            totalMemoryGB: 6.0,
            availableMemoryGB: 3.0,
            deviceName: "iPhone 14"
        )

        let result = HardwareAnalyzer.assess(.llama3_1_70b_gguf, profile: profile)
        XCTAssertEqual(result.fitLevel, .tooLarge,
                       "70B model should be too large even for streaming on 6 GB")
    }

    func testMLXModelCannotStream() {
        // MLX models can't use streaming, so they go directly to tooLarge
        let profile = HardwareProfile(
            totalMemoryGB: 2.0,
            availableMemoryGB: 1.0,
            deviceName: "Tiny Device"
        )

        let result = HardwareAnalyzer.assess(.qwen3_4b, profile: profile)
        // 4B MLX model needs ~3 GB, only 1 GB available, can't stream MLX → tooLarge
        XCTAssertEqual(result.fitLevel, .tooLarge)
    }

    func testGQAAwareKVCache() {
        // Llama 3.1 8B with GQA (8 KV heads): should be much less than non-GQA
        let kvCache = Model.llama3_1_8b_gguf.estimatedKVCacheGB(contextLength: 2048)
        // 2 * 32 layers * 8 kv_heads * 128 head_dim * 2048 seq * 2 bytes / 1 GB
        let expected = Double(2 * 32 * 8 * 128 * 2048 * 2) / 1_073_741_824
        XCTAssertEqual(kvCache, expected, accuracy: 0.001)
        // Should be ~0.25 GB, NOT ~1 GB like full-attention
        XCTAssertLessThan(kvCache, 0.5)
    }

    func testStreamingMemoryEstimate() {
        let streamingGB = Model.llama3_1_8b_gguf.estimatedStreamingMemoryGB
        // Streaming should use much less than full load
        let fullGB = Model.llama3_1_8b_gguf.estimatedRuntimeMemoryGB
        XCTAssertLessThan(streamingGB, fullGB * 0.5,
                          "Streaming memory should be much less than full load")
        // Should be under 1 GB for practical use on iOS
        XCTAssertLessThan(streamingGB, 1.5)
    }

    func testFitLevelIsRunnable() {
        XCTAssertTrue(ModelFitLevel.excellent.isRunnable)
        XCTAssertTrue(ModelFitLevel.good.isRunnable)
        XCTAssertTrue(ModelFitLevel.marginal.isRunnable)
        XCTAssertTrue(ModelFitLevel.streamingRequired.isRunnable)
        XCTAssertFalse(ModelFitLevel.tooLarge.isRunnable)
    }

    func testCompatibleModelsOrdering() {
        let profile = HardwareProfile(
            totalMemoryGB: 8.0,
            availableMemoryGB: 4.0,
            deviceName: "Test"
        )
        let results = HardwareAnalyzer.compatibleModels(profile: profile)

        // tooLarge models should be last
        var seenTooLarge = false
        for r in results {
            if r.fitLevel == .tooLarge {
                seenTooLarge = true
            } else if seenTooLarge {
                XCTFail("Non-tooLarge model found after tooLarge models")
            }
        }
    }
}

// MARK: - BackendRouter Tests

final class BackendRouterTests: XCTestCase {

    @MainActor
    func testMLXModelUsesMLXBackend() {
        let backend = BackendRouter.recommendedBackend(for: .qwen3_1_7b)
        XCTAssertEqual(backend, .mlx)
    }

    @MainActor
    func testGGUFModelOnLargeDeviceUsesLlamaCpp() {
        let profile = HardwareProfile(
            totalMemoryGB: 32.0,
            availableMemoryGB: 20.0,
            deviceName: "Mac"
        )
        let backend = BackendRouter.recommendedBackend(for: .llama3_1_8b_gguf, profile: profile)
        XCTAssertEqual(backend, .llamaCpp)
    }

    @MainActor
    func testGGUFModelOnSmallDeviceUsesStreaming() {
        let profile = HardwareProfile(
            totalMemoryGB: 6.0,
            availableMemoryGB: 3.0,
            deviceName: "iPhone"
        )
        let backend = BackendRouter.recommendedBackend(for: .llama3_1_8b_gguf, profile: profile)
        XCTAssertEqual(backend, .layerStreaming)
    }
}

// MARK: - MemoryBudgetManager Tests

final class MemoryBudgetManagerTests: XCTestCase {

    @MainActor
    func testRecommendedContextReducesUnderPressure() {
        let mgr = MemoryBudgetManager(safetyMarginMB: 200)
        // With plenty of memory, should return base context
        let ctx = mgr.recommendedContextLength(baseContext: 4096)
        XCTAssertGreaterThan(ctx, 0)
        XCTAssertLessThanOrEqual(ctx, 4096)
    }

    @MainActor
    func testPlatformBudgetReturnsValidValues() {
        let budget = MemoryBudgetManager.platformBudget()
        XCTAssertGreaterThan(budget.maxModelSizeGB, 0)
        XCTAssertGreaterThan(budget.maxContextLength, 0)
        XCTAssertGreaterThanOrEqual(budget.maxGPULayers, 0)
    }
}
