import XCTest
@testable import AuraCore

final class ContextWindowOverflowTests: XCTestCase {

    func testFittingGenerationIsNotAnOverflow() {
        XCTAssertNil(ContextWindowOverflow.detect(promptTokens: 1000, generatedTokens: 1048, windowSize: 2048))
        XCTAssertNil(ContextWindowOverflow.detect(promptTokens: 9000, generatedTokens: 9000, windowSize: 0))
    }

    func testOverflowCountsTheTokensThatNoLongerFit() throws {
        let overflow = try XCTUnwrap(ContextWindowOverflow.detect(promptTokens: 1900, generatedTokens: 300,
                                                                  windowSize: 2048))
        XCTAssertEqual(overflow.droppedTokens, 152)
    }

    #if canImport(MLXLLM)
    @MainActor
    func testMLXBackendBoundsItsKVCacheToTheRecommendedWindow() throws {
        let model = try XCTUnwrap(ModelRegistry.shared.models.first { $0.format == .mlx }, "catalog has no MLX entry")
        let backend = MLXBackend(model: model)
        XCTAssertEqual(backend.parameters.maxKVSize, HardwareAnalyzer.recommendedContextWindow(for: model))
        XCTAssertNil(backend.lastContextOverflow)
    }
    #endif
}
