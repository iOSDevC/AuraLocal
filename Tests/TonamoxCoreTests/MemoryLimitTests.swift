import XCTest
@testable import TonamoxCore

final class MemoryLimitTests: XCTestCase {
    private let gib = 1 << 30

    func testLimitIsAvailablePlusFootprint() {
        XCTAssertEqual(HardwareProfile.memoryLimitBytes(available: 2 * gib, footprint: gib,
                                                        physicalMemory: UInt64(8 * gib)), 3 * gib)
    }

    func testLimitIsCappedAt74PercentOfRAM() {
        let reportedOnAnEightGBiPad = 8142 * (1 << 20)
        let limit = HardwareProfile.memoryLimitBytes(available: reportedOnAnEightGBiPad, footprint: gib / 4,
                                                     physicalMemory: UInt64(8 * gib))
        XCTAssertEqual(limit, Int(Double(8 * gib) * 0.74))
    }

    func testUnknownAvailableMemoryStaysUnknown() {
        XCTAssertNil(HardwareProfile.memoryLimitBytes(available: nil, footprint: gib, physicalMemory: UInt64(8 * gib)))
        XCTAssertNil(HardwareProfile.memoryLimitBytes(available: 0, footprint: gib, physicalMemory: UInt64(8 * gib)))
    }

    func testUnknownFootprintFallsBackToAvailableAlone() {
        XCTAssertEqual(HardwareProfile.memoryLimitBytes(available: 2 * gib, footprint: nil,
                                                        physicalMemory: UInt64(8 * gib)), 2 * gib)
    }

    func testFootprintIsReadableHere() throws {
        let footprint = try XCTUnwrap(HardwareProfile.physicalFootprintBytes())
        XCTAssertGreaterThan(footprint, 0)
    }

    // MARK: - Load gate

    private let resident = Model.qwen3_4b
    private let next = Model.medgemma_1_5_4b

    /// A limit that holds `next` alone, or `resident` alone, but not both.
    private var limitForOne: HardwareProfile {
        let gb = next.estimatedRuntimeMemoryGB + resident.estimatedRuntimeMemoryGB * 0.5
        return HardwareProfile(totalMemoryGB: 8, availableMemoryGB: gb, deviceName: "iPhone")
    }

    @MainActor
    func testAResidentModelThatLeavesTooLittleOfTheLimitIsEvictedFirst() throws {
        XCTAssertEqual(next.format, .mlx, "an MLX model cannot fall back to streaming")
        let evictions = ModelManager.evictionsBeforeLoading(next, residents: [resident], budget: 2,
                                                            profile: limitForOne, residentsCountAgainstProfile: true)
        XCTAssertEqual(evictions, 1)
    }

    @MainActor
    func testFreeMemoryReadingsAlreadyLeaveResidentModelsOut() {
        let evictions = ModelManager.evictionsBeforeLoading(next, residents: [resident], budget: 2,
                                                            profile: limitForOne, residentsCountAgainstProfile: false)
        XCTAssertEqual(evictions, 0)
    }

    @MainActor
    func testModelsThatFitTogetherStayAndTheBudgetStillEvicts() {
        let roomy = HardwareProfile(totalMemoryGB: 64, availableMemoryGB: 40, deviceName: "Mac")
        XCTAssertEqual(ModelManager.evictionsBeforeLoading(next, residents: [resident], budget: 2, profile: roomy,
                                                           residentsCountAgainstProfile: true), 0)
        XCTAssertEqual(ModelManager.evictionsBeforeLoading(next, residents: [resident], budget: 1, profile: roomy,
                                                           residentsCountAgainstProfile: true), 1)
    }

    @MainActor
    func testAModelTooLargeAloneIsRefused() {
        let tiny = HardwareProfile(totalMemoryGB: 4, availableMemoryGB: next.estimatedRuntimeMemoryGB * 0.9,
                                   deviceName: "iPhone")
        XCTAssertNil(ModelManager.evictionsBeforeLoading(next, residents: [], budget: 2, profile: tiny,
                                                         residentsCountAgainstProfile: true))
    }

    #if os(macOS)
    func testMacsHaveNoPerAppLimit() {
        XCTAssertNil(HardwareProfile.memoryLimitBytes())
    }
    #endif
}
