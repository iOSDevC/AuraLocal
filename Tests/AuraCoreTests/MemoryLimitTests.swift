import XCTest
@testable import AuraCore

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

    #if os(macOS)
    func testMacsHaveNoPerAppLimit() {
        XCTAssertNil(HardwareProfile.memoryLimitBytes())
    }
    #endif
}
