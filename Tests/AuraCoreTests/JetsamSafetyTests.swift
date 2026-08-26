import XCTest
@testable import AuraCore

/// The bug this locks out: `os_proc_available_memory()` returns 0 **when the process is at or over
/// its jetsam limit**. The old code did `if available > 0 { return it }` and otherwise fell through
/// to `physicalMemory * 0.6` — so a 6 GB iPhone about to be killed reported ~3.6 GB free, and
/// `isUnderPressure` evaluated `3.6 GB < 250 MB` = **false**. Every OOM guard switched itself off at
/// the exact moment it had to fire. Unknown must now mean danger, never "plenty".
final class JetsamSafetyTests: XCTestCase {

    private let margin = 250 * 1024 * 1024   // 250 MB

    // MARK: - unknown must fail SAFE, not open

    func testUnknownMemoryCountsAsPressure() {
        XCTAssertTrue(MemoryBudgetManager.isUnderPressure(available: nil, safetyMargin: margin),
            "unknown is what iOS reports at the jetsam limit — it must trip pressure, not clear it")
    }

    func testUnknownMemoryRefusesAllocation() {
        XCTAssertFalse(
            MemoryBudgetManager.canAllocate(bytes: 1024, available: nil, safetyMargin: margin),
            "refusing a load is recoverable; being jetsammed mid-inference is not")
    }

    func testUnknownMemoryShrinksContextToMinimum() {
        XCTAssertEqual(
            MemoryBudgetManager.recommendedContextLength(baseContext: 8192, available: nil, safetyMargin: margin),
            512)
    }

    /// The precise scenario the old code got backwards.
    func testTheOldFallbackWouldHaveClearedPressure() {
        let sixGBPhoneGuess = Int(6.0 * 1_073_741_824 * 0.6)   // what `physicalMemory * 0.6` produced
        XCTAssertFalse(MemoryBudgetManager.isUnderPressure(available: sixGBPhoneGuess, safetyMargin: margin),
            "precondition: the old guess really did read as 'plenty free'")
        XCTAssertTrue(MemoryBudgetManager.isUnderPressure(available: nil, safetyMargin: margin),
            "the same moment, reported honestly, must read as pressure")
    }

    // MARK: - known values still behave

    func testKnownLowMemoryTripsPressure() {
        XCTAssertTrue(MemoryBudgetManager.isUnderPressure(available: 100 * 1024 * 1024, safetyMargin: margin))
    }

    func testKnownAmpleMemoryDoesNotTripPressure() {
        XCTAssertFalse(MemoryBudgetManager.isUnderPressure(available: 4 * 1024 * 1024 * 1024, safetyMargin: margin))
        XCTAssertTrue(MemoryBudgetManager.canAllocate(bytes: 512 * 1024 * 1024,
                                                      available: 4 * 1024 * 1024 * 1024,
                                                      safetyMargin: margin))
        XCTAssertEqual(MemoryBudgetManager.recommendedContextLength(baseContext: 4096,
                                                                    available: 4 * 1024 * 1024 * 1024,
                                                                    safetyMargin: margin), 4096)
    }

    // MARK: - single source of truth

    #if os(macOS)
    /// On macOS the kernel query must actually answer, or every caller silently degrades to the
    /// conservative unknown path.
    func testMacReportsRealAvailableMemory() throws {
        let bytes = try XCTUnwrap(HardwareProfile.availableMemoryBytes(),
                                  "macOS must resolve available memory from the kernel")
        XCTAssertGreaterThan(bytes, 0)
        XCTAssertLessThan(Double(bytes), Double(ProcessInfo.processInfo.physicalMemory))
    }
    #endif

    /// All three former copies must now agree, because they are one function.
    func testAllCallersShareOneSource() {
        XCTAssertEqual(MemoryBudgetManager.availableMemoryBytes(), HardwareProfile.availableMemoryBytes())
    }
}
