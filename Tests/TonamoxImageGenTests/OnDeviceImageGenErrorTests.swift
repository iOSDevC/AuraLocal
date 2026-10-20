import XCTest
@testable import TonamoxImageGen

final class OnDeviceImageGenErrorTests: XCTestCase {
    private let everyCase: [OnDeviceImageGenError] = [
        .unsupportedPlatform,
        .simulatorUnsupported,
        .modelNotDownloaded(sizeBytes: 1_073_000_000),
        .insufficientMemory(requiredBytes: 2_360_000_000, availableBytes: 1_000_000_000),
        .invalidRequest(field: "steps", reason: "must be 1–50, got 0"),
        .busy,
        .modelLoadFailed("corrupt"),
        .generationFailed("boom"),
        .filteredBySafetyChecker,
        .imagePlaygroundUnavailable,
        .consentDeclined,
    ]

    func testEveryCaseExplainsItselfAndTheWayOut() {
        for error in everyCase {
            XCTAssertFalse(error.errorDescription?.isEmpty ?? true, "\(error)")
            XCTAssertFalse(error.recoverySuggestion?.isEmpty ?? true, "\(error)")
        }
        XCTAssertEqual(Set(everyCase.compactMap(\.errorDescription)).count, everyCase.count)
    }

    func testMemoryMessageNamesBothAmounts() throws {
        let text = try XCTUnwrap(OnDeviceImageGenError.insufficientMemory(requiredBytes: 2_360_000_000,
                                                                           availableBytes: 1_000_000_000)
            .errorDescription)
        XCTAssertTrue(text.hasPrefix("Not enough free memory: needs about "), text)
        XCTAssertTrue(text.hasSuffix(" available."), text)
    }

    func testDeclinedConsentSaysNothingWasSent() {
        XCTAssertEqual(OnDeviceImageGenError.consentDeclined.errorDescription?.hasPrefix("Nothing was sent."), true)
    }
}
