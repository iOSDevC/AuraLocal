import XCTest
@testable import AuraImageGen
#if canImport(ImagePlayground)
import ImagePlayground

final class ImagePlaygroundPolicyTests: XCTestCase {

    func testAllowedStylesNeverIncludeTheExternalProvider() throws {
        guard #available(macOS 15.4, iOS 18.4, visionOS 2.4, *) else { throw XCTSkip("needs 15.4") }
        let ids = ImagePlaygroundPolicy.allowedStyles.map(\.id)
        XCTAssertEqual(ids, ["animation", "illustration", "sketch"])
        XCTAssertFalse(ids.contains("z_external_provider"))
        if #available(macOS 26, iOS 26, visionOS 26, *) {
            XCTAssertFalse(ImagePlaygroundPolicy.allowedStyles.contains(.externalProvider))
        }
        XCTAssertTrue(ImagePlaygroundPolicy.allowedStyles.contains(ImagePlaygroundPolicy.defaultStyle))
    }

    func testProcessingFollowsTheOSMajorVersion() throws {
        guard #available(macOS 15.4, iOS 18.4, visionOS 2.4, *) else { throw XCTSkip("needs 15.4") }
        XCTAssertEqual(ImagePlaygroundPolicy.processing(forMajorVersion: 18), .onDevice)
        XCTAssertEqual(ImagePlaygroundPolicy.processing(forMajorVersion: 26), .onDevice)
        XCTAssertEqual(ImagePlaygroundPolicy.processing(forMajorVersion: 27), .privateCloudCompute)
        XCTAssertEqual(ImagePlaygroundPolicy.processing(forMajorVersion: 28), .privateCloudCompute)
        XCTAssertTrue(ImageGenProcessing.privateCloudCompute.leavesDevice)
        XCTAssertFalse(ImageGenProcessing.onDevice.leavesDevice)
    }

    func testOptionsDisablePersonalization() throws {
        guard #available(macOS 26.4, iOS 26.4, visionOS 26.4, *) else { throw XCTSkip("needs 26.4") }
        XCTAssertEqual(ImagePlaygroundPolicy.options.personalization, .disabled)
    }
}
#endif
