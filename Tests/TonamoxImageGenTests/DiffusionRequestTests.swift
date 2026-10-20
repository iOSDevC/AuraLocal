import XCTest
@testable import TonamoxImageGen

final class DiffusionRequestTests: XCTestCase {

    func testDefaults() {
        let request = DiffusionRequest(prompt: "a fox")
        XCTAssertEqual(request.negativePrompt, "")
        XCTAssertEqual(request.steps, 20)
        XCTAssertEqual(request.guidanceScale, 7.5)
        XCTAssertNil(request.seed)
        XCTAssertEqual(request.imageCount, 1)
        XCTAssertNoThrow(try request.validate())
    }

    func testEveryOutOfRangeFieldIsNamedNotClamped() {
        let cases: [(DiffusionRequest, String)] = [
            (DiffusionRequest(prompt: ""), "prompt"),
            (DiffusionRequest(prompt: " \n\t"), "prompt"),
            (DiffusionRequest(prompt: "a", steps: 0), "steps"),
            (DiffusionRequest(prompt: "a", steps: 51), "steps"),
            (DiffusionRequest(prompt: "a", guidanceScale: 0.99), "guidanceScale"),
            (DiffusionRequest(prompt: "a", guidanceScale: 20.01), "guidanceScale"),
            (DiffusionRequest(prompt: "a", guidanceScale: .nan), "guidanceScale"),
            (DiffusionRequest(prompt: "a", imageCount: 0), "imageCount"),
            (DiffusionRequest(prompt: "a", imageCount: 5), "imageCount"),
        ]
        for (request, field) in cases {
            XCTAssertThrowsError(try request.validate(), "\(request)") { error in
                guard case .invalidRequest(let named, let reason)? = error as? OnDeviceImageGenError else {
                    return XCTFail("\(error)")
                }
                XCTAssertEqual(named, field)
                XCTAssertFalse(reason.isEmpty)
            }
        }
    }

    func testBoundariesAreValid() {
        let valid = [
            DiffusionRequest(prompt: "a", steps: 1, guidanceScale: 1, imageCount: 1),
            DiffusionRequest(prompt: "a", steps: 50, guidanceScale: 20, imageCount: 4),
        ]
        for request in valid {
            XCTAssertNoThrow(try request.validate())
        }
    }

    func testOutOfRangeMessageShowsTheValue() {
        XCTAssertThrowsError(try DiffusionRequest(prompt: "a", steps: 80).validate()) { error in
            XCTAssertEqual((error as? LocalizedError)?.errorDescription, "steps: must be 1–50, got 80")
        }
    }
}
