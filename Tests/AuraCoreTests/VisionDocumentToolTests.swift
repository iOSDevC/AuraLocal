import XCTest
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import AuraCore

final class VisionDocumentToolTests: XCTestCase {

    private static let receipt: [(String, CGFloat)] = [
        ("CAFE AURA", 450), ("Latte $4.50", 340), ("Total $12.75", 230), ("March 3, 2026", 120)
    ]

    // MARK: - Identity / availability

    func testDocumentToolHasAFixedIDAndVisionCategory() {
        let tool = VisionDocumentTool()
        XCTAssertEqual(tool.id, "system.vision.document")
        XCTAssertEqual(tool.category, .vision)
        XCTAssertFalse(tool.summary.isEmpty)
        XCTAssertNotNil(SystemToolRegistry.tool(id: "system.vision.document"))
    }

    func testAvailabilityFollowsTheOSVersion() async {
        let availability = await VisionDocumentTool().availability()
        #if compiler(>=6.2)
        if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) {
            XCTAssertEqual(availability, .available)
            return
        }
        #endif
        XCTAssertFalse(availability.isAvailable)
        XCTAssertNotNil(availability.reason)
    }

    func testAmountsKeepOnlyMoneyMatches() {
        func match(_ kind: VisionDocumentTool.DetectedData.Kind) -> VisionDocumentTool.DetectedData {
            .init(kind: kind, text: "x", boundingBox: nil, amount: kind == .moneyAmount ? 1 : nil,
                  currency: kind == .moneyAmount ? "USD" : nil, date: nil)
        }
        let document = VisionDocumentTool.Document(
            text: "", title: nil, paragraphs: [], lines: [], tables: [],
            detectedData: [match(.date), match(.moneyAmount), match(.phoneNumber)])
        XCTAssertEqual(document.amounts.map(\.kind), [.moneyAmount])
    }

    // MARK: - Reading (iOS 26 / macOS 26)

    #if compiler(>=6.2)
    func testReceiptYieldsLinesAmountsAndADate() async throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            throw XCTSkip("RecognizeDocumentsRequest needs iOS 26 / macOS 26.")
        }
        let image = try XCTUnwrap(VisionFixtures.renderLines(Self.receipt, height: 540))
        let documents = try await VisionDocumentTool().readDocuments(in: image)
        let document = try XCTUnwrap(documents.first { $0.text.localizedCaseInsensitiveContains("total") },
                                     "documents: \(documents.map(\.text))")

        let total = try XCTUnwrap(document.lines.first { $0.text.localizedCaseInsensitiveContains("total") },
                                  "lines: \(document.lines.map(\.text))")
        XCTAssertTrue(VisionFixtures.isNormalized(total.boundingBox), "box: \(total.boundingBox)")
        let latte = try XCTUnwrap(document.lines.first { $0.text.localizedCaseInsensitiveContains("latte") })
        XCTAssertGreaterThan(latte.boundingBox.midY, total.boundingBox.midY, "bottom-left origin")

        let matches = document.detectedData.map {
            "\($0.kind.rawValue) '\($0.text)' \(String(describing: $0.amount)) \(String(describing: $0.boundingBox))"
        }
        let amounts = document.amounts
        let total = try XCTUnwrap(amounts.first { $0.amount == Decimal(string: "12.75") }, "matches: \(matches)")
        XCTAssertTrue(total.text.contains("12.75"), "matches: \(matches); text: \(document.text)")
        XCTAssertTrue(amounts.allSatisfy { $0.currency?.isEmpty == false }, "currencies: \(amounts.map(\.currency))")
        // A span Vision gives no range for has empty text; every other span comes from the document's text.
        XCTAssertTrue(document.detectedData.allSatisfy { $0.text.isEmpty || document.text.contains($0.text) },
                      "matches: \(matches); text: \(document.text)")

        let date = try XCTUnwrap(document.detectedData.first { $0.kind == .date }?.date,
                                 "matches: \(document.detectedData.map { "\($0.kind): \($0.text)" })")
        let day = Calendar.current.dateComponents([.year, .month, .day], from: date)
        XCTAssertEqual([day.year, day.month, day.day], [2026, 3, 3])
    }

    func testReceiptFromDataHonoursEXIFOrientation() async throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            throw XCTSkip("RecognizeDocumentsRequest needs iOS 26 / macOS 26.")
        }
        let upright = try XCTUnwrap(VisionFixtures.renderLines(Self.receipt, height: 540))
        let stored = try XCTUnwrap(VisionFixtures.rotatedCounterClockwise(upright))
        let jpeg = try XCTUnwrap(VisionFixtures.encode(stored, type: .jpeg, orientation: .left))
        let documents = try await VisionDocumentTool().readDocuments(inImageData: jpeg)
        let lines = documents.flatMap(\.lines)
        let total = try XCTUnwrap(lines.first { $0.text.localizedCaseInsensitiveContains("total") },
                                  "lines: \(lines.map(\.text))")
        XCTAssertGreaterThan(total.boundingBox.width, total.boundingBox.height, "horizontal once upright")
    }

    func testBlankImageHasNoText() async throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            throw XCTSkip("RecognizeDocumentsRequest needs iOS 26 / macOS 26.")
        }
        let image = try XCTUnwrap(VisionFixtures.blankImage())
        let documents = try await VisionDocumentTool().readDocuments(in: image)
        XCTAssertTrue(documents.allSatisfy { $0.text.isEmpty && $0.detectedData.isEmpty },
                      "documents: \(documents.map(\.text))")
    }

    func testInvalidDataThrowsInvalidImage() async throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            throw XCTSkip("RecognizeDocumentsRequest needs iOS 26 / macOS 26.")
        }
        do {
            _ = try await VisionDocumentTool().readDocuments(inImageData: Data([0, 1, 2, 3]))
            XCTFail("expected invalidImage")
        } catch VisionDocumentTool.ToolError.invalidImage {
        }
    }
    #endif
}
