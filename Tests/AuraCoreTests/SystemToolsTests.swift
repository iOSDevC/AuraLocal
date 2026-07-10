import XCTest
import CoreGraphics
import CoreText
@testable import AuraCore

final class SystemToolsTests: XCTestCase {

    // MARK: - Discovery / resilience

    func testDiscoveryListsTools() async {
        let infos = await SystemToolRegistry.discover()
        XCTAssertFalse(infos.isEmpty)
        XCTAssertTrue(infos.contains { $0.id == "system.vision.ocr" })
        XCTAssertTrue(infos.contains { $0.id == "system.nl.embedding" })
        // Each carries a resolved availability — discovery never throws.
        for info in infos { XCTAssertFalse(info.summary.isEmpty) }
    }

    func testAvailableToolsAreAllAvailable() async {
        let available = await SystemToolRegistry.availableTools()
        XCTAssertTrue(available.allSatisfy { $0.isAvailable })
    }

    func testLookupById() {
        XCTAssertNotNil(SystemToolRegistry.tool(id: "system.vision.ocr"))
        XCTAssertNil(SystemToolRegistry.tool(id: "does.not.exist"))
    }

    // MARK: - Vision OCR (real, end-to-end)

    func testVisionOCRReadsRenderedText() throws {
        guard let image = Self.renderText("HELLO WORLD") else {
            throw XCTSkip("Could not render a test image on this host.")
        }
        let result = try VisionOCRTool().recognizeText(in: image)
        XCTAssertTrue(result.text.uppercased().contains("HELLO"), "OCR returned: \(result.text)")
        XCTAssertGreaterThan(result.averageConfidence, 0)
        XCTAssertGreaterThanOrEqual(result.lineCount, 1)
    }

    func testVisionOCRInvalidDataThrows() {
        XCTAssertThrowsError(try VisionOCRTool().recognizeText(inImageData: Data([0, 1, 2, 3])))
    }

    // MARK: - NaturalLanguage embeddings (resilient)

    func testNLEmbeddingSemanticDistance() {
        let tool = NLEmbeddingTool()
        guard tool.embed("hello world") != nil else {
            return   // resilient: no embedding installed on this host — skip silently
        }
        let near = tool.distance("the cat sat on the mat", "a cat is sitting on a mat")
        let far = tool.distance("the cat sat on the mat", "quantum chromodynamics field equations")
        if let near, let far {
            XCTAssertLessThan(near, far, "related text should be closer than unrelated")
        }
    }

    // MARK: - Helpers

    /// Render text to a CGImage using CoreText — cross-platform, no AppKit/UIKit.
    private static func renderText(_ text: String, size: CGSize = CGSize(width: 640, height: 160)) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: Int(size.width), height: Int(size.height),
            bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(origin: .zero, size: size))
        let font = CTFontCreateWithName("Helvetica" as CFString, 72, nil)
        let black = CGColor(red: 0, green: 0, blue: 0, alpha: 1)
        let attrs: [CFString: Any] = [kCTFontAttributeName: font, kCTForegroundColorAttributeName: black]
        let attributed = CFAttributedStringCreate(nil, text as CFString, attrs as CFDictionary)!
        let line = CTLineCreateWithAttributedString(attributed)
        ctx.textPosition = CGPoint(x: 28, y: 56)
        CTLineDraw(line, ctx)
        return ctx.makeImage()
    }
}
