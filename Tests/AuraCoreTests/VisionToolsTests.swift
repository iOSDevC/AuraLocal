import XCTest
import CoreGraphics
import CoreText
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import UniformTypeIdentifiers
import Vision
@testable import AuraCore

final class VisionToolsTests: XCTestCase {

    private static let qrPayload = "https://example.com/aura?receipt=42"

    // MARK: - Identity / availability

    func testVisionToolsHaveFixedIDsAndVisionCategory() {
        let tools: [any SystemTool] = [
            VisionImageClassificationTool(), VisionBarcodeTool(), VisionFaceDetectionTool(), VisionOCRTool(),
        ]
        XCTAssertEqual(tools.map(\.id),
                       ["system.vision.classify", "system.vision.barcodes", "system.vision.faces", "system.vision.ocr"])
        XCTAssertTrue(tools.allSatisfy { $0.category == .vision })
        XCTAssertTrue(tools.allSatisfy { !$0.summary.isEmpty && !$0.displayName.isEmpty })
    }

    func testVisionToolsAreAvailableOnThisMac() async {
        let classifyAvailability = await VisionImageClassificationTool().availability()
        let barcodeAvailability = await VisionBarcodeTool().availability()
        let faceAvailability = await VisionFaceDetectionTool().availability()
        XCTAssertEqual(classifyAvailability, .available)
        XCTAssertEqual(barcodeAvailability, .available)
        XCTAssertEqual(faceAvailability, .available)
    }

    // MARK: - Barcodes

    func testBarcodeQRRoundTripReturnsPayload() throws {
        let image = try XCTUnwrap(Self.renderQR(Self.qrPayload))
        let codes = try VisionBarcodeTool().detectBarcodes(in: image)
        let qr = try XCTUnwrap(codes.first { $0.symbology == VNBarcodeSymbology.qr.rawValue })
        XCTAssertEqual(qr.payload, Self.qrPayload)
        XCTAssertTrue(Self.isNormalized(qr.boundingBox), "box: \(qr.boundingBox)")
    }

    func testBarcodeQRRoundTripFromPNGData() throws {
        let image = try XCTUnwrap(Self.renderQR(Self.qrPayload))
        let png = try XCTUnwrap(Self.encode(image, type: .png))
        let codes = try VisionBarcodeTool().detectBarcodes(inImageData: png)
        XCTAssertEqual(codes.map(\.payload), [Self.qrPayload])
    }

    func testBarcodeSymbologyFilterExcludesOtherSymbologies() throws {
        let image = try XCTUnwrap(Self.renderQR(Self.qrPayload))
        let tool = VisionBarcodeTool()
        XCTAssertEqual(try tool.detectBarcodes(in: image, symbologies: [VNBarcodeSymbology.ean13.rawValue]), [])
        let qrOnly = try tool.detectBarcodes(in: image, symbologies: [VNBarcodeSymbology.qr.rawValue])
        XCTAssertEqual(qrOnly.map(\.payload), [Self.qrPayload])
    }

    func testBarcodeUnknownSymbologyThrows() throws {
        let image = try XCTUnwrap(Self.renderQR(Self.qrPayload))
        XCTAssertThrowsError(try VisionBarcodeTool().detectBarcodes(in: image, symbologies: ["NotASymbology"])) { error in
            guard case VisionBarcodeTool.ToolError.unsupportedSymbology(let raw) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(raw, "NotASymbology")
        }
    }

    func testBarcodeSupportedSymbologiesIncludeQR() {
        XCTAssertTrue(VisionBarcodeTool().supportedSymbologies().contains(VNBarcodeSymbology.qr.rawValue))
    }

    func testBarcodeOnBlankImageIsEmpty() throws {
        let image = try XCTUnwrap(Self.blankImage())
        XCTAssertEqual(try VisionBarcodeTool().detectBarcodes(in: image), [])
    }

    func testBarcodeInvalidDataThrows() {
        XCTAssertThrowsError(try VisionBarcodeTool().detectBarcodes(inImageData: Data([0, 1, 2, 3])))
    }

    // MARK: - OCR lines

    func testOCRLinesFindRenderedTextWithNormalizedBoxes() throws {
        let image = try XCTUnwrap(Self.renderLines([("TOTAL 42.50", 280), ("FECHA 2026", 60)]))
        let lines = try VisionOCRTool().recognizeLines(in: image)
        XCTAssertTrue(lines.contains { $0.text.uppercased().contains("TOTAL") }, "OCR returned: \(lines.map(\.text))")
        XCTAssertTrue(lines.contains { $0.text.contains("42.50") }, "OCR returned: \(lines.map(\.text))")
        for line in lines {
            XCTAssertTrue(Self.isNormalized(line.boundingBox), "\(line.text) box: \(line.boundingBox)")
            XCTAssertTrue((0...1).contains(line.confidence), "\(line.text) confidence: \(line.confidence)")
        }
    }

    func testOCRLineBoxesUseVisionBottomLeftOrigin() throws {
        let image = try XCTUnwrap(Self.renderLines([("TOTAL 42.50", 280), ("FECHA 2026", 60)]))
        let lines = try VisionOCRTool().recognizeLines(in: image)
        let upper = try XCTUnwrap(lines.first { $0.text.uppercased().contains("TOTAL") })
        let lower = try XCTUnwrap(lines.first { $0.text.uppercased().contains("FECHA") })
        XCTAssertGreaterThan(upper.boundingBox.midY, lower.boundingBox.midY)
    }

    func testOCRLinesAgreeWithRecognizeText() throws {
        let image = try XCTUnwrap(Self.renderLines([("TOTAL 42.50", 280), ("FECHA 2026", 60)]))
        let tool = VisionOCRTool()
        let lines = try tool.recognizeLines(in: image)
        let recognized = try tool.recognizeText(in: image)
        XCTAssertEqual(lines.map(\.text).joined(separator: "\n"), recognized.text)
        XCTAssertEqual(lines.count, recognized.lineCount)
    }

    func testOCRLinesFromDataHonourEXIFOrientation() throws {
        let upright = try XCTUnwrap(Self.renderLines([("TOTAL 42.50", 280), ("FECHA 2026", 60)]))
        let stored = try XCTUnwrap(Self.rotatedCounterClockwise(upright))
        let jpeg = try XCTUnwrap(Self.encode(stored, type: .jpeg, orientation: .left))
        let lines = try VisionOCRTool().recognizeLines(inImageData: jpeg)
        let upper = try XCTUnwrap(lines.first { $0.text.uppercased().contains("TOTAL") }, "OCR returned: \(lines.map(\.text))")
        let lower = try XCTUnwrap(lines.first { $0.text.uppercased().contains("FECHA") })
        XCTAssertGreaterThan(upper.boundingBox.width, upper.boundingBox.height, "line should be horizontal once upright")
        XCTAssertGreaterThan(upper.boundingBox.midY, lower.boundingBox.midY)
    }

    func testOCRLinesOnBlankImageIsEmpty() throws {
        let image = try XCTUnwrap(Self.blankImage())
        XCTAssertEqual(try VisionOCRTool().recognizeLines(in: image), [])
    }

    func testOCRLinesInvalidDataThrows() {
        XCTAssertThrowsError(try VisionOCRTool().recognizeLines(inImageData: Data([0, 1, 2, 3])))
    }

    // MARK: - Faces

    func testFaceDetectionOnBlankImageIsEmpty() throws {
        let image = try XCTUnwrap(Self.blankImage())
        XCTAssertEqual(try VisionFaceDetectionTool().detectFaces(in: image), [])
    }

    func testFaceDetectionOnPNGDataOfBlankImageIsEmpty() throws {
        let png = try XCTUnwrap(Self.blankImage().flatMap { Self.encode($0, type: .png) })
        XCTAssertEqual(try VisionFaceDetectionTool().detectFaces(inImageData: png), [])
    }

    func testFaceDetectionInvalidDataThrows() {
        XCTAssertThrowsError(try VisionFaceDetectionTool().detectFaces(inImageData: Data([0, 1, 2, 3])))
    }

    // MARK: - Classification

    func testClassificationIsBoundedSortedAndAboveThreshold() throws {
        let image = try XCTUnwrap(Self.syntheticScene())
        let labels = try VisionImageClassificationTool().classify(in: image, maxResults: 3, minimumConfidence: 0.01)
        XCTAssertLessThanOrEqual(labels.count, 3)
        XCTAssertEqual(labels.map(\.confidence), labels.map(\.confidence).sorted(by: >))
        for label in labels {
            XCTAssertFalse(label.identifier.isEmpty)
            XCTAssertTrue((0.01...1).contains(label.confidence), "\(label.identifier): \(label.confidence)")
        }
    }

    func testClassificationWithZeroThresholdFillsMaxResults() throws {
        let image = try XCTUnwrap(Self.syntheticScene())
        let labels = try VisionImageClassificationTool().classify(in: image, maxResults: 4, minimumConfidence: 0)
        XCTAssertEqual(labels.count, 4)
        XCTAssertEqual(Set(labels.map(\.identifier)).count, 4)
    }

    func testClassificationWithNonPositiveMaxResultsIsEmpty() throws {
        let image = try XCTUnwrap(Self.syntheticScene())
        let tool = VisionImageClassificationTool()
        XCTAssertEqual(try tool.classify(in: image, maxResults: 0, minimumConfidence: 0), [])
        XCTAssertEqual(try tool.classify(in: image, maxResults: -1, minimumConfidence: 0), [])
    }

    func testClassificationFromPNGDataIsBounded() throws {
        let png = try XCTUnwrap(Self.syntheticScene().flatMap { Self.encode($0, type: .png) })
        let labels = try VisionImageClassificationTool().classify(inImageData: png, maxResults: 2, minimumConfidence: 0)
        XCTAssertEqual(labels.count, 2)
    }

    func testClassificationInvalidDataThrows() {
        XCTAssertThrowsError(try VisionImageClassificationTool().classify(inImageData: Data([0, 1, 2, 3])))
    }

    // MARK: - Helpers

    private static func isNormalized(_ rect: CGRect) -> Bool {
        let tolerance: CGFloat = 0.001
        return rect.minX >= -tolerance && rect.minY >= -tolerance
            && rect.maxX <= 1 + tolerance && rect.maxY <= 1 + tolerance
            && rect.width > 0 && rect.height > 0
    }

    private static func makeContext(width: Int, height: Int) -> CGContext? {
        CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                  space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    }

    private static func blankImage(width: Int = 640, height: Int = 480) -> CGImage? {
        guard let ctx = makeContext(width: width, height: height) else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()
    }

    /// Each entry is a line of text and its baseline y (CoreGraphics: 0 = bottom).
    private static func renderLines(_ lines: [(String, CGFloat)], width: Int = 800, height: Int = 400) -> CGImage? {
        guard let ctx = makeContext(width: width, height: height) else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        for (text, baseline) in lines {
            draw(text, baseline: baseline, in: ctx)
        }
        return ctx.makeImage()
    }

    private static func draw(_ text: String, baseline: CGFloat, in ctx: CGContext) {
        let font = CTFontCreateWithName("Helvetica" as CFString, 64, nil)
        let black = CGColor(red: 0, green: 0, blue: 0, alpha: 1)
        let attrs: [CFString: Any] = [kCTFontAttributeName: font, kCTForegroundColorAttributeName: black]
        guard let attributed = CFAttributedStringCreate(nil, text as CFString, attrs as CFDictionary) else { return }
        ctx.textPosition = CGPoint(x: 30, y: baseline)
        CTLineDraw(CTLineCreateWithAttributedString(attributed), ctx)
    }

    /// Stores `image` rotated 90° counter-clockwise; EXIF `.left` displays it upright again.
    private static func rotatedCounterClockwise(_ image: CGImage) -> CGImage? {
        guard let ctx = makeContext(width: image.height, height: image.width) else { return nil }
        ctx.translateBy(x: 0, y: CGFloat(image.width))
        ctx.rotate(by: -.pi / 2)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return ctx.makeImage()
    }

    /// A QR code from CoreImage's generator, scaled up and padded with a white quiet zone.
    private static func renderQR(_ payload: String) -> CGImage? {
        let generator = CIFilter.qrCodeGenerator()
        generator.message = Data(payload.utf8)
        generator.correctionLevel = "M"
        guard let code = generator.outputImage else { return nil }
        let scaled = code.transformed(by: CGAffineTransform(scaleX: 10, y: 10))
        let padding: CGFloat = 40
        let canvas = CGRect(x: 0, y: 0, width: scaled.extent.width + 2 * padding, height: scaled.extent.height + 2 * padding)
        let padded = scaled.transformed(by: CGAffineTransform(translationX: padding, y: padding))
            .composited(over: CIImage(color: .white).cropped(to: canvas))
        return CIContext().createCGImage(padded, from: canvas)
    }

    /// Coloured shapes on a gradient: enough structure for the classifier to score, no fixed label expected.
    private static func syntheticScene(width: Int = 512, height: Int = 512) -> CGImage? {
        guard let ctx = makeContext(width: width, height: height),
              let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: [CGColor(red: 0.2, green: 0.5, blue: 0.9, alpha: 1),
                         CGColor(red: 0.95, green: 0.9, blue: 0.7, alpha: 1)] as CFArray,
                locations: [0, 1]) else { return nil }
        ctx.drawLinearGradient(gradient, start: CGPoint(x: 0, y: CGFloat(height)), end: .zero, options: [])
        ctx.setFillColor(CGColor(red: 0.1, green: 0.6, blue: 0.2, alpha: 1))
        ctx.fillEllipse(in: CGRect(x: 60, y: 40, width: 220, height: 160))
        ctx.setFillColor(CGColor(red: 0.8, green: 0.2, blue: 0.2, alpha: 1))
        ctx.fill(CGRect(x: 300, y: 80, width: 140, height: 260))
        return ctx.makeImage()
    }

    private static func encode(_ image: CGImage, type: UTType, orientation: CGImagePropertyOrientation = .up) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, type.identifier as CFString, 1, nil) else {
            return nil
        }
        let properties: [CFString: Any] = [
            kCGImagePropertyOrientation: orientation.rawValue,
            kCGImageDestinationLossyCompressionQuality: 1.0,
        ]
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }
}
