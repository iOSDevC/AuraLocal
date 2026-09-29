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
    private static let junk = Data([0, 1, 2, 3])

    // MARK: - Identity / availability

    func testVisionToolsHaveFixedIDsAndVisionCategory() {
        let tools: [any SystemTool] = [
            VisionImageClassificationTool(), VisionBarcodeTool(), VisionFaceDetectionTool(), VisionOCRTool()
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
        let image = try XCTUnwrap(VisionFixtures.renderQR(Self.qrPayload))
        let codes = try VisionBarcodeTool().detectBarcodes(in: image)
        let qrCode = try XCTUnwrap(codes.first { $0.symbology == VNBarcodeSymbology.qr.rawValue })
        XCTAssertEqual(qrCode.payload, Self.qrPayload)
        XCTAssertTrue(VisionFixtures.isNormalized(qrCode.boundingBox), "box: \(qrCode.boundingBox)")
    }

    func testBarcodeQRRoundTripFromPNGData() throws {
        let image = try XCTUnwrap(VisionFixtures.renderQR(Self.qrPayload))
        let png = try XCTUnwrap(VisionFixtures.encode(image, type: .png))
        let codes = try VisionBarcodeTool().detectBarcodes(inImageData: png)
        XCTAssertEqual(codes.map(\.payload), [Self.qrPayload])
    }

    func testBarcodeSymbologyFilterExcludesOtherSymbologies() throws {
        let image = try XCTUnwrap(VisionFixtures.renderQR(Self.qrPayload))
        let tool = VisionBarcodeTool()
        XCTAssertEqual(try tool.detectBarcodes(in: image, symbologies: [VNBarcodeSymbology.ean13.rawValue]), [])
        let qrOnly = try tool.detectBarcodes(in: image, symbologies: [VNBarcodeSymbology.qr.rawValue])
        XCTAssertEqual(qrOnly.map(\.payload), [Self.qrPayload])
    }

    func testBarcodeUnknownSymbologyThrows() throws {
        let image = try XCTUnwrap(VisionFixtures.renderQR(Self.qrPayload))
        let tool = VisionBarcodeTool()
        XCTAssertThrowsError(try tool.detectBarcodes(in: image, symbologies: ["NotASymbology"])) { error in
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
        let image = try XCTUnwrap(VisionFixtures.blankImage())
        XCTAssertEqual(try VisionBarcodeTool().detectBarcodes(in: image), [])
    }

    func testBarcodeInvalidDataThrowsInvalidImage() {
        XCTAssertThrowsError(try VisionBarcodeTool().detectBarcodes(inImageData: Self.junk)) { error in
            guard case VisionBarcodeTool.ToolError.invalidImage = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    // MARK: - OCR lines

    func testOCRLinesFindRenderedTextWithNormalizedBoxes() throws {
        let image = try XCTUnwrap(VisionFixtures.renderLines([("TOTAL 42.50", 280), ("FECHA 2026", 60)]))
        let lines = try VisionOCRTool().recognizeLines(in: image)
        XCTAssertTrue(lines.contains { $0.text.uppercased().contains("TOTAL") }, "OCR returned: \(lines.map(\.text))")
        XCTAssertTrue(lines.contains { $0.text.contains("42.50") }, "OCR returned: \(lines.map(\.text))")
        for line in lines {
            XCTAssertTrue(VisionFixtures.isNormalized(line.boundingBox), "\(line.text) box: \(line.boundingBox)")
            XCTAssertTrue((0...1).contains(line.confidence), "\(line.text) confidence: \(line.confidence)")
        }
    }

    func testOCRLineBoxesUseVisionBottomLeftOrigin() throws {
        let image = try XCTUnwrap(VisionFixtures.renderLines([("TOTAL 42.50", 280), ("FECHA 2026", 60)]))
        let lines = try VisionOCRTool().recognizeLines(in: image)
        let upper = try XCTUnwrap(lines.first { $0.text.uppercased().contains("TOTAL") })
        let lower = try XCTUnwrap(lines.first { $0.text.uppercased().contains("FECHA") })
        XCTAssertGreaterThan(upper.boundingBox.midY, lower.boundingBox.midY)
    }

    func testOCRLinesAgreeWithRecognizeText() throws {
        let image = try XCTUnwrap(VisionFixtures.renderLines([("TOTAL 42.50", 280), ("FECHA 2026", 60)]))
        let tool = VisionOCRTool()
        let lines = try tool.recognizeLines(in: image)
        let recognized = try tool.recognizeText(in: image)
        XCTAssertEqual(lines.map(\.text).joined(separator: "\n"), recognized.text)
        XCTAssertEqual(lines.count, recognized.lineCount)
    }

    func testOCRLinesFromDataHonourEXIFOrientation() throws {
        let upright = try XCTUnwrap(VisionFixtures.renderLines([("TOTAL 42.50", 280), ("FECHA 2026", 60)]))
        let stored = try XCTUnwrap(VisionFixtures.rotatedCounterClockwise(upright))
        let jpeg = try XCTUnwrap(VisionFixtures.encode(stored, type: .jpeg, orientation: .left))
        let lines = try VisionOCRTool().recognizeLines(inImageData: jpeg)
        let upper = try XCTUnwrap(
            lines.first { $0.text.uppercased().contains("TOTAL") }, "OCR returned: \(lines.map(\.text))")
        let lower = try XCTUnwrap(lines.first { $0.text.uppercased().contains("FECHA") })
        XCTAssertGreaterThan(upper.boundingBox.width, upper.boundingBox.height, "horizontal once upright")
        XCTAssertGreaterThan(upper.boundingBox.midY, lower.boundingBox.midY)
    }

    func testOCRLinesOnBlankImageIsEmpty() throws {
        let image = try XCTUnwrap(VisionFixtures.blankImage())
        XCTAssertEqual(try VisionOCRTool().recognizeLines(in: image), [])
    }

    func testOCRLinesInvalidDataThrowsInvalidImage() {
        XCTAssertThrowsError(try VisionOCRTool().recognizeLines(inImageData: Self.junk)) { error in
            guard case VisionOCRTool.ToolError.invalidImage = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testOCRWithoutLanguagesReadsJapanese() throws {
        let image = try XCTUnwrap(VisionFixtures.renderLines([("東京駅", 200)], font: "HiraginoSans-W6"))
        let tool = VisionOCRTool()
        let lines = try tool.recognizeLines(in: image)
        XCTAssertTrue(lines.contains { $0.text.contains("東京") }, "OCR returned: \(lines.map(\.text))")
        XCTAssertTrue(try tool.recognizeText(in: image).text.contains("東京"))
    }

    // MARK: - Faces

    func testFaceDetectionOnBlankImageIsEmpty() throws {
        let image = try XCTUnwrap(VisionFixtures.blankImage())
        XCTAssertEqual(try VisionFaceDetectionTool().detectFaces(in: image), [])
    }

    func testFaceDetectionOnPNGDataOfBlankImageIsEmpty() throws {
        let png = try XCTUnwrap(VisionFixtures.blankImage().flatMap { VisionFixtures.encode($0, type: .png) })
        XCTAssertEqual(try VisionFaceDetectionTool().detectFaces(inImageData: png), [])
    }

    func testFaceDetectionInvalidDataThrowsInvalidImage() {
        XCTAssertThrowsError(try VisionFaceDetectionTool().detectFaces(inImageData: Self.junk)) { error in
            guard case VisionFaceDetectionTool.ToolError.invalidImage = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testDetectedFaceMapsEachObservationField() {
        let box = CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        let observation = VNFaceObservation(
            requestRevision: VNDetectFaceRectanglesRequest.defaultRevision, boundingBox: box,
            roll: 0.1, yaw: 0.2, pitch: 0.3)
        let face = VisionFaceDetectionTool.detectedFace(from: observation)
        XCTAssertEqual(face.boundingBox, box)
        XCTAssertEqual(face.confidence, observation.confidence)
        XCTAssertEqual(face.roll ?? .nan, 0.1, accuracy: 1e-6)
        XCTAssertEqual(face.yaw ?? .nan, 0.2, accuracy: 1e-6)
        XCTAssertEqual(face.pitch ?? .nan, 0.3, accuracy: 1e-6)
    }

    func testDetectedFaceKeepsMissingAnglesNil() {
        let observation = VNFaceObservation(
            requestRevision: VNDetectFaceRectanglesRequest.defaultRevision, boundingBox: .zero,
            roll: nil, yaw: nil, pitch: nil)
        let face = VisionFaceDetectionTool.detectedFace(from: observation)
        XCTAssertNil(face.roll)
        XCTAssertNil(face.yaw)
        XCTAssertNil(face.pitch)
    }

    // MARK: - Classification

    func testClassificationIsBoundedSortedAndAboveThreshold() throws {
        let image = try XCTUnwrap(VisionFixtures.syntheticScene())
        let labels = try VisionImageClassificationTool().classify(in: image, maxResults: 3, minimumConfidence: 0.01)
        XCTAssertLessThanOrEqual(labels.count, 3)
        XCTAssertEqual(labels.map(\.confidence), labels.map(\.confidence).sorted(by: >))
        for label in labels {
            XCTAssertFalse(label.identifier.isEmpty)
            XCTAssertTrue((0.01...1).contains(label.confidence), "\(label.identifier): \(label.confidence)")
        }
    }

    func testClassificationWithZeroThresholdFillsMaxResults() throws {
        let image = try XCTUnwrap(VisionFixtures.syntheticScene())
        let labels = try VisionImageClassificationTool().classify(in: image, maxResults: 4, minimumConfidence: 0)
        XCTAssertEqual(labels.count, 4)
        XCTAssertEqual(Set(labels.map(\.identifier)).count, 4)
    }

    func testClassificationWithNonPositiveMaxResultsIsEmpty() throws {
        let image = try XCTUnwrap(VisionFixtures.syntheticScene())
        let tool = VisionImageClassificationTool()
        XCTAssertEqual(try tool.classify(in: image, maxResults: 0, minimumConfidence: 0), [])
        XCTAssertEqual(try tool.classify(in: image, maxResults: -1, minimumConfidence: 0), [])
    }

    func testClassificationFromPNGDataIsBounded() throws {
        let png = try XCTUnwrap(VisionFixtures.syntheticScene().flatMap { VisionFixtures.encode($0, type: .png) })
        let labels = try VisionImageClassificationTool().classify(inImageData: png, maxResults: 2, minimumConfidence: 0)
        XCTAssertEqual(labels.count, 2)
    }

    func testClassificationInvalidDataThrowsInvalidImage() {
        XCTAssertThrowsError(try VisionImageClassificationTool().classify(inImageData: Self.junk)) { error in
            guard case VisionImageClassificationTool.ToolError.invalidImage = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }
}
