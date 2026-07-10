import Foundation
import Vision
import CoreGraphics
import ImageIO

/// On-device OCR via Apple's **Vision** framework. Extracts text from a document
/// page or image with no model download — the SLM then reasons over the text.
/// Resilient: no text found returns an empty result (not an error); a bad image
/// throws a typed error.
public struct VisionOCRTool: SystemTool {
    public let id = "system.vision.ocr"
    public let displayName = "On-device OCR (Vision)"
    public let summary = "Extract text from an image or document page using Apple's Vision framework — no model download, works offline."

    public init() {}

    public enum ToolError: LocalizedError {
        case invalidImage
        public var errorDescription: String? {
            switch self {
            case .invalidImage: "The provided data is not a decodable image."
            }
        }
    }

    /// Recognized text plus a couple of quality signals the caller can gate on.
    public struct Recognized: Sendable, Equatable {
        public let text: String
        public let lineCount: Int
        public let averageConfidence: Float   // 0…1
        public var isEmpty: Bool { text.isEmpty }
    }

    public func availability() async -> SystemToolAvailability {
        // Vision text recognition ships on every OS version AuraCore targets.
        .available
    }

    /// OCR a `CGImage`. `accurate` trades speed for quality; `languages` (BCP-47)
    /// can hint recognition (empty = automatic).
    public func recognizeText(
        in image: CGImage,
        languages: [String] = [],
        accurate: Bool = true
    ) throws -> Recognized {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = accurate ? .accurate : .fast
        request.usesLanguageCorrection = true
        if !languages.isEmpty { request.recognitionLanguages = languages }

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])

        let observations = (request.results as? [VNRecognizedTextObservation]) ?? []
        var lines: [String] = []
        var confidenceSum: Float = 0
        for observation in observations {
            if let best = observation.topCandidates(1).first {
                lines.append(best.string)
                confidenceSum += best.confidence
            }
        }
        let average = observations.isEmpty ? 0 : confidenceSum / Float(observations.count)
        return Recognized(
            text: lines.joined(separator: "\n"),
            lineCount: lines.count,
            averageConfidence: average)
    }

    /// OCR raw image bytes (PNG/JPEG/HEIC/…). Throws ``ToolError/invalidImage`` if undecodable.
    public func recognizeText(
        inImageData data: Data,
        languages: [String] = [],
        accurate: Bool = true
    ) throws -> Recognized {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw ToolError.invalidImage
        }
        return try recognizeText(in: image, languages: languages, accurate: accurate)
    }
}
