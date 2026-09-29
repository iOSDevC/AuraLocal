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
    public let category = SystemToolCategory.vision

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
        let observations = try Self.observations(
            using: VisionImageInput.handler(for: image), languages: languages, accurate: accurate)
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
        guard let image = VisionImageInput.decode(data)?.image else {
            throw ToolError.invalidImage
        }
        return try recognizeText(in: image, languages: languages, accurate: accurate)
    }

    // MARK: - Lines with position

    /// One recognized line and where it sits, so a caller can anchor a value (an amount,
    /// a date) to its position. `boundingBox` is normalized 0…1 with Vision's bottom-left origin.
    public struct RecognizedLine: Sendable, Equatable {
        public let text: String
        public let confidence: Float   // 0…1
        public let boundingBox: CGRect
    }

    /// OCR a `CGImage` line by line, in Vision's reading order. Same options as
    /// ``recognizeText(in:languages:accurate:)``.
    public func recognizeLines(
        in image: CGImage,
        languages: [String] = [],
        accurate: Bool = true
    ) throws -> [RecognizedLine] {
        try Self.observations(using: VisionImageInput.handler(for: image), languages: languages, accurate: accurate)
            .compactMap(Self.recognizedLine(from:))
    }

    /// OCR raw image bytes line by line. Boxes follow the image's EXIF orientation, i.e. the
    /// upright image as displayed. Throws ``ToolError/invalidImage`` if undecodable.
    public func recognizeLines(
        inImageData data: Data,
        languages: [String] = [],
        accurate: Bool = true
    ) throws -> [RecognizedLine] {
        guard let handler = VisionImageInput.handler(forImageData: data) else {
            throw ToolError.invalidImage
        }
        return try Self.observations(using: handler, languages: languages, accurate: accurate)
            .compactMap(Self.recognizedLine(from:))
    }

    private static func observations(
        using handler: VNImageRequestHandler,
        languages: [String],
        accurate: Bool
    ) throws -> [VNRecognizedTextObservation] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = accurate ? .accurate : .fast
        request.usesLanguageCorrection = true
        // Without languages Vision assumes en-US and silently misses non-Latin scripts.
        if languages.isEmpty {
            request.automaticallyDetectsLanguage = true
        } else {
            request.recognitionLanguages = languages
        }
        try handler.perform([request])
        return request.results ?? []
    }

    private static func recognizedLine(from observation: VNRecognizedTextObservation) -> RecognizedLine? {
        guard let best = observation.topCandidates(1).first else { return nil }
        return RecognizedLine(text: best.string, confidence: best.confidence, boundingBox: observation.boundingBox)
    }
}
