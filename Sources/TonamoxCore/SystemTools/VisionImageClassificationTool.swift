import Foundation
import Vision
import CoreGraphics

/// On-device image classification via Apple's **Vision** framework. Labels an image
/// against Vision's built-in taxonomy (about 1,300 identifiers such as `document`,
/// `dog`, `beach`) with no model download. Resilient: nothing above the threshold
/// returns an empty result (not an error); a bad image throws a typed error.
public struct VisionImageClassificationTool: SystemTool {
    public let id = "system.vision.classify"
    public let displayName = "Image classification (Vision)"
    public let summary = "Label what an image shows (scene, objects, documents, animals…) with Apple's built-in Vision classifier — no model download, works offline."
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

    /// One label from Vision's taxonomy.
    public struct Classification: Sendable, Equatable {
        public let identifier: String
        public let confidence: Float   // 0…1
    }

    public func availability() async -> SystemToolAvailability {
        #if targetEnvironment(simulator)
        // Seen on the iOS 26.5 Simulator: perform() fails with "Failed to create espresso context".
        return .unavailable(reason: "Vision image classification does not run in the Simulator; use a device.")
        #else
        do {
            let identifiers = try VNClassifyImageRequest().supportedIdentifiers()
            return identifiers.isEmpty
                ? .unavailable(reason: "Vision reports no image classification labels on this device.")
                : .available
        } catch {
            return .unavailable(reason: "Vision image classification is unavailable: \(error.localizedDescription)")
        }
        #endif
    }

    /// Classify a `CGImage`. Returns at most `maxResults` labels whose confidence is at
    /// least `minimumConfidence`, highest first.
    public func classify(
        in image: CGImage,
        maxResults: Int = 5,
        minimumConfidence: Float = 0.1
    ) throws -> [Classification] {
        try classify(using: VisionImageInput.handler(for: image),
                     maxResults: maxResults, minimumConfidence: minimumConfidence)
    }

    /// Classify raw image bytes (PNG/JPEG/HEIC/…). Throws ``ToolError/invalidImage`` if undecodable.
    public func classify(
        inImageData data: Data,
        maxResults: Int = 5,
        minimumConfidence: Float = 0.1
    ) throws -> [Classification] {
        guard let handler = VisionImageInput.handler(forImageData: data) else {
            throw ToolError.invalidImage
        }
        return try classify(using: handler, maxResults: maxResults, minimumConfidence: minimumConfidence)
    }

    private func classify(
        using handler: VNImageRequestHandler,
        maxResults: Int,
        minimumConfidence: Float
    ) throws -> [Classification] {
        let request = VNClassifyImageRequest()
        try handler.perform([request])
        // Vision scores every label in its taxonomy; ties break by identifier so output is stable.
        return (request.results ?? [])
            .filter { $0.confidence >= minimumConfidence }
            .sorted { $0.confidence != $1.confidence ? $0.confidence > $1.confidence : $0.identifier < $1.identifier }
            .prefix(max(0, maxResults))
            .map(Self.classification(from:))
    }

    private static func classification(from observation: VNClassificationObservation) -> Classification {
        Classification(identifier: observation.identifier, confidence: observation.confidence)
    }
}
