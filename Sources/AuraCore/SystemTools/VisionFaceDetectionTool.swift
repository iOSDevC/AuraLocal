import Foundation
import Vision
import CoreGraphics

/// On-device face **detection** via Apple's **Vision** framework: where faces are and
/// how the head is turned. It never identifies or recognizes anyone. Resilient: an
/// image without faces returns an empty result (not an error).
public struct VisionFaceDetectionTool: SystemTool {
    public let id = "system.vision.faces"
    public let displayName = "Face detection (Vision)"
    public let summary = "Locate faces in an image and estimate head pose with Apple's Vision framework — detection only, no identity or recognition, works offline."
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

    /// A detected face. `boundingBox` is normalized 0…1 with Vision's bottom-left origin;
    /// angles are radians and `nil` when Vision could not estimate them.
    public struct DetectedFace: Sendable, Equatable {
        public let boundingBox: CGRect
        public let confidence: Float   // 0…1
        public let roll: Double?
        public let yaw: Double?
        public let pitch: Double?
    }

    public func availability() async -> SystemToolAvailability {
        #if targetEnvironment(simulator)
        // Seen on the iOS 26.5 Simulator: perform() fails with "Could not create inference context".
        return .unavailable(reason: "Vision face detection does not run in the Simulator; use a device.")
        #else
        // Face rectangle detection ships on every OS version AuraCore targets.
        return .available
        #endif
    }

    public func detectFaces(in image: CGImage) throws -> [DetectedFace] {
        try detectFaces(using: VisionImageInput.handler(for: image))
    }

    /// Detect faces in raw image bytes (PNG/JPEG/HEIC/…). Throws ``ToolError/invalidImage`` if undecodable.
    public func detectFaces(inImageData data: Data) throws -> [DetectedFace] {
        guard let handler = VisionImageInput.handler(forImageData: data) else {
            throw ToolError.invalidImage
        }
        return try detectFaces(using: handler)
    }

    private func detectFaces(using handler: VNImageRequestHandler) throws -> [DetectedFace] {
        let request = VNDetectFaceRectanglesRequest()
        try handler.perform([request])
        return (request.results ?? []).map(Self.detectedFace(from:))
    }

    private static func detectedFace(from observation: VNFaceObservation) -> DetectedFace {
        DetectedFace(boundingBox: observation.boundingBox,
                     confidence: observation.confidence,
                     roll: observation.roll?.doubleValue,
                     yaw: observation.yaw?.doubleValue,
                     pitch: observation.pitch?.doubleValue)
    }
}
