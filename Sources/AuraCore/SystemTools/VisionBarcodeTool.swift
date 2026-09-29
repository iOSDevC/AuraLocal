import Foundation
import Vision
import CoreGraphics

/// On-device barcode and QR reading via Apple's **Vision** framework (QR, EAN, UPC,
/// Code 128, PDF417, Aztec, Data Matrix, …). Resilient: no code in the image returns
/// an empty result (not an error); a bad image or unknown symbology throws.
public struct VisionBarcodeTool: SystemTool {
    public let id = "system.vision.barcodes"
    public let displayName = "Barcode & QR reader (Vision)"
    public let summary = "Read QR codes and barcodes (EAN, UPC, Code 128, PDF417, Aztec, Data Matrix…) from an image with Apple's Vision framework — works offline."
    public let category = SystemToolCategory.vision

    public init() {}

    public enum ToolError: LocalizedError {
        case invalidImage
        case unsupportedSymbology(String)
        public var errorDescription: String? {
            switch self {
            case .invalidImage: "The provided data is not a decodable image."
            case .unsupportedSymbology(let raw): "\(raw) is not a barcode symbology Vision supports on this device."
            }
        }
    }

    /// A decoded code. `symbology` is Vision's raw value (e.g. `VNBarcodeSymbologyQR`).
    /// `boundingBox` is normalized 0…1 with Vision's bottom-left origin.
    public struct DetectedBarcode: Sendable, Equatable {
        /// `nil` when the payload is binary rather than text.
        public let payload: String?
        public let symbology: String
        public let boundingBox: CGRect
    }

    public func availability() async -> SystemToolAvailability {
        #if targetEnvironment(simulator)
        // Seen on the iOS 26.5 Simulator: detection either throws or finds nothing, even on a clean QR.
        return .unavailable(reason: "Vision barcode detection does not work in the Simulator; use a device.")
        #else
        return supportedSymbologies().isEmpty
            ? .unavailable(reason: "Vision reports no supported barcode symbologies on this device.")
            : .available
        #endif
    }

    /// Raw values accepted by the `symbologies` filter on this OS (empty if Vision can't report them).
    public func supportedSymbologies() -> [String] {
        ((try? VNDetectBarcodesRequest().supportedSymbologies()) ?? []).map(\.rawValue)
    }

    /// Detect codes in a `CGImage`. `symbologies` (raw values) restricts detection;
    /// empty = every supported symbology.
    public func detectBarcodes(in image: CGImage, symbologies: [String] = []) throws -> [DetectedBarcode] {
        try detectBarcodes(using: VisionImageInput.handler(for: image), symbologies: symbologies)
    }

    /// Detect codes in raw image bytes (PNG/JPEG/HEIC/…). Throws ``ToolError/invalidImage`` if undecodable.
    public func detectBarcodes(inImageData data: Data, symbologies: [String] = []) throws -> [DetectedBarcode] {
        guard let handler = VisionImageInput.handler(forImageData: data) else {
            throw ToolError.invalidImage
        }
        return try detectBarcodes(using: handler, symbologies: symbologies)
    }

    private func detectBarcodes(using handler: VNImageRequestHandler, symbologies: [String]) throws -> [DetectedBarcode] {
        let request = VNDetectBarcodesRequest()
        if !symbologies.isEmpty {
            let supported = Set(try request.supportedSymbologies().map(\.rawValue))
            if let unknown = symbologies.first(where: { !supported.contains($0) }) {
                throw ToolError.unsupportedSymbology(unknown)
            }
            request.symbologies = symbologies.map(VNBarcodeSymbology.init(rawValue:))
        }
        try handler.perform([request])
        return (request.results ?? []).map(Self.detectedBarcode(from:))
    }

    private static func detectedBarcode(from observation: VNBarcodeObservation) -> DetectedBarcode {
        DetectedBarcode(payload: observation.payloadStringValue,
                        symbology: observation.symbology.rawValue,
                        boundingBox: observation.boundingBox)
    }
}
