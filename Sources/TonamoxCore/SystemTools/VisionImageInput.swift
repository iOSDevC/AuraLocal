import Foundation
import Vision
import CoreGraphics
import ImageIO

/// Shared image input for the Vision tools. Encoded data keeps its EXIF orientation,
/// so a camera photo is analysed upright and boxes match the displayed image.
enum VisionImageInput {

    static func handler(for image: CGImage) -> VNImageRequestHandler {
        VNImageRequestHandler(cgImage: image, options: [:])
    }

    /// `nil` when `data` is not a decodable image.
    static func handler(forImageData data: Data) -> VNImageRequestHandler? {
        guard let decoded = decode(data) else { return nil }
        return VNImageRequestHandler(cgImage: decoded.image, orientation: decoded.orientation, options: [:])
    }

    /// First frame of PNG/JPEG/HEIC/… bytes plus its EXIF orientation (`.up` when absent).
    static func decode(_ data: Data) -> (image: CGImage, orientation: CGImagePropertyOrientation)? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            return nil
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let rawOrientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value
        let orientation = rawOrientation.flatMap(CGImagePropertyOrientation.init(rawValue:)) ?? .up
        return (image, orientation)
    }
}
