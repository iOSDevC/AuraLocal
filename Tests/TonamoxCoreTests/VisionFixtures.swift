import XCTest
import CoreGraphics
import CoreText
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import UniformTypeIdentifiers

/// Images synthesized for the Vision tool tests, so they stay offline and deterministic.
enum VisionFixtures {

    static func isNormalized(_ rect: CGRect) -> Bool {
        let tolerance: CGFloat = 0.001
        return rect.minX >= -tolerance && rect.minY >= -tolerance
            && rect.maxX <= 1 + tolerance && rect.maxY <= 1 + tolerance
            && rect.width > 0 && rect.height > 0
    }

    static func makeContext(width: Int, height: Int) -> CGContext? {
        CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                  space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    }

    static func blankImage(width: Int = 640, height: Int = 480) -> CGImage? {
        guard let ctx = makeContext(width: width, height: height) else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()
    }

    /// Each entry is a line of text and its baseline y (CoreGraphics: 0 = bottom).
    static func renderLines(
        _ lines: [(String, CGFloat)], font: String = "Helvetica", width: Int = 800, height: Int = 400
    ) -> CGImage? {
        guard let ctx = makeContext(width: width, height: height) else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        for (text, baseline) in lines {
            draw(text, font: font, baseline: baseline, in: ctx)
        }
        return ctx.makeImage()
    }

    static func draw(_ text: String, font fontName: String, baseline: CGFloat, in ctx: CGContext) {
        let font = CTFontCreateWithName(fontName as CFString, 64, nil)
        let black = CGColor(red: 0, green: 0, blue: 0, alpha: 1)
        let attrs: [CFString: Any] = [kCTFontAttributeName: font, kCTForegroundColorAttributeName: black]
        guard let attributed = CFAttributedStringCreate(nil, text as CFString, attrs as CFDictionary) else { return }
        ctx.textPosition = CGPoint(x: 30, y: baseline)
        CTLineDraw(CTLineCreateWithAttributedString(attributed), ctx)
    }

    /// Stores `image` rotated 90° counter-clockwise; EXIF `.left` displays it upright again.
    static func rotatedCounterClockwise(_ image: CGImage) -> CGImage? {
        guard let ctx = makeContext(width: image.height, height: image.width) else { return nil }
        ctx.translateBy(x: 0, y: CGFloat(image.width))
        ctx.rotate(by: -.pi / 2)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return ctx.makeImage()
    }

    /// A QR code from CoreImage's generator, scaled up and padded with a white quiet zone.
    static func renderQR(_ payload: String) -> CGImage? {
        let generator = CIFilter.qrCodeGenerator()
        generator.message = Data(payload.utf8)
        generator.correctionLevel = "M"
        guard let code = generator.outputImage else { return nil }
        let scaled = code.transformed(by: CGAffineTransform(scaleX: 10, y: 10))
        let padding: CGFloat = 40
        let canvas = CGRect(x: 0, y: 0,
                            width: scaled.extent.width + 2 * padding, height: scaled.extent.height + 2 * padding)
        let padded = scaled.transformed(by: CGAffineTransform(translationX: padding, y: padding))
            .composited(over: CIImage(color: .white).cropped(to: canvas))
        return CIContext().createCGImage(padded, from: canvas)
    }

    /// Coloured shapes on a gradient: enough structure for the classifier to score, no fixed label expected.
    static func syntheticScene(width: Int = 512, height: Int = 512) -> CGImage? {
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

    static func encode(_ image: CGImage, type: UTType, orientation: CGImagePropertyOrientation = .up) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, type.identifier as CFString, 1, nil) else {
            return nil
        }
        let properties: [CFString: Any] = [
            kCGImagePropertyOrientation: orientation.rawValue,
            kCGImageDestinationLossyCompressionQuality: 1.0
        ]
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }
}
