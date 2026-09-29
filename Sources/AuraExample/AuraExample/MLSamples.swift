import Foundation
import CoreGraphics
import CoreText
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import UniformTypeIdentifiers
import AVFoundation
import AuraCore

/// Offline sample inputs so every section can be tried without picking a file.
nonisolated enum MLSamples {
    static let expenses: [TextClassifierTrainer.Example] = expensePairs.map(makeExample)

    private static let expensePairs: [(text: String, label: String)] = [
        ("Almuerzo en el restaurante del centro", "food"), ("Compra semanal en el supermercado", "food"),
        ("Café y croissant en la panadería", "food"), ("Cena con amigos, pizza y cervezas", "food"),
        ("Pedido de comida a domicilio", "food"), ("Fruta y verdura del mercado", "food"),
        ("Lunch at the sushi place", "food"), ("Groceries at the supermarket", "food"),
        ("Coffee and a bagel downtown", "food"), ("Dinner delivery from the Thai restaurant", "food"),
        ("Billete de metro mensual", "transport"), ("Taxi al aeropuerto", "transport"),
        ("Gasolina para el coche", "transport"), ("Tren de alta velocidad a Barcelona", "transport"),
        ("Parking en el centro comercial", "transport"), ("Uber to the office", "transport"),
        ("Monthly bus pass", "transport"), ("Filled up the car with gas", "transport"),
        ("Train ticket to Boston", "transport"), ("Airport parking for three days", "transport"),
        ("Alquiler del piso de octubre", "housing"), ("Factura de la luz", "housing"),
        ("Recibo del agua y la basura", "housing"), ("Cuota de la comunidad de vecinos", "housing"),
        ("Reparación de la caldera", "housing"), ("Monthly rent payment", "housing"),
        ("Electricity bill", "housing"), ("Internet and cable for the apartment", "housing"),
        ("Plumber fixed the kitchen sink", "housing"), ("Home insurance renewal", "housing")
    ]

    private static func makeExample(_ pair: (text: String, label: String)) -> TextClassifierTrainer.Example {
        TextClassifierTrainer.Example(text: pair.text, label: pair.label)
    }

    /// A white canvas with two lines of text and a QR code, encoded as PNG.
    static func receiptWithQRCode() -> Data? {
        let size = CGSize(width: 900, height: 420)
        guard let canvas = CGContext(
            data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let code = qrCode("https://github.com/iOSDevC/AuraLocal") else { return nil }
        canvas.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        canvas.fill(CGRect(origin: .zero, size: size))
        draw("CAFÉ CENTRAL  4,50 EUR", in: canvas, at: CGPoint(x: 40, y: 300))
        draw("Tarjeta ****1234", in: canvas, at: CGPoint(x: 40, y: 220))
        canvas.interpolationQuality = .none
        canvas.draw(code, in: CGRect(x: 560, y: 40, width: 300, height: 300))
        guard let image = canvas.makeImage() else { return nil }
        let output = NSMutableData()
        let pngType = UTType.png.identifier as CFString
        guard let destination = CGImageDestinationCreateWithData(output, pngType, 1, nil) else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        return CGImageDestinationFinalize(destination) ? output as Data : nil
    }

    private static func qrCode(_ text: String) -> CGImage? {
        let generator = CIFilter.qrCodeGenerator()
        generator.message = Data(text.utf8)
        generator.correctionLevel = "M"
        guard let output = generator.outputImage else { return nil }
        return CIContext().createCGImage(output, from: output.extent)
    }

    private static func draw(_ text: String, in canvas: CGContext, at point: CGPoint) {
        let font = CTFontCreateWithName("Helvetica" as CFString, 36, nil)
        let attributes: [CFString: Any] = [
            kCTFontAttributeName: font,
            kCTForegroundColorAttributeName: CGColor(red: 0, green: 0, blue: 0, alpha: 1)
        ]
        guard let attributed = CFAttributedStringCreate(nil, text as CFString, attributes as CFDictionary) else {
            return
        }
        canvas.textPosition = point
        CTLineDraw(CTLineCreateWithAttributedString(attributed), canvas)
    }

    /// Upright (EXIF-applied) preview, so Vision boxes from the `Data` overloads line up with it.
    static func uprightPreview(of data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 1600
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    /// A 3 s, 440 Hz sine tone as a 16 kHz mono WAV in the temporary directory.
    static func toneFile() throws -> URL {
        let sampleRate = 16_000.0
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(sampleRate * 3)),
              let samples = buffer.floatChannelData?[0] else {
            throw CocoaError(.fileWriteUnknown)
        }
        buffer.frameLength = buffer.frameCapacity
        let step = Float(2 * Double.pi * 440 / sampleRate)
        var phase: Float = 0
        for index in 0..<Int(buffer.frameLength) {
            samples[index] = 0.5 * sin(phase)
            phase += step
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("aura-tone-440hz.wav")
        try? FileManager.default.removeItem(at: url)
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
        file.close()
        return url
    }

    static func readImported(_ url: URL) throws -> Data {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        return try Data(contentsOf: url)
    }

    /// Copies a picked file out of its security scope so later async reads keep working.
    static func copyImported(_ url: URL) throws -> URL {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let copy = FileManager.default.temporaryDirectory
            .appendingPathComponent("aura-ml-\(UUID().uuidString)")
            .appendingPathExtension(url.pathExtension)
        try FileManager.default.copyItem(at: url, to: copy)
        return copy
    }
}
