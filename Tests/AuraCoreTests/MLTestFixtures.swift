import XCTest
import AVFoundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import AuraCore

/// Inputs shared by the ML tool tests, synthesized so the tests stay offline and deterministic.
enum MLTestFixtures {

    static let expenseExamples: [TextClassifierTrainer.Example] = [
        .init(text: "Almuerzo en el restaurante", label: "food"),
        .init(text: "Cena con amigos en una pizzería", label: "food"),
        .init(text: "Compra semanal en el supermercado", label: "food"),
        .init(text: "Café y croissant en la panadería", label: "food"),
        .init(text: "Frutas y verduras del mercado", label: "food"),
        .init(text: "Lunch at the sushi place", label: "food"),
        .init(text: "Groceries at the supermarket", label: "food"),
        .init(text: "Pizza delivery for dinner", label: "food"),
        .init(text: "Coffee and a bagel at the bakery", label: "food"),
        .init(text: "Breakfast at the diner", label: "food"),
        .init(text: "Taxi al aeropuerto", label: "transport"),
        .init(text: "Gasolina para el coche", label: "transport"),
        .init(text: "Billete de autobús", label: "transport"),
        .init(text: "Abono mensual del metro", label: "transport"),
        .init(text: "Peaje de la autopista", label: "transport"),
        .init(text: "Uber ride home", label: "transport"),
        .init(text: "Train ticket to the city", label: "transport"),
        .init(text: "Fuel at the gas station", label: "transport"),
        .init(text: "Monthly subway pass", label: "transport"),
        .init(text: "Parking garage downtown", label: "transport"),
        .init(text: "Pago del alquiler del piso", label: "housing"),
        .init(text: "Factura de la luz", label: "housing"),
        .init(text: "Recibo del agua", label: "housing"),
        .init(text: "Cuota de la hipoteca", label: "housing"),
        .init(text: "Gastos de la comunidad de vecinos", label: "housing"),
        .init(text: "Monthly apartment rent", label: "housing"),
        .init(text: "Electricity bill", label: "housing"),
        .init(text: "Water utility bill", label: "housing"),
        .init(text: "Mortgage payment to the bank", label: "housing"),
        .init(text: "Home insurance premium", label: "housing")
    ]

    static func makeWorkDirectory(for testCase: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(testCase)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func writeCSV(_ examples: [TextClassifierTrainer.Example], to url: URL) throws {
        let rows = examples.map { "\"\($0.text.replacingOccurrences(of: "\"", with: "\"\""))\",\($0.label)" }
        try (["text,label"] + rows).joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    }

    /// A 440 Hz tone as a 16 kHz mono WAV.
    static func writeSineWave(to url: URL, seconds: Double) throws -> URL {
        let sampleRate = 16_000.0
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1))
        let frameCount = AVAudioFrameCount(sampleRate * seconds)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount))
        buffer.frameLength = frameCount
        let samples = (0..<Int(frameCount)).map { Float(0.5 * sin(2 * Double.pi * 440 * Double($0) / sampleRate)) }
        let channel = try XCTUnwrap(buffer.floatChannelData?[0])
        samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: samples.count) }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
        return url
    }

    /// A 64×64 solid square with a small grey marker whose position varies by `marker`.
    static func makeSquare(_ color: CGColor, marker: Int) -> CGImage? {
        guard let context = CGContext(
            data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        context.setFillColor(color)
        context.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        context.setFillColor(CGColor(gray: 0.5, alpha: 1))
        context.fill(CGRect(x: marker * 8, y: marker * 6, width: 8, height: 8))
        return context.makeImage()
    }

    static func writeSquare(_ color: CGColor, marker: Int, in folder: URL, named name: String) throws -> URL {
        let url = folder.appendingPathComponent(name).appendingPathExtension("png")
        let image = try XCTUnwrap(makeSquare(color, marker: marker))
        let destination = try XCTUnwrap(
            CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return url
    }
}
