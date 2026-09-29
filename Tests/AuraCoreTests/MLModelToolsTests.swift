import XCTest
import AVFoundation
import CoreGraphics
import CoreML
import ImageIO
import TabularData
import UniformTypeIdentifiers
#if canImport(CreateML)
import CreateML
#endif
@testable import AuraCore

final class MLModelToolsTests: XCTestCase {

    private var workDirectory: URL!

    override func setUpWithError() throws {
        workDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLModelToolsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: workDirectory)
    }

    // MARK: - Sound classification

    func testSoundClassificationOfSynthesizedToneIsBoundedAndSorted() async throws {
        let wav = try Self.writeSineWave(to: workDirectory.appendingPathComponent("tone.wav"), seconds: 6)
        let results = try await SoundClassificationTool().classify(audioFileAt: wav, maxResults: 5)

        XCTAssertFalse(results.isEmpty)
        XCTAssertLessThanOrEqual(results.count, 5)
        XCTAssertTrue(results.allSatisfy { (0...1).contains($0.confidence) }, "\(results)")
        XCTAssertEqual(results.map(\.confidence), results.map(\.confidence).sorted(by: >))
    }

    func testSoundClassificationMeanNeverExceedsPeak() async throws {
        let wav = try Self.writeSineWave(to: workDirectory.appendingPathComponent("tone.wav"), seconds: 6)
        let tool = SoundClassificationTool()
        let everything = try tool.knownSounds().count
        let peaks = try await tool.classify(audioFileAt: wav, maxResults: everything, aggregation: .peak)
        let means = try await tool.classify(audioFileAt: wav, maxResults: everything, aggregation: .mean)

        XCTAssertEqual(peaks.count, everything)
        XCTAssertEqual(means.count, everything)
        let peakByID = Dictionary(uniqueKeysWithValues: peaks.map { ($0.identifier, $0.confidence) })
        XCTAssertTrue(means.allSatisfy { $0.confidence <= (peakByID[$0.identifier] ?? 0) + 1e-9 })
        XCTAssertEqual(means.map(\.confidence), means.map(\.confidence).sorted(by: >))
    }

    func testSoundClassificationWithZeroMaxResultsIsEmpty() async throws {
        let wav = try Self.writeSineWave(to: workDirectory.appendingPathComponent("tone.wav"), seconds: 1)
        let results = try await SoundClassificationTool().classify(audioFileAt: wav, maxResults: 0)
        XCTAssertTrue(results.isEmpty)
    }

    func testSoundClassificationHonoursCancellation() async throws {
        let wav = try Self.writeSineWave(to: workDirectory.appendingPathComponent("long.wav"), seconds: 60)
        let task = Task { try await SoundClassificationTool().classify(audioFileAt: wav) }
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected cancellation")
        } catch is CancellationError {
            // expected
        }
    }

    func testSoundClassificationOfUnreadableFileThrows() async throws {
        let junk = workDirectory.appendingPathComponent("junk.wav")
        try Data([0, 1, 2, 3]).write(to: junk)
        do {
            _ = try await SoundClassificationTool().classify(audioFileAt: junk)
            XCTFail("expected an error")
        } catch SoundClassificationTool.ToolError.unreadableAudio {
            // expected
        }
    }

    func testKnownSoundsAreNonEmptyAndSorted() async throws {
        let tool = SoundClassificationTool()
        let known = try tool.knownSounds()
        XCTAssertFalse(known.isEmpty)
        XCTAssertEqual(known, known.sorted())
        XCTAssertTrue(known.contains("speech"))
        let availability = await tool.availability()
        XCTAssertTrue(availability.isAvailable)
        XCTAssertEqual(tool.id, "system.audio.sounds")
        XCTAssertEqual(tool.category, .audio)
    }

    // MARK: - CoreMLModelTool (no model needed)

    func testCoreMLToolReportsMissingModel() async throws {
        let tool = CoreMLModelTool(modelAt: workDirectory.appendingPathComponent("Missing.mlmodel"))
        XCTAssertEqual(tool.id, "coreml.Missing")
        XCTAssertEqual(tool.category, .customModel)
        let availability = await tool.availability()
        XCTAssertFalse(availability.isAvailable)
        do {
            _ = try await tool.describe()
            XCTFail("expected an error")
        } catch let error as CoreMLModelTool.ToolError {
            XCTAssertEqual(error, .modelNotFound(workDirectory.appendingPathComponent("Missing.mlmodel").path))
        }
    }

    func testCoreMLToolRejectsUnsupportedFormat() async throws {
        let onnx = workDirectory.appendingPathComponent("model.onnx")
        try Data([0]).write(to: onnx)
        let availability = await CoreMLModelTool(modelAt: onnx).availability()
        XCTAssertEqual(availability.reason, CoreMLModelTool.ToolError.unsupportedModelFormat("onnx").errorDescription)
    }

    func testMultiArrayConversionRoundTripsEveryElementType() {
        let values: [Double] = [1, 2, 3, 4, 5, 6]
        let expected = CoreMLModelTool.FeatureValue.multiArray(shape: [2, 3], values: values)
        for dataType in [MLMultiArrayDataType.double, .float32, .float16, .int32] {
            let array = FeatureConversion.makeMultiArray(values, shape: [2, 3], dataType: dataType)
            XCTAssertEqual(array.dataType, dataType)
            XCTAssertEqual(CoreMLModelTool.FeatureValue.multiArray(copying: array), expected)
        }
    }

    func testFeatureValueLiterals() {
        let inputs: [String: CoreMLModelTool.FeatureValue] = [
            "text": "hola", "count": 3, "ratio": 0.5, "vector": [1, 2.5]
        ]
        XCTAssertEqual(inputs["text"], .string("hola"))
        XCTAssertEqual(inputs["count"], .int(3))
        XCTAssertEqual(inputs["ratio"], .double(0.5))
        XCTAssertEqual(inputs["vector"], .doubles([1, 2.5]))
    }

    // MARK: - CSV loading

    func testLoadExamplesFromCSVHandlesQuotesAccentsAndBlankRows() throws {
        let csv = workDirectory.appendingPathComponent("expenses.csv")
        let content = """
        texto,categoria,importe
        "Cena, vino y postre",comida,42.5
        Taxi al aeropuerto,transporte,30
        ,vivienda,0
        "Factura de la luz — ""tarifa"" nocturna",vivienda,61
        """
        try content.write(to: csv, atomically: true, encoding: .utf8)

        let examples = try TextClassifierTrainer.loadExamples(csvAt: csv, textColumn: "texto", labelColumn: "categoria")
        XCTAssertEqual(examples, [
            .init(text: "Cena, vino y postre", label: "comida"),
            .init(text: "Taxi al aeropuerto", label: "transporte"),
            .init(text: "Factura de la luz — \"tarifa\" nocturna", label: "vivienda")
        ])
        XCTAssertThrowsError(try TextClassifierTrainer.loadExamples(csvAt: csv, textColumn: "missing"))
    }

    func testTrainerRejectsBadInputBeforeTraining() async throws {
        let trainer = TextClassifierTrainer()
        let output = workDirectory.appendingPathComponent("Model.mlmodel")
        do {
            _ = try await trainer.train(examples: [.init(text: "rent", label: "housing")], writingModelTo: output)
            XCTFail("expected an error")
        } catch TextClassifierTrainer.ToolError.notEnoughData {
            // expected
        }
        do {
            let textFile = workDirectory.appendingPathComponent("Model.txt")
            _ = try await trainer.train(examples: Self.expenseExamples, writingModelTo: textFile)
            XCTFail("expected an error")
        } catch TextClassifierTrainer.ToolError.invalidOutputURL {
            // expected
        }
        do {
            _ = try await trainer.train(examples: Self.expenseExamples, writingModelTo: output,
                                        validation: .holdOut(fraction: 1.5, seed: 1))
            XCTFail("expected an error")
        } catch TextClassifierTrainer.ToolError.invalidValidationFraction {
            // expected
        }
    }

    // MARK: - Train → classify → describe (Create ML, macOS)

    #if canImport(CreateML)
    func testTrainedTextClassifierClassifiesItsExamplesAndDescribesItsLabels() async throws {
        let csv = workDirectory.appendingPathComponent("expenses.csv")
        try Self.writeCSV(Self.expenseExamples, to: csv)
        let modelURL = workDirectory.appendingPathComponent("Models/Expenses.mlmodel")
        let cache = workDirectory.appendingPathComponent("Compiled", isDirectory: true)
        let expectedLabels = ["food", "housing", "transport"]

        let trainer = TextClassifierTrainer()
        let trainerAvailability = await trainer.availability()
        XCTAssertTrue(trainerAvailability.isAvailable)
        let report = try await trainer.train(csvAt: csv, writingModelTo: modelURL, validation: .disabled)
        XCTAssertEqual(report.classLabels, expectedLabels)
        XCTAssertEqual(report.exampleCount, Self.expenseExamples.count)
        XCTAssertGreaterThanOrEqual(report.trainingAccuracy ?? 0, 0.9)
        XCTAssertNil(report.validationAccuracy)
        XCTAssertTrue(FileManager.default.fileExists(atPath: modelURL.path))

        let classifier = TextClassifierTool(modelAt: modelURL, compiledModelsDirectory: cache)
        XCTAssertEqual(classifier.id, "coreml.text-classifier.Expenses")
        let labels = try await classifier.labels()
        XCTAssertEqual(labels, expectedLabels)
        var correct = 0
        for example in Self.expenseExamples {
            let result = try await classifier.classify(example.text, maxHypotheses: expectedLabels.count)
            if result.label == example.label { correct += 1 }
            XCTAssertEqual(result.hypotheses.values.reduce(0, +), 1, accuracy: 0.01, example.text)
            XCTAssertEqual(result.ranked().first?.label, result.label)
        }
        let accuracy = Double(correct) / Double(Self.expenseExamples.count)
        XCTAssertGreaterThanOrEqual(accuracy, 0.9, "classified \(correct)/\(Self.expenseExamples.count) correctly")
        let blank = try await classifier.classify("   ")
        XCTAssertEqual(blank, .init(label: nil, hypotheses: [:]))

        let runner = CoreMLModelTool(modelAt: modelURL, compiledModelsDirectory: cache)
        XCTAssertEqual(runner.id, "coreml.Expenses")
        let info = try await runner.describe()
        XCTAssertTrue(info.isClassifier)
        XCTAssertEqual(info.classLabels.sorted(), expectedLabels)
        XCTAssertEqual(info.inputs.map(\.name), ["text"])
        XCTAssertEqual(info.inputs.first?.kind, .string)
        XCTAssertEqual(info.author, "AuraLocal")
        let prediction = try await runner.classify(["text": "Factura del gas"])
        XCTAssertTrue(expectedLabels.contains(prediction.label))

        let compiled = try await runner.compiledModelURL()
        XCTAssertEqual(compiled.deletingLastPathComponent().standardizedFileURL, cache.standardizedFileURL)
        let sameModelAgain = try await CoreMLModelTool(modelAt: modelURL, compiledModelsDirectory: cache)
            .compiledModelURL()
        XCTAssertEqual(sameModelAgain, compiled)
    }

    func testTrainingWithHoldOutReportsValidationAccuracy() async throws {
        let report = try await TextClassifierTrainer().train(
            examples: Self.expenseExamples,
            writingModelTo: workDirectory.appendingPathComponent("Holdout.mlmodel"),
            validation: .holdOut(fraction: 0.3, seed: 7))
        let validation = try XCTUnwrap(report.validationAccuracy)
        XCTAssertTrue((0...1).contains(validation))
    }

    func testCoreMLToolRunsAnImageClassifierOnACGImage() async throws {
        let images = workDirectory.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
        let red = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        let blue = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
        let filesByLabel = [
            "red": try (0..<5).map { try Self.writeSquare(red, marker: $0, in: images, named: "red\($0)") },
            "blue": try (0..<5).map { try Self.writeSquare(blue, marker: $0, in: images, named: "blue\($0)") }
        ]
        let modelURL = workDirectory.appendingPathComponent("Colors.mlmodel")
        try MLImageClassifier(trainingData: .filesByLabel(filesByLabel)).write(to: modelURL)

        let tool = CoreMLModelTool(modelAt: modelURL, computeUnits: .cpuOnly,
                                   compiledModelsDirectory: workDirectory.appendingPathComponent("Compiled"))
        let info = try await tool.describe()
        let input = try XCTUnwrap(info.inputs.first)
        XCTAssertEqual(input.kind, .image)
        XCTAssertEqual(input.shape.count, 2)
        XCTAssertNotNil(input.dataType)
        XCTAssertEqual(Set(info.classLabels), ["red", "blue"])
        XCTAssertNotNil(info.predictedProbabilitiesName)

        let image = try XCTUnwrap(Self.makeSquare(red, marker: 2))
        let result = try await tool.classify([input.name: .image(image)])
        XCTAssertEqual(result.label, "red")
        XCTAssertEqual(result.probabilities.values.reduce(0, +), 1, accuracy: 0.01)
        XCTAssertEqual(result.ranked(limit: 1).first?.label, "red")
    }

    func testCoreMLToolConvertsNumericInputsAndRejectsBadOnes() async throws {
        let samples: [Double] = [0, 1, 2, 3, 4, 5, 6, 7]
        let frame: DataFrame = ["x": samples, "y": samples.map { 2 * $0 + 1 }]
        let modelURL = workDirectory.appendingPathComponent("Line.mlmodel")
        try MLLinearRegressor(trainingData: frame, targetColumn: "y").write(to: modelURL)

        let cache = workDirectory.appendingPathComponent("Compiled")
        let tool = CoreMLModelTool(modelAt: modelURL, compiledModelsDirectory: cache)
        let info = try await tool.describe()
        XCTAssertFalse(info.isClassifier)
        XCTAssertEqual(info.inputs.first?.kind, .double)

        let outputs = try await tool.predict(["x": 10])
        guard case .double(let predicted) = outputs["y"] else {
            return XCTFail("unexpected outputs \(outputs)")
        }
        XCTAssertEqual(predicted, 21, accuracy: 0.5)

        do {
            _ = try await tool.predict(["z": 1])
            XCTFail("expected an error")
        } catch let error as CoreMLModelTool.ToolError {
            XCTAssertEqual(error, .unknownInput("z"))
        }
        do {
            _ = try await tool.predict(["x": "ten"])
            XCTFail("expected an error")
        } catch CoreMLModelTool.ToolError.incompatibleInput(let name, _) {
            XCTAssertEqual(name, "x")
        }
        do {
            _ = try await tool.classify(["x": 1])
            XCTFail("expected an error")
        } catch let error as CoreMLModelTool.ToolError {
            XCTAssertEqual(error, .notAClassifier)
        }
    }
    #endif
}

// MARK: - Fixtures

private extension MLModelToolsTests {

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
