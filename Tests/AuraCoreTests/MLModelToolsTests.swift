import XCTest
import CoreML
@testable import AuraCore

final class MLModelToolsTests: XCTestCase {

    private var workDirectory: URL!

    override func setUpWithError() throws {
        workDirectory = try MLTestFixtures.makeWorkDirectory(for: "MLModelToolsTests")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: workDirectory)
    }

    private func tone(seconds: Double, named name: String = "tone.wav") throws -> URL {
        try MLTestFixtures.writeSineWave(to: workDirectory.appendingPathComponent(name), seconds: seconds)
    }

    private func classifyCountingWindows(
        _ url: URL
    ) async throws -> (results: [SoundClassificationTool.Classification], windows: Int) {
        let collector = WindowCollector()
        let results = try await SoundClassificationTool()
            .classify(audioFileAt: url, maxResults: 3, aggregation: .peak, collector: collector)
        return (results, collector.windowCount)
    }

    // MARK: - Sound classification

    func testSoundClassificationOfSynthesizedToneIsBoundedAndSorted() async throws {
        let results = try await SoundClassificationTool().classify(audioFileAt: try tone(seconds: 6), maxResults: 5)

        XCTAssertFalse(results.isEmpty)
        XCTAssertLessThanOrEqual(results.count, 5)
        XCTAssertTrue(results.allSatisfy { (0...1).contains($0.confidence) }, "\(results)")
        XCTAssertEqual(results.map(\.confidence), results.map(\.confidence).sorted(by: >))
    }

    func testSoundClassificationMeanNeverExceedsPeak() async throws {
        let wav = try tone(seconds: 6)
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

    func testSoundClassificationOfAudioShorterThanTheDefaultWindowScoresOneWindow() async throws {
        for seconds in [0.6, 1.3, 2.9] {
            let analysis = try await classifyCountingWindows(try tone(seconds: seconds, named: "short-\(seconds).wav"))
            XCTAssertEqual(analysis.results.count, 3, "\(seconds) s")
            XCTAssertEqual(analysis.windows, 1, "\(seconds) s")
        }
    }

    func testSoundClassificationOfAudioUnderHalfASecondThrowsTooShort() async throws {
        let wav = try tone(seconds: 0.3)
        do {
            _ = try await SoundClassificationTool().classify(audioFileAt: wav)
            XCTFail("expected an error")
        } catch SoundClassificationTool.ToolError.audioTooShort(let seconds, let minimum) {
            XCTAssertEqual(seconds, 0.3, accuracy: 0.001)
            XCTAssertEqual(minimum, 0.5, accuracy: 0.001)
        }
    }

    func testSoundClassificationWithZeroMaxResultsIsEmpty() async throws {
        let results = try await SoundClassificationTool().classify(audioFileAt: try tone(seconds: 1), maxResults: 0)
        XCTAssertTrue(results.isEmpty)
    }

    func testCancellingMidAnalysisStopsTheAnalyzerEarly() async throws {
        // 3 s windows every 1.5 s: a 300 s file has 199 of them.
        let wav = try tone(seconds: 300, named: "long.wav")
        let collector = WindowCollector()
        let task = Task {
            try await SoundClassificationTool()
                .classify(audioFileAt: wav, maxResults: 5, aggregation: .peak, collector: collector)
        }
        let deadline = Date().addingTimeInterval(10)
        while collector.windowCount == 0 && Date() < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        XCTAssertGreaterThan(collector.windowCount, 0, "the analysis never started")
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected cancellation")
        } catch is CancellationError {
            XCTAssertLessThan(collector.windowCount, 100, "the analyzer kept going after the cancel")
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

    func testKnownSoundsAreSortedAndIncludeSpeech() throws {
        let known = try SoundClassificationTool().knownSounds()
        XCTAssertFalse(known.isEmpty)
        XCTAssertEqual(known, known.sorted())
        XCTAssertTrue(known.contains("speech"))
    }

    func testSoundToolIsAvailableWithFixedIDAndAudioCategory() async {
        let tool = SoundClassificationTool()
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

    func testFeatureValueLiteralsMapToMatchingCases() {
        let inputs: [String: CoreMLModelTool.FeatureValue] = [
            "text": "hola", "count": 3, "ratio": 0.5, "vector": [1, 2.5]
        ]
        XCTAssertEqual(inputs["text"], .string("hola"))
        XCTAssertEqual(inputs["count"], .int(3))
        XCTAssertEqual(inputs["ratio"], .double(0.5))
        XCTAssertEqual(inputs["vector"], .doubles([1, 2.5]))
    }

    func testClassificationRankedIsSortedAndClampsTheLimit() {
        let result = CoreMLModelTool.Classification(label: "b", probabilities: ["a": 0.2, "b": 0.7, "c": 0.1])
        XCTAssertEqual(result.ranked().map(\.label), ["b", "a", "c"])
        XCTAssertEqual(result.ranked(limit: 1).map(\.label), ["b"])
        XCTAssertTrue(result.ranked(limit: 0).isEmpty)
        XCTAssertTrue(result.ranked(limit: -1).isEmpty)
    }

    // MARK: - CSV loading and trainer input checks

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
    }

    func testLoadExamplesWithAMissingColumnThrowsInvalidCSV() throws {
        let csv = workDirectory.appendingPathComponent("expenses.csv")
        try MLTestFixtures.writeCSV(MLTestFixtures.expenseExamples, to: csv)
        XCTAssertThrowsError(try TextClassifierTrainer.loadExamples(csvAt: csv, textColumn: "missing")) { error in
            guard case TextClassifierTrainer.ToolError.invalidCSV = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
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
            _ = try await trainer.train(examples: MLTestFixtures.expenseExamples, writingModelTo: textFile)
            XCTFail("expected an error")
        } catch TextClassifierTrainer.ToolError.invalidOutputURL {
            // expected
        }
        do {
            _ = try await trainer.train(examples: MLTestFixtures.expenseExamples, writingModelTo: output,
                                        validation: .holdOut(fraction: 1.5, seed: 1))
            XCTFail("expected an error")
        } catch TextClassifierTrainer.ToolError.invalidValidationFraction {
            // expected
        }
    }
}
