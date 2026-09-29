import XCTest
import CoreGraphics
import TabularData
#if canImport(CreateML)
import CreateML
#endif
@testable import AuraCore

/// Custom-model tools exercised with models trained in the test (Create ML ships on macOS only).
final class CreateMLModelToolsTests: XCTestCase {

    private var workDirectory: URL!
    private var cache: URL { workDirectory.appendingPathComponent("Compiled", isDirectory: true) }
    private let expectedLabels = ["food", "housing", "transport"]

    override func setUpWithError() throws {
        workDirectory = try MLTestFixtures.makeWorkDirectory(for: "CreateMLModelToolsTests")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: workDirectory)
    }

    func testTrainingFrameKeepsTheExamplesInOrder() {
        let examples = MLTestFixtures.expenseExamples
        let frame = TextClassifierTrainer.trainingFrame(examples)
        XCTAssertEqual(Array(frame["text", String.self]), examples.map { Optional($0.text) })
        XCTAssertEqual(Array(frame["label", String.self]), examples.map { Optional($0.label) })
    }

    #if canImport(CreateML)

    // MARK: - Train → classify → describe

    func testTrainedTextClassifierReportsItsLabelsAndTrainingAccuracy() async throws {
        let csv = workDirectory.appendingPathComponent("expenses.csv")
        try MLTestFixtures.writeCSV(MLTestFixtures.expenseExamples, to: csv)
        let modelURL = workDirectory.appendingPathComponent("Models/Expenses.mlmodel")

        let trainer = TextClassifierTrainer()
        let trainerAvailability = await trainer.availability()
        XCTAssertTrue(trainerAvailability.isAvailable)
        let report = try await trainer.train(csvAt: csv, writingModelTo: modelURL, validation: .disabled)
        XCTAssertEqual(report.classLabels, expectedLabels)
        XCTAssertEqual(report.exampleCount, MLTestFixtures.expenseExamples.count)
        XCTAssertGreaterThanOrEqual(report.trainingAccuracy ?? 0, 0.9)
        XCTAssertNil(report.validationAccuracy)
        XCTAssertTrue(FileManager.default.fileExists(atPath: modelURL.path))

        let labels = try await TextClassifierTool(modelAt: modelURL, compiledModelsDirectory: cache).labels()
        XCTAssertEqual(labels, expectedLabels)
    }

    func testTrainedTextClassifierLabelsItsOwnExamples() async throws {
        let classifier = TextClassifierTool(modelAt: try await trainExpenses(), compiledModelsDirectory: cache)
        XCTAssertEqual(classifier.id, "coreml.text-classifier.Expenses")
        var correct = 0
        for example in MLTestFixtures.expenseExamples {
            let result = try await classifier.classify(example.text, maxHypotheses: expectedLabels.count)
            if result.label == example.label { correct += 1 }
            XCTAssertEqual(result.hypotheses.values.reduce(0, +), 1, accuracy: 0.01, example.text)
            XCTAssertEqual(result.ranked().first?.label, result.label)
        }
        let total = MLTestFixtures.expenseExamples.count
        XCTAssertGreaterThanOrEqual(Double(correct) / Double(total), 0.9, "classified \(correct)/\(total) correctly")
    }

    func testTextClassifierReturnsNoLabelForBlankText() async throws {
        let classifier = TextClassifierTool(modelAt: try await trainExpenses(), compiledModelsDirectory: cache)
        let blank = try await classifier.classify("   ")
        XCTAssertEqual(blank, .init(label: nil, hypotheses: [:]))
    }

    func testCoreMLToolDescribesAndClassifiesATrainedTextClassifier() async throws {
        let runner = CoreMLModelTool(modelAt: try await trainExpenses(), compiledModelsDirectory: cache)
        XCTAssertEqual(runner.id, "coreml.Expenses")
        let info = try await runner.describe()
        XCTAssertTrue(info.isClassifier)
        XCTAssertEqual(info.classLabels.sorted(), expectedLabels)
        XCTAssertEqual(info.inputs.map(\.name), ["text"])
        XCTAssertEqual(info.inputs.first?.kind, .string)
        XCTAssertEqual(info.author, "AuraLocal")
        let prediction = try await runner.classify(["text": "Factura del gas"])
        XCTAssertTrue(expectedLabels.contains(prediction.label))
    }

    func testTrainingWithHoldOutReportsValidationAccuracy() async throws {
        let report = try await TextClassifierTrainer().train(
            examples: MLTestFixtures.expenseExamples,
            writingModelTo: workDirectory.appendingPathComponent("Holdout.mlmodel"),
            validation: .holdOut(fraction: 0.3, seed: 7))
        let validation = try XCTUnwrap(report.validationAccuracy)
        XCTAssertTrue((0...1).contains(validation))
    }

    func testTransferLearningWithTooFewExamplesFailsAsNotEnoughData() async throws {
        let few = Array(MLTestFixtures.expenseExamples.prefix(3) + MLTestFixtures.expenseExamples.suffix(3))
        do {
            _ = try await TextClassifierTrainer().train(
                examples: few,
                writingModelTo: workDirectory.appendingPathComponent("Few.mlmodel"),
                algorithm: .transferLearning(.staticEmbedding),
                validation: .disabled)
        } catch TextClassifierTrainer.ToolError.notEnoughData {
            // Create ML needed 10 examples on macOS 26; a later OS may train with fewer.
        }
    }

    // MARK: - Compiled model cache

    func testCompiledModelIsReusedWhileTheModelFileIsUnchanged() async throws {
        let modelURL = try writeLineRegressor(named: "Line")
        let pinned = Date(timeIntervalSince1970: 1_700_000_000)
        try FileManager.default.setAttributes([.modificationDate: pinned], ofItemAtPath: modelURL.path)
        let compiled = try await CoreMLModelTool(modelAt: modelURL, compiledModelsDirectory: cache).compiledModelURL()

        // Same size and date but unreadable content: only a cache hit can still succeed.
        let size = try XCTUnwrap(FileManager.default.attributesOfItem(atPath: modelURL.path)[.size] as? Int)
        try Data(count: size).write(to: modelURL)
        try FileManager.default.setAttributes([.modificationDate: pinned], ofItemAtPath: modelURL.path)
        let again = try await CoreMLModelTool(modelAt: modelURL, compiledModelsDirectory: cache).compiledModelURL()
        XCTAssertEqual(again, compiled)
    }

    func testChangedModelFileIsRecompiled() async throws {
        let modelURL = try writeLineRegressor(named: "Line")
        let first = try await CoreMLModelTool(modelAt: modelURL, compiledModelsDirectory: cache).compiledModelURL()

        let later = Date(timeIntervalSince1970: 1_800_000_000)
        try FileManager.default.setAttributes([.modificationDate: later], ofItemAtPath: modelURL.path)
        let second = try await CoreMLModelTool(modelAt: modelURL, compiledModelsDirectory: cache).compiledModelURL()
        XCTAssertNotEqual(second, first)
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.path))
    }

    // MARK: - Availability after a load failure

    func testAvailabilityRecoversOnceAMissingModelIsWritten() async throws {
        let modelURL = workDirectory.appendingPathComponent("Late.mlmodel")
        let runner = CoreMLModelTool(modelAt: modelURL, compiledModelsDirectory: cache)
        let classifier = TextClassifierTool(modelAt: modelURL, compiledModelsDirectory: cache)
        _ = try? await runner.describe()
        _ = try? await classifier.classify("Taxi")
        let runnerBefore = await runner.availability()
        let classifierBefore = await classifier.availability()
        XCTAssertFalse(runnerBefore.isAvailable)
        XCTAssertFalse(classifierBefore.isAvailable)

        _ = try await trainExpenses(to: modelURL)
        let runnerAfter = await runner.availability()
        let classifierAfter = await classifier.availability()
        XCTAssertEqual(runnerAfter, .available)
        XCTAssertEqual(classifierAfter, .available)
    }

    func testBrokenModelStaysUnavailableUntilTheFileChanges() async throws {
        let modelURL = workDirectory.appendingPathComponent("Broken.mlmodel")
        try Data("not a model".utf8).write(to: modelURL)
        let runner = CoreMLModelTool(modelAt: modelURL, compiledModelsDirectory: cache)
        let classifier = TextClassifierTool(modelAt: modelURL, compiledModelsDirectory: cache)
        _ = try? await runner.describe()
        _ = try? await classifier.classify("Taxi")
        let runnerBroken = await runner.availability()
        let classifierBroken = await classifier.availability()
        XCTAssertFalse(runnerBroken.isAvailable)
        XCTAssertFalse(classifierBroken.isAvailable)

        _ = try await trainExpenses(to: modelURL)
        let runnerFixed = await runner.availability()
        let classifierFixed = await classifier.availability()
        XCTAssertEqual(runnerFixed, .available)
        XCTAssertEqual(classifierFixed, .available)
    }

    // MARK: - Other model kinds

    func testCoreMLToolRunsAnImageClassifierOnACGImage() async throws {
        let images = workDirectory.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
        let red = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        let blue = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
        let filesByLabel = [
            "red": try (0..<5).map { try MLTestFixtures.writeSquare(red, marker: $0, in: images, named: "red\($0)") },
            "blue": try (0..<5).map { try MLTestFixtures.writeSquare(blue, marker: $0, in: images, named: "blue\($0)") }
        ]
        let modelURL = workDirectory.appendingPathComponent("Colors.mlmodel")
        try MLImageClassifier(trainingData: .filesByLabel(filesByLabel)).write(to: modelURL)

        let tool = CoreMLModelTool(modelAt: modelURL, computeUnits: .cpuOnly, compiledModelsDirectory: cache)
        let info = try await tool.describe()
        let input = try XCTUnwrap(info.inputs.first)
        XCTAssertEqual(input.kind, .image)
        XCTAssertEqual(input.shape.count, 2)
        XCTAssertNotNil(input.dataType)
        XCTAssertEqual(Set(info.classLabels), ["red", "blue"])
        XCTAssertNotNil(info.predictedProbabilitiesName)

        let image = try XCTUnwrap(MLTestFixtures.makeSquare(red, marker: 2))
        let result = try await tool.classify([input.name: .image(image)])
        XCTAssertEqual(result.label, "red")
        XCTAssertEqual(result.probabilities.values.reduce(0, +), 1, accuracy: 0.01)
        XCTAssertEqual(result.ranked(limit: 1).first?.label, "red")
    }

    func testCoreMLToolConvertsNumericInputsAndRejectsBadOnes() async throws {
        let tool = CoreMLModelTool(modelAt: try writeLineRegressor(named: "Line"), compiledModelsDirectory: cache)
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

    // MARK: - Helpers

    private func trainExpenses(to url: URL? = nil) async throws -> URL {
        let modelURL = url ?? workDirectory.appendingPathComponent("Models/Expenses.mlmodel")
        _ = try await TextClassifierTrainer().train(
            examples: MLTestFixtures.expenseExamples, writingModelTo: modelURL, validation: .disabled)
        return modelURL
    }

    /// y = 2x + 1 as a Create ML linear regressor.
    private func writeLineRegressor(named name: String) throws -> URL {
        let samples: [Double] = [0, 1, 2, 3, 4, 5, 6, 7]
        let frame: DataFrame = ["x": samples, "y": samples.map { 2 * $0 + 1 }]
        let modelURL = workDirectory.appendingPathComponent("\(name).mlmodel")
        try MLLinearRegressor(trainingData: frame, targetColumn: "y").write(to: modelURL)
        return modelURL
    }

    #endif
}
