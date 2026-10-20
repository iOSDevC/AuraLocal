import CoreGraphics
import CoreML
import ImageIO
import XCTest
@testable import TonamoxImageGen

/// `CoreMLDiffusionEngine` with a fake pipeline: no weights, no Core ML, no network.
final class CoreMLDiffusionEngineTests: XCTestCase {
    private var directories: [URL] = []

    override func tearDown() {
        directories.forEach { try? FileManager.default.removeItem(at: $0) }
        super.tearDown()
    }

    private func resources() throws -> URL {
        let directory = try DiffusionFixtures.downloadedResources()
        directories.append(directory)
        return directory
    }

    private func outputDirectory() throws -> URL {
        let directory = try DiffusionFixtures.temporaryDirectory("CoreMLDiffusionOutput")
        directories.append(directory)
        return directory
    }

    // MARK: Events and output

    func testEventsArePreparingThenEachStepThenDecoding() async throws {
        let runner = FakeRunner()
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)
        let log = EventLog()

        _ = try await engine.generate(DiffusionRequest(prompt: "a fox", steps: 3),
                                      outputDirectory: try outputDirectory()) { log.append($0) }

        XCTAssertEqual(log.all, [.preparingModel, .step(1, of: 3), .step(2, of: 3), .step(3, of: 3), .decoding])
    }

    func testLoadedModelIsReusedUntilUnloaded() async throws {
        let runner = FakeRunner()
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)
        let output = try outputDirectory()
        let second = EventLog()

        _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1), outputDirectory: output)
        _ = try await engine.generate(DiffusionRequest(prompt: "b", steps: 1), outputDirectory: output) {
            second.append($0)
        }
        XCTAssertEqual(runner.loads, 1)
        XCTAssertFalse(second.all.contains(.preparingModel))

        await engine.unload()
        let third = EventLog()
        _ = try await engine.generate(DiffusionRequest(prompt: "c", steps: 1), outputDirectory: output) {
            third.append($0)
        }
        XCTAssertEqual(runner.unloads, 1)
        XCTAssertEqual(runner.loads, 2)
        XCTAssertEqual(third.all.first, .preparingModel)
    }

    func testMissingSeedIsGeneratedAndReturned() async throws {
        let runner = FakeRunner()
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)

        let output = try await engine.generate(DiffusionRequest(prompt: "a", steps: 2),
                                               outputDirectory: try outputDirectory())

        XCTAssertEqual(runner.calls.map(\.seed), [output.seed])
        XCTAssertEqual(output.steps, 2)
    }

    func testRequestIsPassedThroughUnchanged() async throws {
        let runner = FakeRunner()
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)
        let request = DiffusionRequest(prompt: "  a lighthouse  ", steps: 50, seed: 1234, imageCount: 2)

        let output = try await engine.generate(request, outputDirectory: try outputDirectory())

        XCTAssertEqual(output.seed, 1234)
        let expected = FakeRunner.Call(prompt: "  a lighthouse  ", steps: 50, seed: 1234, imageCount: 2)
        XCTAssertEqual(runner.calls, [expected])
    }

    func testEachImageBecomesADecodablePNG() async throws {
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: FakeRunner())
        let directory = try outputDirectory()

        let output = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1, imageCount: 3),
                                               outputDirectory: directory)

        XCTAssertEqual(output.images.map(\.fileURL.lastPathComponent), ["image-0.png", "image-1.png", "image-2.png"])
        for result in output.images {
            XCTAssertEqual(result.fileURL.deletingLastPathComponent().standardizedFileURL,
                           directory.standardizedFileURL)
            let source = try XCTUnwrap(CGImageSourceCreateWithURL(result.fileURL as CFURL, nil))
            XCTAssertEqual(CGImageSourceGetType(source) as String?, "public.png")
            XCTAssertNotNil(CGImageSourceCreateImageAtIndex(source, 0, nil))
        }
    }

    func testDefaultOutputIsAFreshTemporaryFolder() async throws {
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: FakeRunner())

        let first = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1))
        let second = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1))
        let folders = [first, second].compactMap { $0.images.first?.fileURL.deletingLastPathComponent() }
        directories += folders

        XCTAssertEqual(Set(folders).count, 2)
        XCTAssertTrue(folders.allSatisfy { $0.lastPathComponent.hasPrefix("auraimagegen-") })
    }

    func testSafetyFilteredImagesAreSkipped() async throws {
        let runner = FakeRunner()
        runner.outputs = [true, false, true]
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)

        let output = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1, imageCount: 3),
                                               outputDirectory: try outputDirectory())

        XCTAssertEqual(output.images.map(\.fileURL.lastPathComponent), ["image-0.png", "image-2.png"])
    }

    func testEveryImageFilteredThrows() async throws {
        let runner = FakeRunner()
        runner.outputs = [false, false]
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)

        await assertThrows(OnDeviceImageGenError.filteredBySafetyChecker) {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1, imageCount: 2),
                                          outputDirectory: try self.outputDirectory())
        }
    }

    // MARK: Cancellation and concurrency

    func testCancellingStopsAtTheNextStep() async throws {
        let runner = FakeRunner()
        runner.stepDelay = .milliseconds(20)
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)
        let directory = try outputDirectory()
        let log = EventLog()
        let task = Task {
            try await engine.generate(DiffusionRequest(prompt: "a", steps: 50), outputDirectory: directory) {
                log.append($0)
            }
        }
        try await waitUntil { log.all.contains(.step(1, of: 50)) }

        task.cancel()

        do {
            _ = try await task.value
            XCTFail("expected CancellationError")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        let stopped = try XCTUnwrap(runner.stoppedAtStep)
        XCTAssertLessThan(stopped, 50)
        XCTAssertFalse(log.all.contains(.decoding))
    }

    func testCancelledLowMemoryRunUnloadsTheStagesItLeftLoaded() async throws {
        let runner = FakeRunner()
        runner.stepDelay = .milliseconds(20)
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner, memoryMode: .lowMemory)
        let directory = try outputDirectory()
        let log = EventLog()
        let task = Task {
            try await engine.generate(DiffusionRequest(prompt: "a", steps: 50), outputDirectory: directory) {
                log.append($0)
            }
        }
        try await waitUntil { log.all.contains(.step(1, of: 50)) }

        task.cancel()

        let result = await task.result
        XCTAssertThrowsError(try result.get()) { XCTAssertTrue($0 is CancellationError, "\($0)") }
        XCTAssertEqual(runner.unloads, 1)
    }

    func testCancelledResidentRunKeepsTheModelLoaded() async throws {
        let runner = FakeRunner()
        runner.stepDelay = .milliseconds(20)
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner, memoryMode: .resident)
        let directory = try outputDirectory()
        let log = EventLog()
        let task = Task {
            try await engine.generate(DiffusionRequest(prompt: "a", steps: 50), outputDirectory: directory) {
                log.append($0)
            }
        }
        try await waitUntil { log.all.contains(.step(1, of: 50)) }

        task.cancel()

        let result = await task.result
        XCTAssertThrowsError(try result.get())
        XCTAssertEqual(runner.unloads, 0)
        _ = try await engine.generate(DiffusionRequest(prompt: "b", steps: 1), outputDirectory: directory)
        XCTAssertEqual(runner.loads, 1)
    }

    func testFailedLowMemoryRunUnloadsTheStagesItLeftLoaded() async throws {
        let runner = FakeRunner()
        runner.generateError = CocoaError(.featureUnsupported)
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner, memoryMode: .lowMemory)

        do {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1),
                                          outputDirectory: try outputDirectory())
            XCTFail("expected generationFailed")
        } catch OnDeviceImageGenError.generationFailed {
            XCTAssertEqual(runner.unloads, 1)
        }
    }

    func testCancelledBeforeLoadingNeverLoads() async throws {
        let runner = FakeRunner()
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)
        let directory = try outputDirectory()
        let task = Task {
            try? await Task.sleep(for: .seconds(5))
            return try await engine.generate(DiffusionRequest(prompt: "a", steps: 1), outputDirectory: directory)
        }
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("expected CancellationError")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        XCTAssertEqual(runner.loads, 0)
    }

    func testASecondConcurrentCallIsBusy() async throws {
        let runner = FakeRunner()
        let gate = DispatchSemaphore(value: 0)
        runner.gate = gate
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)
        let directory = try outputDirectory()
        let first = Task {
            try await engine.generate(DiffusionRequest(prompt: "a", steps: 1), outputDirectory: directory)
        }
        try await waitUntil { runner.entered }

        await assertThrows(OnDeviceImageGenError.busy) {
            _ = try await engine.generate(DiffusionRequest(prompt: "b", steps: 1), outputDirectory: directory)
        }
        gate.signal()

        let output = try await first.value
        XCTAssertEqual(output.images.count, 1)
        XCTAssertEqual(runner.calls.count, 1)
    }

    // MARK: Invalid output

    func testNonFiniteLatentsStopTheRunAsAFailure() async throws {
        let runner = FakeRunner()
        runner.nonFiniteFromStep = 1
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner, memoryMode: .lowMemory)
        let directory = try outputDirectory()
        let log = EventLog()

        do {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 20), outputDirectory: directory) {
                log.append($0)
            }
            XCTFail("expected generationFailed")
        } catch let OnDeviceImageGenError.generationFailed(detail) {
            XCTAssertTrue(detail.contains("NaN") && detail.contains("step 1 "), detail)
        }
        XCTAssertEqual(runner.stoppedAtStep, 1)
        XCTAssertEqual(log.all, [.preparingModel])
        XCTAssertEqual(runner.unloads, 1)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [])
    }

    #if canImport(StableDiffusion)
    func testLatentCheckRejectsNaNAndInfinity() {
        let finite = MLShapedArray<Float32>(scalars: [0, -4.7, 7.7, 1], shape: [1, 4])
        XCTAssertTrue(StableDiffusionRunner.areFinite([finite, finite]))
        XCTAssertFalse(StableDiffusionRunner.areFinite([finite, MLShapedArray(scalars: [0, .nan], shape: [1, 2])]))
        XCTAssertFalse(StableDiffusionRunner.areFinite([MLShapedArray(scalars: [.infinity, 0], shape: [1, 2])]))
    }
    #endif

    // MARK: Gates

    func testSimulatorRefusesBeforeLoading() async throws {
        let runner = FakeRunner()
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner, runsInSimulator: true)

        XCTAssertEqual(engine.unavailability, .simulatorUnsupported)
        XCTAssertFalse(engine.isReady)
        await assertThrows(OnDeviceImageGenError.simulatorUnsupported) {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1),
                                          outputDirectory: try self.outputDirectory())
        }
        XCTAssertEqual(runner.loads, 0)
    }

    func testDeviceAndMacEnginesAreAvailable() throws {
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: FakeRunner())

        XCTAssertNil(engine.unavailability)
        XCTAssertTrue(engine.isReady)
    }

    func testValidationRunsBeforeAnyOtherCheck() async throws {
        let engine = DiffusionFixtures.engine(resources: URL(fileURLWithPath: "/nonexistent"), runner: nil)

        await assertThrows(OnDeviceImageGenError.invalidRequest(field: "steps", reason: "must be 1–50, got 0")) {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 0))
        }
    }

    func testNoPipelineMeansUnsupportedPlatform() async throws {
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: nil)

        XCTAssertFalse(engine.isReady)
        XCTAssertEqual(engine.unavailability, .unsupportedPlatform)
        await assertThrows(OnDeviceImageGenError.unsupportedPlatform) {
            _ = try await engine.generate(DiffusionRequest(prompt: "a"))
        }
    }

    func testMissingModelThrowsNotDownloaded() async throws {
        let empty = try outputDirectory()
        let runner = FakeRunner()
        let engine = DiffusionFixtures.engine(resources: empty, runner: runner)

        XCTAssertFalse(engine.isReady)
        await assertThrows(OnDeviceImageGenError.modelNotDownloaded(sizeBytes: 1_073_000_000)) {
            _ = try await engine.generate(DiffusionRequest(prompt: "a"))
        }
        XCTAssertEqual(runner.loads, 0)
    }

    func testInsufficientMemoryThrowsBeforeLoading() async throws {
        let runner = FakeRunner()
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner,
                                              availableMemory: { 1_000_000 })
        let required = Int64(CoreMLDiffusionEngine.requiredMemoryBytes(for: .stableDiffusion21Base))

        XCTAssertTrue(engine.isReady)
        let expected = OnDeviceImageGenError.insufficientMemory(requiredBytes: required, availableBytes: 1_000_000)
        await assertThrows(expected) {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1),
                                          outputDirectory: try self.outputDirectory())
        }
        XCTAssertEqual(runner.loads, 0)
    }

    func testUnknownMemoryOnADeviceThrowsBeforeLoading() async throws {
        let runner = FakeRunner()
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner, availableMemory: { nil },
                                              unknownMemoryMeansExhausted: true)
        let required = Int64(CoreMLDiffusionEngine.requiredMemoryBytes(for: .stableDiffusion21Base))

        await assertThrows(.insufficientMemory(requiredBytes: required, availableBytes: 0)) {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1),
                                          outputDirectory: try self.outputDirectory())
        }
        XCTAssertEqual(runner.loads, 0)
    }

    func testMemoryCheckAtTheBoundaryAndWhenUnknown() {
        func passes(_ available: Int?, unknownMeansExhausted: Bool) -> Bool {
            (try? CoreMLDiffusionEngine.checkMemory(availableBytes: available, requiredBytes: 1_000,
                                                    unknownMeansExhausted: unknownMeansExhausted)) != nil
        }
        XCTAssertTrue(passes(1_000, unknownMeansExhausted: true))
        XCTAssertFalse(passes(999, unknownMeansExhausted: false))
        XCTAssertFalse(passes(nil, unknownMeansExhausted: true))
        XCTAssertTrue(passes(nil, unknownMeansExhausted: false))
    }

    func testOnlyMacTreatsUnknownMemoryAsRoom() {
        #if os(macOS)
        XCTAssertFalse(CoreMLDiffusionEngine.unknownMemoryIsJetsamLimit)
        #else
        XCTAssertTrue(CoreMLDiffusionEngine.unknownMemoryIsJetsamLimit)
        #endif
        XCTAssertFalse(CoreMLDiffusionEngine.isSimulatorBuild)
    }

    func testRequiredMemoryFollowsTheDiffusionHeuristic() {
        let required = CoreMLDiffusionEngine.requiredMemoryBytes(for: .stableDiffusion21Base)
        XCTAssertEqual(Double(required), 1_073_000_000 * 2.2, accuracy: 1_000)
    }

    func testMemoryModeResolutionAtTheBoundary() {
        let required = 1_000
        func resolve(_ mode: DiffusionMemoryMode, _ available: Int?) -> DiffusionMemoryMode {
            CoreMLDiffusionEngine.resolvedMemoryMode(mode, availableBytes: available, requiredBytes: required)
        }
        XCTAssertEqual(resolve(.automatic, 2_000), .resident)
        XCTAssertEqual(resolve(.automatic, 1_999), .lowMemory)
        XCTAssertEqual(resolve(.automatic, nil), .lowMemory)
        XCTAssertEqual(resolve(.resident, 0), .resident)
        XCTAssertEqual(resolve(.lowMemory, 1_000_000), .lowMemory)
    }

    func testResolvedModeReachesThePipeline() async throws {
        let log = FactoryLog()
        let required = CoreMLDiffusionEngine.requiredMemoryBytes(for: .stableDiffusion21Base)
        let roomy = DiffusionFixtures.engine(resources: try resources(), runner: FakeRunner(),
                                             availableMemory: { 2 * required }, factoryLog: log)
        let tight = DiffusionFixtures.engine(resources: try resources(), runner: FakeRunner(),
                                             availableMemory: { 2 * required - 1 }, factoryLog: log)
        let directory = try outputDirectory()

        _ = try await roomy.generate(DiffusionRequest(prompt: "a", steps: 1), outputDirectory: directory)
        _ = try await tight.generate(DiffusionRequest(prompt: "a", steps: 1), outputDirectory: directory)

        XCTAssertEqual(log.flags, [false, true], "reduceMemory: resident, then low memory")
    }

    func testLoadFailureIsReportedAndRetriedNextCall() async throws {
        let runner = FakeRunner()
        runner.loadError = CocoaError(.fileReadCorruptFile)
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)
        let directory = try outputDirectory()

        do {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1), outputDirectory: directory)
            XCTFail("expected modelLoadFailed")
        } catch let OnDeviceImageGenError.modelLoadFailed(detail) {
            XCTAssertFalse(detail.isEmpty)
        }
        runner.loadError = nil
        _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1), outputDirectory: directory)
        XCTAssertEqual(runner.loads, 2)
    }

    func testPipelineErrorBecomesGenerationFailed() async throws {
        let runner = FakeRunner()
        runner.generateError = CocoaError(.featureUnsupported)
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner)

        do {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 1),
                                          outputDirectory: try outputDirectory())
            XCTFail("expected generationFailed")
        } catch let OnDeviceImageGenError.generationFailed(detail) {
            XCTAssertFalse(detail.isEmpty)
        }
    }

    // MARK: Helpers

    private func waitUntil(timeout: Duration = .seconds(5), _ condition: @Sendable () -> Bool) async throws {
        let deadline = ContinuousClock.now + timeout
        while !condition(), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        if !condition() { throw ConditionTimeout() }
    }

    private func assertThrows(_ expected: OnDeviceImageGenError, file: StaticString = #filePath, line: UInt = #line,
                              _ body: () async throws -> Void) async {
        do {
            try await body()
            XCTFail("expected \(expected)", file: file, line: line)
        } catch {
            XCTAssertEqual(error as? OnDeviceImageGenError, expected, file: file, line: line)
        }
    }
}

private struct ConditionTimeout: Error {}
