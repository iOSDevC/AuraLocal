import Synchronization
import XCTest
import TonamoxCore
@testable import TonamoxImageGen

/// Counts readings so a test can turn the device critical after a number of denoising steps.
private final class StepReadings: Sendable {
    private let calls = Mutex(0)
    private let criticalAfter: Int

    init(criticalAfter: Int) {
        self.criticalAfter = criticalAfter
    }

    func read() -> DeviceConditions.Reading {
        let call = calls.withLock { count -> Int in
            count += 1
            return count
        }
        return DeviceConditions.Reading(thermalPressure: call > criticalAfter ? .critical : .nominal)
    }
}

/// Backgrounding and the device guard stop a Core ML diffusion run between steps, with a fake pipeline.
@MainActor
final class DiffusionInterruptionTests: XCTestCase {
    private let background = Notification.Name("test.didEnterBackground")
    private let foreground = Notification.Name("test.willEnterForeground")
    private let notifications = NotificationCenter()

    private func resources() throws -> URL {
        removedAfterTest(try DiffusionFixtures.downloadedResources())
    }

    private func outputDirectory() throws -> URL {
        removedAfterTest(try DiffusionFixtures.temporaryDirectory("DiffusionInterruption"))
    }

    private func removedAfterTest(_ directory: URL) -> URL {
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    private func makeLifecycle() -> BackgroundLifecycle {
        BackgroundLifecycle(notificationCenter: notifications, didEnterBackground: background,
                            willEnterForeground: foreground)
    }

    func testBackgroundingStopsARunningGeneration() async throws {
        let runner = FakeRunner()
        runner.stepDelay = .milliseconds(20)
        let lifecycle = makeLifecycle()
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner, lifecycle: lifecycle)
        let directory = try outputDirectory()
        let log = EventLog()
        let pending = Task {
            try await engine.generate(DiffusionRequest(prompt: "a", steps: 50), outputDirectory: directory) {
                log.append($0)
            }
        }
        try await waitForStep(log)

        notifications.post(name: background, object: nil)

        let result = await pending.result
        XCTAssertThrowsError(try result.get()) { error in
            XCTAssertEqual(error as? GenerationInterrupted,
                           GenerationInterrupted(reason: .appBackgrounded, partialText: ""), "\(error)")
        }
        XCTAssertLessThan(try XCTUnwrap(runner.stoppedAtStep), 50)
        XCTAssertFalse(log.all.contains(.decoding))
    }

    func testGenerationIsRefusedWhileInTheBackground() async throws {
        let runner = FakeRunner()
        let lifecycle = makeLifecycle()
        lifecycle.startObserving()
        notifications.post(name: background, object: nil)
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner, lifecycle: lifecycle)

        do {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 2), outputDirectory: try outputDirectory())
            XCTFail("expected GenerationInterrupted")
        } catch let stop as GenerationInterrupted {
            XCTAssertEqual(stop.reason, .appBackgrounded)
        }
        XCTAssertEqual(runner.loads, 0)

        notifications.post(name: foreground, object: nil)
        let output = try await engine.generate(DiffusionRequest(prompt: "a", steps: 2),
                                               outputDirectory: try outputDirectory())
        XCTAssertEqual(output.images.count, 1)
    }

    func testCriticalHeatStopsAtTheNextCheckedStep() async throws {
        let runner = FakeRunner()
        let readings = StepReadings(criticalAfter: 3)
        var policy = DeviceConditions(checkInterval: 1)
        policy.readingOverride = { readings.read() }
        let engine = DiffusionFixtures.engine(resources: try resources(), runner: runner, deviceGuard: policy,
                                              lifecycle: makeLifecycle())

        do {
            _ = try await engine.generate(DiffusionRequest(prompt: "a", steps: 20),
                                          outputDirectory: try outputDirectory())
            XCTFail("expected GenerationInterrupted")
        } catch let stop as GenerationInterrupted {
            guard case .deviceConditions(let assessment) = stop.reason else { return XCTFail("\(stop)") }
            XCTAssertEqual(assessment.causes, [.thermal(.critical)])
        }
        // Reading 1 is the check before the run; readings 2 and 3 are steps 1 and 2; step 3 is critical.
        XCTAssertEqual(runner.stoppedAtStep, 3)
    }

    private func waitForStep(_ log: EventLog) async throws {
        let deadline = ContinuousClock.now + .seconds(5)
        while !log.all.contains(.step(1, of: 50)), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertTrue(log.all.contains(.step(1, of: 50)), "the run never reached step 1")
    }
}
