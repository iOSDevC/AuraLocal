import Synchronization
import XCTest
@testable import AuraCore

/// A reading the test changes while a generation runs.
final class ReadingSource: Sendable {
    private let current: Mutex<DeviceConditions.Reading>

    init(_ reading: DeviceConditions.Reading = .init()) {
        current = Mutex(reading)
    }

    func set(_ reading: DeviceConditions.Reading) {
        current.withLock { $0 = reading }
    }

    func read() -> DeviceConditions.Reading {
        current.withLock { $0 }
    }
}

/// Collects `onWarning` calls.
final class WarningLog: Sendable {
    private let entries = Mutex<[DeviceConditions.Assessment]>([])

    func append(_ assessment: DeviceConditions.Assessment) {
        entries.withLock { $0.append(assessment) }
    }

    var all: [DeviceConditions.Assessment] { entries.withLock { $0 } }
}

@MainActor
final class DeviceConditionsTests: XCTestCase {

    private let defaults = DeviceConditions()

    // MARK: - Thresholds

    func testDefaultThermalThresholdsWarnAtSeriousAndBlockAtCritical() {
        XCTAssertEqual(defaults.assess(.init(thermalPressure: .nominal)).severity, .normal)
        XCTAssertEqual(defaults.assess(.init(thermalPressure: .fair)).severity, .normal)
        XCTAssertEqual(defaults.assess(.init(thermalPressure: .serious)).severity, .warn)
        let critical = defaults.assess(.init(thermalPressure: .critical))
        XCTAssertEqual(critical.severity, .block)
        XCTAssertEqual(critical.causes, [.thermal(.critical)])
    }

    func testNilThermalThresholdsIgnoreHeat() {
        let policy = DeviceConditions(warnAtThermalState: nil, blockAtThermalState: nil)
        XCTAssertEqual(policy.assess(.init(thermalPressure: .critical)).severity, .normal)
    }

    func testLowPowerModeWarnsByDefaultAndCanBlockOrBeIgnored() {
        let lowPower = DeviceConditions.Reading(isLowPowerModeEnabled: true)
        XCTAssertEqual(defaults.assess(lowPower).severity, .warn)
        XCTAssertEqual(DeviceConditions(lowPowerMode: .block).assess(lowPower).severity, .block)
        XCTAssertEqual(DeviceConditions(lowPowerMode: .normal).assess(lowPower).causes, [])
    }

    func testBatteryIsIgnoredUnlessThresholdsAreSet() {
        let flat = DeviceConditions.Reading(batteryLevel: 0.01, isCharging: false)
        XCTAssertEqual(defaults.assess(flat).severity, .normal)
    }

    func testBatteryThresholdsApplyOnlyWhileUnplugged() {
        let policy = DeviceConditions(warnBelowBatteryLevel: 0.2, blockBelowBatteryLevel: 0.05)
        XCTAssertEqual(policy.assess(.init(batteryLevel: 0.5, isCharging: false)).severity, .normal)
        XCTAssertEqual(policy.assess(.init(batteryLevel: 0.15, isCharging: false)).severity, .warn)
        XCTAssertEqual(policy.assess(.init(batteryLevel: 0.04, isCharging: false)).causes, [.lowBattery(0.04)])
        XCTAssertEqual(policy.assess(.init(batteryLevel: 0.04, isCharging: false)).severity, .block)
        XCTAssertEqual(policy.assess(.init(batteryLevel: 0.04, isCharging: true)).severity, .normal)
        XCTAssertEqual(policy.assess(.init(batteryLevel: 0.04, isCharging: nil)).severity, .normal,
                       "an unknown charging state must not block")
    }

    func testTheHighestCauseSetsTheLevelAndEveryCauseIsListed() {
        let policy = DeviceConditions(warnBelowBatteryLevel: 0.2)
        let reading = DeviceConditions.Reading(thermalPressure: .critical, isLowPowerModeEnabled: true,
                                               batteryLevel: 0.1, isCharging: false)
        let assessment = policy.assess(reading)
        XCTAssertEqual(assessment.severity, .block)
        XCTAssertEqual(assessment.causes, [.thermal(.critical), .lowPowerMode, .lowBattery(0.1)])
        XCTAssertEqual(assessment.snapshot, reading)
    }

    // MARK: - Check cadence

    func testChecksEveryIntervalAndWarnsOncePerTransition() {
        let warnings = WarningLog()
        let source = ReadingSource(.init(thermalPressure: .serious))
        var check = DeviceConditionsCheck(policy: DeviceConditions(checkInterval: 3,
                                                                   onWarning: { warnings.append($0) }))

        let firstSix = (1...6).map { _ in check.tick { source.read() } }
        XCTAssertEqual(firstSix.compactMap { $0 }, [])
        XCTAssertEqual(warnings.all.count, 1, "ticks 3 and 6 are both warn; only the first transition reports")

        source.set(.init(thermalPressure: .nominal))
        _ = (1...3).map { _ in check.tick { source.read() } }
        source.set(.init(thermalPressure: .serious))
        _ = (1...3).map { _ in check.tick { source.read() } }
        XCTAssertEqual(warnings.all.count, 2)

        source.set(.init(thermalPressure: .critical))
        let blocked = (1...3).map { _ in check.tick { source.read() } }.compactMap { $0 }
        XCTAssertEqual(blocked.map(\.severity), [.block])
    }

    func testIntervalBelowOneChecksEveryTick() {
        var check = DeviceConditionsCheck(policy: DeviceConditions(checkInterval: 0))
        XCTAssertNotNil(check.tick { .init(thermalPressure: .critical) })
    }

    // MARK: - Generation

    private func makeEngine(_ backend: SlowFakeBackend, source: ReadingSource, interval: Int = 2,
                        warnings: WarningLog = WarningLog()) -> AuraEngine {
        let lifecycle = BackgroundLifecycle(notificationCenter: NotificationCenter(), didEnterBackground: nil,
                                            willEnterForeground: nil)
        let engine = AuraEngine(model: .dolphin3_qwen25_1_5b_gguf, backend: backend, lifecycle: lifecycle)
        var policy = DeviceConditions(checkInterval: interval, onWarning: { warnings.append($0) })
        policy.readingOverride = { source.read() }
        engine.deviceGuard = policy
        return engine
    }

    func testCriticalHeatMidGenerationStopsWithTheTextSoFar() async throws {
        let backend = SlowFakeBackend()
        let source = ReadingSource()
        let engine = makeEngine(backend, source: source)
        let pending = Task { @MainActor in
            try await engine.generate(prompt: "x", maxTokens: 100_000) { _ in }
        }
        try await waitUntil { backend.producedTokens >= 5 }

        source.set(.init(thermalPressure: .critical))

        let result = await pending.result
        XCTAssertThrowsError(try result.get()) { error in
            guard let stop = error as? GenerationInterrupted,
                  case .deviceConditions(let assessment) = stop.reason else { return XCTFail("\(error)") }
            XCTAssertEqual(assessment.severity, .block)
            XCTAssertTrue(stop.partialText.hasPrefix("t0 t1 t2 t3 t4 "), stop.partialText)
        }
        XCTAssertFalse(backend.isGenerating)
        XCTAssertLessThan(backend.producedTokens, 100_000)
    }

    func testABlockedDeviceRefusesToStart() async {
        let backend = SlowFakeBackend()
        let engine = makeEngine(backend, source: ReadingSource(.init(thermalPressure: .critical)))

        do {
            _ = try await engine.generate(prompt: "x", maxTokens: 4) { _ in }
            XCTFail("expected GenerationInterrupted")
        } catch let stop as GenerationInterrupted {
            XCTAssertEqual(stop.partialText, "")
        } catch {
            XCTFail("\(error)")
        }
        XCTAssertEqual(backend.generations, 0)
    }

    func testWarnLetsTheGenerationFinishAndReportsIt() async throws {
        let warnings = WarningLog()
        let engine = makeEngine(SlowFakeBackend(), source: ReadingSource(.init(isLowPowerModeEnabled: true)),
                            warnings: warnings)

        let reply = try await engine.generate(prompt: "x", maxTokens: 6) { _ in }

        XCTAssertEqual(reply, "t0 t1 t2 t3 t4 t5 ")
        XCTAssertEqual(warnings.all.map(\.causes), [[.lowPowerMode]])
    }

    func testNoGuardMeansNoChecks() async throws {
        let backend = SlowFakeBackend()
        let engine = makeEngine(backend, source: ReadingSource(.init(thermalPressure: .critical)))
        engine.deviceGuard = nil

        let reply = try await engine.generate(prompt: "x", maxTokens: 3) { _ in }

        XCTAssertEqual(reply, "t0 t1 t2 ")
    }
}
