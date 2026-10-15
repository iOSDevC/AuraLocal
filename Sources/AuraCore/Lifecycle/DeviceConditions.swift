import Foundation
#if os(iOS)
import UIKit
#endif

// MARK: - DeviceConditions

/// Opt-in guard that stops long generations on a hot or nearly flat device.
///
/// Thermal state and Low Power Mode are always readable. The battery is read only when the app itself has turned
/// on `UIDevice.isBatteryMonitoringEnabled`: that switch is process-wide, so AuraLocal never flips it. Without it,
/// or on a Mac, the battery thresholds are ignored.
///
/// The device is assessed before a generation starts and then every ``checkInterval`` streamed tokens (text) or
/// denoising steps (images). At ``Level/warn`` the generation continues and ``onWarning`` is called; at
/// ``Level/block`` it is cancelled and the call throws ``GenerationInterrupted`` with the text produced so far.
///
/// ```swift
/// llm.deviceGuard = DeviceConditions(onWarning: { print($0.causes) })
/// ```
public struct DeviceConditions: Sendable {

    public enum Level: Int, Sendable, Comparable, CaseIterable {
        case normal
        case warn
        case block

        public static func < (lhs: Level, rhs: Level) -> Bool { lhs.rawValue < rhs.rawValue }
    }

    /// One condition that raised the level.
    public enum Cause: Sendable, Equatable {
        case thermal(ProcessInfo.ThermalState)
        case lowPowerMode
        /// Battery level, 0...1, while unplugged.
        case lowBattery(Float)
    }

    /// The device at one moment.
    public struct Reading: Sendable, Equatable {
        /// `ProcessInfo.thermalState`.
        public var thermalPressure: ProcessInfo.ThermalState
        public var isLowPowerModeEnabled: Bool
        /// 0...1, or `nil` when battery monitoring is off or the level is unknown.
        public var batteryLevel: Float?
        /// `nil` when battery monitoring is off or the state is unknown.
        public var isCharging: Bool?

        public init(thermalPressure: ProcessInfo.ThermalState = .nominal, isLowPowerModeEnabled: Bool = false,
                    batteryLevel: Float? = nil, isCharging: Bool? = nil) {
            self.thermalPressure = thermalPressure
            self.isLowPowerModeEnabled = isLowPowerModeEnabled
            self.batteryLevel = batteryLevel
            self.isCharging = isCharging
        }

        /// Thermal state and Low Power Mode, from any thread. The battery fields are `nil`.
        public static func system() -> Reading {
            let info = ProcessInfo.processInfo
            return Reading(thermalPressure: info.thermalState, isLowPowerModeEnabled: info.isLowPowerModeEnabled)
        }

        /// ``system()`` plus the battery, when the app has enabled battery monitoring (iOS only).
        @MainActor public static func current() -> Reading {
            var reading = system()
            #if os(iOS)
            let device = UIDevice.current
            if device.isBatteryMonitoringEnabled {
                reading.batteryLevel = device.batteryLevel >= 0 ? device.batteryLevel : nil
                switch device.batteryState {
                case .charging, .full: reading.isCharging = true
                case .unplugged: reading.isCharging = false
                case .unknown: reading.isCharging = nil
                @unknown default: reading.isCharging = nil
                }
            }
            #endif
            return reading
        }
    }

    public struct Assessment: Sendable, Equatable {
        public let severity: Level
        /// Every condition at warn or block, in the order thermal, Low Power Mode, battery.
        public let causes: [Cause]
        public let snapshot: Reading

        var summary: String {
            causes.map(\.summary).joined(separator: ", ")
        }
    }

    /// Thermal state that warns; `nil` never warns on heat.
    public var warnAtThermalState: ProcessInfo.ThermalState?
    /// Thermal state that blocks; `nil` never blocks on heat.
    public var blockAtThermalState: ProcessInfo.ThermalState?
    /// What Low Power Mode raises the level to.
    public var lowPowerMode: Level
    /// Battery level (0...1) below which an unplugged device warns; `nil` ignores the battery.
    public var warnBelowBatteryLevel: Float?
    /// Battery level (0...1) below which an unplugged device blocks; `nil` never blocks on the battery.
    public var blockBelowBatteryLevel: Float?
    /// Tokens or denoising steps between checks. Values below 1 count as 1.
    public var checkInterval: Int
    /// Called when a check moves the level to ``Level/warn``: on the main actor for text and for the check before
    /// an image run, then on the image engine's queue for the checks between denoising steps.
    public var onWarning: (@Sendable (Assessment) -> Void)?

    /// Replaces the real readings (tests).
    package var readingOverride: (@Sendable () -> Reading)?

    public init(warnAtThermalState: ProcessInfo.ThermalState? = .serious,
                blockAtThermalState: ProcessInfo.ThermalState? = .critical,
                lowPowerMode: Level = .warn,
                warnBelowBatteryLevel: Float? = nil,
                blockBelowBatteryLevel: Float? = nil,
                checkInterval: Int = 8,
                onWarning: (@Sendable (Assessment) -> Void)? = nil) {
        self.warnAtThermalState = warnAtThermalState
        self.blockAtThermalState = blockAtThermalState
        self.lowPowerMode = lowPowerMode
        self.warnBelowBatteryLevel = warnBelowBatteryLevel
        self.blockBelowBatteryLevel = blockBelowBatteryLevel
        self.checkInterval = checkInterval
        self.onWarning = onWarning
    }

    /// The level `reading` reaches under these thresholds.
    public func assess(_ reading: Reading) -> Assessment {
        var raised: [(level: Level, cause: Cause)] = []
        let thermal = reading.thermalPressure.rawValue
        if let block = blockAtThermalState, thermal >= block.rawValue {
            raised.append((.block, .thermal(reading.thermalPressure)))
        } else if let warn = warnAtThermalState, thermal >= warn.rawValue {
            raised.append((.warn, .thermal(reading.thermalPressure)))
        }
        if reading.isLowPowerModeEnabled, lowPowerMode > .normal {
            raised.append((lowPowerMode, .lowPowerMode))
        }
        if let battery = reading.batteryLevel, reading.isCharging == false {
            if let block = blockBelowBatteryLevel, battery < block {
                raised.append((.block, .lowBattery(battery)))
            } else if let warn = warnBelowBatteryLevel, battery < warn {
                raised.append((.warn, .lowBattery(battery)))
            }
        }
        let level = raised.map(\.level).max() ?? .normal
        return Assessment(severity: level, causes: raised.map(\.cause), snapshot: reading)
    }

    // MARK: Readings

    /// Full reading, battery included. Text generation checks run on the main actor.
    @MainActor package func currentReading() -> Reading {
        readingOverride?() ?? .current()
    }

    /// Live thermal state and Low Power Mode with the battery of `start`: off the main actor the battery cannot be
    /// read, so image generation keeps the value read when the run began.
    package func reading(batteryFrom start: Reading) -> Reading {
        if let readingOverride { return readingOverride() }
        var reading = Reading.system()
        reading.batteryLevel = start.batteryLevel
        reading.isCharging = start.isCharging
        return reading
    }
}

// MARK: - Check

/// Per-run state of a ``DeviceConditions`` guard: counts tokens or steps, reports warn transitions, and returns
/// the assessment that blocks.
package struct DeviceConditionsCheck: Sendable {
    package let policy: DeviceConditions
    private var ticks = 0
    private var lastLevel = DeviceConditions.Level.normal

    package init(policy: DeviceConditions) {
        self.policy = policy
    }

    /// Assesses `reading` now; the assessment when it blocks, else `nil`.
    package mutating func evaluate(_ reading: DeviceConditions.Reading) -> DeviceConditions.Assessment? {
        let assessment = policy.assess(reading)
        if assessment.severity == .warn, lastLevel != .warn {
            policy.onWarning?(assessment)
        }
        lastLevel = assessment.severity
        return assessment.severity == .block ? assessment : nil
    }

    /// Counts one token or step and assesses every ``DeviceConditions/checkInterval`` of them.
    package mutating func tick(_ reading: () -> DeviceConditions.Reading) -> DeviceConditions.Assessment? {
        ticks += 1
        guard ticks % max(1, policy.checkInterval) == 0 else { return nil }
        return evaluate(reading())
    }
}

// MARK: - Descriptions

extension DeviceConditions.Cause {
    var summary: String {
        switch self {
        case .thermal(let state): "thermal state \(state.label)"
        case .lowPowerMode: "Low Power Mode on"
        case .lowBattery(let level): "battery at \(Int((level * 100).rounded()))%"
        }
    }
}

extension ProcessInfo.ThermalState {
    var label: String {
        switch self {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}
