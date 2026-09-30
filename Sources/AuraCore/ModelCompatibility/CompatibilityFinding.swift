import Foundation

/// How much a finding matters. Sorted blockers first.
public enum FindingSeverity: Int, Sendable, Comparable, CaseIterable {
    /// The model will not load or run in AuraLocal.
    case blocker
    /// It runs, but with a limitation the user should know about.
    case caveat
    /// Context only.
    case info

    public static func < (lhs: FindingSeverity, rhs: FindingSeverity) -> Bool { lhs.rawValue < rhs.rawValue }

    public var label: String {
        switch self {
        case .blocker: "Blocker"
        case .caveat: "Caveat"
        case .info: "Info"
        }
    }

    public var systemImage: String {
        switch self {
        case .blocker: "xmark.octagon.fill"
        case .caveat: "exclamationmark.triangle.fill"
        case .info: "info.circle"
        }
    }
}

/// One typed, human-readable result of a compatibility rule.
public struct CompatibilityFinding: Sendable, Equatable, Identifiable {
    /// The rule that produced it (e.g. `mlx.weight-prefixes`).
    public let rule: String
    public let level: FindingSeverity
    public let title: String
    /// The evidence: which file, key or pinned source line decides it.
    public let detail: String

    public init(rule: String, level: FindingSeverity, title: String, detail: String) {
        self.rule = rule
        self.level = level
        self.title = title
        self.detail = detail
    }

    public var id: String { "\(rule)|\(title)" }

    static func blocker(_ rule: String, _ title: String, _ detail: String) -> CompatibilityFinding {
        CompatibilityFinding(rule: rule, level: .blocker, title: title, detail: detail)
    }

    static func caveat(_ rule: String, _ title: String, _ detail: String) -> CompatibilityFinding {
        CompatibilityFinding(rule: rule, level: .caveat, title: title, detail: detail)
    }

    static func info(_ rule: String, _ title: String, _ detail: String) -> CompatibilityFinding {
        CompatibilityFinding(rule: rule, level: .info, title: title, detail: detail)
    }

    // Without a rule id: ``CompatibilityRule/findings(for:)`` stamps it.
    static func blocker(_ title: String, _ detail: String) -> CompatibilityFinding { blocker("", title, detail) }
    static func caveat(_ title: String, _ detail: String) -> CompatibilityFinding { caveat("", title, detail) }
    static func info(_ title: String, _ detail: String) -> CompatibilityFinding { info("", title, detail) }
}

/// The overall answer for one repository on one device.
public enum CompatibilityVerdict: String, Sendable, CaseIterable {
    case runnable
    case runnableWithCaveats
    case notRunnable
    /// The repository could not be read (network, 404, token).
    case unknown

    public var label: String {
        switch self {
        case .runnable: "Runs"
        case .runnableWithCaveats: "Runs, with caveats"
        case .notRunnable: "Won't run"
        case .unknown: "Unknown"
        }
    }

    public var systemImage: String {
        switch self {
        case .runnable: "checkmark.circle.fill"
        case .runnableWithCaveats: "exclamationmark.triangle.fill"
        case .notRunnable: "xmark.octagon.fill"
        case .unknown: "questionmark.circle"
        }
    }

    public var isRunnable: Bool { self == .runnable || self == .runnableWithCaveats }
}
