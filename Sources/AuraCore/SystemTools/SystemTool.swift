import Foundation

// MARK: - Availability

/// Whether a native system tool can run on THIS device right now. Resilience is
/// first-class: a tool that isn't available (OS too old, permission denied, no
/// hardware) reports *why* instead of failing at call time.
public enum SystemToolAvailability: Sendable, Equatable {
    case available
    case unavailable(reason: String)

    public var isAvailable: Bool {
        if case .available = self { return true }
        return false
    }

    public var reason: String? {
        if case .unavailable(let r) = self { return r }
        return nil
    }
}

// MARK: - SystemTool

/// A capability backed by an on-device Apple framework (Vision, NaturalLanguage,
/// PDFKit, Speech, Translation, …) that can **support a small local model**: the
/// SLM stays the reasoner while the OS does the perception/precision work.
///
/// Discovery + resilience live here; each concrete tool adds its own typed
/// execution API. Availability is `async` so tools that must ask for a permission
/// (camera, speech) fit the same shape as pure-compute ones.
public protocol SystemTool: Sendable {
    /// Stable identifier, e.g. `system.vision.ocr`.
    var id: String { get }
    var displayName: String { get }
    /// One line describing what the tool gives the model (used in UI / tool listings).
    var summary: String { get }

    /// Check availability on the current device. Must never throw — return
    /// `.unavailable(reason:)` instead.
    func availability() async -> SystemToolAvailability
}

// MARK: - Registry / discovery

/// Identifies which native tools are usable on the current device and surfaces
/// them so an SLM (or the app) can lean on the OS instead of a downloaded model.
public enum SystemToolRegistry {

    /// Every known native tool (some may be unavailable on this device/OS).
    public static let all: [any SystemTool] = [
        VisionOCRTool(),
        NLEmbeddingTool(),
    ]

    /// A discovered tool with its resolved availability.
    public struct ToolInfo: Sendable, Identifiable {
        public let id: String
        public let displayName: String
        public let summary: String
        public let availability: SystemToolAvailability
        public var isAvailable: Bool { availability.isAvailable }
    }

    /// Probe every tool's availability (resilient — never throws). Order matches ``all``.
    public static func discover() async -> [ToolInfo] {
        var infos: [ToolInfo] = []
        for tool in all {
            let availability = await tool.availability()
            infos.append(ToolInfo(
                id: tool.id, displayName: tool.displayName,
                summary: tool.summary, availability: availability))
        }
        return infos
    }

    /// Only the tools that are actually usable right now.
    public static func availableTools() async -> [ToolInfo] {
        await discover().filter(\.isAvailable)
    }

    /// Look up a concrete tool by id (nil if unknown).
    public static func tool(id: String) -> (any SystemTool)? {
        all.first { $0.id == id }
    }
}
