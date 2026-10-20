import Foundation

/// Thrown when Tonamox itself stops a generation before the model finished: the app moved to the background,
/// or a ``DeviceConditions`` guard reached ``DeviceConditions/Level/block``.
///
/// ``partialText`` holds what was produced and is always truncated. A streaming call has already yielded that
/// text before it finishes with this error.
public struct GenerationInterrupted: Error, LocalizedError, Sendable, Equatable {

    public enum Trigger: Sendable, Equatable {
        /// The app moved to the background (iOS), or was already there when the generation was asked for.
        case appBackgrounded
        /// The ``DeviceConditions`` guard assessed the device at ``DeviceConditions/Level/block``.
        case deviceConditions(DeviceConditions.Assessment)
    }

    public let reason: Trigger
    /// What the model produced before the stop. Empty when nothing was produced, and always empty for images.
    public let partialText: String

    public init(reason: Trigger, partialText: String) {
        self.reason = reason
        self.partialText = partialText
    }

    public var errorDescription: String? {
        switch reason {
        case .appBackgrounded:
            "Generation stopped because the app moved to the background."
        case .deviceConditions(let assessment):
            "Generation stopped: \(assessment.summary)."
        }
    }
}
