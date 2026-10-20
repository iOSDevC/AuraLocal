import Foundation

/// Errors thrown by the Tonamox inference pipeline.
public enum TonamoxError: LocalizedError {
    /// The model has not been loaded yet — call a factory method or
    /// ``ModelManager/load(_:tools:temperature:topP:onProgress:)`` first.
    case modelNotLoaded
    /// The provided image could not be converted to the format required by the vision model.
    case imageProcessingFailed
    /// The model returned an unexpected or invalid result.
    case invalidResponse(String)
    /// A remote provider returned a non-success HTTP status (code, body snippet).
    case remoteFailed(Int, String)
    /// No API key is configured for the selected remote provider.
    case remoteAuthMissing
    /// No remote provider could be reached (offline, or all endpoints down).
    case remoteUnreachable
    /// The user declined to escalate this request to a remote model.
    case escalationDeclined
    /// Context compression failed before a remote call.
    case compressionFailed(String)
    /// The model needs more memory than this device has, even with layer-streaming.
    case modelTooLarge(String)

    public var errorDescription: String? {
        switch self {
            case .modelNotLoaded:
                return "Model is not loaded. Initialize Tonamox first."
            case .imageProcessingFailed:
                return "Failed to process the provided image."
            case .invalidResponse(let detail):
                return "Invalid model response: \(detail)"
            case .remoteFailed(let code, let body):
                return "Remote provider failed (HTTP \(code)): \(body)"
            case .remoteAuthMissing:
                return "No API key configured for this remote provider."
            case .remoteUnreachable:
                return "No remote provider is reachable."
            case .escalationDeclined:
                return "Escalation to a remote model was declined."
            case .compressionFailed(let detail):
                return "Context compression failed: \(detail)"
            case .modelTooLarge(let detail):
                return "Model too large for this device: \(detail)"
        }
    }
}

/// A load option the model's backend cannot honour, or a sampling value out of range. Thrown before anything is
/// downloaded or loaded.
public enum ModelConfigurationError: LocalizedError, Equatable, Sendable {
    /// Tools were passed for a model whose backend runs no tool-calling loop for it: an MLX vision or document
    /// model, or an MLX model on the iOS Simulator. GGUF models and MLX text models run tools.
    case toolsUnsupported(BackendKind)
    /// A temperature that is negative, infinite or NaN.
    case invalidTemperature(Float)
    /// A top-p outside `0 < topP <= 1`, or NaN.
    case invalidTopP(Float)

    public var errorDescription: String? {
        switch self {
            case .toolsUnsupported(let kind):
                return "The \(kind.rawValue) backend cannot call tools with this model. Tools run on GGUF models "
                    + "and on MLX text models; load one of those, or pass none."
            case .invalidTemperature(let value):
                return "Temperature must be a finite number of at least 0 (0 is greedy); got \(value)."
            case .invalidTopP(let value):
                return "Top-p must be greater than 0 and at most 1; got \(value)."
        }
    }
}
