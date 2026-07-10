import Foundation

/// Errors thrown by the AuraLocal inference pipeline.
public enum AuraError: LocalizedError {
    /// The model has not been loaded yet — call a factory method or ``ModelManager/load(_:onProgress:)`` first.
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

    public var errorDescription: String? {
        switch self {
            case .modelNotLoaded:
                return "Model is not loaded. Initialize AuraLocal first."
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
        }
    }
}
