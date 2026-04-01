import Foundation

/// Errors thrown by the AuraLocal inference pipeline.
public enum AuraError: LocalizedError {
    /// The model has not been loaded yet — call a factory method or ``ModelManager/load(_:onProgress:)`` first.
    case modelNotLoaded
    /// The provided image could not be converted to the format required by the vision model.
    case imageProcessingFailed
    /// The model returned an unexpected or invalid result.
    case invalidResponse(String)
    
    public var errorDescription: String? {
        switch self {
            case .modelNotLoaded:
                return "Model is not loaded. Initialize AuraLocal first."
            case .imageProcessingFailed:
                return "Failed to process the provided image."
            case .invalidResponse(let detail):
                return "Invalid model response: \(detail)"
        }
    }
}
