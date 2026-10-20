import Foundation

// MARK: - ModelCapabilities

/// What a model takes and does, read from its catalog metadata and from the backend that would run it. Pickers use
/// it for badges, and to offer image input or document extraction only where they work.
///
/// ```swift
/// let caps = model.capabilities
/// if caps.contains(.vision) { /* offer an image attachment */ }
/// if caps.contains(.documentExtraction) { /* offer extractDocument(_:) */ }
/// ```
public struct ModelCapabilities: OptionSet, Sendable, Hashable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    /// Holds a conversation from text alone: ``Model/Category/text`` and ``Model/Category/vision`` models.
    public static let text = ModelCapabilities(rawValue: 1 << 0)
    /// Takes an image with the prompt (``AuraLocal/analyze(_:image:maxTokens:)``): vision and document models in
    /// MLX format. The llama.cpp backends drop the image, so a GGUF model never has it.
    public static let vision = ModelCapabilities(rawValue: 1 << 1)
    /// An image model with a ``Model/defaultDocumentPrompt``, which ``AuraLocal/extractDocument(_:maxTokens:)``
    /// needs.
    public static let documentExtraction = ModelCapabilities(rawValue: 1 << 2)
    /// Its backend runs a tool-calling loop when loaded with tools: every GGUF model, and MLX text models where MLX
    /// runs (not the iOS Simulator). An MLX vision or document model loaded with tools throws
    /// ``ModelConfigurationError/toolsUnsupported(_:)``.
    public static let tools = ModelCapabilities(rawValue: 1 << 3)
}

public extension Model {

    /// What this model takes and does on the backend ``BackendRouter`` picks for it.
    var capabilities: ModelCapabilities {
        var result: ModelCapabilities
        switch category {
        case .text:              result = [.text]
        case .vision:            result = [.text, .vision]
        case .visionSpecialized: result = [.vision]
        }
        if !BackendRouter.acceptsImages(for: self) {
            result.remove(.vision)
        }
        if result.contains(.vision), defaultDocumentPrompt != nil {
            result.insert(.documentExtraction)
        }
        if BackendRouter.runsToolLoop(for: self) {
            result.insert(.tools)
        }
        return result
    }
}
