import Foundation

// MARK: - HFPipelineFilter

/// Keeps the Hugging Face search results whose declared pipeline tag fits one use, such as chat. It reads only the
/// tag a repo declares; whether the weights load is still the compatibility checker's call.
public struct HFPipelineFilter: Sendable, Hashable {
    /// Pipeline tags that pass, e.g. `"text-generation"`.
    public let pipelineTags: Set<String>
    /// Whether a repo that declares no pipeline passes. Many GGUF uploads declare none.
    public let admitsUndeclared: Bool

    public init(pipelineTags: Set<String>, admitsUndeclared: Bool = true) {
        self.pipelineTags = pipelineTags
        self.admitsUndeclared = admitsUndeclared
    }

    public func admits(_ hit: HFModelHit) -> Bool {
        guard let tag = hit.pipelineTag else { return admitsUndeclared }
        return pipelineTags.contains(tag)
    }

    /// Models a chat loads: text generation and image + text to text, plus any-to-any, which Gemma 4 GGUF repos
    /// declare. Repos without a tag pass.
    public static let chat = HFPipelineFilter(pipelineTags: ["text-generation", "image-text-to-text", "any-to-any"])
}
