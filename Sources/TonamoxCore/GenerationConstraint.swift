import Foundation

// MARK: - GenerationConstraint

/// Restricts what one generation may emit, enforced token by token in the llama.cpp sampler.
///
/// Only ``BackendKind/llamaCpp`` (a GGUF model loaded in full) enforces it. Every other backend throws
/// ``GenerationConstraintError/unsupportedBackend(_:)`` instead of returning unconstrained text.
public enum GenerationConstraint: Sendable, Equatable {
    /// A JSON object, using the `json.gbnf` grammar bundled with LocalLLMClient.
    case json
    /// A custom GBNF grammar; `root` names its start rule.
    case grammar(gbnf: String, root: String = "root")
}

// MARK: - GenerationConstraintError

/// Why a constrained generation could not run.
public enum GenerationConstraintError: LocalizedError, Equatable {
    /// The backend cannot enforce a ``GenerationConstraint``.
    case unsupportedBackend(BackendKind)
    /// llama.cpp rejected the GBNF grammar, or it was empty.
    case invalidGrammar(root: String)

    public var errorDescription: String? {
        switch self {
            case .unsupportedBackend(let kind):
                return "The \(kind.rawValue) backend cannot enforce a JSON/GBNF generation constraint; "
                    + "only a GGUF model loaded in full on llama.cpp can."
            case .invalidGrammar(let root):
                return "llama.cpp rejected the GBNF grammar: it is empty, has a syntax error or an undefined "
                    + "rule, lacks a '\(root)' rule, or is left-recursive."
        }
    }
}
