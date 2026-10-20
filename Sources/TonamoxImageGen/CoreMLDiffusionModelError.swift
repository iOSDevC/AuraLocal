import Foundation

/// Why searching, inspecting or adopting a Core ML Stable Diffusion repo from Hugging Face failed. A separate enum
/// from ``OnDeviceImageGenError`` so existing exhaustive switches over it keep compiling. Cancellation throws
/// `CancellationError`, not a case here.
public enum CoreMLDiffusionModelError: LocalizedError, Equatable, Sendable {
    /// Not an `owner/repo` id or a Hugging Face repository URL.
    case invalidRepoID(String)
    /// The search API answered with a non-2xx status, or `nil` for an unreadable body.
    case searchFailed(status: Int?)
    /// The listing answered 401/403/404 (or another non-2xx status), or it had no commit sha (`status` `nil`).
    case repositoryUnavailable(repoID: String, status: Int?)
    /// The request did not reach Hugging Face.
    case network(detail: String)
    /// The variant failed the compatibility rules, or its size or image side is unknown.
    case notRunnable(repoID: String, folder: String, reasons: [String])

    public var errorDescription: String? {
        switch self {
        case .invalidRepoID(let input):
            "“\(input)” is not a Hugging Face repository."
        case .searchFailed(let status):
            "The Hugging Face search failed\(Self.statusSuffix(status))."
        case .repositoryUnavailable(let repoID, let status):
            "\(repoID) could not be read\(Self.statusSuffix(status))."
        case .network(let detail):
            "Hugging Face could not be reached: \(detail)"
        case .notRunnable(let repoID, let folder, let reasons):
            "\(repoID)/\(folder) can't run here: \(reasons.joined(separator: "; "))"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .invalidRepoID:
            "Enter it as owner/repo, or paste the repository's page URL."
        case .searchFailed:
            "Try again later, or inspect a repository by its id."
        // Anonymous Hugging Face answers 401 for a mistyped id too; a gated repo lists fine and is rule R17.
        case .repositoryUnavailable(_, let status) where [401, 403, 404].contains(status):
            "Not found, or private: check the id. TonamoxImageGen sends no Hugging Face token."
        case .repositoryUnavailable:
            "Check the id, then try again."
        case .network:
            "Check the connection and try again."
        case .notRunnable:
            "Pick a variant marked Runs, or another model."
        }
    }

    private static func statusSuffix(_ status: Int?) -> String {
        status.map { " (HTTP \($0))" } ?? ""
    }
}
