import Foundation

/// Why a model download failed. A separate enum rather than new ``AuraError`` cases, so existing exhaustive
/// switches over ``AuraError`` in apps keep compiling.
public enum ModelDownloadError: LocalizedError, Equatable, Sendable {
    /// HTTP 401/403 for a gated repo, an invalid token, or a token without access.
    case unauthorized(repoID: String)
    /// HTTP 404, or Hugging Face's `RepoNotFound` / `EntryNotFound` / `RevisionNotFound` error code.
    case notFound(repoID: String, path: String)
    /// HTTP 401 with no error code to an anonymous request: Hugging Face answers a missing, renamed or private
    /// repo this way, so the two cannot be told apart without a token.
    case notFoundOrPrivate(repoID: String)
    /// Not enough free space for the remaining bytes plus a 20 % margin.
    case insufficientSpace(neededBytes: Int64, availableBytes: Int64)
    /// Any other unexpected HTTP status.
    case http(status: Int)
    /// The repo listing matched no weights or no `config.json`, so there is nothing loadable to download.
    case incompleteRepo(repoID: String)
    /// A finished file does not have the size the listing announced; it is not moved into place.
    case sizeMismatch(path: String, expected: Int64, actual: Int64)
    /// A model this API cannot download: a split or subfolder GGUF quant, or a user's local file.
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .unauthorized(let repoID):
            "\(repoID) is gated or private. Accept its license on huggingface.co, then add a read token in Settings."
        case .notFound(let repoID, let path):
            path.isEmpty ? "\(repoID) was not found on Hugging Face." : "\(path) was not found in \(repoID)."
        case .notFoundOrPrivate(let repoID):
            "\(repoID) does not exist, was renamed, or is private. A private repo needs a read token with access."
        case .insufficientSpace(let needed, let available):
            "Not enough free space: needs \(Self.bytesText(needed)), \(Self.bytesText(available)) available."
        case .http(let status):
            "Hugging Face returned HTTP \(status)."
        case .incompleteRepo(let repoID):
            "\(repoID) has no weights or config.json to download."
        case .sizeMismatch(let path, let expected, let actual):
            "\(path) is \(actual) bytes, expected \(expected). Discard it and download again."
        case .unsupported(let reason):
            reason
        }
    }

    /// The error for a failed response; `nil` for 2xx. `sentToken` tells an anonymous 401 (missing or private
    /// repo) from a rejected token.
    init?(response: HTTPURLResponse, repoID: String, path: String, sentToken: Bool) {
        let code = response.value(forHTTPHeaderField: "X-Error-Code")
        switch (response.statusCode, code) {
        case (200..<300, _): return nil
        case (_, "GatedRepo"): self = .unauthorized(repoID: repoID)
        case (_, "RepoNotFound"), (_, "RevisionNotFound"): self = .notFound(repoID: repoID, path: "")
        case (_, "EntryNotFound"), (404, _): self = .notFound(repoID: repoID, path: path)
        case (401, nil) where !sentToken: self = .notFoundOrPrivate(repoID: repoID)
        case (401, _), (403, _): self = .unauthorized(repoID: repoID)
        default: self = .http(status: response.statusCode)
        }
    }

    init?(response: HTTPURLResponse, request: URLRequest, repoID: String, path: String) {
        self.init(response: response, repoID: repoID, path: path,
                  sentToken: request.value(forHTTPHeaderField: "Authorization") != nil)
    }

    /// Worth another attempt: server-side or throttling statuses. Auth, missing files and bad sizes are not.
    var isTransient: Bool {
        guard case .http(let status) = self else { return false }
        return status >= 500 || status == 408 || status == 429
    }

    private static func bytesText(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

/// A download whose `.partial` file another writer changed before it was moved into place. A separate enum rather
/// than a new ``ModelDownloadError`` case, so existing exhaustive switches over that enum in apps keep compiling.
public enum DownloadConflictError: LocalizedError, Equatable, Sendable {
    /// The finished `.partial` of `path` was gone or no longer had the size just written when it was about to be moved
    /// into place: another download of the same file, in this process or another one, moved, deleted or rewrote it.
    /// AuraLocal's own downloads wait for each other, so this points at another tool or an older build writing the
    /// same folder. Nothing was moved into place; downloading again resumes or restarts the file.
    case partialChanged(path: String)

    public var errorDescription: String? {
        switch self {
        case .partialChanged(let path):
            "\(path) was changed by another download of the same file before it could be moved into place. "
                + "Nothing was replaced; download it again."
        }
    }
}
