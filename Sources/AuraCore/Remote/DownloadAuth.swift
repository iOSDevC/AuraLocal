import Foundation

/// Supplies HTTP auth headers for a download request, resolved from the URL's host.
/// Lets any downloader authenticate gated/private sources without host-specific
/// logic scattered around.
public protocol DownloadAuthorizing: Sendable {
    /// Headers to attach for `url` (empty when no auth applies).
    func headers(for url: URL) -> [String: String]
}

public extension DownloadAuthorizing {
    /// Attach this authorizer's headers to an existing request.
    func authorize(_ request: inout URLRequest) {
        guard let url = request.url else { return }
        for (field, value) in headers(for: url) {
            request.setValue(value, forHTTPHeaderField: field)
        }
    }

    /// A new request for `url` with auth headers already applied.
    func request(for url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        authorize(&request)
        return request
    }
}

/// Resolves a per-host bearer token from the Keychain (reuses ``KeychainStore``).
///
/// Store a token under account `download.huggingface` for Hugging Face, or
/// `download.<host>` for any other host. Nothing is added when no token exists,
/// so public downloads are unaffected.
public struct KeychainDownloadAuth: DownloadAuthorizing {
    public init() {}

    /// Keychain account for the Hugging Face token.
    public static let huggingFaceAccount = "download.huggingface"

    public func headers(for url: URL) -> [String: String] {
        guard let host = url.host?.lowercased() else { return [:] }

        // Hugging Face gated/private repos: token ONLY on the API host — never the
        // signed CDN (cdn-lfs.huggingface.co), which gets a pre-signed URL. Exact
        // host match avoids leaking the token to the CDN on redirect.
        if host == "huggingface.co" || host == "hf.co" {
            if let token = KeychainStore.read(for: Self.huggingFaceAccount) {
                return ["Authorization": "Bearer \(token)"]
            }
            return [:]
        }

        // Any other host: a bearer token stored as `download.<host>`.
        if let token = KeychainStore.read(for: "download.\(host)") {
            return ["Authorization": "Bearer \(token)"]
        }
        return [:]
    }
}

/// An authorizer that never adds headers — for tests or explicit opt-out.
public struct NoDownloadAuth: DownloadAuthorizing {
    public init() {}
    public func headers(for url: URL) -> [String: String] { [:] }
}
