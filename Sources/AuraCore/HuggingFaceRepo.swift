import Foundation

// MARK: - RemoteGGUFFile

/// One downloadable `.gguf` file discovered inside a Hugging Face repository. `path` is repo-relative and may
/// contain subfolders (e.g. `MTP/model-Q8_0.gguf`); `resolveURL` is the ready-to-download direct URL that
/// ``Model/fromURL(_:displayName:)`` accepts. The host app shows these so the user can pick a quant.
public struct RemoteGGUFFile: Sendable, Equatable, Identifiable {
    /// Repo-relative path, slashes preserved (e.g. `quants/model.Q4_K_M.gguf`).
    public let path: String
    /// File size in bytes as reported by the tree API, if present.
    public let sizeBytes: Int?
    /// Direct `…/resolve/<rev>/<path>` download URL — feed straight into `Model.fromURL`.
    public let resolveURL: URL

    public init(path: String, sizeBytes: Int?, resolveURL: URL) {
        self.path = path
        self.sizeBytes = sizeBytes
        self.resolveURL = resolveURL
    }

    public var id: String { resolveURL.absoluteString }
    /// The bare file name (last path component).
    public var filename: String { (path as NSString).lastPathComponent }
    /// The quantization token parsed from the file name (e.g. `Q4_K_M`, `IQ3_XS`, `BF16`), if recognizable.
    public var quantLabel: String? { HuggingFaceRepo.quantLabel(for: filename) }
    /// Human-readable size (e.g. `1.6 GB`), or `nil` if the tree API omitted it. Formatting lives here, not
    /// in the view.
    public var sizeText: String? {
        guard let sizeBytes else { return nil }
        return ByteCountFormatter.string(fromByteCount: Int64(sizeBytes), countStyle: .file)
    }
}

// MARK: - HuggingFaceRepo

/// Turns a Hugging Face **repository** URL (the page a user lands on, e.g.
/// `https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF`) into the list of `.gguf` files it offers, so the host
/// app can present quant choices instead of demanding a direct `…/resolve/…/file.gguf` URL. Pure parsing +
/// resolve-URL building are separated from the network fetch so they're unit-testable against canned JSON.
public enum HuggingFaceRepo {

    /// A resolved repository coordinate.
    public struct Ref: Sendable, Equatable {
        public let owner: String
        public let repo: String
        public let revision: String
    }

    /// Segments that are HF *routes*, not repo owners — reject them so `/datasets/x/y`, `/models`, etc. aren't
    /// mistaken for a model repo.
    private static let reservedOwners: Set<String> = [
        "datasets", "models", "spaces", "organizations", "settings", "api", "join", "login", "docs",
        "papers", "collections", "blog", "posts", "learn", "tasks", "pricing"
    ]

    /// The hosts Hugging Face serves from — the canonical domain and its official short form (`hf.co`), which
    /// HF hands out in share links.
    static func isHuggingFaceHost(_ host: String) -> Bool {
        host == "huggingface.co" || host == "www.huggingface.co" || host == "hf.co" || host == "www.hf.co"
    }

    /// Parse a **repository** URL into `owner/repo[@revision]`. Returns `nil` for a direct file URL
    /// (`/resolve/` or `/blob/`), a non-`huggingface.co` host, or anything without an `owner/repo` pair — those
    /// are handled by `Model.fromURL`. Accepts a bare repo URL and the `/tree/<rev>[/<subdir>]` browse form.
    public static func parse(_ urlString: String) -> Ref? {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let comps = URLComponents(string: trimmed),
              comps.scheme?.lowercased() == "https",
              let host = comps.host?.lowercased(), isHuggingFaceHost(host)
        else { return nil }

        let segs = comps.path.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        guard segs.count >= 2 else { return nil }
        // A direct file URL is not a repo URL — let `Model.fromHuggingFaceURL` own those.
        if segs.contains("resolve") || segs.contains("blob") { return nil }

        let owner = segs[0], repo = segs[1]
        guard !owner.isEmpty, !repo.isEmpty, !reservedOwners.contains(owner.lowercased()) else { return nil }

        // Optional `/tree/<rev>` selects a branch/commit; otherwise default to `main`.
        var revision = "main"
        if segs.count >= 4, segs[2] == "tree" { revision = segs[3] }

        return Ref(owner: owner, repo: repo, revision: revision)
    }

    /// The tree API endpoint listing every file in the repo (recursive → includes subfolders).
    public static func treeAPIURL(_ ref: Ref) -> URL? {
        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "huggingface.co"
        comps.path = "/api/models/\(ref.owner)/\(ref.repo)/tree/\(ref.revision)"
        comps.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        return comps.url
    }

    /// Build the direct download URL for a repo-relative file path, percent-encoding each segment while keeping
    /// the `/` separators (HF serves subfolder files at `…/resolve/<rev>/<sub>/<file>`).
    public static func resolveURL(owner: String, repo: String, revision: String, path: String) -> URL? {
        let segments = ([owner, repo, "resolve", revision]
            + path.split(separator: "/", omittingEmptySubsequences: true).map(String.init))
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? $0 }
        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "huggingface.co"
        comps.percentEncodedPath = "/" + segments.joined(separator: "/")
        return comps.url
    }

    /// A single entry in the tree API response.
    private struct TreeEntry: Decodable {
        let type: String
        let path: String
        let size: Int?
    }

    /// Decode a tree API JSON payload into the downloadable `.gguf` files, smallest first. Non-files,
    /// non-`.gguf`, and unresolvable paths are dropped. Pure — no network — so it's directly testable.
    public static func ggufFiles(fromTreeJSON data: Data,
                                 owner: String, repo: String, revision: String) -> [RemoteGGUFFile] {
        guard let entries = try? JSONDecoder().decode([TreeEntry].self, from: data) else { return [] }
        return entries.compactMap { entry -> RemoteGGUFFile? in
            guard entry.type == "file",
                  entry.path.lowercased().hasSuffix(".gguf"),
                  entry.path.count > ".gguf".count
            else { return nil }
            let segments = entry.path.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
            // Defense-in-depth: never let a crafted tree path traverse out of this repo.
            guard !segments.contains("..") else { return nil }
            // Multi-part shards (`…-00001-of-00003.gguf`) need every part present in one directory; the
            // single-file loader can't assemble them, so don't offer picks that would fail after a big
            // download. (Uncommon for the small on-device models this targets.)
            guard !isShard((entry.path as NSString).lastPathComponent) else { return nil }
            guard let url = resolveURL(owner: owner, repo: repo, revision: revision, path: entry.path)
            else { return nil }
            return RemoteGGUFFile(path: entry.path, sizeBytes: entry.size, resolveURL: url)
        }
        .sorted { lhs, rhs in
            // Smallest quant first (a nil size sorts last); tie-break on path for stable ordering.
            (lhs.sizeBytes ?? .max, lhs.path) < (rhs.sizeBytes ?? .max, rhs.path)
        }
    }

    /// True when a file name is one part of a multi-part GGUF (`…-00001-of-00003.gguf`).
    static func isShard(_ filename: String) -> Bool {
        let base = (filename as NSString).deletingPathExtension
        let parts = base.split(separator: "-")
        guard parts.count >= 3, parts[parts.count - 2] == "of" else { return false }
        let index = parts[parts.count - 3], total = parts[parts.count - 1]
        return index.count == 5 && total.count == 5 && Int(index) != nil && Int(total) != nil
    }

    /// Parse the quantization token out of a GGUF file name — e.g. `…-Q4_K_M.gguf` → `Q4_K_M`,
    /// `…-IQ3_XS.gguf` → `IQ3_XS`, `…-BF16.gguf` → `BF16`. Returns `nil` when none is recognizable.
    public static func quantLabel(for filename: String) -> String? {
        let base = (filename as NSString).deletingPathExtension
        let pattern = "(?i)(IQ[0-9][A-Z0-9_]*|Q[0-9](_[A-Z0-9]+)*|BF16|F16|F32)"
        guard let range = base.range(of: pattern, options: .regularExpression) else { return nil }
        return base[range].uppercased()
    }

    /// Why a repository enumeration failed. A dedicated type so the message reaches the UI clean, without the
    /// generic `AuraError.invalidResponse` "Invalid model response:" prefix.
    public enum RepoError: LocalizedError, Equatable {
        case notARepositoryURL
        case notFound
        case http(Int)
        case noResponse

        public var errorDescription: String? {
            switch self {
            case .notARepositoryURL: "Not a Hugging Face repository URL."
            case .notFound: "Repository not found on Hugging Face."
            case .http(let code): "Hugging Face returned HTTP \(code)."
            case .noResponse: "No response from Hugging Face."
            }
        }
    }

    /// Fetch a repo's `.gguf` files from the Hugging Face tree API. Throws ``RepoError`` on a non-repo URL or a
    /// non-200 response (404 → not found); rethrows transport errors. Off the main actor via `URLSession`.
    public static func listGGUF(repoURL: String) async throws -> [RemoteGGUFFile] {
        guard let ref = parse(repoURL), let api = treeAPIURL(ref) else {
            throw RepoError.notARepositoryURL
        }
        var request = URLRequest(url: api)
        request.timeoutInterval = 20
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw RepoError.noResponse }
        guard http.statusCode == 200 else {
            throw http.statusCode == 404 ? RepoError.notFound : RepoError.http(http.statusCode)
        }
        return ggufFiles(fromTreeJSON: data, owner: ref.owner, repo: ref.repo, revision: ref.revision)
    }
}
