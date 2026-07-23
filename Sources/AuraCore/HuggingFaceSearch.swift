import Foundation

// MARK: - HFModelHit

/// One result from a HuggingFace model search.
public struct HFModelHit: Sendable, Identifiable, Equatable {
    /// `owner/repo`.
    public let id: String
    public let likes: Int
    public let downloads: Int
    /// e.g. `"text-to-image"`, `"text-generation"`; `nil` if the repo doesn't declare one.
    public let pipelineTag: String?
    /// The repo's tags (e.g. `"diffusers"`, `"gguf"`, `"mlx"`).
    public let tags: [String]
    /// Whether HuggingFace gates the repo (needs a login + accepted license to download).
    public let gated: Bool

    public init(id: String, likes: Int, downloads: Int,
                pipelineTag: String?, tags: [String], gated: Bool) {
        self.id = id
        self.likes = likes
        self.downloads = downloads
        self.pipelineTag = pipelineTag
        self.tags = tags
        self.gated = gated
    }

    public var owner: String { id.split(separator: "/").first.map(String.init) ?? id }

    /// A diffusion / image-generation model — its runtime memory profile is NOT the LLM one
    /// (text encoder + VAE + activations push the peak to ~2× the weights). Drives the fit multiplier.
    public var isDiffusion: Bool {
        pipelineTag == "text-to-image" || tags.contains("diffusers") || tags.contains("text-to-image")
    }

    /// Best-guess on-device kind for the compatibility filter.
    public var kind: ModelKind { isDiffusion ? .diffusion : .llm }
}

// MARK: - HuggingFaceSearch

public enum HuggingFaceSearch {

    public enum Sort: String, Sendable, CaseIterable {
        case downloads, likes, lastModified, trendingScore
        public var label: String {
            switch self {
            case .downloads:     "Downloads"
            case .likes:         "Likes"
            case .lastModified:  "Recent"
            case .trendingScore: "Trending"
            }
        }
    }

    /// Search HuggingFace models by keyword. Network wrapper around ``models(fromSearchJSON:)``.
    public static func search(
        _ query: String,
        limit: Int = 30,
        sort: Sort = .downloads,
        session: URLSession = .shared
    ) async throws -> [HFModelHit] {
        guard let url = searchURL(query: query, limit: limit, sort: sort) else { return [] }
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse else {
            throw HuggingFaceRepo.RepoError.noResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            throw HuggingFaceRepo.RepoError.http(http.statusCode)
        }
        return models(fromSearchJSON: data)
    }

    /// Build the `/api/models` search URL. `?search=` + `full=true` (so `tags`/`gated` come back).
    public static func searchURL(query: String, limit: Int, sort: Sort) -> URL? {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "huggingface.co"
        comps.path = "/api/models"
        comps.queryItems = [
            URLQueryItem(name: "search", value: trimmed),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "sort", value: sort.rawValue),
            URLQueryItem(name: "full", value: "true"),
        ]
        return comps.url
    }

    /// Decode a `/api/models` search payload. Pure — no network — so it is directly testable.
    public static func models(fromSearchJSON data: Data) -> [HFModelHit] {
        guard let raw = try? JSONDecoder().decode([SearchEntry].self, from: data) else { return [] }
        return raw.map { e in
            HFModelHit(
                id: e.id,
                likes: e.likes ?? 0,
                downloads: e.downloads ?? 0,
                pipelineTag: e.pipeline_tag,
                tags: e.tags ?? [],
                gated: Self.isGated(e.gated))
        }
    }

    /// HF's `gated` is polymorphic: `false`, `"auto"`, `"manual"`, or absent. Anything non-false gates.
    static func isGated(_ value: GatedValue?) -> Bool {
        switch value {
        case .none:            return false
        case .bool(let b):     return b
        case .string(let s):   return s.lowercased() != "false"
        }
    }

    // MARK: - Wire types

    private struct SearchEntry: Decodable {
        let id: String
        let likes: Int?
        let downloads: Int?
        let pipeline_tag: String?
        let tags: [String]?
        let gated: GatedValue?
    }

    /// `gated` decodes as either a Bool or a String across HF responses.
    enum GatedValue: Decodable {
        case bool(Bool)
        case string(String)
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let b = try? c.decode(Bool.self) { self = .bool(b) }
            else { self = .string((try? c.decode(String.self)) ?? "false") }
        }
    }
}
