import Foundation
import TonamoxCore

/// Every Core ML Stable Diffusion folder of one repo, judged for one device at one commit.
public struct CoreMLDiffusionRepoReport: Sendable, Equatable {
    public let repoID: String
    /// The commit every probe and a later download are pinned to.
    public let revision: String
    /// Sorted by folder path.
    public let variants: [CoreMLDiffusionVariant]
    /// Set when `variants` is empty.
    public let unsupportedReason: String?

    /// Runs over runs-with-caveats, the Neural Engine over the GPU, then the smaller download; `nil` without variants.
    public var recommended: CoreMLDiffusionVariant? {
        variants.min(by: CoreMLDiffusionRules.isPreferred)
    }
}

/// A search result screened by file names only: no sizes and no probe, so no verdict yet.
public struct CoreMLDiffusionSearchHit: Sendable, Equatable, Identifiable {
    public let id: String
    public let revision: String?
    public let downloads: Int
    public let likes: Int
    public let isGated: Bool
    /// Folders that pass the layout rules.
    public let candidateFolders: [String]
    /// Why no folder qualified, when that is Core ML specific (zip, `.mlpackage`, …).
    public let unsupportedReason: String?
}

/// Finds Core ML Stable Diffusion repos on Hugging Face and judges them for a device without downloading weights:
/// one listing per repo, then each folder's Unet `metadata.json` (~4 KB), both pinned to the listing's commit.
///
/// What leaves the device: the search text and repo ids, sent to huggingface.co. No token, cookie or
/// `Authorization` header is ever sent, so gated repos are reported, not opened.
public struct CoreMLDiffusionInspector: Sendable {
    public static let maxProbedFolders = 4
    /// The package's deployment floor, used when judging another platform than the running one.
    static let floorMajor: [TargetOS: Int] = [.iOS: 18, .macOS: 15]

    private let session: URLSession
    private let requestTimeout: TimeInterval

    public init(session: URLSession = .shared, requestTimeout: TimeInterval = 60) {
        self.session = session
        self.requestTimeout = requestTimeout
    }

    /// Core ML text-to-image repos for `query` (empty is allowed; a pasted repo URL searches its id), most downloaded
    /// first. Sends `query` to huggingface.co; never a token. Three searches run, by the `coreml` tag, by the
    /// `core-ml` tag and by name together with "coreml", so repos with either tag or none show up; a failing search is
    /// ignored unless all fail.
    public func search(_ query: String, limit: Int = 50) async throws -> [CoreMLDiffusionSearchHit] {
        let urls = Self.searchURLs(query: query, limit: limit)
        async let tagged = hitsOutcome(urls.first)
        async let otherTag = hitsOutcome(urls.dropFirst().first)
        async let named = hitsOutcome(urls.dropFirst(2).first)
        let outcomes = await [tagged, otherTag, named]
        try Task.checkCancellation()
        let lists = outcomes.compactMap { try? $0.get() }
        if lists.isEmpty, case .failure(let error)? = outcomes.first { throw error }
        return Self.merged(lists)
    }

    /// The repo's listing at its current commit, then up to ``maxProbedFolders`` Unet probes pinned to that commit
    /// (non-root folders first). A failed probe leaves that variant unknown (rule R1); it does not throw.
    public func inspect(_ repo: String, on target: DevicePreset = .thisDevice()) async throws
        -> CoreMLDiffusionRepoReport {
        guard let requested = ModelCompatibilityChecker.repoID(from: repo), let url = Self.listingURL(requested) else {
            throw CoreMLDiffusionModelError.invalidRepoID(repo)
        }
        let body = try await data(from: url) { .repositoryUnavailable(repoID: requested, status: $0) }
        guard let listing = HFRepoInfo.parse(body), let revision = listing.revision, Self.isCommit(revision),
              ModelCompatibilityChecker.repoID(from: listing.repoID) == listing.repoID else {
            throw CoreMLDiffusionModelError.repositoryUnavailable(repoID: requested, status: nil)
        }
        let folders = CoreMLDiffusionLayout.folders(in: listing.files)
        let probes = listing.gatedMode == nil
            ? try await probe(folders, repoID: listing.repoID, revision: revision) : [:]
        let osMajor = Self.osMajor(for: target.platform)
        let skipped = "Not probed: only the first \(Self.maxProbedFolders) folders of a repo are."
        let variants = folders.map { folder in
            CoreMLDiffusionRules.evaluate(folder, unet: probes[folder.path]?.unet, listing: listing, on: target,
                                          osMajor: osMajor, unetProblem: probes[folder.path]?.problem ?? skipped)
        }
        return CoreMLDiffusionRepoReport(
            repoID: listing.repoID, revision: revision, variants: variants,
            unsupportedReason: variants.isEmpty ? CoreMLDiffusionLayout.unsupportedReason(in: listing.files) : nil)
    }

    // MARK: Search

    static func searchURLs(query: String, limit: Int) -> [URL] {
        let typed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let term = ModelCompatibilityChecker.repoID(from: typed) ?? typed
        let common = [
            URLQueryItem(name: "pipeline_tag", value: "text-to-image"),
            URLQueryItem(name: "full", value: "true"),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "limit", value: String(min(max(limit, 1), 200))),
        ]
        let text = term.isEmpty ? [] : [URLQueryItem(name: "search", value: term)]
        let byTag = [URLQueryItem(name: "filter", value: "coreml")] + common + text
        let byOtherTag = [URLQueryItem(name: "filter", value: "core-ml")] + common + text
        // Search terms must all appear in the id; alone, a common word's top hits are all diffusers repos.
        let named = term.lowercased().contains("coreml") ? term : "\(term) coreml"
        let byName = common + [URLQueryItem(name: "search", value: named.trimmingCharacters(in: .whitespaces))]
        return [byTag, byOtherTag, byName].compactMap(apiURL)
    }

    /// Core ML candidates only: a plain diffusers repo has neither a folder nor a Core ML specific reason.
    static func hits(fromSearchJSON data: Data) -> [CoreMLDiffusionSearchHit] {
        guard let entries = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return [] }
        return entries.compactMap(makeHit)
    }

    static func makeHit(_ entry: [String: Any]) -> CoreMLDiffusionSearchHit? {
        guard let id = (entry["id"] as? String) ?? (entry["modelId"] as? String),
              ModelCompatibilityChecker.repoID(from: id) == id else { return nil }
        let names = (entry["siblings"] as? [[String: Any]] ?? []).compactMap { $0["rfilename"] as? String }
        let files = names.map { RepoFile(path: $0, sizeBytes: nil) }
        let folders = CoreMLDiffusionLayout.folders(in: files)
        let reason = folders.isEmpty ? CoreMLDiffusionLayout.unsupportedReason(in: files) : nil
        guard !folders.isEmpty || (reason.map { $0 != CoreMLDiffusionLayout.genericReason } ?? false) else {
            return nil
        }
        return CoreMLDiffusionSearchHit(
            id: id, revision: entry["sha"] as? String, downloads: entry["downloads"] as? Int ?? 0,
            likes: entry["likes"] as? Int ?? 0, isGated: isGated(entry["gated"]),
            candidateFolders: folders.map(\.path), unsupportedReason: reason)
    }

    /// First list's order kept; later lists only add ids not seen yet.
    static func merged(_ lists: [[CoreMLDiffusionSearchHit]]) -> [CoreMLDiffusionSearchHit] {
        var seen = Set<String>()
        return lists.joined().filter { seen.insert($0.id).inserted }
    }

    /// `gated` is `false`, `"auto"` or `"manual"`.
    static func isGated(_ raw: Any?) -> Bool {
        if let flag = raw as? Bool { return flag }
        if let mode = raw as? String { return mode.lowercased() != "false" }
        return false
    }

    private func hitsOutcome(_ url: URL?) async -> Result<[CoreMLDiffusionSearchHit], any Error> {
        guard let url else { return .success([]) }
        do {
            let body = try await data(from: url) { .searchFailed(status: $0) }
            return .success(Self.hits(fromSearchJSON: body))
        } catch {
            return .failure(error)
        }
    }

    // MARK: Probe

    private typealias ProbeOutcome = (unet: CoreMLUnetFacts?, problem: String?)

    private func probe(_ folders: [CoreMLDiffusionFolder], repoID: String,
                       revision: String) async throws -> [String: ProbeOutcome] {
        let chosen = folders
            .sorted { ($0.isRepoRoot ? 1 : 0, $0.path) < ($1.isRepoRoot ? 1 : 0, $1.path) }
            .prefix(Self.maxProbedFolders)
        return try await withThrowingTaskGroup(of: (String, ProbeOutcome).self) { group in
            for folder in chosen {
                group.addTask { (folder.path, try await self.probeUnet(folder, repoID: repoID, revision: revision)) }
            }
            var outcomes: [String: ProbeOutcome] = [:]
            for try await (path, outcome) in group {
                outcomes[path] = outcome
            }
            return outcomes
        }
    }

    /// Throws only on cancellation.
    private func probeUnet(_ folder: CoreMLDiffusionFolder, repoID: String,
                           revision: String) async throws -> ProbeOutcome {
        let parts = repoID.split(separator: "/").map(String.init)
        guard parts.count == 2, let url = HuggingFaceRepo.resolveURL(owner: parts[0], repo: parts[1],
                                                                     revision: revision,
                                                                     path: folder.unetMetadataPath) else {
            return (nil, "No URL for \(folder.unetMetadataPath).")
        }
        do {
            let body = try await data(from: url) { .repositoryUnavailable(repoID: repoID, status: $0) }
            guard let facts = CoreMLUnetFacts.parse(body) else { return (nil, "metadata.json could not be parsed.") }
            return (facts, nil)
        } catch let error as CoreMLDiffusionModelError {
            if case .repositoryUnavailable(_, let status?) = error {
                return (nil, "metadata.json answered HTTP \(status).")
            }
            return (nil, error.errorDescription)
        }
    }

    // MARK: HTTP

    /// GET without credentials or cookies. Non-2xx throws `failure(status)`; cancellation throws
    /// `CancellationError`; any other transport error throws ``CoreMLDiffusionModelError/network(detail:)``.
    private func data(from url: URL, failure: (Int) -> CoreMLDiffusionModelError) async throws -> Data {
        var request = URLRequest(url: url, timeoutInterval: requestTimeout)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpShouldHandleCookies = false
        let body: Data
        let response: URLResponse
        do {
            (body, response) = try await session.data(for: request)
        } catch let error as URLError where error.code == .cancelled {
            throw CancellationError()
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            if Task.isCancelled { throw CancellationError() }
            throw CoreMLDiffusionModelError.network(detail: error.localizedDescription)
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else { throw failure(status) }
        return body
    }

    // MARK: URLs

    static func apiURL(_ items: [URLQueryItem]) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/api/models"
        components.queryItems = items
        return components.url
    }

    static func listingURL(_ repoID: String) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/api/models/\(repoID)"
        components.queryItems = [URLQueryItem(name: "blobs", value: "true")]
        return components.url
    }

    /// A full commit sha: the revision goes into download URLs and the folder's `.revision` marker.
    static func isCommit(_ revision: String) -> Bool {
        revision.count == 40 && revision.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }

    /// The running OS when `platform` is this one; otherwise (and under Mac Catalyst) the package's floor.
    static func osMajor(for platform: TargetOS) -> Int {
        #if os(macOS)
        let running = TargetOS.macOS
        #else
        let running = TargetOS.iOS
        #endif
        guard platform == running, !ProcessInfo.processInfo.isMacCatalystApp else {
            return floorMajor[platform] ?? 18
        }
        return ProcessInfo.processInfo.operatingSystemVersion.majorVersion
    }
}
