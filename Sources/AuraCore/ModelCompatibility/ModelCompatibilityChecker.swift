import Foundation

/// Answers "will AuraLocal, with its pinned runtimes, load and run this Hugging Face repo on this device — and if
/// not, why". It reads the repo listing, `config.json`, the safetensors weight map and, through HTTP Range reads,
/// the first megabytes of a GGUF file or a single safetensors file; nothing is downloaded in full.
///
/// ```swift
/// let checker = ModelCompatibilityChecker()
/// let report = await checker.check("mlx-community/Qwen3.5-27B-4bit", on: .mac32GB)
/// print(report.status.label, report.headline)
/// ```
///
/// Network failures, 401/403 and 404 become findings (verdict ``CompatibilityVerdict/unknown``), never throws.
public struct ModelCompatibilityChecker: Sendable {
    public let transport: URLSession
    public let authorizer: any DownloadAuthorizing
    /// Bytes read from the start of a GGUF file. The architecture keys come first; the tokenizer that follows
    /// may not fit, which only marks the header incomplete.
    public let ggufHeaderBytes: Int
    public let requestTimeout: TimeInterval

    /// Unreferenced safetensors files whose headers are read to count their tensors.
    static let maxExtraHeaders = 4
    static let safetensorsProbeBytes = 256 << 10

    public init(transport: URLSession = .shared,
                authorizer: any DownloadAuthorizing = KeychainDownloadAuth(),
                ggufHeaderBytes: Int = 4 << 20,
                requestTimeout: TimeInterval = 60) {
        self.transport = transport
        self.authorizer = authorizer
        self.ggufHeaderBytes = ggufHeaderBytes
        self.requestTimeout = requestTimeout
    }

    /// Fetch and evaluate in one go.
    public func check(_ repo: String, on target: DevicePreset = .thisDevice()) async -> CompatibilityReport {
        CompatibilityEvaluator.evaluate(await snapshot(of: repo), on: target)
    }

    /// Fetch everything the rules need. Cache this and re-run ``CompatibilityEvaluator/evaluate(_:on:)`` to
    /// judge another device.
    public func snapshot(of repo: String) async -> RepoSnapshot {
        guard let repoID = Self.repoID(from: repo), let listingURL = Self.listingURL(repoID) else {
            return RepoSnapshot(repoID: repo, problems: [
                FetchProblem(subject: .repository, statusCode: nil, message: "Not a Hugging Face repository id: \(repo)"),
            ])
        }
        var snapshot = RepoSnapshot(repoID: repoID)
        let listingData: Data
        switch await get(listingURL, subject: .repository) {
        case .failure(let problem):
            snapshot.problems.append(problem)
            return snapshot
        case .success(let data):
            listingData = data
        }
        guard let listing = HFRepoInfo.parse(listingData) else {
            snapshot.problems.append(FetchProblem(subject: .repository, statusCode: nil,
                                                  message: "Unexpected repository listing format"))
            return snapshot
        }
        snapshot.listing = listing

        let topLevel = Set(listing.files.filter(\.isTopLevel).map(\.path))
        let groups = GGUFQuantGroup.groups(in: listing.files)
        let sample = (groups.first { !$0.isSplit } ?? groups.first)?.firstPath
        let shippedSafetensors = listing.files.filter { $0.isSafetensors && $0.isTopLevel }.map(\.path)

        async let config = fileIfListed("config.json", in: topLevel, repoID: repoID, subject: .config)
        async let index = fileIfListed("model.safetensors.index.json", in: topLevel, repoID: repoID, subject: .weightIndex)
        async let gguf = ggufHeader(repoID: repoID, path: sample)

        var problems: [FetchProblem] = []
        let indexResult = await index
        snapshot.weightMap = Self.unwrap(indexResult, into: &problems).flatMap(RepoSnapshot.weightMap(fromIndexJSON:))
        let readsShippedHeaders = indexResult == nil || snapshot.isWeightMapStale
        async let shipped = mergedSafetensorsHeader(
            repoID: repoID, paths: readsShippedHeaders ? Array(shippedSafetensors.prefix(Self.maxExtraHeaders)) : [])
        snapshot.configuration = Self.unwrap(await config, into: &problems).flatMap(ModelConfigFacts.parse)
        let firstShard = snapshot.liveWeightMap.flatMap { Set($0.values).min() }
        let needsShardMetadata = CompatibilityRules.verdictDependsOnWeightMetadata(snapshot)
        async let shard = safetensorsHeader(repoID: repoID, path: needsShardMetadata ? firstShard : nil)

        snapshot.ggufMetadata = Self.unwrap(await gguf, into: &problems)
        snapshot.ggufSamplePath = sample
        snapshot.singleFileHeader = Self.unwrap(await shipped, into: &problems)
        snapshot.weightMetadata = Self.unwrap(await shard, into: &problems)?.metadata
        if let weightMap = snapshot.liveWeightMap {
            snapshot.extraTensorCounts = await extraTensorCounts(repoID: repoID, files: listing.files, weightMap: weightMap)
        }
        snapshot.problems = problems
        return snapshot
    }

    private static func unwrap<T>(_ result: Result<T, FetchProblem>?, into problems: inout [FetchProblem]) -> T? {
        switch result {
        case .success(let value)?: return value
        case .failure(let problem)?: problems.append(problem); return nil
        case nil: return nil
        }
    }

    // MARK: Repo ids

    /// `owner/repo` from a bare id or a huggingface.co / hf.co URL; `nil` otherwise.
    public static func repoID(from input: String) -> String? {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if let ref = HuggingFaceRepo.parse(trimmed) { return "\(ref.owner)/\(ref.repo)" }
        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy(isValidSegment) else { return nil }
        return trimmed
    }

    private static func isValidSegment(_ segment: Substring) -> Bool {
        !segment.isEmpty && segment != "." && segment != ".."
            && segment.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || "-_.".contains($0)) }
    }

    static func listingURL(_ repoID: String) -> URL? {
        URL(string: "https://huggingface.co/api/models/\(repoID)?blobs=true")
    }

    static func fileURL(_ repoID: String, path: String) -> URL? {
        let parts = repoID.split(separator: "/").map(String.init)
        guard parts.count == 2 else { return nil }
        return HuggingFaceRepo.resolveURL(owner: parts[0], repo: parts[1], revision: "main", path: path)
    }

    // MARK: Fetching

    private func fileIfListed(_ path: String, in names: Set<String>, repoID: String,
                              subject: FetchedResource) async -> Result<Data, FetchProblem>? {
        guard names.contains(path), let url = Self.fileURL(repoID, path: path) else { return nil }
        return await get(url, subject: subject)
    }

    private func ggufHeader(repoID: String, path: String?) async -> Result<GGUFHeader, FetchProblem>? {
        guard let path, let url = Self.fileURL(repoID, path: path) else { return nil }
        switch await readPrefix(url, byteCount: ggufHeaderBytes, subject: .ggufHeader) {
        case .failure(let problem):
            return .failure(problem)
        case .success(let data):
            do {
                return .success(try GGUFHeaderParser.parse(data))
            } catch {
                return .failure(FetchProblem(subject: .ggufHeader, statusCode: nil,
                                             message: "\(path): \(error.localizedDescription)"))
            }
        }
    }

    private func safetensorsHeader(repoID: String, path: String?) async -> Result<SafetensorsHeader, FetchProblem>? {
        guard let path, let url = Self.fileURL(repoID, path: path) else { return nil }
        let probe: Data
        switch await readPrefix(url, byteCount: Self.safetensorsProbeBytes, subject: .safetensorsHeader) {
        case .failure(let problem): return .failure(problem)
        case .success(let data): probe = data
        }
        guard let needed = SafetensorsHeader.requiredByteCount(prefix: probe) else {
            return .failure(FetchProblem(subject: .safetensorsHeader, statusCode: nil, message: "\(path): not a safetensors header"))
        }
        var full = probe
        if needed > probe.count {
            switch await readPrefix(url, byteCount: needed, subject: .safetensorsHeader) {
            case .failure(let problem): return .failure(problem)
            case .success(let data): full = data
            }
        }
        guard let header = SafetensorsHeader.parse(full) else {
            return .failure(FetchProblem(subject: .safetensorsHeader, statusCode: nil, message: "\(path): unreadable safetensors header"))
        }
        return .success(header)
    }

    /// The headers of `paths` as one: every tensor name, and the first file's `__metadata__`.
    private func mergedSafetensorsHeader(repoID: String, paths: [String]) async -> Result<SafetensorsHeader, FetchProblem>? {
        guard !paths.isEmpty else { return nil }
        let results = await withTaskGroup(of: (Int, Result<SafetensorsHeader, FetchProblem>?).self) { group in
            for (position, path) in paths.enumerated() {
                group.addTask { await (position, safetensorsHeader(repoID: repoID, path: path)) }
            }
            return await group.reduce(into: [:]) { byPosition, item in byPosition[item.0] = item.1 }
        }
        var headers: [SafetensorsHeader] = []
        for position in paths.indices {
            switch results[position] ?? nil {
            case .success(let header)?: headers.append(header)
            case .failure(let problem)?: return .failure(problem)
            case nil: continue
            }
        }
        guard let first = headers.first else { return nil }
        return .success(SafetensorsHeader(tensorNames: headers.flatMap(\.tensorNames).sorted(), metadata: first.metadata))
    }

    private func extraTensorCounts(repoID: String, files: [RepoFile], weightMap: [String: String]) async -> [String: Int] {
        let referenced = Set(weightMap.values)
        let extras = files.filter { $0.isSafetensors && $0.isTopLevel && !referenced.contains($0.path) }
            .prefix(Self.maxExtraHeaders).map(\.path)
        return await withTaskGroup(of: (String, Int?).self) { group in
            for path in extras {
                group.addTask { await tensorCount(repoID: repoID, path: path) }
            }
            return await group.reduce(into: [:]) { counts, item in counts[item.0] = item.1 }
        }
    }

    private func tensorCount(repoID: String, path: String) async -> (String, Int?) {
        guard case .success(let header)? = await safetensorsHeader(repoID: repoID, path: path) else { return (path, nil) }
        return (path, header.tensorNames.count)
    }

    private func get(_ url: URL, subject: FetchedResource) async -> Result<Data, FetchProblem> {
        var request = URLRequest(url: url, timeoutInterval: requestTimeout)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        authorizer.authorize(&request)
        do {
            let (data, response) = try await transport.data(for: request)
            if let problem = Self.problem(for: response, subject: subject) { return .failure(problem) }
            return .success(data)
        } catch {
            return .failure(FetchProblem(subject: subject, statusCode: nil, message: error.localizedDescription))
        }
    }

    /// The first `byteCount` bytes of a file. Streams and stops at the limit, so a server that ignores `Range`
    /// still costs only `byteCount` bytes.
    private func readPrefix(_ url: URL, byteCount: Int, subject: FetchedResource) async -> Result<Data, FetchProblem> {
        var request = URLRequest(url: url, timeoutInterval: requestTimeout)
        request.setValue("bytes=0-\(max(byteCount, 1) - 1)", forHTTPHeaderField: "Range")
        authorizer.authorize(&request)
        do {
            let (stream, response) = try await transport.bytes(for: request)
            if let problem = Self.problem(for: response, subject: subject) { return .failure(problem) }
            return .success(try await Self.collect(stream, limit: byteCount))
        } catch {
            return .failure(FetchProblem(subject: subject, statusCode: nil, message: error.localizedDescription))
        }
    }

    private static func collect(_ stream: URLSession.AsyncBytes, limit: Int) async throws -> Data {
        var buffer = Data()
        buffer.reserveCapacity(limit)
        for try await byte in stream {
            buffer.append(byte)
            if buffer.count >= limit { break }
        }
        return buffer
    }

    static func problem(for response: URLResponse, subject: FetchedResource) -> FetchProblem? {
        guard let http = response as? HTTPURLResponse else {
            return FetchProblem(subject: subject, statusCode: nil, message: "\(subject.rawValue): no HTTP response")
        }
        let code = http.statusCode
        let reason: String
        switch HuggingFaceRepo.RepoError(status: code) {
        case nil: return nil
        case .authRequired?: reason = "HTTP \(code) — gated or private, a Hugging Face token is required"
        case .notFound?: reason = "not found (HTTP 404)"
        default: reason = "HTTP \(code)"
        }
        return FetchProblem(subject: subject, statusCode: code, message: "\(subject.rawValue): \(reason)")
    }
}

extension FetchProblem: Error {}
