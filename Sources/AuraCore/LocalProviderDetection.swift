import Foundation
import os

// MARK: - Public data model

/// A local inference provider AuraLocal can discover on the machine.
public enum LocalProviderKind: String, Sendable, Codable {
    /// Ollama native API server (default port 11434).
    case ollama
    /// llama.cpp `llama-server`, OpenAI-compatible (default port 8080, `/v1`).
    case llamaServer
}

/// A configurable endpoint to probe for one provider.
public struct LocalProviderEndpoint: Sendable, Hashable {
    public let kind: LocalProviderKind
    /// Base URL, without a trailing slash. Ollama: `http://localhost:11434`.
    /// llama-server: `http://127.0.0.1:8080/v1` (must include the `/v1` segment).
    public let baseURL: URL

    public init(kind: LocalProviderKind, baseURL: URL) {
        self.kind = kind
        self.baseURL = baseURL
    }

    /// Ollama on its default port.
    public static let ollamaDefault = LocalProviderEndpoint(
        kind: .ollama, baseURL: URL(string: "http://localhost:11434")!)

    /// llama.cpp `llama-server` on its default port and `/v1` path.
    public static let llamaDefault = LocalProviderEndpoint(
        kind: .llamaServer, baseURL: URL(string: "http://127.0.0.1:8080/v1")!)

    /// Both default providers.
    public static let defaults: [LocalProviderEndpoint] = [.ollamaDefault, .llamaDefault]
}

/// A model exposed by a detected provider, normalized across the two APIs.
public struct LocalProviderModel: Sendable, Codable, Identifiable, Hashable {
    public var id: String { name }
    /// Ollama tag (e.g. `llama3:latest`) or llama-server model id.
    public let name: String
    /// On-disk size in bytes (Ollama only; `nil` for llama-server).
    public let sizeBytes: Int64?
    /// Quantization level (Ollama only; `nil` otherwise).
    public let quantization: String?
    /// Trained context length (llama-server only; `nil` for Ollama).
    public let contextLength: Int?

    public init(name: String, sizeBytes: Int64? = nil, quantization: String? = nil, contextLength: Int? = nil) {
        self.name = name
        self.sizeBytes = sizeBytes
        self.quantization = quantization
        self.contextLength = contextLength
    }
}

/// The result of probing one provider. Always returned — `reachable` tells the story.
public struct LocalProviderStatus: Sendable, Identifiable, Hashable {
    public var id: LocalProviderKind { kind }
    public let kind: LocalProviderKind
    public let baseURL: URL
    /// `false` when the server is down, timed out, or returned unparseable data.
    public let reachable: Bool
    /// Server version (Ollama only; `nil` for llama-server, which exposes none).
    public let version: String?
    public let models: [LocalProviderModel]

    /// Reachable AND exposing at least one model — safe to offer in a picker.
    public var isAvailable: Bool { reachable && !models.isEmpty }

    public init(
        kind: LocalProviderKind,
        baseURL: URL,
        reachable: Bool,
        version: String?,
        models: [LocalProviderModel]
    ) {
        self.kind = kind
        self.baseURL = baseURL
        self.reachable = reachable
        self.version = version
        self.models = models
    }
}

// MARK: - Detector

/// Discovers running local providers (Ollama, llama.cpp `llama-server`) and the
/// models they expose.
///
/// Detection NEVER throws: a down or unreachable server yields a
/// ``LocalProviderStatus`` with `reachable == false`. This is a deliberate
/// deviation from AuraLocal's `async throws` factories — a stopped server is a
/// normal result here, not an error.
@MainActor
public enum LocalProviderDetector {

    private static let logger = Logger(subsystem: "dev.auralocal", category: "ProviderDetection")

    /// Probe every endpoint concurrently. Returns one status per endpoint, in the
    /// same order as `endpoints`. Never throws.
    public static func detectAll(
        endpoints: [LocalProviderEndpoint] = LocalProviderEndpoint.defaults,
        timeout: TimeInterval = 2.0
    ) async -> [LocalProviderStatus] {
        await withTaskGroup(of: (Int, LocalProviderStatus).self) { group in
            for (index, endpoint) in endpoints.enumerated() {
                group.addTask { (index, await detect(endpoint, timeout: timeout)) }
            }
            var collected = [(Int, LocalProviderStatus)]()
            for await pair in group { collected.append(pair) }
            return collected.sorted { $0.0 < $1.0 }.map(\.1)
        }
    }

    /// Probe a single endpoint. Never throws.
    public static func detect(
        _ endpoint: LocalProviderEndpoint,
        timeout: TimeInterval = 2.0
    ) async -> LocalProviderStatus {
        let session = makeSession(timeout: timeout)
        defer { session.invalidateAndCancel() }
        do {
            switch endpoint.kind {
            case .ollama:      return try await probeOllama(endpoint, session: session)
            case .llamaServer: return try await probeLlamaServer(endpoint, session: session)
            }
        } catch {
            logger.debug("Provider \(endpoint.kind.rawValue, privacy: .public) not detected: \(error.localizedDescription, privacy: .public)")
            return LocalProviderStatus(
                kind: endpoint.kind, baseURL: endpoint.baseURL,
                reachable: false, version: nil, models: [])
        }
    }

    // MARK: - Probes

    private static func probeOllama(_ endpoint: LocalProviderEndpoint, session: URLSession) async throws -> LocalProviderStatus {
        // /api/version confirms the server is up.
        let version = try await getJSON(
            OllamaVersionDTO.self,
            from: endpoint.baseURL.appending(path: "api/version"),
            session: session).version
        // Models are best-effort: a fresh Ollama install has none.
        let models: [LocalProviderModel]
        if let tags = try? await getJSON(OllamaTagsDTO.self, from: endpoint.baseURL.appending(path: "api/tags"), session: session) {
            models = tags.models.map {
                LocalProviderModel(
                    name: $0.name, sizeBytes: $0.size,
                    quantization: $0.details?.quantizationLevel, contextLength: nil)
            }
        } else {
            models = []
        }
        logger.info("Detected Ollama \(version, privacy: .public) with \(models.count) model(s)")
        return LocalProviderStatus(
            kind: .ollama, baseURL: endpoint.baseURL,
            reachable: true, version: version, models: models)
    }

    private static func probeLlamaServer(_ endpoint: LocalProviderEndpoint, session: URLSession) async throws -> LocalProviderStatus {
        // baseURL already includes /v1, so append only "models".
        let list = try await getJSON(
            LlamaModelsDTO.self,
            from: endpoint.baseURL.appending(path: "models"),
            session: session)
        let models = list.data.map {
            LocalProviderModel(
                name: $0.id, sizeBytes: $0.meta?.size,
                quantization: nil, contextLength: $0.meta?.contextLength)
        }
        logger.info("Detected llama-server with \(models.count) model(s)")
        return LocalProviderStatus(
            kind: .llamaServer, baseURL: endpoint.baseURL,
            reachable: true, version: nil, models: models)
    }

    // MARK: - Helpers

    /// A dedicated, short-timeout, ephemeral session so a probe fails fast and
    /// never inherits the download session's long timeout or connectivity wait.
    private static func makeSession(timeout: TimeInterval) -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = timeout
        config.timeoutIntervalForResource = timeout
        config.waitsForConnectivity = false
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config)
    }

    private static func getJSON<T: Decodable>(_ type: T.Type, from url: URL, session: URLSession) async throws -> T {
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(T.self, from: data)
    }
}

// MARK: - Wire DTOs (private)

private struct OllamaVersionDTO: Decodable { let version: String }

private struct OllamaTagsDTO: Decodable { let models: [OllamaModelDTO] }

private struct OllamaModelDTO: Decodable {
    let name: String
    let size: Int64?
    let details: OllamaDetailsDTO?
}

private struct OllamaDetailsDTO: Decodable {
    let quantizationLevel: String?
    enum CodingKeys: String, CodingKey { case quantizationLevel = "quantization_level" }
}

private struct LlamaModelsDTO: Decodable { let data: [LlamaModelDTO] }

private struct LlamaModelDTO: Decodable {
    let id: String
    let meta: LlamaMetaDTO?
}

private struct LlamaMetaDTO: Decodable {
    let contextLength: Int?
    let size: Int64?
    enum CodingKeys: String, CodingKey {
        case contextLength = "n_ctx_train"
        case size
    }
}
