import Foundation

// MARK: - ModelCatalog

/// Top-level JSON envelope for the model catalog.
struct ModelCatalog: Codable, Sendable {
    let schemaVersion: Int
    let models: [Model]
}

// MARK: - ModelRegistry

/// Central registry that owns the canonical list of available models.
///
/// On first access, loads the bundled `models.json` from the framework's
/// resource bundle. Call ``refreshFromRemote(url:)`` to fetch an updated
/// catalog from a GitHub raw URL (or any HTTPS endpoint).
///
/// ```swift
/// // All models (bundled + any remote additions)
/// let models = ModelRegistry.shared.models
///
/// // Lookup by stable id
/// if let m = ModelRegistry.shared.model(id: "qwen3_1_7b") { … }
///
/// // Trigger remote update
/// try await ModelRegistry.shared.refreshFromRemote(
///     url: URL(string: "https://raw.githubusercontent.com/user/AuraLocal/main/models.json")!
/// )
/// ```
public final class ModelRegistry: @unchecked Sendable {

    // MARK: - Singleton

    /// Shared registry instance. Loads bundled catalog on first access.
    public static let shared: ModelRegistry = {
        let registry = ModelRegistry()
        registry.loadBundled()
        return registry
    }()

    // MARK: - State

    /// The current list of registered models.
    public private(set) var models: [Model] = []

    /// Fast lookup by model id.
    private var index: [String: Model] = [:]

    private let lock = NSLock()

    /// The maximum schema version this client understands.
    private static let supportedSchemaVersion = 1

    // MARK: - Init

    private init() {}

    // MARK: - Bundled catalog

    /// Loads models from the bundled `models.json` resource.
    private func loadBundled() {
        guard let url = Bundle.module.url(forResource: "models", withExtension: "json") else {
            assertionFailure("models.json not found in Bundle.module")
            return
        }
        do {
            let data = try Data(contentsOf: url)
            let catalog = try JSONDecoder().decode(ModelCatalog.self, from: data)
            guard catalog.schemaVersion <= Self.supportedSchemaVersion else {
                return // future schema — ignore
            }
            lock.lock()
            models = catalog.models
            rebuildIndex()
            lock.unlock()
        } catch {
            assertionFailure("Failed to decode bundled models.json: \(error)")
        }
    }

    // MARK: - Remote refresh

    /// Fetches an updated catalog from a remote URL and merges new/updated models.
    ///
    /// Remote models are merged additively: new ids are appended, existing ids
    /// are updated. Models present in the bundled catalog but absent from the
    /// remote are **not** removed (offline-safe).
    ///
    /// The remote response is cached to disk with a 24-hour TTL.
    public func refreshFromRemote(url: URL) async throws {
        let (data, _) = try await URLSession.shared.data(from: url)
        let catalog = try JSONDecoder().decode(ModelCatalog.self, from: data)

        guard catalog.schemaVersion <= Self.supportedSchemaVersion else {
            return // future schema — skip silently
        }

        // Cache to disk for offline fallback
        if let cacheURL = remoteCacheURL {
            try? data.write(to: cacheURL, options: .atomic)
        }

        merge(catalog.models)
    }

    /// Loads the cached remote catalog if available and not expired.
    public func loadCachedRemote(maxAge: TimeInterval = 86_400) {
        guard let cacheURL = remoteCacheURL,
              FileManager.default.fileExists(atPath: cacheURL.path) else { return }

        // Check age
        if let attrs = try? FileManager.default.attributesOfItem(atPath: cacheURL.path),
           let modified = attrs[.modificationDate] as? Date,
           Date().timeIntervalSince(modified) > maxAge {
            return // expired
        }

        guard let data = try? Data(contentsOf: cacheURL),
              let catalog = try? JSONDecoder().decode(ModelCatalog.self, from: data),
              catalog.schemaVersion <= Self.supportedSchemaVersion else { return }

        merge(catalog.models)
    }

    // MARK: - Lookup

    /// Look up a model by its stable identifier.
    public func model(id: String) -> Model? {
        lock.lock()
        defer { lock.unlock() }
        return index[id]
    }

    /// Look up a model by its HuggingFace repository ID.
    public func model(repoID: String) -> Model? {
        lock.lock()
        defer { lock.unlock() }
        return models.first { $0.repoID == repoID }
    }

    // MARK: - Registration

    /// Programmatically register a model at runtime (e.g. user-provided).
    public func register(_ model: Model) {
        lock.lock()
        if let idx = models.firstIndex(where: { $0.id == model.id }) {
            models[idx] = model
        } else {
            models.append(model)
        }
        index[model.id] = model
        lock.unlock()
    }

    // MARK: - Private

    private func merge(_ incoming: [Model]) {
        lock.lock()
        for model in incoming {
            if let idx = models.firstIndex(where: { $0.id == model.id }) {
                models[idx] = model
            } else {
                models.append(model)
            }
            index[model.id] = model
        }
        lock.unlock()
    }

    private func rebuildIndex() {
        index = Dictionary(models.map { ($0.id, $0) }, uniquingKeysWith: { _, new in new })
    }

    private var remoteCacheURL: URL? {
        guard let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            return nil
        }
        return caches.appendingPathComponent("aura_models_remote.json")
    }
}
