import Foundation

/// The sized ``Model`` behind each download from a search result, so it can be registered again after a relaunch.
/// Files on disk stay the source of truth; a record only describes them.
///
/// Record only models built by ``CatalogEntry/decodedModel()``: a ``Model/fromURL(_:displayName:)`` model has
/// size 0, which the memory guard always rates as fitting.
public struct DownloadRecords: @unchecked Sendable {
    /// Records in `UserDefaults.standard`.
    public static let standard = DownloadRecords()

    static let defaultsKey = "dev.auralocal.downloadRecords"
    /// Where the Model Finder example kept records before they moved to AuraCore.
    static let legacyKey = "downloadRecords"

    private static let encoder = JSONEncoder()
    private static let decoder = JSONDecoder()
    /// Every read-modify-write of the stored dictionary holds this, so concurrent writers don't drop records.
    /// Static because instances on the same `UserDefaults` share the dictionary.
    private static let lock = NSLock()

    // The reason for @unchecked: UserDefaults is thread-safe per call but not marked Sendable in this SDK;
    // updates that span calls hold `lock`.
    private let defaults: UserDefaults

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    // MARK: Records

    /// File identity, not model id: `gguf:<repo>/<file>` or `mlx:<repo>`. Two ids for one file share a key.
    public static func key(for model: Model) -> String {
        switch model.format {
        case .gguf: "gguf:\(model.repoID)/\(model.ggufFilename ?? "")"
        case .mlx: "mlx:\(model.repoID)"
        }
    }

    /// Stores `model` under its file key, replacing any record for the same file. Call it when a download starts,
    /// so a partial file can resume with its size after a relaunch.
    public func remember(_ model: Model) {
        guard let data = try? Self.encoder.encode(model) else { return }
        Self.lock.withLock {
            var stored = all()
            stored[Self.key(for: model)] = data
            defaults.set(stored, forKey: Self.defaultsKey)
        }
    }

    /// Drops the record for `model`'s file, if any. Leaves the files and the registry alone.
    public func forget(_ model: Model) {
        Self.lock.withLock {
            var stored = all()
            guard stored.removeValue(forKey: Self.key(for: model)) != nil else { return }
            defaults.set(stored, forKey: Self.defaultsKey)
        }
    }

    /// Every record that still decodes, sorted by id.
    public var models: [Model] {
        snapshot().values.compactMap(Self.decode).sorted { $0.id < $1.id }
    }

    /// The record for one file; a `nil` filename means the MLX snapshot of `repoID`.
    public func model(repoID: String, ggufFilename: String?) -> Model? {
        let key = ggufFilename.map { "gguf:\(repoID)/\($0)" } ?? "mlx:\(repoID)"
        return Self.decode(snapshot()[key])
    }

    /// Whether a record has `model`'s id. A catalog model whose file was downloaded from search keeps its own id,
    /// so it does not count as recorded.
    public func contains(_ model: Model) -> Bool {
        models.contains { $0.id == model.id }
    }

    // MARK: Registry

    /// Registers each recorded model that is fully on disk, unless a registry entry already points at the same
    /// file, and drops records with nothing on disk (deleted, or purged from Caches before migration). A partial
    /// file keeps its record for Resume.
    /// Returns the models now listed under their own id.
    @discardableResult
    public func registerDownloaded() -> [Model] {
        registerDownloaded(isOnDisk: { $0.isDownloaded }, bytesOnDisk: { ModelDownloads.bytesOnDisk(for: $0) })
    }

    /// The ``registerDownloaded()`` rule for one model, after its download finished. Returns whether it is listed
    /// under its own id.
    @discardableResult
    public func registerIfDownloaded(_ model: Model) -> Bool {
        registerIfDownloaded(model, isOnDisk: { $0.isDownloaded })
    }

    /// Forgets the record and unregisters the model. A no-op for the registry when `model` comes from a catalog.
    public func remove(_ model: Model) {
        forget(model)
        ModelRegistry.shared.unregister(id: model.id)
    }

    @discardableResult
    func registerDownloaded(isOnDisk: (Model) -> Bool, bytesOnDisk: (Model) -> Int64) -> [Model] {
        Self.lock.withLock {
            var stored = all()
            var listed: [Model] = []
            var pruned = false
            for (key, data) in stored {
                guard let model = Self.decode(data) else { continue }
                if isOnDisk(model) {
                    if Self.list(model) { listed.append(model) }
                } else if bytesOnDisk(model) == 0 {
                    stored[key] = nil
                    pruned = true
                }
            }
            if pruned { defaults.set(stored, forKey: Self.defaultsKey) }
            return listed.sorted { $0.id < $1.id }
        }
    }

    func registerIfDownloaded(_ model: Model, isOnDisk: (Model) -> Bool) -> Bool {
        guard isOnDisk(model) else { return false }
        return Self.list(model)
    }

    /// Registers `model` unless the registry already has its file: a catalog entry for the same file already loads
    /// it and shows as downloaded, and a second id would list one file twice in the pickers.
    private static func list(_ model: Model) -> Bool {
        let registry = ModelRegistry.shared
        let fileKey = key(for: model)
        if let existing = registry.models.first(where: { key(for: $0) == fileKey }) {
            return existing.id == model.id
        }
        registry.register(model)
        return true
    }

    // MARK: Storage

    private func snapshot() -> [String: Data] {
        Self.lock.withLock { all() }
    }

    /// Call only while holding `lock`: the one-time legacy copy is a write too.
    private func all() -> [String: Data] {
        if let stored = defaults.dictionary(forKey: Self.defaultsKey) as? [String: Data] { return stored }
        // One-time copy from the old key; the old key stays for older builds of the example.
        guard let legacy = defaults.dictionary(forKey: Self.legacyKey) as? [String: Data] else { return [:] }
        defaults.set(legacy, forKey: Self.defaultsKey)
        return legacy
    }

    private static func decode(_ data: Data?) -> Model? {
        data.flatMap { try? decoder.decode(Model.self, from: $0) }
    }
}
