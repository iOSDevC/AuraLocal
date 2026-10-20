import Foundation

/// The Hugging Face image models the user adopted, so they are listed again after a relaunch. Files on disk stay
/// the source of truth: a record describes a model, ``CoreMLDiffusionModel/isDownloaded`` says whether it is there.
/// The built-in model is always listed and never stored.
public struct CoreMLDiffusionModelStore: @unchecked Sendable {
    /// Records in `UserDefaults.standard`.
    public static let standard = CoreMLDiffusionModelStore()

    static let defaultsKey = "dev.auralocal.coreMLDiffusionModels"

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

    /// Stores `model` under its id, replacing an earlier record of the same folder. Call it when a download
    /// starts, so a partial download is listed after a relaunch. Ignores the built-in model.
    public func remember(_ model: CoreMLDiffusionModel) {
        guard !model.isBuiltIn, let data = try? Self.encoder.encode(model) else { return }
        Self.lock.withLock {
            var stored = records()
            stored[model.id] = data
            defaults.set(stored, forKey: Self.defaultsKey)
        }
    }

    /// Drops the record; leaves the files alone (``CoreMLDiffusionModel/delete()`` removes them).
    public func forget(_ model: CoreMLDiffusionModel) {
        Self.lock.withLock {
            var stored = records()
            guard stored.removeValue(forKey: model.id) != nil else { return }
            defaults.set(stored, forKey: Self.defaultsKey)
        }
    }

    /// The built-in model first, then every record that still decodes, by display name.
    public var all: [CoreMLDiffusionModel] {
        let stored = Self.lock.withLock { records() }.values.compactMap(Self.decode)
            .sorted { ($0.displayName, $0.id) < ($1.displayName, $1.id) }
        return [.stableDiffusion21Base] + stored
    }

    /// The built-in model or a record, by ``CoreMLDiffusionModel/id``.
    public func model(id: String) -> CoreMLDiffusionModel? {
        if id == CoreMLDiffusionModel.stableDiffusion21Base.id { return .stableDiffusion21Base }
        return Self.decode(Self.lock.withLock { records()[id] })
    }

    // MARK: Storage

    /// Call only while holding `lock`.
    private func records() -> [String: Data] {
        defaults.dictionary(forKey: Self.defaultsKey) as? [String: Data] ?? [:]
    }

    private static func decode(_ data: Data?) -> CoreMLDiffusionModel? {
        data.flatMap { try? decoder.decode(CoreMLDiffusionModel.self, from: $0) }
    }
}
