import Foundation
import Dispatch
import TonamoxCore
import LocalLLMClientCore
#if os(iOS) || os(tvOS)
import UIKit
#endif

// MARK: - ModelLoadState

/// Observable load state for a single model, published by ``ModelManager``.
///
/// Use ``isActive`` to determine whether a progress indicator should be shown.
public enum ModelLoadState: Equatable {
    /// No load has been requested yet (or the model was evicted).
    case idle
    /// The model weights are being downloaded; `progress` contains a human-readable status string.
    case downloading(progress: String)
    /// Download complete — the model is being loaded into memory.
    case loading
    /// The model is loaded and ready for inference.
    case ready
    /// Loading failed with the given error description.
    case failed(String)

    /// `true` when the model is actively downloading or loading.
    public var isActive: Bool {
        switch self {
            case .downloading, .loading: return true
            default: return false
        }
    }
}

// MARK: - ModelManager

/// Centralized model lifecycle manager with LRU caching and memory-pressure eviction.
///
/// `ModelManager` is the recommended way to load models across your app.
/// It provides four key guarantees:
///
/// 1. **LRU cache** — keeps up to ``memoryBudget`` instances in RAM, one per model and sampling setting,
///    automatically evicting the least-recently-used when the budget is exceeded.
/// 2. **In-flight deduplication** — concurrent requests for the same model and sampling
///    share a single download/load `Task`, avoiding redundant work.
/// 3. **Memory-pressure handling** — listens for OS memory warnings
///    (`DispatchSource` + `UIApplication` notifications) and evicts models proactively.
/// 4. **Dual-backend support** — transparently routes MLX and GGUF models
///    to the appropriate backend via ``BackendRouter``.
///
/// ```swift
/// // Load any model (MLX or GGUF — routing is automatic)
/// let llm = try await ModelManager.shared.load(.qwen3_1_7b)
/// let llm7b = try await ModelManager.shared.load(.llama3_1_8b_gguf)
///
/// // Observe per-model state in SwiftUI
/// @ObservedObject var manager = ModelManager.shared
/// let state = manager.state(for: .llama3_1_8b_gguf) // .idle | .downloading | .loading | .ready | .failed
/// ```
@MainActor
public final class ModelManager: ObservableObject {

    // MARK: - Singleton

    public static let shared = ModelManager()

    // MARK: - Published state

    @Published public private(set) var states: [Model: ModelLoadState] = [:]

    // MARK: - Private storage

    /// One cache slot per model and sampling: a GGUF's sampler is fixed when its context is created.
    private struct InstanceKey: Hashable {
        let base: Model
        let sampling: SamplingSettings
    }

    private var cache:    [InstanceKey: Tonamox] = [:]
    private var lruOrder: [InstanceKey] = []

    /// In-flight load tasks — deduplicates concurrent requests for the same model and sampling.
    private var inFlight: [InstanceKey: Task<Tonamox, Error>] = [:]

    /// GGUF downloader instance (shared across downloads).
    public let ggufDownloader = GGUFModelDownloader()

    /// Max models to keep in RAM simultaneously (adaptive, based on device RAM).
    public private(set) var memoryBudget: Int

    private init() {
        self.memoryBudget = Self.detectMemoryBudget()
        observeMemoryPressure()
    }

    // MARK: - Memory pressure

    /// Registers for OS memory warnings and evicts LRU models to free RAM.
    private func observeMemoryPressure() {
        #if os(iOS) || os(tvOS)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleMemoryPressure()
            }
        }
        #endif

        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: .main
        )
        source.setEventHandler { [weak self] in
            Task { @MainActor [weak self] in
                self?.handleMemoryPressure()
            }
        }
        source.resume()
        memoryPressureSource = source
    }

    private var memoryPressureSource: Any?

    private func handleMemoryPressure() {
        evictAllButMostRecent()
    }

    /// Evict every loaded model except the most-recently-used one, freeing RAM while keeping
    /// the active model warm. Owned here because it needs the private LRU order; called both
    /// on OS memory pressure and by ``BackgroundLifecycle`` when backgrounding aggressively.
    public func evictAllButMostRecent() {
        while cache.count > 1, let lru = lruOrder.last {
            evict(lru)
        }
    }

    // MARK: - Public API

    /// Load a model and return the instance.
    /// Returns immediately from cache if already loaded with the same sampling.
    /// Deduplicates concurrent loads — multiple callers requesting the same model and sampling
    /// share a single in-flight load Task.
    ///
    /// For GGUF models, this automatically downloads the GGUF file from HuggingFace
    /// if not already present, then loads via the llama.cpp backend.
    ///
    /// `temperature` and `topP` hold for the instance's lifetime; `nil` keeps the backend default, and temperature
    /// 0 is greedy. Each distinct setting is its own cached instance, a separate load of the model. `tools` need
    /// a GGUF model or an MLX text model. Both are checked before anything is downloaded and throw
    /// ``ModelConfigurationError``. An MLX instance with tools runs on the weights of a cached instance of the same
    /// model when there is one, instead of loading them again.
    public func load(
        _ model: Model,
        tools: [any LLMTool] = [],
        temperature: Float? = nil,
        topP: Float? = nil,
        onProgress: (@MainActor (String) -> Void)? = nil
    ) async throws -> Tonamox {
        try BackendRouter.requireToolSupport(tools, for: model)
        let sampling = try SamplingSettings.resolve(temperature: temperature, topP: topP, for: model)

        // Tool-enabled sessions are always freshly built and never cached/deduped — a function-calling
        // instance must not be served from (or pollute) the plain shared cache.
        guard tools.isEmpty else {
            return try await performLoad(model, sampling: sampling, tools: tools, onProgress: onProgress)
        }
        let key = InstanceKey(base: model, sampling: sampling)

        // Return cached immediately
        if let cached = cache[key] {
            touch(key)
            return cached
        }

        // Deduplicate: if this model is already loading with this sampling, join the existing Task
        if let existing = inFlight[key] {
            return try await existing.value
        }

        // Create a new load Task and register it
        let task = Task<Tonamox, Error> { @MainActor [weak self] in
            guard let self else { throw TonamoxError.modelNotLoaded }
            defer { self.inFlight[key] = nil }
            return try await self.performLoad(model, sampling: sampling, onProgress: onProgress)
        }
        inFlight[key] = task
        return try await task.value
    }

    /// Current load state for a model. While another sampling setting of a loaded model loads, this reports that
    /// load's progress, and returns to `.ready` if it fails or is cancelled.
    public func state(for model: Model) -> ModelLoadState {
        states[model] ?? .idle
    }

    /// Release a specific instance obtained from `load`: unload its llama.cpp context, and — only if it is the
    /// SHARED cached instance for its model — drop the cache entry so the next `load` rebuilds instead of serving
    /// a torn-down zombie. A tool-enabled instance is never cached, so this just unloads it (identity check via
    /// `===` distinguishes the two). Use this from `TonamoxSession` teardown instead of a bare `unload()`.
    public func release(_ instance: Tonamox) {
        instance.engine.unload()
        let key = InstanceKey(base: instance.model, sampling: instance.sampling)
        if cache[key] === instance {
            drop(key)
        }
    }

    /// Evict a model from memory, at every sampling setting it is loaded with.
    public func evict(_ model: Model) {
        for key in cache.keys where key.base == model {
            evict(key)
        }
        states[model] = .idle
    }

    /// Cancels an in-flight load, including the network download (cancelling the load Task alone doesn't reach
    /// the URLSession transfer). No-op if already loaded or never started.
    public func cancelLoad(_ model: Model) {
        ggufDownloader.cancel()
        for (key, task) in inFlight where key.base == model {
            task.cancel()
            inFlight[key] = nil
        }
        settle(model, otherwise: .idle)
    }

    /// Whether the model download (if any) is currently paused.
    public var isDownloadPaused: Bool { ggufDownloader.isPaused }
    /// Whether a model file is actively downloading (false while paused).
    public var isDownloading: Bool { ggufDownloader.isDownloading }

    /// Pause the in-flight model download, retaining progress. The load Task stays suspended in `download(...)`.
    public func pauseDownload() { ggufDownloader.pause() }
    /// Resume a paused model download from where it stopped.
    public func resumeDownload() { ggufDownloader.resume() }

    /// Evict all loaded models.
    public func evictAll() {
        for (_, instance) in cache {
            instance.engine.unload()
        }
        cache     = [:]
        lruOrder  = []
        for key in states.keys { states[key] = .idle }
    }

    /// Whether the model is currently loaded and ready, at any sampling setting.
    public func isReady(_ model: Model) -> Bool {
        isLoaded(model)
    }

    /// Whether the model is already loaded in memory, at any sampling setting.
    public func isLoaded(_ model: Model) -> Bool {
        cache.keys.contains { $0.base == model }
    }

    /// The ``BackendKind`` that will be used for a model on this device.
    public func recommendedBackend(for model: Model) -> BackendKind {
        BackendRouter.recommendedBackend(for: model)
    }

    // MARK: - Load

    private func performLoad(
        _ model: Model,
        sampling: SamplingSettings,
        tools: [any LLMTool] = [],
        onProgress: (@MainActor (String) -> Void)? = nil
    ) async throws -> Tonamox {
        if let shared = try await toolInstanceOnLoadedWeights(model, sampling: sampling, tools: tools) {
            return shared
        }
        // Refuse a model that can't fit even with streaming — BEFORE downloading gigabytes or
        // risking a jetsam kill mid-load. HardwareAnalyzer only *flags* .tooLarge; this is where
        // it's enforced, together with the models already resident.
        let residents = lruOrder.filter { cache[$0] != nil }
        guard let evictions = Self.evictionsBeforeLoading(
            model, residents: residents.map(\.base), budget: memoryBudget, profile: .current(),
            residentsCountAgainstProfile: Self.profileIsMemoryLimit
        ) else {
            let msg = "\(model.displayName) needs more memory than this device has, even with streaming."
            settle(model, otherwise: .failed(msg))
            throw TonamoxError.modelTooLarge(msg)
        }
        for lru in residents.suffix(evictions).reversed() {
            evict(lru)
        }

        let progress: @MainActor (String) -> Void = { [weak self] p in
            self?.states[model] = .downloading(progress: p)
            onProgress?(p)
        }
        // engine.load reports both download progress (MLX fetches inside it) and load messages ("… ready"); only the
        // former is a download, so the state no longer flips back to .downloading after .loading.
        let engineProgress: @MainActor (String) -> Void = { [weak self] p in
            self?.states[model] = Self.isDownloadMessage(p) ? .downloading(progress: p) : .loading
            onProgress?(p)
        }

        do {
            // GGUF models: download first if needed
            if model.format == .gguf {
                states[model] = .downloading(progress: "Checking \(model.displayName)...")
                _ = try await ggufDownloader.download(model: model, onProgress: progress)
            }

            states[model] = .loading

            // Create Tonamox instance — BackendRouter picks the right backend; `tools` (if any) enable
            // GGUF function-calling.
            let instance = Tonamox(model: model, sampling: sampling, tools: tools)
            try await instance.engine.load(onProgress: engineProgress)

            // Only plain (tool-less) instances are cached; tool-enabled ones are caller-owned.
            if tools.isEmpty {
                let key = InstanceKey(base: model, sampling: sampling)
                cache[key] = instance
                touch(key)
            }
            states[model] = .ready
            return instance
        } catch is CancellationError {
            settle(model, otherwise: .idle)   // user cancelled the download/load — not a failure
            throw CancellationError()
        } catch {
            settle(model, otherwise: .failed(error.localizedDescription))
            throw error
        }
    }

    /// An MLX instance with `tools` on the weights a cached instance of `model` holds: nothing to download, load or
    /// evict. It keeps those weights in memory after that instance is evicted, until it is released too.
    private func toolInstanceOnLoadedWeights(_ model: Model, sampling: SamplingSettings,
                                             tools: [any LLMTool]) async throws -> Tonamox? {
        #if canImport(MLXLLM)
        guard !tools.isEmpty,
              let loaded = cache.first(where: { $0.key.base == model })?.value.engine.backend as? MLXBackend,
              let backend = loaded.sharingWeights(sampling: sampling, tools: tools) else { return nil }
        let instance = Tonamox(engine: TonamoxEngine(model: model, backend: backend), sampling: sampling)
        try await instance.engine.load { _ in }
        return instance
        #else
        return nil
        #endif
    }

    /// A load that fails or is cancelled leaves the model `.ready` while another setting of it is still cached.
    /// Backends prefix download progress with "Downloading"; every other load message means the weights are on disk.
    nonisolated static func isDownloadMessage(_ message: String) -> Bool {
        message.hasPrefix("Downloading")
    }

    private func settle(_ model: Model, otherwise state: ModelLoadState) {
        states[model] = isLoaded(model) ? .ready : state
    }

    // MARK: - LRU

    private func touch(_ key: InstanceKey) {
        lruOrder.removeAll { $0 == key }
        lruOrder.insert(key, at: 0)
    }

    private func evict(_ key: InstanceKey) {
        cache[key]?.engine.unload()
        drop(key)
    }

    /// Forgets one cached instance without unloading it. The model stays `.ready` while another setting of it is
    /// still cached.
    private func drop(_ key: InstanceKey) {
        cache[key] = nil
        lruOrder.removeAll { $0 == key }
        if !isLoaded(key.base) {
            states[key.base] = .idle
        }
    }

    // MARK: - Memory detection

    /// On iOS ``HardwareProfile/current()`` reports the app's memory limit, which does not shrink while models are
    /// loaded; on macOS it reads free memory, which already leaves them out.
    #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
    static let profileIsMemoryLimit = true
    #else
    static let profileIsMemoryLimit = false
    #endif

    /// How many of `residents` (most recently used first) to evict before loading `model`, or `nil` when it does
    /// not fit even alone. The ``memoryBudget`` count leaves room for the new model; when
    /// `residentsCountAgainstProfile`, the estimated memory of the models that stay is also taken off the profile,
    /// evicting the least recently used until `model` is no longer ``ModelFitLevel/tooLarge`` beside them.
    nonisolated static func evictionsBeforeLoading(
        _ model: Model,
        residents: [Model],
        budget: Int,
        profile: HardwareProfile,
        residentsCountAgainstProfile: Bool
    ) -> Int? {
        func fits(beside kept: ArraySlice<Model>) -> Bool {
            let used = residentsCountAgainstProfile ? kept.reduce(0) { $0 + $1.estimatedRuntimeMemoryGB } : 0
            let room = HardwareProfile(totalMemoryGB: profile.totalMemoryGB,
                                       availableMemoryGB: max(0, profile.availableMemoryGB - used),
                                       deviceName: profile.deviceName,
                                       memoryBandwidthGBs: profile.memoryBandwidthGBs)
            return HardwareAnalyzer.assess(model, profile: room).fitLevel != .tooLarge
        }
        guard fits(beside: []) else { return nil }
        var kept = min(residents.count, max(budget - 1, 0))
        while kept > 0, !fits(beside: residents.prefix(kept)) {
            kept -= 1
        }
        return residents.count - kept
    }

    private static func detectMemoryBudget() -> Int {
        // Unknown → keep exactly one model resident. The OS reports unknown when the process is at
        // its jetsam limit, which is the worst possible moment to cache a second model.
        guard let available = availableMemoryBytes() else { return 1 }
        let usable    = max(0, available - 2 * 1024 * 1024 * 1024) // reserve 2 GB for OS
        let budget    = max(1, Int(usable / (1_500 * 1024 * 1024))) // ~1.5 GB per model
        return min(budget, 4)
    }

    /// Delegates to ``HardwareProfile/availableMemoryBytes()`` — the single place that asks the OS.
    private static func availableMemoryBytes() -> Int? {
        HardwareProfile.availableMemoryBytes()
    }
}

// MARK: - ModelLoadingOverlay (SwiftUI)

#if canImport(SwiftUI)
import SwiftUI

/// Drop-in overlay that shows model load progress.
/// Renders nothing when the model is idle or ready.
public struct ModelLoadingOverlay: View {
    let model: Model
    @ObservedObject private var manager = ModelManager.shared

    public init(model: Model) { self.model = model }

    public var body: some View {
        let s = manager.state(for: model)
        if s.isActive {
            VStack(spacing: 12) {
                ProgressView().scaleEffect(1.2)
                switch s {
                    case .downloading(let p):
                        Text(p)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    case .loading:
                        Text("Loading \(model.displayName)…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    default:
                        EmptyView()
                }
            }
            .padding(24)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
    }
}
#endif
