import Foundation
import Dispatch
import AuraCore
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
/// 1. **LRU cache** — keeps up to ``memoryBudget`` models in RAM,
///    automatically evicting the least-recently-used when the budget is exceeded.
/// 2. **In-flight deduplication** — concurrent requests for the same model
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

    private var cache:    [Model: AuraLocal] = [:]
    private var lruOrder: [Model] = []

    /// In-flight load tasks — deduplicates concurrent requests for the same model.
    private var inFlight: [Model: Task<AuraLocal, Error>] = [:]

    /// GGUF downloader instance (shared across downloads).
    /// Only available on real devices — GGUF is not supported on the iOS Simulator.
    #if !targetEnvironment(simulator)
    public let ggufDownloader = GGUFModelDownloader()
    #endif

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
        while cache.count > 1, let lru = lruOrder.last {
            evict(lru)
        }
    }

    // MARK: - Public API

    /// Load a model and return the instance.
    /// Returns immediately from cache if already loaded.
    /// Deduplicates concurrent loads — multiple callers requesting the same model
    /// share a single in-flight load Task.
    ///
    /// For GGUF models, this automatically downloads the GGUF file from HuggingFace
    /// if not already present, then loads via the llama.cpp backend.
    public func load(
        _ model: Model,
        tools: [any LLMTool] = [],
        onProgress: (@MainActor (String) -> Void)? = nil
    ) async throws -> AuraLocal {
        // Tool-enabled sessions are always freshly built and never cached/deduped — a function-calling
        // instance must not be served from (or pollute) the plain shared cache keyed only by model.
        guard tools.isEmpty else {
            return try await performLoad(model, tools: tools, onProgress: onProgress)
        }

        // Return cached immediately
        if let cached = cache[model] {
            touch(model)
            return cached
        }

        // Deduplicate: if this model is already loading, join the existing Task
        if let existing = inFlight[model] {
            return try await existing.value
        }

        // Create a new load Task and register it
        let task = Task<AuraLocal, Error> { @MainActor [weak self] in
            guard let self else { throw AuraError.modelNotLoaded }
            defer { self.inFlight[model] = nil }
            return try await self.performLoad(model, onProgress: onProgress)
        }
        inFlight[model] = task
        return try await task.value
    }

    /// Current load state for a model.
    public func state(for model: Model) -> ModelLoadState {
        states[model] ?? .idle
    }

    /// Release a specific instance obtained from `load`: unload its llama.cpp context, and — only if it is the
    /// SHARED cached instance for its model — drop the cache entry so the next `load` rebuilds instead of serving
    /// a torn-down zombie. A tool-enabled instance is never cached, so this just unloads it (identity check via
    /// `===` distinguishes the two). Use this from `AuraSession` teardown instead of a bare `unload()`.
    public func release(_ instance: AuraLocal) {
        instance.engine.unload()
        if cache[instance.model] === instance {
            cache[instance.model] = nil
            lruOrder.removeAll { $0 == instance.model }
            states[instance.model] = .idle
        }
    }

    /// Evict a specific model from memory.
    public func evict(_ model: Model) {
        if let instance = cache[model] {
            instance.engine.unload()
        }
        cache[model] = nil
        lruOrder.removeAll { $0 == model }
        states[model] = .idle
    }

    /// Cancels an in-flight load, including the network download (cancelling the load Task alone doesn't reach
    /// the URLSession transfer). No-op if already loaded or never started.
    public func cancelLoad(_ model: Model) {
        ggufDownloader.cancel()
        if let task = inFlight[model] {
            task.cancel()
            inFlight[model] = nil
        }
        states[model] = .idle
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

    /// Whether the model is currently loaded and ready.
    public func isReady(_ model: Model) -> Bool {
        cache[model] != nil
    }

    /// Whether the model is already loaded in memory.
    public func isLoaded(_ model: Model) -> Bool {
        cache[model] != nil
    }

    /// The ``BackendKind`` that will be used for a model on this device.
    public func recommendedBackend(for model: Model) -> BackendKind {
        BackendRouter.recommendedBackend(for: model)
    }

    // MARK: - Load

    private func performLoad(
        _ model: Model,
        tools: [any LLMTool] = [],
        onProgress: (@MainActor (String) -> Void)? = nil
    ) async throws -> AuraLocal {
        // Evict LRU if over budget
        while cache.count >= memoryBudget, let lru = lruOrder.last {
            evict(lru)
        }

        let progress: @MainActor (String) -> Void = { [weak self] p in
            self?.states[model] = .downloading(progress: p)
            onProgress?(p)
        }

        do {
            // GGUF models: download first if needed
            if model.format == .gguf {
                #if !targetEnvironment(simulator)
                states[model] = .downloading(progress: "Checking \(model.displayName)...")
                _ = try await ggufDownloader.download(model: model, onProgress: progress)
                #else
                throw AuraError.invalidResponse("GGUF models are not supported on the iOS Simulator. Use MLX models instead.")
                #endif
            }

            states[model] = .loading

            // Create AuraLocal instance — BackendRouter picks the right backend; `tools` (if any) enable
            // GGUF function-calling.
            let instance = AuraLocal(model: model, tools: tools)
            try await instance.engine.load(onProgress: progress)

            // Only plain (tool-less) instances are cached; tool-enabled ones are caller-owned.
            if tools.isEmpty {
                cache[model] = instance
                touch(model)
            }
            states[model] = .ready
            return instance
        } catch is CancellationError {
            states[model] = .idle   // user cancelled the download/load — not a failure
            throw CancellationError()
        } catch {
            states[model] = .failed(error.localizedDescription)
            throw error
        }
    }

    // MARK: - LRU

    private func touch(_ model: Model) {
        lruOrder.removeAll { $0 == model }
        lruOrder.insert(model, at: 0)
    }

    // MARK: - Memory detection

    private static func detectMemoryBudget() -> Int {
        let available = availableMemoryBytes()
        let usable    = max(0, available - 2 * 1024 * 1024 * 1024) // reserve 2 GB for OS
        let budget    = max(1, Int(usable / (1_500 * 1024 * 1024))) // ~1.5 GB per model
        return min(budget, 4)
    }

    private static func availableMemoryBytes() -> Int {
#if os(iOS) || os(tvOS) || os(watchOS)
        let available = os_proc_available_memory()
        if available > 0 { return Int(available) }
#endif
        return Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.6)
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
