import SwiftUI
import AuraCore
import AuraImageGen

// MARK: - OnDeviceImageGenView
//
// Text-to-image on this device with Core ML Stable Diffusion (AuraImageGen/CoreMLDiffusionEngine).
// The built-in SD 2.1 base, or a Core ML folder adopted from Hugging Face (ImageModelFinderSheet).
// Only search, inspection and the one-time download touch the network; prompts and images stay on the device.

struct OnDeviceImageGenView: View {
    @Bindable var state: ImageGenState
    @State private var confirmingDownload = false
    @State private var confirmingDelete = false
    @State private var showsAdvanced = false
    @State private var showsFinder = false

    var body: some View {
        Form {
            modelSection
            if state.isDownloaded && state.unavailability == nil {
                promptSection
                advancedSection
                generateSection
            }
            FailureSection(state: state)
            ResultsSection(state: state)
        }
        .formStyle(.grouped)
        .onAppear { state.refreshDisk() }
        .alert("Download the image model?", isPresented: $confirmingDownload) {
            Button("Cancel", role: .cancel) {}
            Button("Download") { state.startDownload() }
        } message: {
            Text("About \(downloadSize) from huggingface.co (\(state.selectedModel.repoID)). No account needed. "
                 + "License: \(state.selectedModel.licenseName), which may restrict some uses; read it on the "
                 + "model card.")
        }
        .sheet(item: $state.selected) { GeneratedImageDetail(item: $0) }
        .sheet(isPresented: $showsFinder) { ImageModelFinderSheet(state: state) }
    }

    // MARK: Model card

    private var modelSection: some View {
        Section("Model") {
            Picker("Model", selection: modelSelection) {
                ForEach(state.availableModels) { model in
                    Text("\(state.title(for: model)) · \(state.isDownloaded(model) ? "Ready" : "Not downloaded")")
                        .tag(model.id)
                }
            }
            .disabled(state.isGenerating || state.isDownloading)
            VStack(alignment: .leading, spacing: 4) {
                Text(modelCaption).font(.caption).foregroundStyle(.secondary)
                Link("License: \(state.selectedModel.licenseName)", destination: state.selectedModel.licenseURL)
                    .font(.caption)
            }
            Button {
                showsFinder = true
            } label: {
                Label("Find on Hugging Face…", systemImage: "magnifyingglass")
            }
            .disabled(state.isGenerating || state.isDownloading)
            Label("Runs on this device. Your prompts and images never leave it.", systemImage: "lock.shield")
                .font(.callout)
            downloadControls
        }
    }

    private var modelSelection: Binding<String> {
        Binding {
            state.selectedModel.id
        } set: { id in
            if let model = state.availableModels.first(where: { $0.id == id }) { state.select(model) }
        }
    }

    /// Side, size, compute unit and where it comes from.
    private var modelCaption: String {
        let model = state.selectedModel
        let compute = model.preferredComputeUnits == .neuralEngine ? "Neural Engine" : "GPU"
        let origin = model.isBuiltIn ? "built in" : "\(model.repoID) · \(model.folder)"
        return "\(model.imageSide)×\(model.imageSide) · \(downloadSize) · \(compute) · \(origin)"
    }

    private var downloadSize: String {
        ByteCountFormatter.string(fromByteCount: state.selectedModel.approximateDownloadBytes, countStyle: .file)
    }

    @ViewBuilder private var downloadControls: some View {
        if let unavailability = state.unavailability {
            Label {
                VStack(alignment: .leading, spacing: 4) {
                    Text(unavailability.errorDescription ?? "")
                    if let recovery = unavailability.recoverySuggestion {
                        Text(recovery).font(.caption)
                    }
                }
            } icon: {
                Image(systemName: "slash.circle")
            }
            .foregroundStyle(.orange)
            if state.isDownloaded || !state.selectedModel.isBuiltIn { deleteButton }
        } else if state.isDownloading {
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(state.downloadedBytes), total: Double(max(state.downloadTotal, 1)))
                Text("\(ByteCountFormatter.string(fromByteCount: state.downloadedBytes, countStyle: .file)) of "
                     + "\(ByteCountFormatter.string(fromByteCount: state.downloadTotal, countStyle: .file))")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Button("Cancel download", role: .cancel) { state.cancelDownload() }
            Text("The partial download is kept; Download resumes it.")
                .font(.caption).foregroundStyle(.secondary)
        } else if state.isDownloaded {
            HStack {
                Label("Ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                Spacer()
                deleteButton
            }
        } else {
            Button {
                confirmingDownload = true
            } label: {
                Label("Download…", systemImage: "arrow.down.circle")
            }
            if let free = state.freeDiskBytes {
                Text("\(ByteCountFormatter.string(fromByteCount: free, countStyle: .file)) free on this device.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !state.selectedModel.isBuiltIn { deleteButton }
        }
    }

    /// "Remove" for a Hugging Face model that is not (fully) downloaded, so its record and partial files can go too.
    private var deleteButton: some View {
        let action = state.isDownloaded ? "Delete" : "Remove"
        return Button(action, role: .destructive) { confirmingDelete = true }
            .disabled(state.isGenerating)
            .confirmationDialog("\(action) \(state.title(for: state.selectedModel))?", isPresented: $confirmingDelete,
                                titleVisibility: .visible) {
                Button(action, role: .destructive) { Task { await state.deleteModel() } }
            } message: {
                Text(deleteMessage)
            }
    }

    private var deleteMessage: String {
        let cache = " Core ML may keep a compiled copy in the app's caches until the system clears them."
        if !state.isDownloaded {
            return "Removes it from the list and deletes any partial download; the built-in model is selected."
        }
        return state.selectedModel.isBuiltIn
            ? "Frees the \(downloadSize) download. Generating images again needs the full download." + cache
            : "Frees the \(downloadSize) download and removes it from the list; the built-in model is selected." + cache
    }

    // MARK: Prompt

    private var promptSection: some View {
        Section("Prompt") {
            TextField("Describe the image", text: $state.prompt, axis: .vertical)
                .lineLimit(2...6)
            Picker("Style preset", selection: $state.style) {
                ForEach(StylePreset.allCases) { Text($0.rawValue).tag($0) }
            }
            if state.style != .plain {
                Text("Adds “\(state.style.suffix)” to the prompt.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var advancedSection: some View {
        Section {
            DisclosureGroup("Advanced", isExpanded: $showsAdvanced) {
                TextField("Negative prompt (what to avoid)", text: $state.negativePrompt, axis: .vertical)
                    .lineLimit(1...3)
                Stepper("Steps: \(state.steps)", value: $state.steps, in: 1...50)
                Stepper("Guidance: \(state.guidance.formatted(.number.precision(.fractionLength(1))))",
                        value: $state.guidance, in: 1...20, step: 0.5)
                seedField
                Stepper("Images per run: \(state.imageCount)", value: $state.imageCount, in: 1...4)
                Picker("Compute", selection: $state.computeUnits) {
                    Text("Neural Engine").tag(DiffusionComputeUnits.neuralEngine)
                    Text("GPU").tag(DiffusionComputeUnits.gpu)
                }
                .disabled(state.isGenerating)
                Picker("Memory", selection: $state.memoryMode) {
                    Text("Automatic").tag(DiffusionMemoryMode.automatic)
                    Text("Keep model loaded").tag(DiffusionMemoryMode.resident)
                    Text("Low memory").tag(DiffusionMemoryMode.lowMemory)
                }
                .disabled(state.isGenerating)
                Text("Keep model loaded makes repeat runs faster but holds about 2 GB; low memory loads each "
                     + "stage on demand.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var seedField: some View {
        #if os(iOS)
        TextField("Seed (blank = random)", text: $state.seedText)
            .keyboardType(.numberPad)
        #else
        TextField("Seed (blank = random)", text: $state.seedText)
        #endif
    }

    // MARK: Generate

    private var generateSection: some View {
        Section {
            if state.isGenerating {
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: state.stepFraction)
                    Text(state.phaseText).font(.caption).foregroundStyle(.secondary)
                }
                Button("Cancel", role: .cancel) { state.cancelGeneration() }
            } else {
                Button {
                    state.generate()
                } label: {
                    Label("Generate", systemImage: "wand.and.stars")
                }
                .disabled(state.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                if !state.phaseText.isEmpty {
                    Text(state.phaseText).font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("Generating unloads the chat model first to free memory.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

// MARK: - Shared sections

/// The last error, with its recovery suggestion.
struct FailureSection: View {
    let state: ImageGenState

    var body: some View {
        if let failure = state.failure {
            Section {
                Label {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(failure.message).textSelection(.enabled)
                        if let recovery = failure.recovery {
                            Text(recovery).font(.caption)
                        }
                    }
                } icon: {
                    Image(systemName: "xmark.octagon")
                }
                .foregroundStyle(.red)
            }
        }
    }
}

/// The session's images, newest first; tap one for details and Share.
struct ResultsSection: View {
    let state: ImageGenState
    private static let columns = [GridItem(.adaptive(minimum: 150), spacing: 8)]

    var body: some View {
        if !state.results.isEmpty {
            Section {
                LazyVGrid(columns: Self.columns, spacing: 8) {
                    ForEach(state.results) { item in
                        Button {
                            state.selected = item
                        } label: {
                            thumbnail(item)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(item.prompt)
                    }
                }
                .padding(.vertical, 4)
            } header: {
                HStack {
                    Text("Results")
                    Spacer()
                    Button("Clear", role: .destructive) { state.clearResults() }
                        .font(.caption)
                }
            }
        }
    }

    private func thumbnail(_ item: GeneratedImage) -> some View {
        Color.clear
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                Image(platformImage: item.image)
                    .resizable()
                    .scaledToFill()
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

/// One image with what produced it, and Share. Saving goes through the share sheet's "Save Image", which needs
/// the NSPhotoLibraryAddUsageDescription key set in the project.
struct GeneratedImageDetail: View {
    let item: GeneratedImage
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Image(platformImage: item.image)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    Text(item.prompt).textSelection(.enabled)
                    if !item.settings.isEmpty {
                        Text(item.settings).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Label(item.provenance, systemImage: item.leftDevice ? "cloud" : "iphone")
                        .font(.callout)
                    Text(item.engineName).font(.caption).foregroundStyle(.secondary)
                    ShareLink(item: item.fileURL) {
                        Label("Share", systemImage: "square.and.arrow.up")
                    }
                }
                .padding()
            }
            .navigationTitle("Image")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
        #if os(macOS)
        .frame(minWidth: 480, minHeight: 560)
        #endif
    }
}

// MARK: - Style presets

/// Prompt suffixes, not a model feature: the model only ever sees the final text.
enum StylePreset: String, CaseIterable, Identifiable {
    case plain = "None"
    case photo = "Photo"
    case illustration = "Illustration"
    case watercolor = "Watercolor"
    case sketch = "Sketch"
    case pixelArt = "Pixel art"

    var id: Self { self }

    var suffix: String {
        switch self {
        case .plain: ""
        case .photo: "professional photograph, natural light, sharp focus, high detail"
        case .illustration: "digital illustration, clean lines, vibrant colors"
        case .watercolor: "watercolor painting, soft washes, paper texture"
        case .sketch: "pencil sketch, graphite, cross-hatching, monochrome"
        case .pixelArt: "pixel art, 16-bit, limited palette"
        }
    }

    func apply(to prompt: String) -> String {
        let base = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return suffix.isEmpty || base.isEmpty ? base : "\(base), \(suffix)"
    }
}

// MARK: - Results

struct GeneratedImage: Identifiable {
    let id = UUID()
    let image: PlatformImage
    let fileURL: URL
    let prompt: String
    /// Seed, steps and timing; empty when the engine does not report them.
    let settings: String
    let engineName: String
    let provenance: String
    let leftDevice: Bool
}

struct FailureNotice {
    let message: String
    let recovery: String?
}

// MARK: - ImageGenState

/// Model choice, download, generation and the shared results grid for the Image tab.
@Observable
final class ImageGenState {
    private static let selectedModelKey = "auraExample.imageModelID"
    private let store = CoreMLDiffusionModelStore.standard

    // Model
    private(set) var selectedModel: CoreMLDiffusionModel
    private(set) var availableModels: [CoreMLDiffusionModel]
    private var downloadedIDs: Set<String> = []
    /// Set while `select` applies a model's compute unit, so that change does not replace the engine twice.
    @ObservationIgnored private var isSelecting = false

    // Download
    private(set) var isDownloaded = false
    private(set) var downloadedBytes: Int64 = 0
    private(set) var downloadTotal: Int64 = 0
    private(set) var freeDiskBytes: Int64?
    private var downloadTask: Task<Void, Never>?
    var isDownloading: Bool { downloadTask != nil }

    // Inputs
    var prompt = "a lighthouse on a cliff at sunset, dramatic sky"
    var negativePrompt = ""
    var style = StylePreset.plain
    var steps = 20
    var guidance = 7.5
    var seedText = ""
    var imageCount = 1
    var computeUnits = DiffusionComputeUnits.neuralEngine {
        didSet { if computeUnits != oldValue && !isSelecting { replaceEngine() } }
    }
    var memoryMode = DiffusionMemoryMode.automatic {
        didSet { if memoryMode != oldValue { replaceEngine() } }
    }

    // Run
    private(set) var isGenerating = false
    private(set) var phaseText = ""
    private(set) var stepFraction = 0.0
    private(set) var failure: FailureNotice?
    var results: [GeneratedImage] = []
    var selected: GeneratedImage?
    private var engine: CoreMLDiffusionEngine
    private var generateTask: Task<Void, Never>?

    init() {
        let saved = UserDefaults.standard.string(forKey: Self.selectedModelKey)
        let model = saved.flatMap { CoreMLDiffusionModelStore.standard.model(id: $0) } ?? .stableDiffusion21Base
        selectedModel = model
        availableModels = CoreMLDiffusionModelStore.standard.all
        engine = CoreMLDiffusionEngine(model: model, computeUnits: model.preferredComputeUnits)
        steps = model.recommendedSteps
        guidance = Double(model.recommendedGuidance)
        computeUnits = model.preferredComputeUnits
    }

    /// Set in the Simulator (and on visionOS): the engine cannot run there whatever is downloaded.
    var unavailability: OnDeviceImageGenError? {
        engine.unavailability
    }

    // MARK: Model

    /// Applies the model's defaults and swaps the engine. Ignored while a generation or a download runs.
    func select(_ model: CoreMLDiffusionModel) {
        guard !isGenerating, !isDownloading, model != selectedModel else { return }
        selectedModel = model
        UserDefaults.standard.set(model.id, forKey: Self.selectedModelKey)
        steps = model.recommendedSteps
        guidance = Double(model.recommendedGuidance)
        isSelecting = true
        computeUnits = model.preferredComputeUnits
        isSelecting = false
        phaseText = ""
        replaceEngine()
        refreshDisk()
    }

    /// Lists a model chosen in the Hugging Face finder, selects it and starts its download where the engine can
    /// run (in the Simulator it is only listed and selected).
    func adopt(_ model: CoreMLDiffusionModel) {
        store.remember(model)
        availableModels = store.all
        select(model)
        if unavailability == nil, selectedModel == model, !model.isDownloaded { startDownload() }
    }

    func isDownloaded(_ model: CoreMLDiffusionModel) -> Bool {
        downloadedIDs.contains(model.id)
    }

    /// The display name, plus the folder when another listed model has the same name.
    func title(for model: CoreMLDiffusionModel) -> String {
        let shared = availableModels.filter { $0.displayName == model.displayName }.count > 1
        return shared ? "\(model.displayName) (\(model.folder))" : model.displayName
    }

    // MARK: Download

    func refreshDisk() {
        isDownloaded = selectedModel.isDownloaded
        downloadedIDs = Set(availableModels.filter(\.isDownloaded).map(\.id))
        freeDiskBytes = ModelDownloads.availableDiskBytes()
    }

    func startDownload() {
        guard downloadTask == nil else { return }
        failure = nil
        downloadedBytes = 0
        let model = selectedModel
        downloadTotal = model.approximateDownloadBytes
        downloadTask = Task {
            do {
                try await model.download { done, total in
                    Task { @MainActor in self.downloadProgressed(done, of: total) }
                }
            } catch is CancellationError {
                // The partial files stay for the next Download.
            } catch {
                self.show(error)
            }
            self.downloadTask = nil
            self.refreshDisk()
        }
    }

    private func downloadProgressed(_ done: Int64, of total: Int64) {
        downloadedBytes = done
        downloadTotal = total
    }

    func cancelDownload() {
        downloadTask?.cancel()
    }

    /// A Hugging Face model is also dropped from the list, and the built-in model is selected.
    func deleteModel() async {
        do {
            try await delete(selectedModel)
        } catch {
            show(error)
        }
    }

    /// Deletes `model`'s files, unloading the engine first when it runs `model`. A Hugging Face model is also
    /// dropped from the list; when it was selected, the built-in model is selected. The Models tab uses this too.
    func delete(_ model: CoreMLDiffusionModel) async throws {
        if model == selectedModel { await engine.unload() }
        defer { refreshDisk() }
        try model.delete()
        guard !model.isBuiltIn else { return }
        store.forget(model)
        availableModels = store.all
        if model == selectedModel { select(.stableDiffusion21Base) }
    }

    /// Whether a download or a generation uses `model` now.
    func isBusy(with model: CoreMLDiffusionModel) -> Bool {
        model == selectedModel && (isDownloading || isGenerating)
    }

    // MARK: Generate

    func generate() {
        guard !isGenerating else { return }
        failure = nil
        let seedInput = seedText.trimmingCharacters(in: .whitespaces)
        let seed = UInt32(seedInput)
        guard seedInput.isEmpty || seed != nil else {
            failure = FailureNotice(message: "The seed must be a whole number from 0 to \(UInt32.max).",
                                    recovery: "Leave it blank for a random seed.")
            return
        }
        // A loaded chat model would compete with the pipeline for the app's memory budget.
        ModelManager.shared.evictAll()
        let request = DiffusionRequest(prompt: style.apply(to: prompt), negativePrompt: negativePrompt, steps: steps,
                                       guidanceScale: Float(guidance), seed: seed, imageCount: imageCount)
        let engine = engine
        isGenerating = true
        phaseText = "Starting…"
        stepFraction = 0
        generateTask = Task {
            do {
                let output = try await engine.generate(request) { event in
                    Task { @MainActor in self.apply(event) }
                }
                self.finish(output, request: request)
            } catch is CancellationError {
                self.phaseText = "Cancelled."
            } catch {
                self.phaseText = ""
                self.show(error)
            }
            self.isGenerating = false
            self.generateTask = nil
            self.refreshDisk()
        }
    }

    func cancelGeneration() {
        generateTask?.cancel()
    }

    private func apply(_ event: DiffusionEvent) {
        guard isGenerating else { return }
        switch event {
        case .preparingModel:
            phaseText = "Preparing the model — the first run on this device can take a few minutes"
            stepFraction = 0
        case .step(let step, let total):
            phaseText = "Step \(step) of \(total)"
            stepFraction = Double(step) / Double(max(total, 1))
        case .decoding:
            phaseText = "Decoding"
            stepFraction = 1
        }
    }

    private func finish(_ output: DiffusionOutput, request: DiffusionRequest) {
        let elapsed = output.elapsed.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated,
                                                      fractionalPart: .show(length: 1)))
        let guidanceText = Double(request.guidanceScale).formatted(.number.precision(.fractionLength(1)))
        let settings = "Seed \(output.seed) · \(output.steps) steps · guidance \(guidanceText) · \(elapsed)"
        let items = output.images.map { result in
            GeneratedImage(image: result.image, fileURL: result.fileURL, prompt: request.prompt, settings: settings,
                           engineName: selectedModel.displayName, provenance: "Generated on device",
                           leftDevice: false)
        }
        results.insert(contentsOf: items, at: 0)
        phaseText = "\(items.count) image\(items.count == 1 ? "" : "s") in \(elapsed)."
    }

    // MARK: Results

    func add(_ item: GeneratedImage) {
        results.insert(item, at: 0)
    }

    /// Removes the session's PNGs; each run wrote into its own temporary folder.
    func clearResults() {
        let folders = Set(results.map { $0.fileURL.deletingLastPathComponent() })
        folders.forEach { try? FileManager.default.removeItem(at: $0) }
        results = []
        phaseText = ""
    }

    func clearFailure() {
        failure = nil
    }

    func show(_ error: any Error) {
        let localized = error as? any LocalizedError
        failure = FailureNotice(message: localized?.errorDescription ?? error.localizedDescription,
                                recovery: localized?.recoverySuggestion)
    }

    // MARK: Lifecycle

    /// Background apps cannot use the GPU and are first in line for jetsam: stop at the next step and free the
    /// pipeline.
    func enterBackground() {
        generateTask?.cancel()
        let engine = engine
        Task { await engine.unload() }
    }

    private func replaceEngine() {
        let previous = engine
        engine = CoreMLDiffusionEngine(model: selectedModel, computeUnits: computeUnits, memoryMode: memoryMode)
        Task { await previous.unload() }
    }
}
