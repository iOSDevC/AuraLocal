import SwiftUI
import AuraCore
import AuraImageGen

// MARK: - OnDeviceImageGenView
//
// Text-to-image on this device with Core ML Stable Diffusion (AuraImageGen/CoreMLDiffusionEngine).
// Only the one-time model download touches the network; prompts and images stay on the device.

struct OnDeviceImageGenView: View {
    @Bindable var state: ImageGenState
    @State private var confirmingDownload = false
    @State private var confirmingDelete = false
    @State private var showsAdvanced = false

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
            Text("About \(downloadSize) from huggingface.co (\(state.preset.repoID)). No account needed. "
                 + "License: \(state.preset.licenseName), which restricts some uses; read it on the model card.")
        }
        .sheet(item: $state.selected) { GeneratedImageDetail(item: $0) }
    }

    // MARK: Model card

    private var modelSection: some View {
        Section("Model") {
            VStack(alignment: .leading, spacing: 4) {
                Text(state.preset.displayName).font(.headline)
                Text("\(downloadSize) · \(state.preset.imageSide)×\(state.preset.imageSide) · Hugging Face")
                    .font(.caption).foregroundStyle(.secondary)
                Link("License: \(state.preset.licenseName)", destination: state.preset.licenseURL)
                    .font(.caption)
            }
            Label("Runs on this device. Your prompts and images never leave it.", systemImage: "lock.shield")
                .font(.callout)
            downloadControls
        }
    }

    private var downloadSize: String {
        ByteCountFormatter.string(fromByteCount: state.preset.approximateDownloadBytes, countStyle: .file)
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
            if state.isDownloaded { deleteButton }
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
        }
    }

    private var deleteButton: some View {
        Button("Delete", role: .destructive) { confirmingDelete = true }
            .disabled(state.isGenerating)
            .confirmationDialog("Delete the image model?", isPresented: $confirmingDelete,
                                titleVisibility: .visible) {
                Button("Delete", role: .destructive) { Task { await state.deleteModel() } }
            } message: {
                Text("Frees about \(downloadSize). Generating images again needs the full download.")
            }
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
                Stepper("Steps: \(state.steps)", value: $state.steps, in: 10...50)
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

/// One image with what produced it, and Share. There is no direct save to Photos: the app has no photo-library
/// usage description.
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

/// Download, generation and the shared results grid for the Image tab.
@Observable
final class ImageGenState {
    let preset = CoreMLDiffusionModel.stableDiffusion21Base

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
        didSet { if computeUnits != oldValue { replaceEngine() } }
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
    private var engine = CoreMLDiffusionEngine()
    private var generateTask: Task<Void, Never>?

    /// Set in the Simulator (and on visionOS): the engine cannot run there whatever is downloaded.
    var unavailability: OnDeviceImageGenError? {
        engine.unavailability
    }

    // MARK: Download

    func refreshDisk() {
        isDownloaded = preset.isDownloaded
        freeDiskBytes = ModelDownloads.availableDiskBytes()
    }

    func startDownload() {
        guard downloadTask == nil else { return }
        failure = nil
        downloadedBytes = 0
        downloadTotal = preset.approximateDownloadBytes
        let preset = preset
        downloadTask = Task {
            do {
                try await preset.download { done, total in
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

    func deleteModel() async {
        await engine.unload()
        do {
            try preset.delete()
        } catch {
            show(error)
        }
        refreshDisk()
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
                           engineName: preset.displayName, provenance: "Generated on device", leftDevice: false)
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
        engine = CoreMLDiffusionEngine(model: preset, computeUnits: computeUnits, memoryMode: memoryMode)
        Task { await previous.unload() }
    }
}
