import SwiftUI
import AuraCore

// MARK: - ModelSection

public struct ModelSection: View {
    let title: String
    let icon: String
    let color: Color
    let models: [Model]
    let compatibilityMap: [Model: ModelCompatibility]
    let onTest: ((Model) -> Void)?

    public init(title: String, icon: String, color: Color, models: [Model], onTest: ((Model) -> Void)? = nil) {
        self.title = title
        self.icon = icon
        self.color = color
        self.models = models
        self.onTest = onTest
        let results = HardwareAnalyzer.compatibleModels(from: models)
        self.compatibilityMap = Dictionary(uniqueKeysWithValues: results.map { ($0.model, $0) })
    }

    public var body: some View {
        Section {
            ForEach(models, id: \.self) { model in
                let compat = compatibilityMap[model]
                ModelRow(model: model, compatibility: compat, onTest: onTest)
            }
        } header: {
            Label(title, systemImage: icon)
                .foregroundStyle(color)
        }
    }
}

// MARK: - ModelRow

/// A row with download, cancel and delete actions. Downloads go through ``ModelDownloads`` so files land where the
/// backends load them from, with the Hugging Face token, a free-space check and resume from `.partial`. The row
/// claims its file in ``ModelDownloadCenter`` so the Hugging Face browser never writes the same `.partial`.
struct ModelRow: View {
    let model: Model
    let compatibility: ModelCompatibility?
    var onTest: ((Model) -> Void)? = nil

    @State private var downloadState: DownloadState = .idle
    @State private var downloadPercent: Int = 0
    @State private var errorMessage: String?
    @State private var isDownloaded: Bool = false
    @State private var downloadTask: Task<Void, Never>?

    private var center: ModelDownloadCenter { .shared }

    private enum DownloadState {
        case idle
        case downloading
        case failed
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                // Model info
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(model.displayName)
                            .font(.subheadline.weight(.medium))
                        BackendBadge(model: model)
                        if let compatibility {
                            FitBadge(fitLevel: compatibility.fitLevel)
                            SpeedBadge(compatibility: compatibility)
                        }
                    }
                    Text(formattedSize)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if let onTest {
                    Button { onTest(model) } label: {
                        Text("Test")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.tint)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .overlay(Capsule().stroke(.tint, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                }

                // Actions
                actionView
            }

            // Progress bar
            if downloadState == .downloading {
                ProgressView(value: Double(downloadPercent), total: 100)
                    .tint(.accentColor)
                    .animation(.default, value: downloadPercent)
            }

            // Error
            if let errorMessage {
                Text(errorMessage)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 2)
        .opacity(compatibility?.fitLevel == .tooLarge ? 0.5 : 1.0)
        .onAppear { isDownloaded = model.isDownloaded }
        .onChange(of: center.revision) { isDownloaded = model.isDownloaded }
    }

    // MARK: - Action View

    @ViewBuilder
    private var actionView: some View {
        switch downloadState {
        case .idle:
            if isDownloaded {
                HStack(spacing: 8) {
                    Label("Ready", systemImage: "checkmark.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.green)

                    Button(role: .destructive) { deleteModel() } label: {
                        Image(systemName: "trash")
                            .font(.caption)
                            .foregroundStyle(.red)
                            .padding(5)
                            .background(.red.opacity(0.1), in: Circle())
                    }
                    .accessibilityLabel("Delete \(model.displayName)")
                }
            } else if compatibility?.fitLevel == .tooLarge {
                Label("Too Large", systemImage: "xmark.circle")
                    .font(.caption2)
                    .foregroundStyle(.red)
            } else if center.isWritingElsewhere(model) {
                Label("Downloading elsewhere", systemImage: "arrow.down.circle.dotted")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Button { startDownload() } label: {
                    Text("Download")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(.tint, in: Capsule())
                }
            }

        case .downloading:
            HStack(spacing: 8) {
                Text("\(downloadPercent)%")
                    .font(.caption.weight(.semibold).monospacedDigit())
                    .foregroundStyle(.tint)
                    .contentTransition(.numericText())
                Button { downloadTask?.cancel() } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Cancel download")
            }

        case .failed:
            Button { startDownload() } label: {
                Text("Retry")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(.orange, in: Capsule())
            }
        }
    }

    // MARK: - Download

    private func startDownload() {
        let key = DownloadRecords.key(for: model)
        guard center.claim(key) else {
            errorMessage = "This file is downloading elsewhere in the app."
            return
        }
        downloadState = .downloading
        downloadPercent = 0
        errorMessage = nil
        let model = self.model
        downloadTask = Task {
            do {
                _ = try await ModelDownloads.download(model) { done, total in
                    let percent = total > 0 ? Int(done * 100 / total) : 0
                    Task { @MainActor in downloadPercent = percent }
                }
                downloadState = .idle
                isDownloaded = model.isDownloaded
            } catch {
                // Cancellation can surface as CancellationError or URLError(.cancelled); the partial file stays on
                // disk either way and the next download resumes from it.
                if Task.isCancelled {
                    downloadState = .idle
                } else {
                    errorMessage = error.localizedDescription
                    downloadState = .failed
                }
            }
            downloadTask = nil
            center.release(key)
            center.noteDiskChange()
        }
    }

    // MARK: - Delete

    private func deleteModel() {
        do {
            try center.delete(model)
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
        isDownloaded = model.isDownloaded
    }

    // MARK: - Helpers

    private var formattedSize: String {
        let mb = model.approximateSizeMB
        if mb >= 1000 {
            return String(format: "%.1f GB", Double(mb) / 1024.0)
        }
        return "\(mb) MB"
    }
}

// MARK: - BackendBadge

struct BackendBadge: View {
    let model: Model

    var body: some View {
        let backend = BackendRouter.recommendedBackend(for: model)
        Text(backend.label)
            .font(.system(size: 9, weight: .semibold))
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(backend.color.opacity(0.15), in: Capsule())
            .foregroundStyle(backend.color)
    }
}

private extension BackendKind {
    var label: String {
        switch self {
        case .mlx:            "MLX"
        case .llamaCpp:       "GGUF"
        case .layerStreaming:  "STREAM"
        case .remote:          "REMOTE"
        case .hybrid:          "HYBRID"
        }
    }

    var color: Color {
        switch self {
        case .mlx:            .blue
        case .llamaCpp:       .purple
        case .layerStreaming:  .orange
        case .remote:          .teal
        case .hybrid:          .indigo
        }
    }
}

// MARK: - SpeedBadge

/// The other half of the fit answer. A model can be "Excellent" on memory and still
/// take four minutes to reply; showing only the memory verdict hides that.
/// Silent when the chip's bandwidth is unknown — no estimate beats a made-up one.
struct SpeedBadge: View {
    let compatibility: ModelCompatibility

    var body: some View {
        if let tps = compatibility.estimatedDecodeTokensPerSecond {
            Text(String(format: "~%.0f tok/s", tps))
                .font(.caption2.weight(.medium))
                .foregroundStyle(speedColor)
                .help("Estimated decode speed on this device — \(compatibility.speedLevel.label)")
        }
    }

    private var speedColor: Color {
        switch compatibility.speedLevel {
        case .fast, .usable: .green
        case .slow:          .orange
        case .unusable:      .red
        case .unknown:       .secondary
        }
    }
}

// MARK: - FitBadge

struct FitBadge: View {
    let fitLevel: ModelFitLevel

    var body: some View {
        Label(fitLevel.label, systemImage: fitLevel.systemImage)
            .font(.caption2.weight(.medium))
            .foregroundStyle(fitColor)
    }

    private var fitColor: Color {
        switch fitLevel {
        case .excellent:         .green
        case .good:              .blue
        case .marginal:          .orange
        case .streamingRequired: .purple
        case .tooLarge:          .red
        }
    }
}
