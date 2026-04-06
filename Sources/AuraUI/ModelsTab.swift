import SwiftUI
import AuraCore

// MARK: - ModelSection

public struct ModelSection: View {
    let title: String
    let icon: String
    let color: Color
    let models: [Model]
    let compatibilityMap: [Model: ModelCompatibility]

    public init(title: String, icon: String, color: Color, models: [Model]) {
        self.title = title
        self.icon = icon
        self.color = color
        self.models = models
        let results = HardwareAnalyzer.compatibleModels(from: models)
        self.compatibilityMap = Dictionary(uniqueKeysWithValues: results.map { ($0.model, $0) })
    }

    public var body: some View {
        Section {
            ForEach(models, id: \.self) { model in
                let compat = compatibilityMap[model]
                ModelRow(model: model, compatibility: compat)
            }
        } header: {
            Label(title, systemImage: icon)
                .foregroundStyle(color)
        }
    }
}

// MARK: - ModelRow

/// A row with download, select, and delete actions powered by ChunkedDownloader.
struct ModelRow: View {
    let model: Model
    let compatibility: ModelCompatibility?

    @State private var downloadState: DownloadState = .idle
    @State private var downloadPercent: Int = 0
    @State private var errorMessage: String?
    @State private var isDownloaded: Bool = false

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
                        }
                    }
                    Text(formattedSize)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Spacer()

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
        .onAppear { isDownloaded = checkDownloaded() }
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
                }
            } else if compatibility?.fitLevel == .tooLarge {
                Label("Too Large", systemImage: "xmark.circle")
                    .font(.caption2)
                    .foregroundStyle(.red)
            } else {
                Button { Task { await download() } } label: {
                    Text("Download")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(.tint, in: Capsule())
                }
            }

        case .downloading:
            Text("\(downloadPercent)%")
                .font(.caption.weight(.semibold).monospacedDigit())
                .foregroundStyle(.tint)
                .contentTransition(.numericText())

        case .failed:
            Button { Task { await download() } } label: {
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

    private func download() async {
        downloadState = .downloading
        downloadPercent = 0
        errorMessage = nil

        let repoID = model.rawValue
        let sanitized = repoID.replacingOccurrences(of: "/", with: "--")
        let modelDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appending(path: "huggingface/hub/models--\(sanitized)/snapshots/main", directoryHint: .isDirectory)

        do {
            try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        } catch {
            fail(error.localizedDescription)
            return
        }

        // Fetch file list
        guard let apiURL = URL(string: "https://huggingface.co/api/models/\(repoID)/tree/main") else {
            fail("Invalid URL")
            return
        }

        let apiData: Data
        let apiResponse: URLResponse
        do {
            (apiData, apiResponse) = try await URLSession.shared.data(from: apiURL)
        } catch {
            fail(error.localizedDescription)
            return
        }

        if let http = apiResponse as? HTTPURLResponse, http.statusCode != 200 {
            fail("API HTTP \(http.statusCode)")
            return
        }

        struct HFFile: Codable {
            let path: String
            let size: Int?
            let type: String?
        }

        guard let files = try? JSONDecoder().decode([HFFile].self, from: apiData) else {
            fail("Invalid response")
            return
        }

        let downloadable = files.filter { $0.type == "file" }
        guard !downloadable.isEmpty else { fail("Empty repo"); return }

        let totalBytes = downloadable.compactMap(\.size).reduce(0, +)
        var completedBytes: Int64 = 0

        let downloader = ChunkedDownloader()

        for file in downloadable {
            let fileURL = modelDir.appending(path: file.path)
            let fileSize = Int64(file.size ?? 0)

            if FileManager.default.fileExists(atPath: fileURL.path()) {
                completedBytes += fileSize
                downloadPercent = totalBytes > 0 ? Int(completedBytes * 100 / Int64(totalBytes)) : 0
                continue
            }

            let parent = fileURL.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)

            guard let dlURL = URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(file.path)") else {
                continue
            }

            let captured = completedBytes
            do {
                try await downloader.download(from: dlURL, to: fileURL, fileName: file.path) { progress in
                    let globalBytes = captured + progress.bytesDownloaded
                    let pct = totalBytes > 0 ? Int(globalBytes * 100 / Int64(totalBytes)) : 0
                    Task { @MainActor in
                        downloadPercent = pct
                    }
                }
            } catch {
                fail("\((file.path as NSString).lastPathComponent): \(error.localizedDescription)")
                return
            }

            completedBytes += fileSize
        }

        downloadState = .idle
        isDownloaded = true
        errorMessage = nil
    }

    // MARK: - Delete

    private func deleteModel() {
        ModelManager.shared.evict(model)
        let sanitized = model.rawValue.replacingOccurrences(of: "/", with: "--")
        let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appending(path: "huggingface/hub/models--\(sanitized)", directoryHint: .isDirectory)
        try? FileManager.default.removeItem(at: dir)
        isDownloaded = false
        errorMessage = nil
    }

    // MARK: - Helpers

    private func fail(_ msg: String) {
        errorMessage = msg
        downloadState = .failed
    }

    private func checkDownloaded() -> Bool {
        let sanitized = model.rawValue.replacingOccurrences(of: "/", with: "--")
        let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appending(path: "huggingface/hub/models--\(sanitized)/snapshots/main", directoryHint: .isDirectory)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dir.path(), isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        return ((try? FileManager.default.contentsOfDirectory(atPath: dir.path()))?.isEmpty == false)
    }

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
        }
    }

    var color: Color {
        switch self {
        case .mlx:            .blue
        case .llamaCpp:       .purple
        case .layerStreaming:  .orange
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
