import SwiftUI
import AuraCore

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

/// A single row showing model name, size, download status, backend, and fit level.
struct ModelRow: View {
    let model: Model
    let compatibility: ModelCompatibility?

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(model.displayName)
                        .font(.subheadline.weight(.medium))
                    BackendBadge(model: model)
                }
                Text(model.rawValue)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text(formattedSize)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                if model.isDownloaded {
                    Label("Ready", systemImage: "checkmark.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.green)
                } else {
                    Label("Not downloaded", systemImage: "arrow.down.circle")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                if let compatibility {
                    FitBadge(fitLevel: compatibility.fitLevel)
                }
            }
        }
        .padding(.vertical, 2)
        .opacity(compatibility?.fitLevel == .tooLarge ? 0.5 : 1.0)
    }

    private var formattedSize: String {
        let mb = model.approximateSizeMB
        if mb >= 1000 {
            let gb = Double(mb) / 1024.0
            return String(format: "%.1f GB", gb)
        }
        return "\(mb) MB"
    }
}

// MARK: - BackendBadge

/// Compact badge showing which inference backend will be used.
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

/// Compact badge showing hardware compatibility for a model.
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
