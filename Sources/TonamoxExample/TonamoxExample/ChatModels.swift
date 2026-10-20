import SwiftUI
import TonamoxCore

// MARK: - Model lists

/// The models Chat and Voice offer: downloaded ones only, so picking a model never starts a download.
enum ChatModels {

    /// Downloaded models that have every capability in `required`, by name. A model that takes neither text nor
    /// images here (a document model in GGUF) is left out; Models still lists it for deletion.
    static func downloaded(requiring required: ModelCapabilities = []) -> [Model] {
        Model.allModels
            .filter {
                $0.isDownloaded && $0.capabilities.isSuperset(of: required)
                    && !$0.capabilities.isDisjoint(with: [.text, .vision])
            }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
    }

    /// The model stored under `id` when it is still downloaded and loads now, else the first one that does.
    static func preferred(id: String, requiring required: ModelCapabilities = []) -> Model? {
        let choices = downloaded(requiring: required)
        let usable = choices.filter { ModelBlocker.check($0) == nil }
        return usable.first { $0.id == id } ?? usable.first
    }

    /// The catalog models in `models` that fit this device, downloaded or not; models added from Hugging Face are
    /// left out. The Simulator lists GGUF only, since MLX can't run there.
    static func catalog(_ models: [Model]) -> [Model] {
        let recorded = Set(DownloadRecords.standard.models.map(\.id))
        let profile = HardwareProfile.current()
        return models.filter { model in
            #if targetEnvironment(simulator)
            guard model.format == .gguf else { return false }
            #endif
            return !recorded.contains(model.id) && HardwareAnalyzer.assess(model, profile: profile).fitLevel.isRunnable
        }
    }
}

// MARK: - ModelBlocker

/// Why a downloaded model can't load right now; the same rule TonamoxUI's Try it applies.
enum ModelBlocker {
    case simulatorMLX
    case tooLarge

    var label: String {
        switch self {
        case .simulatorMLX: "device only"
        case .tooLarge: "too large now"
        }
    }

    var explanation: String {
        switch self {
        case .simulatorMLX: "MLX can't run on the Simulator; use a GGUF model or a device."
        case .tooLarge: "Too large for this device's free memory right now."
        }
    }

    static func check(_ model: Model) -> ModelBlocker? {
        #if targetEnvironment(simulator)
        if model.format == .mlx { return .simulatorMLX }
        #endif
        let fit = HardwareAnalyzer.assess(model).fitLevel
        if fit == .tooLarge || (model.format == .mlx && fit == .streamingRequired) { return .tooLarge }
        return nil
    }
}

// MARK: - Capability badges

/// One capability as a badge.
struct CapabilityBadge: Identifiable {
    let capability: ModelCapabilities
    let title: String
    let symbol: String

    var id: String { title }

    /// In display order.
    static let all = [
        CapabilityBadge(capability: .text, title: "Text", symbol: "text.bubble"),
        CapabilityBadge(capability: .vision, title: "Vision", symbol: "eye"),
        CapabilityBadge(capability: .documentExtraction, title: "OCR", symbol: "doc.viewfinder"),
        CapabilityBadge(capability: .tools, title: "Tools", symbol: "wrench.and.screwdriver"),
    ]
}

extension ModelCapabilities {
    /// "Text · Vision · Tools", for menus and accessibility.
    var summary: String {
        CapabilityBadge.all.filter { contains($0.capability) }.map(\.title).joined(separator: " · ")
    }
}

/// One capsule per capability the model has.
struct CapabilityBadges: View {
    let capabilities: ModelCapabilities

    var body: some View {
        HStack(spacing: 4) {
            ForEach(CapabilityBadge.all.filter { capabilities.contains($0.capability) }) { badge in
                Label(badge.title, systemImage: badge.symbol)
                    .labelStyle(.titleAndIcon)
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.accentColor.opacity(0.12), in: Capsule())
                    .foregroundStyle(Color.accentColor)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Capabilities: \(capabilities.summary)")
    }
}
