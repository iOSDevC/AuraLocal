import SwiftUI
import AuraCore

// MARK: - MLToolsTab
//
// Usage reference for AuraCore's on-device ML tools, each built on an Apple framework
// (Vision, NaturalLanguage, SoundAnalysis, Core ML, Create ML) and run on this device.

struct MLToolsTab: View {
    private enum Pane: String, CaseIterable, Identifiable {
        case tools = "Tools", text = "Text", image = "Image", audio = "Audio", train = "Train"
        var id: String { rawValue }
    }

    @State private var selection: Pane = .tools

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker("Section", selection: $selection) {
                    ForEach(Pane.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .padding(.top, 8)

                switch selection {
                case .tools: MLToolListView()
                case .text: MLTextToolsView()
                case .image: MLImageToolsView()
                case .audio: MLAudioToolsView()
                case .train: MLTrainingView()
                }
            }
            .navigationTitle("On-device ML")
        }
    }
}

// MARK: - Tools (discovery)

private struct MLToolListView: View {
    @State private var tools: [SystemToolRegistry.ToolInfo] = []

    var body: some View {
        List {
            ForEach(SystemToolCategory.allCases, id: \.self) { category in
                Section(category.displayName) {
                    let members = tools.filter { $0.category == category }
                    if category == .customModel && members.isEmpty {
                        Text("CoreMLModelTool and TextClassifierTool wrap one model file each, so they are "
                             + "created with its URL instead of being listed here. The Train section builds one.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    ForEach(members) { MLToolRow(tool: $0) }
                }
            }
        }
        .task { tools = await SystemToolRegistry.discover() }
    }
}

private struct MLToolRow: View {
    let tool: SystemToolRegistry.ToolInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(tool.displayName, systemImage: tool.isAvailable ? "checkmark.circle.fill" : "xmark.circle")
                .foregroundStyle(tool.isAvailable ? .green : .orange)
                .font(.subheadline.weight(.semibold))
            Text(tool.id).font(.caption.monospaced()).foregroundStyle(.secondary)
            Text(tool.availability.reason ?? tool.summary).font(.caption)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Shared pieces

/// One tool's result, or why it could not run on this device.
nonisolated enum MLOutcome<Value: Sendable>: Sendable {
    case done(Value)
    case unavailable(String)
    case failed(String)

    var output: Value? {
        if case .done(let value) = self { return value }
        return nil
    }

    /// Checks `availability()` first: some tools report unavailable (e.g. in the Simulator)
    /// instead of failing loudly.
    static func run(_ tool: some SystemTool, _ body: () async throws -> Value) async -> MLOutcome<Value> {
        if let reason = await tool.availability().reason { return .unavailable(reason) }
        do {
            return .done(try await body())
        } catch {
            return .failed(error.localizedDescription)
        }
    }
}

struct MLOutcomeView<Value: Sendable, Content: View>: View {
    let outcome: MLOutcome<Value>
    @ViewBuilder let makeContent: (Value) -> Content

    var body: some View {
        switch outcome {
        case .done(let value): makeContent(value)
        case .unavailable(let reason): Label(reason, systemImage: "slash.circle").foregroundStyle(.orange)
        case .failed(let reason): Label(reason, systemImage: "xmark.octagon").foregroundStyle(.red)
        }
    }
}

nonisolated struct MLBox: Sendable {
    let rect: CGRect
    let tint: Color
}

/// Draws Vision boxes (normalized, bottom-left origin) over the displayed image.
struct MLBoxOverlay: View {
    let boxes: [MLBox]

    var body: some View {
        GeometryReader { geometry in
            ForEach(Array(boxes.enumerated()), id: \.offset) { _, box in
                Rectangle()
                    .stroke(box.tint, lineWidth: 2)
                    .frame(width: box.rect.width * geometry.size.width,
                           height: box.rect.height * geometry.size.height)
                    .position(x: box.rect.midX * geometry.size.width,
                              y: (1 - box.rect.midY) * geometry.size.height)
            }
        }
    }
}

nonisolated enum MLFormat {
    static func percent(_ fraction: Double) -> String {
        String(format: "%.0f %%", fraction * 100)
    }

    static func fixed(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}
