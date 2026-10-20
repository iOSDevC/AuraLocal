import SwiftUI
import TonamoxCore

/// The reference panels the Hybrid tab opens as bottom sheets.
enum HybridInfoSheet: String, Identifiable {
    case onDeviceTools, integrate
    var id: String { rawValue }
}

/// Native Apple-framework tools discovered on this device, read-only.
struct OnDeviceToolsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var discovered: [SystemToolRegistry.ToolInfo] = []
    @State private var isLoading = true

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    if isLoading {
                        ProgressView().frame(maxWidth: .infinity)
                    }
                    ForEach(discovered) { tool in
                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text(tool.displayName).font(.callout)
                                Spacer()
                                Image(systemName: tool.isAvailable ? "checkmark.circle.fill" : "xmark.circle")
                                    .foregroundStyle(tool.isAvailable ? .green : .secondary)
                                    .accessibilityLabel(tool.isAvailable ? "Available" : "Unavailable")
                            }
                            Text(tool.isAvailable ? tool.summary : (tool.availability.reason ?? "Unavailable"))
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                } footer: {
                    Text("Native Apple-framework tools that support the local models — no model download. Discovered per device via SystemToolRegistry.")
                }
            }
            .navigationTitle("On-device tools")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .task {
                if discovered.isEmpty { discovered = await SystemToolRegistry.discover() }
                isLoading = false
            }
        }
        .hybridSheetSizing()
    }
}

/// How to use the same features from a package or the `tonamox` CLI.
struct IntegrateView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    row("Swift Package", ".package(url: \"https://github.com/iOSDevC/AuraLocal\", revision: \"<commit SHA>\")")
                    row("CLI · detect providers", "tonamox providers")
                    row("CLI · ask a bigger model", "tonamox ask \"…\" [--provider openai|anthropic]")
                    row("CLI · native OCR", "tonamox ocr image.png")
                } footer: {
                    Text("Pin the package by revision: a version range cannot resolve while Tonamox pins LocalLLMClient by revision. The same features run headlessly via the `tonamox` CLI (build: scripts/build-cli.sh) and this app ships as a .dmg (scripts/build-app.sh). `tonamox ask` uses your llama-server / Ollama by default and a cloud API only when you name one; export OPENAI_API_KEY or ANTHROPIC_API_KEY for the CLI.")
                }
            }
            .navigationTitle("Integrate")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .hybridSheetSizing()
    }

    private func row(_ title: String, _ code: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(code).font(.caption.monospaced()).textSelection(.enabled)
        }
    }
}

private extension View {
    /// A bottom sheet on iPhone (half height, draggable to full); a sized window sheet on the Mac.
    func hybridSheetSizing() -> some View {
        // Room below the last row, clear of the sheet's rounded bottom edge.
        let padded = contentMargins(.bottom, 32, for: .scrollContent)
        #if os(macOS)
        return padded.frame(minWidth: 460, minHeight: 380)
        #else
        // An opaque background: the default glass let the tab behind show through the text.
        return padded
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
            .presentationBackground(Color(uiColor: .systemGroupedBackground))
        #endif
    }
}
