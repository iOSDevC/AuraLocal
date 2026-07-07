import SwiftUI
import AuraCore

/// Identifiable target so a domain model can be presented via `.sheet(item:)`.
struct DomainTestTarget: Identifiable {
    let id = UUID()
    let model: Model
    let domain: Model.Domain
}

/// Quick playground to test a domain-specialized model with curated example
/// prompts. Loads the (downloaded) model and streams its response.
struct DomainTestView: View {
    let model: Model
    let domain: Model.Domain

    @Environment(\.dismiss) private var dismiss

    @State private var prompt = ""
    @State private var response = ""
    @State private var status = ""
    @State private var errorText: String?
    @State private var isRunning = false

    private var isDownloaded: Bool { model.isDownloaded }

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    LabeledContent(model.displayName, value: domain.displayName)
                    if !isDownloaded {
                        Label("Not downloaded yet — download it from the Models list to run a test.",
                              systemImage: "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }

                Section("Example prompts") {
                    ForEach(domain.samplePrompts, id: \.self) { sample in
                        Button { prompt = sample } label: {
                            Text(sample)
                                .font(.caption)
                                .multilineTextAlignment(.leading)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .buttonStyle(.plain)
                    }
                }

                Section("Prompt") {
                    TextEditor(text: $prompt)
                        .font(.callout)
                        .frame(minHeight: 80)
                }

                Section {
                    Button { Task { await run() } } label: {
                        if isRunning {
                            HStack {
                                ProgressView()
                                Text(status.isEmpty ? "Running…" : status)
                            }
                        } else {
                            Label("Run test", systemImage: "play.fill")
                        }
                    }
                    .disabled(prompt.isEmpty || isRunning || !isDownloaded)
                }

                if !response.isEmpty {
                    Section("Response") {
                        Text(response)
                            .font(.callout.monospaced())
                            .textSelection(.enabled)
                    }
                }

                if let errorText {
                    Section {
                        Label(errorText, systemImage: "xmark.octagon")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }

                if domain == .medicine {
                    Section {
                        Text("Educational only — not medical advice.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Test · \(domain.displayName)")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .frame(minWidth: 460, minHeight: 560)
    }

    @MainActor
    private func run() async {
        isRunning = true
        response = ""
        errorText = nil
        status = "Loading model…"
        defer { isRunning = false; status = "" }
        do {
            let llm = try await ModelManager.shared.load(model) { status = $0 }
            status = "Generating…"
            for try await token in llm.stream(prompt, systemPrompt: domain.testSystemPrompt, maxTokens: 512) {
                response += token
            }
        } catch {
            errorText = error.localizedDescription
        }
    }
}
