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
    @State private var remoteReceipt: String?
    @State private var showSettings = false
    @ObservedObject private var consentGate = HybridSettings.shared.consent

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

                Section {
                    Button { Task { await runRemote() } } label: {
                        Label("Ask a bigger model (remote)", systemImage: "cloud.bolt")
                    }
                    .disabled(prompt.isEmpty || isRunning)
                    if let remoteReceipt {
                        Text(remoteReceipt)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Hybrid escalation")
                } footer: {
                    Text("Sends this prompt to your own bigger local model (llama-server / Ollama), compressing context to save tokens. Falls back to nothing if none is running.")
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
                ToolbarItem {
                    Button { showSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
            .sheet(isPresented: $showSettings) { HybridSettingsView() }
            .sheet(item: $consentGate.pending) { request in
                ConsentSheet(request: request, gate: consentGate)
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

    /// Escalate the prompt to the user's own bigger model (llama-server / Ollama),
    /// compressing context to save tokens. Fail-closed: shows a hint if none runs.
    @MainActor
    private func runRemote() async {
        isRunning = true
        response = ""
        errorText = nil
        remoteReceipt = nil
        status = "Finding a target…"
        defer { isRunning = false; status = "" }

        let policy = HybridSettings.shared.policy
        let candidates = await HybridEscalator.candidateTargets(policy: policy)
        guard let target = candidates.first else {
            errorText = "No provider available. Start Ollama/llama-server, or add a cloud key in Settings (⚙) and allow cloud."
            return
        }

        // Cloud targets require per-conversation consent (LAN is trusted).
        if !target.isLocalNetwork {
            let preview = ContextCompressor().compress(
                context: prompt, question: prompt,
                budgetTokens: max(256, target.contextLength ?? 8192))
            let cost = CostLedger.projectedCost(
                target: target, inputTokens: max(1, prompt.count / 4), maxOutput: 512)
            let approved = await HybridSettings.shared.consent.requestConsent(
                target: target, preview: preview, projectedCostUSD: cost)
            guard approved else {
                errorText = "Escalation declined — keeping local."
                return
            }
        }
        status = "Asking \(target.provider.displayName)…"
        do {
            let result = try await HybridEscalator().escalate(
                to: target,
                systemPrompt: domain.testSystemPrompt,
                context: "",
                question: prompt,
                maxTokens: 512,
                redactPII: !target.isLocalNetwork) { response = $0 }

            var receipt = "via \(result.providerName) · \(target.modelID)"
            if let usage = result.usage {
                receipt += " · \(usage.inputTokens) in / \(usage.outputTokens) out"
            }
            let c = result.compression
            if c.originalTokens > c.compressedTokens {
                receipt += " · compressed \(c.originalTokens)→\(c.compressedTokens) (\(String(format: "%.1f", c.factor))×)"
            }
            if result.redactedPIICount > 0 {
                receipt += " · redacted \(result.redactedPIICount) PII"
            }
            if result.fromCache {
                receipt += " · (cached, $0)"
            }
            remoteReceipt = receipt
        } catch {
            errorText = error.localizedDescription
        }
    }
}
