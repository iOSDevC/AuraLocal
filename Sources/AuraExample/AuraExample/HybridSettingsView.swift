import SwiftUI
import AuraCore

// MARK: - Settings

/// Configure remote escalation: policy, BYOK cloud keys (stored in Keychain),
/// and a session cost readout.
struct HybridSettingsView: View {
    @ObservedObject private var settings = HybridSettings.shared
    @ObservedObject private var ledger = CostLedger.shared
    @Environment(\.dismiss) private var dismiss

    /// When embedded in a tab or the macOS Settings scene, hide the sheet's "Done" button.
    var isEmbedded = false

    @State private var anthropicKey = ""
    @State private var openAIKey = ""
    @State private var gitHubModelsKey = ""
    @State private var hfDownloadKey = ""

    private var costCap: Binding<Double> {
        Binding(
            get: { NSDecimalNumber(decimal: settings.policy.costCapUSDPerSession).doubleValue },
            set: { settings.policy.costCapUSDPerSession = Decimal($0) })
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Escalation policy") {
                    Picker("Mode", selection: $settings.policy.mode) {
                        Text("Off").tag(EscalationPolicy.Mode.off)
                        Text("Ask each time").tag(EscalationPolicy.Mode.askEachTime)
                        Text("Auto (LAN) + ask cloud").tag(EscalationPolicy.Mode.autoWithConsentMemory)
                    }
                    Toggle("Allow cloud providers", isOn: $settings.policy.allowCloud)
                    HStack {
                        Text("Cost cap")
                        Spacer()
                        TextField("USD/session", value: costCap, format: .number)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 90)
                        Text("USD/session").font(.caption).foregroundStyle(.secondary)
                    }
                }

                Section {
                    keyRow(title: "Anthropic (Claude)", account: "cloud.anthropic", field: $anthropicKey)
                    keyRow(title: "OpenAI", account: "cloud.openai", field: $openAIKey)
                    keyRow(title: "GitHub Models (Copilot)", account: "cloud.github-models", field: $gitHubModelsKey)
                } header: {
                    Text("Cloud API keys (BYOK)")
                } footer: {
                    Text("Keys are stored in the Keychain (this device only) — never in code, files, or logs. GitHub Models rides your GitHub/Copilot account: use a fine-grained token with the models:read permission.")
                }

                Section {
                    keyRow(title: "Hugging Face token",
                           account: KeychainDownloadAuth.huggingFaceAccount,
                           field: $hfDownloadKey)
                } header: {
                    Text("Model download tokens")
                } footer: {
                    Text("A Hugging Face access token lets AuraLocal download gated or private GGUF repos. Stored in the Keychain.")
                }

                Section("This session") {
                    LabeledContent("Remote tokens", value: "\(ledger.sessionTokens)")
                    LabeledContent("Cost", value: String(format: "$%.4f",
                        NSDecimalNumber(decimal: ledger.sessionCostUSD).doubleValue))
                    LabeledContent("Escalations", value: "\(ledger.records.count)")
                }
            }
            .navigationTitle("Hybrid settings")
            .toolbar {
                if !isEmbedded {
                    ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
                }
            }
        }
        .frame(minWidth: 460, minHeight: 520)
    }

    @ViewBuilder
    private func keyRow(title: String, account: String, field: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                if KeychainStore.hasKey(for: account) {
                    Label("Configured", systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(.green)
                    Button("Clear") { KeychainStore.delete(for: account) }
                        .font(.caption)
                }
            }
            HStack {
                SecureField("Paste API key", text: field)
                Button("Save") {
                    try? KeychainStore.save(field.wrappedValue, for: account)
                    field.wrappedValue = ""
                }
                .disabled(field.wrappedValue.isEmpty)
            }
        }
    }
}

// MARK: - Consent sheet

/// Shown before any cloud send: the exact compressed payload, projected cost,
/// and the provider's retention note. The user chooses to send or keep local.
struct ConsentSheet: View {
    let request: UIConsentGate.Request
    let gate: UIConsentGate

    var body: some View {
        NavigationStack {
            Form {
                Section("Escalate to") {
                    LabeledContent("Provider", value: request.target.provider.displayName)
                    LabeledContent("Model", value: request.target.modelID)
                    LabeledContent("Where", value: request.target.isLocalNetwork ? "Your machine" : "Cloud")
                    Text(request.target.provider.retentionNote)
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Cost") {
                    LabeledContent("Projected", value: request.cost == 0
                        ? "Free (local)"
                        : String(format: "$%.4f", NSDecimalNumber(decimal: request.cost).doubleValue))
                    if request.preview.originalTokens > request.preview.compressedTokens {
                        LabeledContent("Compression",
                            value: "\(request.preview.originalTokens)→\(request.preview.compressedTokens) (\(String(format: "%.1f", request.preview.factor))×)")
                    }
                }
                if !request.preview.keptText.isEmpty {
                    Section("Payload being sent") {
                        Text(request.preview.keptText)
                            .font(.caption).textSelection(.enabled)
                    }
                }
            }
            .navigationTitle("Send to remote?")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Keep local") { gate.resolve(false) } }
                ToolbarItem(placement: .confirmationAction) { Button("Send") { gate.resolve(true) } }
            }
            .interactiveDismissDisabled()
            .onDisappear { gate.resolve(false) }   // safety: never leak the continuation
        }
        .frame(minWidth: 420, minHeight: 420)
    }
}
