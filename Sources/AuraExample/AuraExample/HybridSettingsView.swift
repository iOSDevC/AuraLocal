import SwiftUI
import AuraCore
import AuraUI

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
    @State private var hasRetiredGitHubToken = false

    /// GitHub Models was retired on 2026-07-30; AuraLocal no longer reads this account.
    private static let retiredGitHubModelsAccount = "cloud.github-models"

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
                    Picker("Redact secrets & PII", selection: $settings.policy.piiRedaction) {
                        Text("Cloud only").tag(EscalationPolicy.RedactionScope.cloudOnly)
                        Text("All targets").tag(EscalationPolicy.RedactionScope.allTargets)
                        Text("Never").tag(EscalationPolicy.RedactionScope.never)
                    }
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
                    if hasRetiredGitHubToken {
                        retiredGitHubModelsRow
                    }
                } header: {
                    Text("Cloud API keys (BYOK)")
                } footer: {
                    Text("Keys are stored in the Keychain (this device only) — never in code, files, or logs.")
                }

                HuggingFaceTokenSection()

                Section {
                    LabeledContent("Remote tokens", value: "\(ledger.sessionTokens)")
                    LabeledContent("Cost", value: sessionCostText)
                    LabeledContent("Escalations", value: "\(ledger.sessionRecordCount)")
                    Button("Start new session") { ledger.startNewSession() }
                        .disabled(ledger.sessionRecordCount == 0)
                } header: {
                    Text("This session")
                } footer: {
                    Text("The cost cap counts spend since the session started. Starting a new one keeps the history.")
                }
            }
            .navigationTitle("Hybrid settings")
            .toolbar {
                if !isEmbedded {
                    ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 460, minHeight: 520)
        #endif
        .onAppear { hasRetiredGitHubToken = KeychainStore.hasKey(for: Self.retiredGitHubModelsAccount) }
    }

    private var sessionCostText: String {
        let priced = String(format: "$%.4f", NSDecimalNumber(decimal: ledger.sessionCostUSD).doubleValue)
        let unpriced = ledger.unpricedRecordCount
        return unpriced == 0 ? priced : "\(priced) + \(unpriced) unpriced"
    }

    private var retiredGitHubModelsRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("GitHub Models")
            Text("GitHub retired GitHub Models on 2026-07-30. A token for it is still stored in the Keychain and is no longer used.")
                .font(.caption).foregroundStyle(.secondary)
            Button("Delete stored GitHub token", role: .destructive) {
                KeychainStore.delete(for: Self.retiredGitHubModelsAccount)
                hasRetiredGitHubToken = KeychainStore.hasKey(for: Self.retiredGitHubModelsAccount)
            }
            .font(.caption)
            .buttonStyle(.borderless)   // only the button deletes, not a tap anywhere on the row
        }
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

/// Shown before any cloud send: the exact payload with its redaction count, the
/// projected cost, and the provider's retention note. The user chooses to send or keep local.
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
                    LabeledContent("Projected", value: projectedCostText)
                    if request.cost == nil {
                        Text("No price is set for this model, so the cost cap cannot check it.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let offer = request.offer {
                        LabeledContent("Session spend", value: Self.sessionSpendText(offer))
                    }
                    if isOverCap {
                        Text("Sending goes over your session cost cap.")
                            .font(.caption).foregroundStyle(.orange)
                    }
                    if request.preview.originalTokens > request.preview.compressedTokens {
                        LabeledContent("Compression",
                            value: "\(request.preview.originalTokens)→\(request.preview.compressedTokens) (\(String(format: "%.1f", request.preview.factor))×)")
                    }
                }
                if let payload = request.payload {
                    payloadSection(payload)
                } else if !request.preview.keptText.isEmpty {
                    Section("Context being sent") {
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
        #if os(macOS)
        .frame(minWidth: 420, minHeight: 420)
        #endif
    }

    private func payloadSection(_ payload: EscalationPayload) -> some View {
        Section {
            LabeledContent("Redaction", value: Self.redactionText(payload))
            LabeledContent("Max output", value: "\(payload.maxTokens) tokens")
            if let system = payload.systemPrompt {
                VStack(alignment: .leading, spacing: 4) {
                    Text("System").font(.caption.bold())
                    Text(system).font(.caption.monospaced()).textSelection(.enabled)
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("User").font(.caption.bold())
                Text(payload.userMessage).font(.caption.monospaced()).textSelection(.enabled)
            }
        } header: {
            Text("Exact payload")
        } footer: {
            Text("Sent exactly as shown: \(Self.byteCount(payload)) bytes of text. "
                 + "If this exact request was answered earlier this session, "
                 + "that answer is reused and nothing is sent.")
        }
    }

    private static func redactionText(_ payload: EscalationPayload) -> String {
        guard payload.redactionEnabled else { return "Off" }
        switch payload.redactedPIICount {
        case 0: return "On · nothing found"
        case 1: return "On · 1 item replaced"
        default: return "On · \(payload.redactedPIICount) items replaced"
        }
    }

    private static func byteCount(_ payload: EscalationPayload) -> Int {
        (payload.systemPrompt?.utf8.count ?? 0) + payload.userMessage.utf8.count
    }

    private var isOverCap: Bool {
        guard let offer = request.offer else { return false }
        if offer.reason == .costCapped { return true }
        guard let cost = request.cost else { return false }
        return cost > 0 && offer.sessionSpentUSD + cost > offer.costCapUSD
    }

    private static func sessionSpendText(_ offer: EscalationOffer) -> String {
        let spent = String(format: "$%.4f", NSDecimalNumber(decimal: offer.sessionSpentUSD).doubleValue)
        let cap = String(format: "$%.2f", NSDecimalNumber(decimal: offer.costCapUSD).doubleValue)
        let unpriced = offer.unpricedRecordCount == 0 ? "" : " + \(offer.unpricedRecordCount) unpriced"
        return "\(spent)\(unpriced) of \(cap) cap"
    }

    private var projectedCostText: String {
        guard let cost = request.cost else { return "Cost unknown" }
        if cost == 0 { return request.target.isLocalNetwork ? "Free (local)" : "$0" }
        return String(format: "$%.4f", NSDecimalNumber(decimal: cost).doubleValue)
    }
}
