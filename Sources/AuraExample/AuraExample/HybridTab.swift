import SwiftUI
import AuraCore

// MARK: - Hybrid Tab
//
// A first-class home for the hybrid local+remote line — previously reachable only
// via Models -> pick a domain model -> Test -> gear. Surfaces escalation, the
// detected local providers, the on-device system tools, and the cost history —
// all over already-public AuraCore APIs (read-only panels + one escalation entry).

struct HybridTab: View {
    @ObservedObject private var ledger = CostLedger.shared
    @ObservedObject private var consentGate = HybridSettings.shared.consent

    // Quick ask (escalation)
    @State private var prompt = ""
    @State private var response = ""
    @State private var receipt: String?
    @State private var errorText: String?
    @State private var isRunning = false

    // Read-only discovery panels
    @State private var providers: [LocalProviderStatus] = []
    @State private var isProbing = false

    @State private var showSettings = false
    @State private var infoSheet: HybridInfoSheet?

    var body: some View {
        NavigationStack {
            Form {
                askSection
                providersSection
                historySection
                referenceSection
            }
            .navigationTitle("Hybrid")
            .toolbar {
                ToolbarItem {
                    Button { showSettings = true } label: { Image(systemName: "gearshape") }
                }
            }
            .sheet(isPresented: $showSettings) { HybridSettingsView() }
            .sheet(item: $infoSheet) { sheet in
                switch sheet {
                case .onDeviceTools: OnDeviceToolsView()
                case .integrate: IntegrateView()
                }
            }
            .sheet(item: $consentGate.pending) { request in
                ConsentSheet(request: request, gate: consentGate)
            }
            .task {
                if providers.isEmpty { await refreshProviders() }
            }
        }
        #if os(macOS)
        .frame(minWidth: 460, minHeight: 560)   // wider than an iPhone screen, so Mac only
        #endif
    }

    // MARK: - Quick ask

    @ViewBuilder
    private var askSection: some View {
        Section {
            TextEditor(text: $prompt)
                .font(.callout)
                .frame(minHeight: 70)
            Button { Task { await runRemote() } } label: {
                if isRunning {
                    HStack { ProgressView(); Text("Asking…") }
                } else {
                    Label("Ask a bigger model", systemImage: "cloud.bolt")
                }
            }
            .disabled(prompt.isEmpty || isRunning)

            if let receipt {
                Text(receipt).font(.caption2).foregroundStyle(.secondary)
            }
            if !response.isEmpty {
                Text(response).font(.callout.monospaced()).textSelection(.enabled)
            }
            if let errorText {
                Label(errorText, systemImage: "xmark.octagon").font(.caption).foregroundStyle(.red)
            }
        } header: {
            Text("Ask a bigger model")
        } footer: {
            Text("Local-first: prefers your own llama-server / Ollama, then cloud (if allowed and a key is set). "
                 + "Context is compressed to save tokens; cloud sends redact obvious secrets by default "
                 + "and ask consent first, showing the exact payload.")
        }
    }

    @MainActor
    private func runRemote() async {
        isRunning = true; response = ""; errorText = nil; receipt = nil
        defer { isRunning = false }

        let policy = HybridSettings.shared.policy
        let candidates = await HybridEscalator.candidateTargets(policy: policy)
        guard let target = candidates.first else {
            errorText = "No provider available. Start Ollama/llama-server, or add an Anthropic or OpenAI key in Settings (⚙) and allow cloud."
            return
        }

        // Built once: the consent sheet shows exactly what is sent.
        let escalator = HybridEscalator()
        let payload = escalator.preparePayload(
            for: target, context: "", question: prompt, maxTokens: 512,
            keepRatio: policy.keepRatio, redactPII: policy.redactsPII(for: target))

        // Cloud targets require explicit consent; LAN is trusted.
        if !target.isLocalNetwork {
            let cost = ledger.projectedCost(
                target: target, inputTokens: payload.estimatedInputTokens, maxOutput: payload.maxTokens)
            let offer = EscalationOffer(
                reason: .userRequested, projectedCostUSD: cost,
                sessionSpentUSD: ledger.sessionCostUSD, unpricedRecordCount: ledger.unpricedRecordCount,
                costCapUSD: policy.costCapUSDPerSession, payload: payload)
            let approved = await HybridSettings.shared.consent.requestConsent(
                target: target, preview: payload.compression, offer: offer)
            guard approved else { errorText = "Escalation declined — keeping local."; return }
        }

        do {
            let result = try await escalator.escalate(payload, to: target) { response = $0 }
            var r = "via \(result.providerName) · \(target.modelID)"
            if let u = result.usage { r += " · \(u.inputTokens) in / \(u.outputTokens) out" }
            let c = result.compression
            if c.originalTokens > c.compressedTokens {
                r += " · compressed \(c.originalTokens)→\(c.compressedTokens) (\(String(format: "%.1f", c.factor))×)"
            }
            if result.redactedPIICount > 0 { r += " · redacted \(result.redactedPIICount) PII" }
            if result.fromCache { r += " · (cached, $0)" }
            receipt = r
        } catch {
            errorText = error.localizedDescription
        }
    }

    // MARK: - Local providers (read-only)

    @ViewBuilder
    private var providersSection: some View {
        Section {
            if providers.allSatisfy({ !$0.reachable }) {
                Text("No local provider detected. Start Ollama (:11434) or llama-server (:8080).")
                    .font(.caption).foregroundStyle(.secondary)
            }
            ForEach(providers) { provider in providerRow(provider) }
            Button { Task { await refreshProviders() } } label: {
                if isProbing {
                    HStack { ProgressView(); Text("Probing…") }
                } else {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
            .disabled(isProbing)
        } header: {
            HStack {
                Text("Providers on your network")
                Spacer()
                networkPill
            }
        }
    }

    private func providerRow(_ p: LocalProviderStatus) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(p.kind == .ollama ? "Ollama" : "llama-server").bold()
                Spacer()
                if p.isAvailable {
                    Label("\(p.models.count) model\(p.models.count == 1 ? "" : "s")",
                          systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(.green)
                } else if p.reachable {
                    Label("Reachable, no models", systemImage: "circle")
                        .font(.caption).foregroundStyle(.orange)
                } else {
                    Label("Not running", systemImage: "circle.slash")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Text(p.baseURL.absoluteString).font(.caption2).foregroundStyle(.secondary)
            if let version = p.version {
                Text("v\(version)").font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private var networkPill: some View {
        let online = NetworkMonitor.shared.isOnline
        return Label(online ? "Online" : "Offline", systemImage: online ? "wifi" : "wifi.slash")
            .font(.caption2).foregroundStyle(online ? .green : .secondary)
    }

    @MainActor
    private func refreshProviders() async {
        isProbing = true; defer { isProbing = false }
        providers = await LocalProviderDetector.detectAll()
    }

    // MARK: - Reference (opened as bottom sheets)

    private var referenceSection: some View {
        Section {
            Button { infoSheet = .onDeviceTools } label: {
                Label("On-device tools", systemImage: "wrench.and.screwdriver")
            }
            Button { infoSheet = .integrate } label: {
                Label("Integrate", systemImage: "shippingbox")
            }
        } header: {
            Text("Reference")
        }
    }

    // MARK: - Escalation history (read-only)

    @ViewBuilder
    private var historySection: some View {
        Section {
            if ledger.records.isEmpty {
                Text("No escalations yet.").font(.caption).foregroundStyle(.secondary)
            }
            ForEach(ledger.records) { record in
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text(record.provider).font(.caption).bold()
                        Spacer()
                        Text(Self.costText(record.costUSD))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    HStack(spacing: 8) {
                        if let usage = record.usage {
                            Text("\(usage.inputTokens) in / \(usage.outputTokens) out").font(.caption2)
                        }
                        if let ratio = record.compressionRatio {
                            Text("compressed \(String(format: "%.1f", 1 / max(ratio, 0.0001)))×").font(.caption2)
                        }
                    }
                    .foregroundStyle(.secondary)
                }
            }
        } header: {
            HStack {
                Text("Escalation history")
                Spacer()
                if !ledger.records.isEmpty {
                    Text("\(ledger.sessionTokens) tok this session").font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }

    private static func costText(_ cost: Decimal?) -> String {
        guard let cost else { return "unpriced" }
        return cost == 0 ? "$0" : String(format: "$%.4f", NSDecimalNumber(decimal: cost).doubleValue)
    }
}
