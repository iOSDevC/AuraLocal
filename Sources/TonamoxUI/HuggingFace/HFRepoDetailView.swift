import SwiftUI
import TonamoxCore
#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

/// Everything the checker found about one repo on the search's device, blockers first, plus a Download section
/// when the repo runs there. Fetches the repo the first time it appears. Brings no navigation container.
public struct HFRepoDetailView: View {
    let repoID: String
    let search: HFModelSearch
    @State private var showsToken = false
    @State private var trial: Model?

    public init(repoID: String, search: HFModelSearch) {
        self.repoID = repoID
        self.search = search
    }

    public var body: some View {
        Group {
            if let report = search.reports[repoID] {
                HFReportForm(result: report, showsToken: $showsToken, trial: $trial)
            } else {
                ProgressView("Checking \(repoID)…")
            }
        }
        .task(id: repoID) { await search.loadReport(for: repoID) }
        .navigationTitle(repoID)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Hugging Face token", systemImage: "key") { showsToken = true }
            }
        }
        // Sheets live here, not in the Form's rows, which a List may recycle while scrolling.
        .sheet(isPresented: $showsToken) { HFTokenSheet() }
        .sheet(item: $trial) { ModelTrialView(candidate: $0) }
    }
}

/// The checker's report as a form.
struct HFReportForm: View {
    let result: CompatibilityReport
    @Binding var showsToken: Bool
    @Binding var trial: Model?
    @State private var copied = false

    var body: some View {
        Form {
            summary
            if let suggestion = result.suggestedEntry,
               result.status == .runnable || result.status == .runnableWithCaveats {
                HFDownloadSection(result: result, suggestion: suggestion, showsToken: $showsToken, trial: $trial)
                    .id(result.repoID)
            }
            Section("Findings") {
                ForEach(result.findings) { FindingRow(item: $0) }
            }
            modelSection
            memorySection
            if !result.quantFits.isEmpty {
                Section("Quants on \(result.target.displayName)") {
                    ForEach(result.quantFits) { QuantRow(line: $0) }
                }
            }
            licenseSection
            actions
        }
        .formStyle(.grouped)
    }

    private var summary: some View {
        Section {
            VerdictBadge(status: result.status)
                .font(.title3)
            Text(result.headline)
                .font(.headline)
            LabeledContent("Device", value: result.target.displayName)
            LabeledContent("Memory budget", value: result.target.budgetText)
            Text(result.target.source)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var modelSection: some View {
        Section("Model") {
            LabeledContent("Format", value: result.weightFormat.label)
            if let category = result.modelCategory {
                LabeledContent("Loads as", value: category == .vision ? "Vision (image + text)" : "Text")
            }
            if let family = result.overview.family { LabeledContent("Architecture", value: family) }
            if let layers = result.overview.layers { LabeledContent("Layers", value: "\(layers)") }
            if let context = result.overview.trainedContext { LabeledContent("Trained context", value: "\(context) tokens") }
            if let heads = result.overview.kvHeads { LabeledContent("KV heads", value: "\(heads)") }
            if let dim = result.overview.headDim { LabeledContent("Head dim", value: "\(dim)") }
            LabeledContent("Vision weights", value: result.overview.hasVision ? "Yes" : "No")
            if let quantization = result.overview.quantization { LabeledContent("Quantization", value: quantization) }
        }
    }

    @ViewBuilder private var memorySection: some View {
        if let fit = result.weightsFit {
            Section("Memory on \(result.target.displayName)") {
                if let bytes = result.weightsBytes { LabeledContent("Weights", value: CompatibilityReport.gigabytesText(bytes)) }
                LabeledContent("Fit", value: fit.summary)
                if let speed = fit.decodeText {
                    LabeledContent("Decode estimate", value: speed)
                }
            }
        }
    }

    private var licenseSection: some View {
        Section("License") {
            LabeledContent("License", value: result.licenseID ?? "Not declared")
            if let name = result.licenseName { LabeledContent("Name", value: name) }
            LabeledContent("Gated", value: result.gatedMode.map { "Yes (\($0))" } ?? "No")
        }
    }

    private var actions: some View {
        Section {
            if let entry = result.suggestedEntry {
                Button(copied ? "Copied" : "Copy catalog entry", systemImage: "doc.on.doc") {
                    Self.copy(entry.jsonText())
                    copied = true
                }
                DisclosureGroup("models.json entry") {
                    Text(entry.jsonText())
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                }
            } else {
                Label("No catalog entry: the model can't run here.", systemImage: "doc.badge.ellipsis")
                    .foregroundStyle(.secondary)
            }
            if let url = result.huggingFaceURL {
                Link(destination: url) {
                    Label("Open on Hugging Face", systemImage: "safari")
                }
            }
        }
    }

    private static func copy(_ text: String) {
        #if canImport(AppKit)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #elseif canImport(UIKit)
        UIPasteboard.general.string = text
        #endif
    }
}

struct FindingRow: View {
    let item: CompatibilityFinding

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(item.title, systemImage: item.level.systemImage)
                .foregroundStyle(tint)
                .font(.body.weight(item.level == .blocker ? .semibold : .regular))
            Text(item.detail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(item.level.label): \(item.title). \(item.detail)")
    }

    private var tint: Color {
        switch item.level {
        case .blocker: .red
        case .caveat: .orange
        case .info: .primary
        }
    }
}

struct QuantRow: View {
    let line: QuantFit

    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(line.option.label ?? line.option.firstPath)
                    .font(.body.monospaced())
                Text(CompatibilityReport.gigabytesText(line.option.totalBytes)
                     + (line.option.isSplit ? " · \(line.option.paths.count) parts, not loadable" : ""))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text(line.memory.summary)
                .font(.caption)
                .foregroundStyle(line.isUsable && line.memory.rating != .tooLarge ? Color.primary : Color.secondary)
        }
        .accessibilityElement(children: .combine)
    }
}
