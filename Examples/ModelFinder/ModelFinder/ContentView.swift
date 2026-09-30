import SwiftUI
import AuraCore

struct ContentView: View {
    @State private var finder = SearchSession()
    @State private var selection: String?

    var body: some View {
        NavigationSplitView {
            ResultsList(finder: finder, selection: $selection)
                .navigationTitle("Model Finder")
                #if os(macOS)
                .navigationSplitViewColumnWidth(min: 380, ideal: 440)
                #endif
        } detail: {
            detail
        }
        .task { finder.search() }
    }

    @ViewBuilder private var detail: some View {
        if let selection {
            Group {
                if let result = finder.reports[selection] {
                    ReportDetailView(result: result)
                } else {
                    ProgressView("Checking \(selection)…")
                }
            }
            .task(id: selection) { await finder.loadReport(for: selection) }
        } else {
            ContentUnavailableView(
                "Pick a model", systemImage: "cpu",
                description: Text("Search Hugging Face, then select a result to see whether it runs here and why."))
        }
    }
}

// MARK: - Results

struct ResultsList: View {
    @Bindable var finder: SearchSession
    @Binding var selection: String?

    var body: some View {
        List(selection: $selection) {
            Section {
                SearchControls(finder: finder)
            }
            if let error = finder.searchError {
                Label(error, systemImage: "wifi.exclamationmark")
                    .foregroundStyle(.red)
            }
            Section(finder.matches.isEmpty ? "" : "\(finder.visibleMatches.count) results") {
                ForEach(finder.visibleMatches) { match in
                    ResultRow(match: match, outcome: finder.reports[match.id])
                        .tag(match.id)
                        .task(id: match.id) { await finder.loadReport(for: match.id) }
                }
            }
        }
        .searchable(text: $finder.query, prompt: "Search Hugging Face models")
        .onSubmit(of: .search) { finder.search() }
        .onChange(of: finder.formatChoice) { finder.search() }
        .onChange(of: finder.ordering) { finder.search() }
        .overlay {
            if finder.isSearching && finder.matches.isEmpty {
                ProgressView("Searching…")
            } else if !finder.isSearching && finder.matches.isEmpty && finder.searchError == nil {
                ContentUnavailableView.search(text: finder.query)
            }
        }
    }
}

struct SearchControls: View {
    @Bindable var finder: SearchSession

    var body: some View {
        Picker("Format", selection: $finder.formatChoice) {
            ForEach(FormatFilter.allCases) { Text($0.title).tag($0) }
        }
        .pickerStyle(.segmented)

        Picker("Sort", selection: $finder.ordering) {
            ForEach(HuggingFaceSearch.Sort.allCases, id: \.self) { Text($0.label).tag($0) }
        }

        Picker("Device", selection: $finder.deviceID) {
            ForEach(finder.devices) { target in
                Text("\(target.displayName) · \(target.budgetText)").tag(target.id)
            }
        }
        .accessibilityHint("Judges every result against this device's memory budget")

        Text(finder.currentTarget.source)
            .font(.caption)
            .foregroundStyle(.secondary)

        Toggle("Hide models that won't run", isOn: $finder.onlyRunnable)
    }
}

struct ResultRow: View {
    let match: HFModelHit
    let outcome: CompatibilityReport?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(match.id)
                .font(.body.weight(.medium))
                .lineLimit(1)
                .truncationMode(.middle)
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                VerdictBadge(status: outcome?.status)
                Text(outcome?.headline ?? "Checking…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            HStack(spacing: 12) {
                if let outcome { Text(outcome.weightFormat.label) }
                Label("\(match.downloads)", systemImage: "arrow.down.circle")
                Label("\(match.likes)", systemImage: "heart")
                if match.gated { Label("Gated", systemImage: "lock.fill") }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityText)
    }

    private var accessibilityText: String {
        guard let outcome else { return "\(match.id). Checking compatibility." }
        return "\(match.id). \(outcome.status.label). \(outcome.headline)."
            + (match.gated ? " Gated repository." : "")
    }
}

// MARK: - Badge

struct VerdictBadge: View {
    let status: CompatibilityVerdict?

    var body: some View {
        if let status {
            Label(status.label, systemImage: status.systemImage)
                .labelStyle(.titleAndIcon)
                .font(.caption.weight(.semibold))
                .foregroundStyle(tint(status))
                .accessibilityLabel("Verdict: \(status.label)")
        } else {
            ProgressView()
                .controlSize(.small)
                .accessibilityLabel("Checking")
        }
    }

    private func tint(_ value: CompatibilityVerdict) -> Color {
        switch value {
        case .runnable: .green
        case .runnableWithCaveats: .orange
        case .notRunnable: .red
        case .unknown: .secondary
        }
    }
}
