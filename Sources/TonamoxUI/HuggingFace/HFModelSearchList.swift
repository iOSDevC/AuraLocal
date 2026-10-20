import SwiftUI
import TonamoxCore

/// Hugging Face search results with a verdict for the chosen device on every row. Selecting a row sets
/// `selection` to its repo id; show ``HFRepoDetailView`` for it with a split view's detail column or a stack's
/// `navigationDestination(item:)`. Brings no navigation container of its own.
public struct HFModelSearchList: View {
    @Bindable var search: HFModelSearch
    @Binding var selection: String?

    public init(search: HFModelSearch, selection: Binding<String?>) {
        self.search = search
        self._selection = selection
    }

    public var body: some View {
        List(selection: $selection) {
            Section {
                HFSearchControls(search: search)
            }
            if let error = search.searchError {
                Label(error, systemImage: "wifi.exclamationmark")
                    .foregroundStyle(.red)
            }
            Section(search.matches.isEmpty ? "" : "\(search.visibleMatches.count) results") {
                ForEach(search.visibleMatches) { match in
                    HFResultRow(match: match, outcome: search.reports[match.id])
                        .tag(match.id)
                        .task(id: match.id) { await search.loadReport(for: match.id) }
                }
            }
        }
        .searchable(text: $search.query, prompt: "Search Hugging Face models")
        .onSubmit(of: .search) { search.search() }
        .onChange(of: search.formatChoice) { search.search() }
        .onChange(of: search.ordering) { search.search() }
        .overlay {
            if search.isSearching && search.matches.isEmpty {
                ProgressView("Searching…")
            } else if !search.isSearching && search.matches.isEmpty && search.searchError == nil {
                ContentUnavailableView.search(text: search.query)
            }
        }
    }
}

struct HFSearchControls: View {
    @Bindable var search: HFModelSearch

    var body: some View {
        Picker("Format", selection: $search.formatChoice) {
            ForEach(HFWeightFilter.allCases) { Text($0.title).tag($0) }
        }
        .pickerStyle(.segmented)

        Picker("Sort", selection: $search.ordering) {
            ForEach(HuggingFaceSearch.Sort.allCases, id: \.self) { Text($0.label).tag($0) }
        }

        if search.devices.count > 1 {
            Picker("Device", selection: $search.deviceID) {
                ForEach(search.devices) { target in
                    Text("\(target.displayName) · \(target.budgetText)").tag(target.id)
                }
            }
            .accessibilityHint("Judges every result against this device's memory budget")
        } else {
            LabeledContent(search.currentTarget.displayName, value: search.currentTarget.budgetText)
        }

        VStack(alignment: .leading, spacing: 2) {
            Text(search.currentTarget.source)
            if let filter = search.pipelineFilter {
                Text("Shows \(filter.pipelineTags.sorted().joined(separator: ", "))"
                     + (filter.admitsUndeclared ? " and repos that declare no task." : "."))
            }
            #if targetEnvironment(simulator)
            Text("Simulator: memory is the Mac's, not an iPhone's.")
            #endif
        }
        .font(.caption)
        .foregroundStyle(.secondary)

        Toggle("Compatible only", isOn: $search.compatibleOnly)
            .accessibilityHint("Hides results rated Won't run; unchecked results stay visible")
    }
}

struct HFResultRow: View {
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
