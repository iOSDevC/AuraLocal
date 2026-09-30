import SwiftUI
import AuraCore

// MARK: - ModelSearchSheet
//
// A HuggingFace model search with a device-compatibility filter, in the spirit of LM Studio / Bionic.
// Each row shows ModelCompatibilityChecker's verdict: whether AuraLocal can load the repo at all (format, pinned
// runtimes, image pipelines through AuraImageGen) and whether it fits this device — a 6.8 GB FLUX peaks ~2× its
// weights. Reports are fetched lazily per visible row, so the list stays responsive.

struct ModelSearchSheet: View {
    /// Called when the user picks a model: (repo id, base-model guess for diffusion).
    let onSelect: (String, String?) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var vm = SearchVM()

    private let profile = HardwareProfile.current()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                controls
                Divider()
                list
            }
            .navigationTitle("Search HuggingFace")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
        }
        .frame(minWidth: 460, minHeight: 520)
    }

    private var controls: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search models (e.g. flux schnell, qwen gguf)", text: $vm.query)
                    .textFieldStyle(.plain)
                    .onSubmit { Task { await vm.search(profile: profile) } }
                if vm.isSearching { ProgressView().controlSize(.small) }
            }
            .padding(8)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))

            HStack {
                Picker("Sort", selection: $vm.sort) {
                    ForEach(HuggingFaceSearch.Sort.allCases, id: \.self) { Text($0.label).tag($0) }
                }
                .pickerStyle(.menu).fixedSize()
                Spacer()
                Toggle(isOn: $vm.onlyCompatible) {
                    Text("Only what runs · \(Int(profile.availableMemoryGB.rounded())) GB free")
                        .font(.caption)
                }
                .toggleStyle(.switch).controlSize(.mini)
            }
            .onChange(of: vm.sort) { Task { await vm.search(profile: profile) } }
        }
        .padding()
    }

    @ViewBuilder private var list: some View {
        if let error = vm.error {
            ContentUnavailableView("Search failed", systemImage: "wifi.exclamationmark", description: Text(error))
        } else if vm.visibleHits.isEmpty && !vm.isSearching {
            ContentUnavailableView("No models", systemImage: "magnifyingglass",
                                   description: Text("Try a different query."))
        } else {
            List(vm.visibleHits) { hit in
                ModelSearchRow(hit: hit, verdict: vm.reports[hit.id]) {
                    onSelect(hit.id, baseModelGuess(hit))
                    dismiss()
                }
                .task { await vm.loadReport(for: hit, profile: profile) }
            }
            .listStyle(.plain)
        }
    }

    /// Diffusion repos need a --base-model; guess it from the name (schnell vs dev). nil for LLMs.
    private func baseModelGuess(_ hit: HFModelHit) -> String? {
        guard hit.isDiffusion else { return nil }
        let lower = hit.id.lowercased()
        if lower.contains("schnell") { return "schnell" }
        if lower.contains("dev") { return "dev" }
        return "schnell"
    }
}

// MARK: - Row

private struct ModelSearchRow: View {
    let hit: HFModelHit
    let verdict: CompatibilityReport?
    let use: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(hit.id).font(.callout.weight(.medium)).lineLimit(1).truncationMode(.middle)
                Spacer()
                Button("Use", action: use).buttonStyle(.borderedProminent).controlSize(.small)
                    .disabled(verdict?.status == .notRunnable)
            }
            HStack(spacing: 10) {
                badge(hit.isDiffusion ? "photo" : "text.bubble", hit.isDiffusion ? "image" : "text")
                if hit.gated { badge("lock.fill", "gated", tint: .orange) }
                Label("\(hit.likes)", systemImage: "heart").labelStyle(.titleAndIcon)
                Label(compact(hit.downloads), systemImage: "arrow.down.circle")
                Spacer()
                verdictLabel
            }
            .font(.caption).foregroundStyle(.secondary)
            if let verdict {
                Text(verdict.headline).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
            }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder private var verdictLabel: some View {
        if let verdict {
            Label(verdict.status.label, systemImage: verdict.status.systemImage)
                .font(.caption.weight(.medium))
                .foregroundStyle(color(for: verdict.status))
        } else {
            HStack(spacing: 3) { ProgressView().controlSize(.mini); Text("checking…") }
        }
    }

    private func badge(_ icon: String, _ text: String, tint: Color = .secondary) -> some View {
        Label(text, systemImage: icon).foregroundStyle(tint)
    }
    private func color(for status: CompatibilityVerdict) -> Color {
        switch status {
        case .runnable: .green
        case .runnableWithCaveats: .orange
        case .notRunnable: .red
        case .unknown: .secondary
        }
    }
    private func compact(_ n: Int) -> String {
        n >= 1_000_000 ? String(format: "%.1fM", Double(n) / 1e6)
        : n >= 1_000 ? String(format: "%.1fk", Double(n) / 1e3) : "\(n)"
    }
}

// MARK: - View model

@Observable
final class SearchVM {
    var query = "flux schnell mflux"
    var sort: HuggingFaceSearch.Sort = .downloads
    var onlyCompatible = false
    var isSearching = false
    var error: String?

    private(set) var hits: [HFModelHit] = []
    /// repo id → the checker's report (once fetched).
    private(set) var reports: [String: CompatibilityReport] = [:]
    private let inspector = ModelCompatibilityChecker()

    var visibleHits: [HFModelHit] {
        guard onlyCompatible else { return hits }
        return hits.filter { reports[$0.id].map { $0.status != .notRunnable } ?? true }   // keep unknown until checked
    }

    @MainActor
    func search(profile: HardwareProfile) async {
        error = nil; isSearching = true; defer { isSearching = false }
        do {
            hits = try await HuggingFaceSearch.search(query, limit: 30, sort: sort)
            reports = [:]
        } catch {
            self.error = error.localizedDescription
            hits = []
        }
    }

    @MainActor
    func loadReport(for hit: HFModelHit, profile: HardwareProfile) async {
        guard reports[hit.id] == nil else { return }
        let report = await inspector.check(hit.id, on: .thisDevice(profile: profile))
        guard !Task.isCancelled else { return }
        reports[hit.id] = report
    }
}
