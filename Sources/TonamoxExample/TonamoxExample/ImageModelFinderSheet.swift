import SwiftUI
import TonamoxCore
import TonamoxImageGen

// MARK: - ImageModelFinderSheet
//
// Finds Core ML Stable Diffusion repos on Hugging Face and judges each for this device with
// TonamoxImageGen/CoreMLDiffusionInspector: a listing and one ~4 KB metadata.json per folder, no weights.
// "Use" adds the chosen folder to the Image tab's picker and starts its download (not in the Simulator).
// A pasted owner/repo or repository URL can be inspected directly, whether or not the search lists it.

struct ImageModelFinderSheet: View {
    let state: ImageGenState
    @State private var browser = ImageModelFinderVM()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack {
                        Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                        TextField("Search (e.g. dreamshaper, sd-turbo)", text: $browser.query)
                            .onSubmit { Task { await browser.search() } }
                            #if os(iOS)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            #endif
                        if browser.isSearching { ProgressView().controlSize(.small) }
                    }
                    Toggle("Compatible only", isOn: $browser.onlyCompatible)
                    if let error = browser.error {
                        Label(error, systemImage: "xmark.octagon").foregroundStyle(.red).font(.callout)
                    }
                } footer: {
                    Text("Searching sends your text to huggingface.co. Verdicts are for \(browser.target.displayName), "
                         + "budget \(browser.target.budgetText).")
                }
                Section {
                    if let repoID = browser.pastedRepoID {
                        NavigationLink(value: repoID) {
                            Label("Inspect \(repoID)", systemImage: "doc.text.magnifyingglass")
                        }
                    }
                    ForEach(browser.visibleHits) { hit in
                        NavigationLink(value: hit.id) {
                            ImageModelSearchRow(hit: hit, report: browser.reports[hit.id],
                                                failure: browser.failures[hit.id])
                        }
                        .disabled(hit.candidateFolders.isEmpty)
                        .task { if !hit.candidateFolders.isEmpty { await browser.loadReport(for: hit.id) } }
                    }
                    if browser.visibleHits.isEmpty && !browser.isSearching {
                        Text("No Core ML Stable Diffusion repos found.").foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Hugging Face image models")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .navigationDestination(for: String.self) { id in
                ImageModelVariantsView(report: browser.reports[id], failure: browser.failures[id]) { variant in
                    adopt(variant)
                }
                .task { await browser.loadReport(for: id) }
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
            .task { if browser.hits.isEmpty { await browser.search() } }
        }
        #if os(macOS)
        .frame(minWidth: 520, minHeight: 600)
        #endif
    }

    private func adopt(_ variant: CoreMLDiffusionVariant) {
        do {
            state.adopt(try CoreMLDiffusionModel(variant: variant))
            dismiss()
        } catch {
            state.show(error)
            dismiss()
        }
    }
}

// MARK: - Rows and detail

private struct ImageModelSearchRow: View {
    let hit: CoreMLDiffusionSearchHit
    let report: CoreMLDiffusionRepoReport?
    let failure: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(hit.id).font(.callout.weight(.medium)).lineLimit(2)
            HStack(spacing: 8) {
                Label("\(hit.downloads)", systemImage: "arrow.down.circle")
                if hit.isGated { Label("Gated", systemImage: "lock") }
            }
            .font(.caption).foregroundStyle(.secondary)
            verdict.font(.caption)
        }
    }

    @ViewBuilder private var verdict: some View {
        if let reason = hit.unsupportedReason {
            Label(reason, systemImage: "xmark.octagon").foregroundStyle(.secondary)
        } else if let failure {
            Label(failure, systemImage: "questionmark.circle").foregroundStyle(.secondary)
        } else if let best = report?.recommended {
            Label("\(best.verdict.label) · \(best.folder.path.isEmpty ? "(root)" : best.folder.path)",
                  systemImage: best.verdict.systemImage)
                .foregroundStyle(Self.color(best.verdict))
        } else if let reason = report?.unsupportedReason {
            Label(reason, systemImage: "xmark.octagon").foregroundStyle(.secondary)
        } else {
            Label("Checking…", systemImage: "hourglass").foregroundStyle(.secondary)
        }
    }

    static func color(_ verdict: CompatibilityVerdict) -> Color {
        switch verdict {
        case .runnable: .green
        case .runnableWithCaveats: .orange
        case .notRunnable: .red
        case .unknown: .secondary
        }
    }
}

private struct ImageModelVariantsView: View {
    let report: CoreMLDiffusionRepoReport?
    let failure: String?
    let onUse: (CoreMLDiffusionVariant) -> Void
    @State private var pending: CoreMLDiffusionVariant?

    var body: some View {
        List {
            if let report {
                Section {
                    Text("Revision \(report.revision.prefix(12)) · downloads are pinned to it.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let reason = report.unsupportedReason {
                        Label(reason, systemImage: "xmark.octagon")
                    }
                }
                ForEach(report.variants) { variant in
                    variantSection(variant, isRecommended: variant.id == report.recommended?.id)
                }
            } else if let failure {
                Label(failure, systemImage: "xmark.octagon").foregroundStyle(.red)
            } else {
                ProgressView("Inspecting…")
            }
        }
        .navigationTitle(report?.repoID ?? "Model")
        .alert("Use this model?", isPresented: isConfirming, presenting: pending) { variant in
            Button("Cancel", role: .cancel) {}
            Button("Use") { onUse(variant) }
        } message: { variant in
            Text("About \(Self.size(variant.folder.downloadBytes)) from huggingface.co (\(variant.repoID)). "
                 + "No account needed. License: \(variant.licenseName); read it on the model card. "
                 + "Prompts and images stay on this device.")
        }
    }

    private var isConfirming: Binding<Bool> {
        Binding { pending != nil } set: { if !$0 { pending = nil } }
    }

    private func variantSection(_ variant: CoreMLDiffusionVariant, isRecommended: Bool) -> some View {
        Section {
            Label(variant.verdict.label, systemImage: variant.verdict.systemImage)
                .foregroundStyle(ImageModelSearchRow.color(variant.verdict))
            Text(Self.facts(variant)).font(.caption).foregroundStyle(.secondary)
            ForEach(variant.findings) { finding in
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(finding.title).font(.callout)
                        Text(finding.detail).font(.caption).foregroundStyle(.secondary)
                    }
                } icon: {
                    Image(systemName: finding.level.systemImage)
                }
            }
            Link("License: \(variant.licenseName)", destination: variant.licenseURL).font(.caption)
            Button(Self.isBuiltIn(variant) ? "Use the built-in model" : "Use this model") { pending = variant }
                .disabled(!variant.verdict.isRunnable)
        } header: {
            Text((variant.folder.path.isEmpty ? "(repo root)" : variant.folder.path)
                 + (isRecommended ? " · recommended" : ""))
        }
    }

    private static func facts(_ variant: CoreMLDiffusionVariant) -> String {
        let side = variant.unet?.imageSide.map { "\($0)×\($0)" } ?? "size unknown"
        let compute = variant.preferredComputeUnits == .neuralEngine ? "Neural Engine" : "GPU"
        let fit = variant.fit?.label ?? "fit unknown"
        return "\(size(variant.folder.downloadBytes)) · \(side) · \(compute) · \(fit) · "
            + "\(variant.recommendedSteps) steps"
    }

    /// The built-in model's own folder, which adopting maps to the built-in model.
    private static func isBuiltIn(_ variant: CoreMLDiffusionVariant) -> Bool {
        (try? CoreMLDiffusionModel(variant: variant))?.isBuiltIn ?? false
    }

    private static func size(_ bytes: Int64?) -> String {
        bytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "unknown size"
    }
}

// MARK: - ImageModelFinderVM

@Observable
final class ImageModelFinderVM {
    var query = ""
    var onlyCompatible = false
    private(set) var isSearching = false
    private(set) var error: String?
    private(set) var hits: [CoreMLDiffusionSearchHit] = []
    /// repo id → report, once inspected.
    private(set) var reports: [String: CoreMLDiffusionRepoReport] = [:]
    private(set) var failures: [String: String] = [:]
    /// Measured once; in the Simulator this is the Mac's memory, so fit verdicts there mean little.
    let target = DevicePreset.thisDevice()
    private let inspector = CoreMLDiffusionInspector()

    /// The query as a repo id, when it is `owner/repo` or a Hugging Face repository URL.
    var pastedRepoID: String? {
        ModelCompatibilityChecker.repoID(from: query)
    }

    /// "Compatible only" keeps hits not yet inspected, so rows can still load their verdict.
    var visibleHits: [CoreMLDiffusionSearchHit] {
        guard onlyCompatible else { return hits }
        return hits.filter { hit in
            guard !hit.candidateFolders.isEmpty, !hit.isGated else { return false }
            return reports[hit.id]?.recommended.map { $0.verdict.isRunnable } ?? true
        }
    }

    func search() async {
        error = nil
        isSearching = true
        defer { isSearching = false }
        do {
            hits = try await inspector.search(query)
        } catch is CancellationError {
            return
        } catch {
            self.error = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            hits = []
        }
    }

    func loadReport(for repoID: String) async {
        guard reports[repoID] == nil, failures[repoID] == nil else { return }
        do {
            reports[repoID] = try await inspector.inspect(repoID, on: target)
        } catch is CancellationError {
            return
        } catch {
            let localized = error as? LocalizedError
            failures[repoID] = [localized?.errorDescription, localized?.recoverySuggestion]
                .compactMap { $0 }.joined(separator: " ")
        }
    }
}
