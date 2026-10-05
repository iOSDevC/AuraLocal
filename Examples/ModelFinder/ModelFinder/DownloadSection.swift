import SwiftUI
import AuraCore

/// Download controls for a repo the checker rated runnable: quant choice, size and free space, progress,
/// Cancel/Resume and errors.
struct DownloadSection: View {
    let result: CompatibilityReport
    let suggestion: CatalogEntry
    @Bindable var transfers: DownloadCenter
    let openSettings: () -> Void

    @State private var quantID: String?
    @State private var pendingPlan: DownloadPlan?
    @State private var planError: String?
    @State private var planIsGated = false
    @State private var freeBytes: Int64?

    var body: some View {
        Section {
            if isGGUF {
                quantPicker
            }
            if let candidate {
                status(for: candidate)
                controls(for: candidate)
            } else {
                Text("No single-file quant fits this device.")
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("Download")
        } footer: {
            footer
        }
        .onAppear { quantID = quantID ?? defaultQuantID }
        .task(id: planKey) { await refreshPlan() }
    }

    // MARK: Model

    private var isGGUF: Bool { suggestion.weightFormat == .gguf }

    /// Single-file, top-level quants that are not rated too large; shards and subfolders cannot load.
    private var quantChoices: [QuantFit] {
        result.quantFits.filter {
            $0.isUsable && $0.memory.rating != .tooLarge && !$0.option.firstPath.contains("/")
        }
    }

    private var defaultQuantID: String? {
        let suggested = quantChoices.first { $0.option.firstPath == suggestion.ggufFilename }
        return (suggested ?? QuantFit.recommended(in: quantChoices))?.id
    }

    private var candidate: Model? {
        guard isGGUF else { return suggestion.decodedModel() }
        guard let choice = quantChoices.first(where: { $0.id == quantID }) else { return nil }
        return suggestion.replacingQuant(choice.option)?.decodedModel()
    }

    private var planKey: String {
        "\(candidate?.id ?? "-")#\(transfers.diskRevision)"
    }

    // MARK: Views

    @ViewBuilder private var quantPicker: some View {
        if !quantChoices.isEmpty {
            Picker("Quant", selection: $quantID) {
                ForEach(quantChoices) { choice in
                    Text("\(choice.option.label ?? choice.option.firstPath) · "
                         + ByteText.text(choice.option.totalBytes))
                        .tag(Optional(choice.id))
                }
            }
            .disabled(transfers.isBusy)
        }
    }

    @ViewBuilder private func status(for candidate: Model) -> some View {
        switch transfers.phase(of: candidate) {
        case .downloading(let done, let total):
            VStack(alignment: .leading, spacing: 6) {
                if total > 0 {
                    ProgressView(value: Double(done), total: Double(total))
                    Text("\(ByteText.text(done)) of \(ByteText.text(total))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                } else {
                    ProgressView("Preparing…")
                }
            }
            .accessibilityElement(children: .combine)
        case .failed(let message):
            errorText(message, gated: message == DownloadMessages.gated)
        case .finished(let url):
            Label("Downloaded", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
            Text(url.path)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        case .idle:
            if candidate.isDownloaded {
                Label("Downloaded. Open the Downloads tab to try it.", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else if let planError {
                errorText(planError, gated: planIsGated)
            } else if let pendingPlan {
                Text(sizeLine(pendingPlan))
                    .font(.callout.monospacedDigit())
            } else {
                ProgressView("Checking size…")
            }
        }
    }

    @ViewBuilder private func controls(for candidate: Model) -> some View {
        let phase = transfers.phase(of: candidate)
        if case .downloading = phase {
            Button("Cancel", systemImage: "stop.circle", role: .cancel) { transfers.cancel() }
        } else if !candidate.isDownloaded {
            let resumes = (pendingPlan?.bytesOnDisk ?? 0) > 0
            Button(resumes ? "Resume download" : "Download", systemImage: "arrow.down.circle") {
                transfers.start(candidate)
            }
            .disabled(transfers.isBusy || pendingPlan == nil)
            if transfers.isBusy {
                Text("Another download is running.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if resumes {
                Button("Delete partial download", systemImage: "trash", role: .destructive) {
                    try? transfers.delete(candidate)
                }
                .disabled(transfers.isBusy)
            }
        }
    }

    private func errorText(_ message: String, gated: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(message, systemImage: gated ? "lock.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(gated ? Color.orange : Color.red)
            if gated {
                Button("Add a token in Settings", systemImage: "key") { openSettings() }
                if let url = result.huggingFaceURL {
                    Link("Accept the license on huggingface.co", destination: url)
                }
            }
        }
        // Two controls in one Form row: borderless keeps each tappable on its own.
        .buttonStyle(.borderless)
    }

    @ViewBuilder private var footer: some View {
        VStack(alignment: .leading, spacing: 4) {
            if !isGGUF {
                Text("Fetches config, tokenizer and safetensors from main only.")
            }
            #if targetEnvironment(simulator)
            if !isGGUF {
                Text("MLX can't run on the Simulator; Try it needs a device.")
            }
            #endif
            Text("Downloads stop if the app stays in the background; Resume continues from the partial file.")
        }
    }

    private func sizeLine(_ plan: DownloadPlan) -> String {
        let free = freeBytes.map { " · \(ByteText.text($0)) free" } ?? ""
        if plan.bytesOnDisk > 0 {
            return "Resume (\(ByteText.text(plan.bytesOnDisk)) of \(ByteText.text(plan.totalBytes)) on disk)\(free)"
        }
        return "Download \(ByteText.text(plan.totalBytes))\(free)"
    }

    // MARK: Plan

    private func refreshPlan() async {
        pendingPlan = nil
        planError = nil
        planIsGated = false
        freeBytes = ModelDownloads.availableDiskBytes()
        guard let candidate, !candidate.isDownloaded else { return }
        do {
            pendingPlan = try await ModelDownloads.plan(for: candidate)
        } catch is CancellationError {
            return
        } catch {
            planError = DownloadMessages.text(for: error)
            planIsGated = DownloadMessages.isGated(error)
        }
    }
}
