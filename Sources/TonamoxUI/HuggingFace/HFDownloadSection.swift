import SwiftUI
import TonamoxCore

/// Download controls for a repo the checker rated runnable: quant choice, size and free space, progress,
/// Cancel/Resume, errors and Try it.
struct HFDownloadSection: View {
    let result: CompatibilityReport
    let suggestion: CatalogEntry
    @Binding var showsToken: Bool
    @Binding var trial: Model?

    @State private var quantID: String?
    @State private var pendingPlan: DownloadPlan?
    @State private var planError: String?
    @State private var planIsGated = false
    @State private var freeBytes: Int64?
    @State private var deleteError: String?

    private var center: ModelDownloadCenter { .shared }

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

    /// Also re-plans when the token sheet closes, so a saved token clears a gated error.
    private var planKey: String {
        "\(candidate?.id ?? "-")#\(center.revision)#\(showsToken)"
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
            .disabled(center.isBusy)
        }
    }

    @ViewBuilder private func status(for candidate: Model) -> some View {
        switch center.phase(of: candidate) {
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
        case .finished, .idle:
            if candidate.isDownloaded {
                downloaded(candidate)
            } else if let planError {
                errorText(planError, gated: planIsGated)
            } else if let pendingPlan {
                Text(sizeLine(pendingPlan))
                    .font(.callout.monospacedDigit())
            } else {
                ProgressView("Checking size…")
            }
        }
        if let deleteError {
            Label(deleteError, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
        }
    }

    @ViewBuilder private func downloaded(_ candidate: Model) -> some View {
        Label("Downloaded.", systemImage: "checkmark.circle.fill")
            .foregroundStyle(.green)
        if let file = candidate.resolvedFileURL {
            Text(file.path)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        let blocker = RunBlocker.check(candidate, requiresSize: true)
        Button("Try it", systemImage: "play.circle") { trial = candidate }
            .disabled(blocker != nil)
        if let blocker {
            Text(blocker.explanation)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private func controls(for candidate: Model) -> some View {
        let phase = center.phase(of: candidate)
        if case .downloading = phase {
            Button("Cancel", systemImage: "stop.circle", role: .cancel) { center.cancel() }
        } else if !candidate.isDownloaded {
            let resumes = (pendingPlan?.bytesOnDisk ?? 0) > 0
            let elsewhere = center.isWritingElsewhere(candidate)
            Button(resumes ? "Resume download" : "Download", systemImage: "arrow.down.circle") {
                deleteError = nil
                center.start(candidate, records: true)
            }
            .disabled(center.isBusy || elsewhere || pendingPlan == nil)
            if elsewhere {
                Text("This file is downloading elsewhere in the app.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if center.isBusy {
                Text("Another download is running.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if resumes {
                Button("Delete partial download", systemImage: "trash", role: .destructive) {
                    delete(candidate)
                }
                .disabled(center.isBusy || elsewhere)
            }
        }
    }

    private func errorText(_ message: String, gated: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(message, systemImage: gated ? "lock.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(gated ? Color.orange : Color.red)
            if gated {
                Button("Add a token", systemImage: "key") { showsToken = true }
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
                Text("MLX can't run on the Simulator; it needs a device.")
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

    private func delete(_ candidate: Model) {
        do {
            try center.delete(candidate)
            deleteError = nil
        } catch {
            deleteError = error.localizedDescription
        }
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
