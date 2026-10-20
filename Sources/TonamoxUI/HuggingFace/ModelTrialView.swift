import SwiftUI
import TonamoxCore
#if canImport(UIKit)
import UIKit
#endif

/// Why a downloaded model can't be loaded right now, shared by the pickers and Try it.
enum RunBlocker {
    case simulatorMLX
    case tooLarge
    case unsized

    /// Short label for a picker row.
    var label: String {
        switch self {
        case .simulatorMLX: "device only"
        case .tooLarge: "too large now"
        case .unsized: "size unknown"
        }
    }

    /// Full sentence for Try it.
    var explanation: String {
        switch self {
        case .simulatorMLX: "MLX can't run on the Simulator; Try it needs a device."
        case .tooLarge: "Too large for this device's free memory right now."
        case .unsized: "Try it needs the size from a search result; download it from Search."
        }
    }

    /// `nil` when `model` may load now. Size 0 is only rejected when `requiresSize`: the load-time memory guard
    /// can't judge an unsized model.
    static func check(_ model: Model, requiresSize: Bool = false) -> RunBlocker? {
        if requiresSize && model.approximateSizeMB <= 0 { return .unsized }
        #if targetEnvironment(simulator)
        if model.format == .mlx { return .simulatorMLX }
        #endif
        let fit = HardwareAnalyzer.assess(model).fitLevel
        if fit == .tooLarge || (model.format == .mlx && fit == .streamingRequired) { return .tooLarge }
        return nil
    }
}

/// Loads a downloaded model, asks one short question and unloads it again.
struct ModelTrialView: View {
    let candidate: Model
    @Environment(\.dismiss) private var dismiss

    @State private var reply = ""
    @State private var stage = "Loading…"
    @State private var failure: String?
    @State private var run: Task<Void, Never>?
    @State private var isFinished = false
    @State private var stopReason: String?

    private static let prompt = "Reply with one short sentence: what are you?"

    var body: some View {
        NavigationStack {
            Form {
                Section("Prompt") { Text(Self.prompt) }
                Section("Reply") {
                    if let failure {
                        Label(failure, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                    } else if reply.isEmpty {
                        ProgressView(stage)
                    } else {
                        Text(reply).textSelection(.enabled)
                    }
                }
                Section {
                    Text(stage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(candidate.resolvedFileURL?.path ?? candidate.repoID)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            .formStyle(.grouped)
            .navigationTitle(candidate.displayName)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        stop(reason: nil)
                        dismiss()
                    }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 440, minHeight: 360)
        #endif
        .onAppear { start() }
        .onDisappear { stop(reason: nil) }
        #if canImport(UIKit)
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
            stop(reason: "Stopped: the system reported low memory.")
        }
        #endif
    }

    private func start() {
        guard run == nil else { return }
        if let blocker = RunBlocker.check(candidate, requiresSize: true) {
            failure = blocker.explanation
            isFinished = true
            stage = "Not loaded."
            return
        }
        let target = candidate
        run = Task {
            var outcome = ""
            do {
                let started = ContinuousClock.now
                let session = try await ModelManager.shared.load(target)
                // A cancelled load can still finish; don't generate with it.
                try Task.checkCancellation()
                let loaded = ContinuousClock.now
                stage = "Loaded in \(Self.seconds(loaded - started)). Generating…"
                let text = try await session.chat(Self.prompt, maxTokens: 32)
                try Task.checkCancellation()
                reply = text.trimmingCharacters(in: .whitespacesAndNewlines)
                outcome = "Loaded in \(Self.seconds(loaded - started)), replied in "
                    + "\(Self.seconds(ContinuousClock.now - loaded))."
            } catch is CancellationError {
                outcome = "Cancelled."
            } catch {
                failure = error.localizedDescription
            }
            // Unload whatever happens, so a test never keeps gigabytes resident; report it only once done.
            ModelManager.shared.evict(target)
            isFinished = true
            if let stopReason { failure = stopReason + " The model was unloaded." }
            stage = outcome.isEmpty ? "Unloaded." : outcome + " Unloaded."
        }
    }

    /// Cancelling `run` alone does not stop the load: `ModelManager.load` awaits its own unstructured task.
    private func stop(reason: String?) {
        guard run != nil, !isFinished else { return }
        if let reason { stopReason = reason }
        ModelManager.shared.cancelLoad(candidate)
        run?.cancel()
    }

    private static func seconds(_ elapsed: Duration) -> String {
        elapsed.formatted(.units(allowed: [.seconds, .milliseconds], width: .narrow, maximumUnitCount: 2))
    }
}
