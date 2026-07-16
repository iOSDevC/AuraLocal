import SwiftUI
import AuraCore

// MARK: - ModelEvalView
//
// "Does this model do what I need, on THIS device?"
// HardwareAnalyzer answers whether a model FITS; it can't say whether it's any
// GOOD for your purpose. This runs the domain's curated prompts, measures the
// objective signals (fit, time-to-first-token, tok/s), and lets the user judge
// each answer — the capability verdict stays with the person, not a self-scoring
// model. Everything runs on-device.

struct ModelEvalView: View {
    let model: Model
    let domain: Model.Domain

    @State private var results: [EvalResult] = []
    @State private var isRunning = false
    @State private var status = ""
    @State private var errorText: String?

    private let profile = HardwareProfile.current()

    private var compatibility: ModelCompatibility {
        HardwareAnalyzer.assess(model, profile: profile)
    }

    private var fit: ModelFitLevel { compatibility.fitLevel }

    var body: some View {
        Form {
            fitSection
            runSection
            ForEach($results) { $result in
                resultSection(result: $result)
            }
            if !results.isEmpty { verdictSection }
            if let errorText {
                Section {
                    Label(errorText, systemImage: "xmark.octagon")
                        .font(.caption).foregroundStyle(.red)
                }
            }
        }
        .navigationTitle("Evaluate")
        .frame(minWidth: 460, minHeight: 560)
    }

    private var speedColor: Color {
        switch compatibility.speedLevel {
        case .fast, .usable: .green
        case .slow:          .orange
        case .unusable:      .red
        case .unknown:       .secondary
        }
    }

    // MARK: - Fit (does it even run here?)

    @ViewBuilder
    private var fitSection: some View {
        Section {
            LabeledContent("Model", value: model.displayName)
            HStack {
                Text("Runs on this device")
                Spacer()
                Label(fit.label, systemImage: fit.systemImage)
                    .font(.caption)
                    .foregroundStyle(fit.isRunnable ? .green : .red)
            }
            LabeledContent("Estimated RAM", value: String(format: "%.1f GB", model.estimatedRuntimeMemoryGB))

            // Fitting is not the same as being usable. Predicted BEFORE download,
            // so a 20 GB model that would crawl can be skipped, not discovered
            // the hard way. The run below measures the real number.
            if let tps = compatibility.estimatedDecodeTokensPerSecond {
                HStack {
                    Text("Predicted speed")
                    Spacer()
                    Text(String(format: "~%.0f tok/s · %@", tps, compatibility.speedLevel.label))
                        .font(.caption)
                        .foregroundStyle(speedColor)
                }
                if let seconds = compatibility.estimatedResponseSeconds(tokens: 500) {
                    LabeledContent("500-token answer",
                                   value: seconds < 90
                                        ? String(format: "~%.0f s", seconds)
                                        : String(format: "~%.1f min", seconds / 60))
                }
            }
        } header: {
            Text("Hardware fit")
        } footer: {
            Text("Fit only says whether it can run. Capability — whether it's good enough for \(domain.displayName.lowercased()) — is what you judge below.")
        }
    }

    // MARK: - Run

    @ViewBuilder
    private var runSection: some View {
        Section {
            Button { Task { await runEvaluation() } } label: {
                if isRunning {
                    HStack { ProgressView(); Text(status.isEmpty ? "Running…" : status) }
                } else {
                    Label("Run evaluation (\(domain.samplePrompts.count) prompts)", systemImage: "checklist")
                }
            }
            .disabled(isRunning || !model.isDownloaded)

            if !model.isDownloaded {
                Label("Download the model first from the Models list.", systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
        } footer: {
            Text("Runs this domain's curated prompts on-device and measures speed. Nothing leaves the device.")
        }
    }

    // MARK: - Per-prompt result + your rating

    @ViewBuilder
    private func resultSection(result: Binding<EvalResult>) -> some View {
        Section {
            Text(result.wrappedValue.prompt)
                .font(.caption).foregroundStyle(.secondary)

            if let error = result.wrappedValue.error {
                Label(error, systemImage: "xmark.octagon").font(.caption).foregroundStyle(.red)
            } else {
                Text(result.wrappedValue.output)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)

                HStack(spacing: 12) {
                    if let ttft = result.wrappedValue.ttft {
                        metric("TTFT", String(format: "%.1fs", ttft))
                    }
                    if let tps = result.wrappedValue.tokensPerSecond {
                        metric("Speed", String(format: "≈%.0f tok/s", tps))
                    }
                    metric("Total", String(format: "%.1fs", result.wrappedValue.elapsed))
                }

                HStack {
                    Text("Useful for your purpose?").font(.caption)
                    Spacer()
                    Button { result.wrappedValue.rating = true } label: {
                        Image(systemName: result.wrappedValue.rating == true ? "hand.thumbsup.fill" : "hand.thumbsup")
                    }
                    .buttonStyle(.plain)
                    Button { result.wrappedValue.rating = false } label: {
                        Image(systemName: result.wrappedValue.rating == false ? "hand.thumbsdown.fill" : "hand.thumbsdown")
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func metric(_ name: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(name).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.caption.monospaced())
        }
    }

    // MARK: - Verdict

    @ViewBuilder
    private var verdictSection: some View {
        let rated   = results.filter { $0.rating != nil }
        let useful  = results.filter { $0.rating == true }
        let speeds  = results.compactMap(\.tokensPerSecond)
        let avgSpeed = speeds.isEmpty ? nil : speeds.reduce(0, +) / Double(speeds.count)

        Section {
            LabeledContent("Hardware fit", value: fit.label)
            if let avgSpeed {
                LabeledContent("Average speed", value: String(format: "≈%.0f tok/s", avgSpeed))
            }
            LabeledContent("You rated useful", value: "\(useful.count) / \(rated.count) rated")
            if !rated.isEmpty {
                Label(
                    useful.count * 2 >= rated.count
                        ? "Looks suitable for \(domain.displayName.lowercased()) on this device."
                        : "Falls short for \(domain.displayName.lowercased()) — try a larger model, or escalate to a bigger one from the Hybrid tab.",
                    systemImage: useful.count * 2 >= rated.count ? "checkmark.seal" : "exclamationmark.triangle"
                )
                .font(.caption)
                .foregroundStyle(useful.count * 2 >= rated.count ? .green : .orange)
            }
        } header: {
            Text("Verdict")
        } footer: {
            Text("Speed is measured; suitability is your rating. Token counts are estimated from characters.")
        }
    }

    // MARK: - Evaluation run

    @MainActor
    private func runEvaluation() async {
        isRunning = true
        errorText = nil
        results = domain.samplePrompts.map { EvalResult(prompt: $0) }
        status = "Loading model…"
        defer { isRunning = false; status = "" }

        let llm: AuraLocal
        do {
            llm = try await ModelManager.shared.load(model) { status = $0 }
        } catch {
            errorText = error.localizedDescription
            return
        }

        for index in results.indices {
            status = "Prompt \(index + 1) of \(results.count)…"
            let start = Date()
            var firstToken: Date?
            var text = ""
            do {
                for try await token in llm.stream(results[index].prompt,
                                                  systemPrompt: domain.testSystemPrompt,
                                                  maxTokens: 256) {
                    if firstToken == nil { firstToken = Date() }
                    text += token
                    results[index].output = text
                }
                let elapsed = Date().timeIntervalSince(start)
                let ttft    = firstToken.map { $0.timeIntervalSince(start) }
                // Token count estimated from characters (~4 chars/token).
                let approxTokens = max(1, text.count / 4)
                let genTime = elapsed - (ttft ?? 0)
                results[index].elapsed = elapsed
                results[index].ttft    = ttft
                results[index].tokensPerSecond = genTime > 0 ? Double(approxTokens) / genTime : nil
            } catch {
                results[index].error = error.localizedDescription
            }
        }
    }
}

// MARK: - EvalResult

struct EvalResult: Identifiable {
    let id = UUID()
    let prompt: String
    var output = ""
    var ttft: TimeInterval?
    var elapsed: TimeInterval = 0
    var tokensPerSecond: Double?
    var error: String?
    /// The user's own judgment — nil until rated.
    var rating: Bool?
}
