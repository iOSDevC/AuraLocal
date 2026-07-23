import SwiftUI
import AuraImageGen
#if canImport(UniformTypeIdentifiers)
import UniformTypeIdentifiers
#endif

// MARK: - ImageGenTab
//
// Visual harness for text-to-image + lora.safetensors, driven by AuraImageGen/MFluxEngine.
// macOS-only: mflux (FLUX on MLX) is a Python CLI we exec as a subprocess. On iOS the tab
// shows why it can't run instead of pretending. Two assisted steps, because FLUX has two
// real prerequisites the way LM Studio / Bionic surface them:
//   1. Install mflux (button runs `uv tool install mflux`, output visible).
//   2. Pick a model — default to an UNGATED pre-quantized repo so nothing needs an HF login.

struct ImageGenTab: View {
    var body: some View {
        #if os(macOS)
        ImageGenContent()
        #else
        ContentUnavailableView(
            "Image generation is macOS-only",
            systemImage: "photo.badge.exclamationmark",
            description: Text("FLUX needs Apple Silicon + mflux (~9 GB models). It can't run on iOS."))
        #endif
    }
}

#if os(macOS)
private struct ImageGenContent: View {
    // Default to an ungated, pre-quantized schnell so first run needs no HuggingFace login.
    @State private var model = "dhairyashil/FLUX.1-schnell-mflux-4bit"
    @State private var baseModel = "schnell"
    @State private var prompt = "a red fox in snow, cinematic lighting, highly detailed"
    @State private var steps = 4
    @State private var seedText = ""
    @State private var lowRAM = false
    @State private var loras: [LoRA] = []
    @State private var showingLoRAImporter = false

    @State private var engineReady = MFluxEngine().isAvailable
    @State private var installing = false
    @State private var installLog = ""

    @State private var isGenerating = false
    @State private var status = ""
    @State private var resultImage: NSImage?
    @State private var resultURL: URL?
    @State private var errorText: String?

    private let engine = MFluxEngine()

    var body: some View {
        Form {
            engineSection
            if engineReady {
                modelSection
                promptSection
                loraSection
                advancedSection
                generateSection
            }
            if let errorText {
                Section {
                    Label(errorText, systemImage: "xmark.octagon")
                        .foregroundStyle(.red).textSelection(.enabled)
                }
            }
            if let resultImage {
                Section("Result") {
                    Image(nsImage: resultImage)
                        .resizable().scaledToFit()
                        .frame(maxWidth: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    if let resultURL {
                        Button("Reveal in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([resultURL])
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Image")
        .fileImporter(isPresented: $showingLoRAImporter,
                      allowedContentTypes: safetensorsTypes, allowsMultipleSelection: true) { result in
            if case .success(let urls) = result {
                loras += urls.map { LoRA(url: $0) }
            }
        }
    }

    // MARK: engine status + assisted install

    private var engineSection: some View {
        Section("Engine") {
            if engineReady {
                Label("mflux is ready", systemImage: "checkmark.seal.fill").foregroundStyle(.green)
            } else {
                Label("mflux is not installed", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text("Runs `\(MFluxEngine.installHint)`. Needs [uv](https://docs.astral.sh/uv/) on your PATH.")
                    .font(.caption).foregroundStyle(.secondary)
                Button {
                    Task { await installMflux() }
                } label: {
                    if installing { ProgressView().controlSize(.small) }
                    else { Label("Install mflux", systemImage: "arrow.down.circle") }
                }
                .disabled(installing)
                if !installLog.isEmpty {
                    Text(installLog).font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary).textSelection(.enabled).lineLimit(6)
                }
            }
        }
    }

    private var modelSection: some View {
        Section("Model") {
            TextField("HF repo or alias", text: $model)
            Picker("Architecture", selection: $baseModel) {
                Text("schnell (fast)").tag("schnell")
                Text("dev (quality)").tag("dev")
            }
            Text("Default is an ungated pre-quantized repo — no HuggingFace login needed. "
                 + "The `schnell`/`dev` aliases pull black-forest-labs' gated repos and require auth.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var promptSection: some View {
        Section("Prompt") {
            TextField("Describe the image", text: $prompt, axis: .vertical).lineLimit(2...6)
        }
    }

    private var loraSection: some View {
        Section("LoRA (.safetensors)") {
            Button {
                showingLoRAImporter = true
            } label: { Label("Add LoRA…", systemImage: "plus.circle") }
            ForEach(loras.indices, id: \.self) { i in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(loras[i].url.lastPathComponent).font(.callout).lineLimit(1)
                        Spacer()
                        Button(role: .destructive) { loras.remove(at: i) } label: {
                            Image(systemName: "trash")
                        }.buttonStyle(.borderless)
                    }
                    HStack {
                        Text("scale").font(.caption).foregroundStyle(.secondary)
                        Slider(value: Binding(
                            get: { Double(loras[i].scale) },
                            set: { loras[i].scale = Float($0) }), in: 0...2)
                        Text(String(format: "%.2f", loras[i].scale))
                            .font(.caption.monospacedDigit()).frame(width: 40)
                    }
                }
            }
            if loras.isEmpty {
                Text("No LoRA — the base model runs unmodified.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var advancedSection: some View {
        Section("Options") {
            Stepper("Steps: \(steps)", value: $steps, in: 1...50)
            TextField("Seed (optional)", text: $seedText)
            Toggle("Low-RAM mode (slower, smaller peak)", isOn: $lowRAM)
        }
    }

    private var generateSection: some View {
        Section {
            Button {
                Task { await generate() }
            } label: {
                if isGenerating {
                    HStack { ProgressView().controlSize(.small); Text(status.isEmpty ? "Generating…" : status) }
                } else {
                    Label("Generate", systemImage: "wand.and.stars")
                }
            }
            .disabled(isGenerating || prompt.trimmingCharacters(in: .whitespaces).isEmpty)
            if isGenerating {
                Text("First run downloads the model (several GB) — this can take minutes.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: actions

    private func installMflux() async {
        installing = true; installLog = "Installing…"; defer { installing = false }
        let result = await runShell("/usr/bin/env", ["uv", "tool", "install", "mflux"])
        installLog = result.suffix(600).description
        engineReady = MFluxEngine().isAvailable
    }

    private func generate() async {
        errorText = nil; resultImage = nil; resultURL = nil
        isGenerating = true; status = "Preparing…"; defer { isGenerating = false }
        let request = ImageGenRequest(
            prompt: prompt, model: model, baseModel: baseModel.isEmpty ? nil : baseModel,
            steps: steps, seed: UInt64(seedText),
            quantize: nil,   // the default repo is already quantized
            lowRAM: lowRAM, loras: loras)
        do {
            let result = try await engine.generate(request)
            resultImage = result.image
            resultURL = result.fileURL
        } catch {
            errorText = error.localizedDescription
        }
    }

    private var safetensorsTypes: [UTType] {
        [UTType(filenameExtension: "safetensors") ?? .data]
    }

    /// Run a command, returning combined stdout+stderr. Non-sandboxed macOS only.
    private func runShell(_ launch: String, _ args: [String]) async -> String {
        await withCheckedContinuation { cont in
            DispatchQueue.global().async {
                let p = Process()
                p.executableURL = URL(fileURLWithPath: launch)
                p.arguments = args
                let pipe = Pipe()
                p.standardOutput = pipe; p.standardError = pipe
                do {
                    try p.run()
                    let data = pipe.fileHandleForReading.readDataToEndOfFile()
                    p.waitUntilExit()
                    cont.resume(returning: String(decoding: data, as: UTF8.self))
                } catch {
                    cont.resume(returning: "Failed to launch: \(error.localizedDescription)")
                }
            }
        }
    }
}
#endif
