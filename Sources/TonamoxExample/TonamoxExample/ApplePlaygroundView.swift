#if os(iOS)
import SwiftUI
import TonamoxCore
import TonamoxImageGen
import ImagePlayground

// MARK: - ApplePlaygroundView
//
// Apple's Image Playground sheet as a second engine. From iOS 27 Apple runs it on Private Cloud Compute, so the
// prompt leaves the device: every presentation asks first, and nothing is remembered. Styles come from
// ImagePlaygroundPolicy, which leaves out the ChatGPT (external provider) style.

struct ApplePlaygroundView: View {
    @Bindable var state: ImageGenState
    @Environment(\.supportsImagePlayground) private var supportsImagePlayground
    @State private var prompt = "a cozy cabin in the snow at night"
    @State private var style = ImagePlaygroundPolicy.defaultStyle
    @State private var askingConsent = false
    @State private var presenting = false
    private let processing = ImagePlaygroundPolicy.processing

    var body: some View {
        Form {
            if supportsImagePlayground {
                Section("Prompt") {
                    TextField("Describe the image", text: $prompt, axis: .vertical)
                        .lineLimit(2...6)
                    Picker("Style", selection: $style) {
                        ForEach(ImagePlaygroundPolicy.allowedStyles, id: \.id) { Text($0.id.capitalized).tag($0) }
                    }
                }
                createSection
            } else {
                unavailableSection
            }
            FailureSection(state: state)
            ResultsSection(state: state)
        }
        .formStyle(.grouped)
        .alert("Send to Apple's servers?", isPresented: $askingConsent) {
            Button("Cancel", role: .cancel) { state.show(OnDeviceImageGenError.consentDeclined) }
            Button("Continue to Apple") { presenting = true }
        } message: {
            Text("Image Playground runs on Apple Private Cloud Compute. Your prompt, and anything you add or pick in "
                 + "Apple's sheet, leaves this device. Apple applies daily limits. Tonamox can't see or control "
                 + "what happens inside Apple's sheet.")
        }
        .imagePlaygroundSheet(isPresented: $presenting, concepts: concepts, onCompletion: importResult)
        // Explicit list: the default `in:` is `.all`, which includes the ChatGPT style.
        .imagePlaygroundGenerationStyle(style, in: ImagePlaygroundPolicy.allowedStyles)
        .modifier(PlaygroundOptions())
        .sheet(item: $state.selected) { GeneratedImageDetail(item: $0) }
    }

    private var concepts: [ImagePlaygroundConcept] {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? [] : [.text(text)]
    }

    private var createSection: some View {
        Section {
            Button {
                state.clearFailure()
                if processing.leavesDevice {
                    askingConsent = true
                } else {
                    presenting = true
                }
            } label: {
                Label("Create with Apple", systemImage: "sparkles")
            }
            if processing.leavesDevice {
                Text("Runs on Apple's servers (Private Cloud Compute). You'll be asked before anything is sent.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                Text("Runs on device (Apple Image Playground; ChatGPT style excluded).")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var unavailableSection: some View {
        Section {
            Label(OnDeviceImageGenError.imagePlaygroundUnavailable.errorDescription ?? "",
                  systemImage: "exclamationmark.triangle")
            Text(OnDeviceImageGenError.imagePlaygroundUnavailable.recoverySuggestion ?? "")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    /// Apple hands over a temporary file the app must move or delete; it joins the same grid as on-device images.
    private func importResult(_ url: URL) {
        do {
            let folder = FileManager.default.temporaryDirectory
                .appendingPathComponent("auraexample-playground-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let destination = folder.appendingPathComponent(url.lastPathComponent)
            try FileManager.default.moveItem(at: url, to: destination)
            guard let image = PlatformImage(contentsOfFile: destination.path) else {
                throw CocoaError(.fileReadCorruptFile, userInfo: [NSFilePathErrorKey: destination.path])
            }
            let place = processing.leavesDevice ? "Private Cloud Compute" : "on device"
            state.add(GeneratedImage(image: image, fileURL: destination, prompt: prompt,
                                     settings: "Style \(style.id.capitalized)", engineName: "Apple Image Playground",
                                     provenance: "Apple Image Playground (\(place))",
                                     leftDevice: processing.leavesDevice))
        } catch {
            state.show(error)
        }
    }
}

/// Personalization off on iOS 26.4+, where the options API exists.
private struct PlaygroundOptions: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 26.4, *) {
            content.imagePlaygroundOptions(ImagePlaygroundPolicy.options)
        } else {
            content
        }
    }
}
#endif
