import SwiftUI
import AuraCore
import PhotosUI

// MARK: - Vision Tab
//
// §3.4: ViewThatFits for adaptive layout (wide vs. narrow).
// §4.2: .hoverEffect() on interactive elements.

public struct VisionTab: View {
    @StateObject private var vm = VisionViewModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var selectedModel: Model = .qwen35_0_8b
    @State private var runMode: AuraLocal.VisionRunMode = .standard
    @State private var customPrompt: String = "Describe this image in detail."

    public init() {}

    public var body: some View {
        // Read so the picker is rebuilt when a download finishes or a model is deleted.
        let revision = ModelDownloadCenter.shared.revision
        NavigationStack {
            ScrollView {
                // §3.4: ViewThatFits — wide layout shows image + controls side by side;
                // narrow layout stacks them vertically.
                ViewThatFits(in: .horizontal) {
                    wideLayout
                    narrowLayout
                }
                .padding()
            }
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle("Vision")
            .background(Color.groupedBackground)
        }
        .onChange(of: revision) { resetSelectionIfRemoved() }
    }

    // MARK: - Wide Layout (iPad landscape, macOS, visionOS wide window)

    private var wideLayout: some View {
        HStack(alignment: .top, spacing: 20) {
            VStack(spacing: 16) {
                imagePickerSection
                runButtonSection
            }
            .frame(minWidth: 280, maxWidth: 400)

            VStack(spacing: 16) {
                controlsSection
                StatusSection(progress: vm.progress, output: vm.output)
            }
            .frame(minWidth: 300)
        }
    }

    // MARK: - Narrow Layout (iPhone, compact width)

    private var narrowLayout: some View {
        VStack(spacing: 16) {
            imagePickerSection
            controlsSection
            runButtonSection
            StatusSection(progress: vm.progress, output: vm.output)
        }
    }

    // MARK: - Shared Components

    private var imagePickerSection: some View {
        PhotosPicker(selection: $pickerItem, matching: .images) {
            ImagePickerPreview(image: vm.selectedImage)
        }
#if !os(macOS)
        .hoverEffect()
#endif
        .onChange(of: pickerItem) { _, item in
            Task {
                if let data = try? await item?.loadTransferable(type: Data.self) {
                    vm.selectedImage = PlatformImage(data: data)
                    vm.output = ""
                }
            }
        }
    }

    private var controlsSection: some View {
        GroupBox {
            VStack(spacing: 12) {
                modelPicker

                Picker("Mode", selection: $runMode) {
                    ForEach(AuraLocal.VisionRunMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                TextField("Custom prompt...", text: $customPrompt, axis: .vertical)
                    .lineLimit(2...4)
                    .font(.subheadline)
                    .padding(8)
                    .background(Color.tertiaryGroupedBackground, in: RoundedRectangle(cornerRadius: 8))
            }
        } label: {
            Label("Settings", systemImage: "slider.horizontal.3")
                .font(.subheadline.weight(.semibold))
        }
    }

    /// Downloaded models that fit now first; downloaded ones too large for free memory are left out.
    private var modelPicker: some View {
        let all = Model.visionModels
        let downloaded = all.filter { $0.isDownloaded && HardwareAnalyzer.assess($0).fitLevel != .tooLarge }
        let available = all.filter { !$0.isDownloaded }
        return HStack {
            Text("Model")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer()
            Picker("Model", selection: $selectedModel) {
                if !downloaded.isEmpty {
                    Section("Downloaded") {
                        ForEach(downloaded, id: \.self) { Text($0.displayName).tag($0) }
                    }
                }
                Section("Available") {
                    ForEach(available, id: \.self) { Text($0.displayName).tag($0) }
                }
            }
        }
    }

    /// A deleted model leaves the registry; running with it would silently download it again.
    private func resetSelectionIfRemoved() {
        guard ModelRegistry.shared.model(id: selectedModel.id) == nil else { return }
        selectedModel = Model.visionModels.first {
            $0.isDownloaded && HardwareAnalyzer.assess($0).fitLevel != .tooLarge
        } ?? .qwen35_0_8b
    }

    private var runButtonSection: some View {
        RunButton(
            title: "Analyze Image",
            subtitle: selectedModel.displayName,
            isDownloaded: selectedModel.isDownloaded,
            isLoading: vm.isLoading,
            color: .blue
        ) {
            guard let img = vm.selectedImage else { return }
            await vm.run(
                model: selectedModel,
                image: img,
                prompt: customPrompt,
                mode: runMode
            )
        }
        .disabled(vm.selectedImage == nil)
    }
}

#Preview {
    VisionTab()
}
