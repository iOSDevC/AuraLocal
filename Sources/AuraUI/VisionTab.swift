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
        .hoverEffect()
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
                LabeledPicker("Model", selection: $selectedModel, items: Model.visionModels)

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
