import SwiftUI
import AuraCore
import PhotosUI

// MARK: - OCR Tab

/// Document extraction. A model that is not on this device shows a ``ModelDownloadPlaceholder`` in place of the
/// run button until the user downloads it.
public struct OCRTab: View {
    @StateObject private var vm = OCRViewModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var selectedModel: Model = .fastVLM_0_5b_fp16
    /// The downloaded-model fallback runs on the first appearance only, so a later one keeps the user's pick.
    @State private var choseStartModel = false
    private let autoDownload: Bool

    /// - Parameter autoDownload: keep the run button for a model that is not downloaded and download it on the first
    ///   run, as the tab did before it showed a download placeholder.
    public init(autoDownload: Bool = false) {
        self.autoDownload = autoDownload
    }
    
    public var body: some View {
        // Read so the run button returns when a download finishes.
        let _ = ModelDownloadCenter.shared.revision
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    
                    // Image picker
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
                    
                    // Model picker
                    GroupBox {
                        VStack(alignment: .leading, spacing: 8) {
                            LabeledPicker("Model", selection: $selectedModel, items: Model.specializedModels)
                            
                            // Model description
                            if case .visionSpecialized(let docTags) = selectedModel.purpose {
                                Label(
                                    docTags ? "Outputs DocTags → converted to Markdown" : "Outputs structured JSON",
                                    systemImage: docTags ? "doc.richtext" : "curlybraces"
                                )
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                        }
                    } label: {
                        Label("Model", systemImage: "cpu")
                            .font(.subheadline.weight(.semibold))
                    }
                    
                    runButtonSection
                    
                    // Output
                    StatusSection(progress: vm.progress, output: vm.output)
                }
                .padding()
            }
            .navigationTitle("OCR / Document")
            .background(Color.groupedBackground)
        }
        .onAppear {
            guard !choseStartModel else { return }
            choseStartModel = true
            if !selectedModel.isDownloaded {
                selectedModel = Model.specializedModels.first(where: \.isDownloaded) ?? selectedModel
            }
        }
    }

    @ViewBuilder private var runButtonSection: some View {
        if autoDownload || selectedModel.isDownloaded {
            RunButton(
                title: "Extract Document",
                subtitle: "\(selectedModel.approximateSizeMB) MB",
                isDownloaded: selectedModel.isDownloaded,
                isLoading: vm.isLoading,
                color: .orange
            ) {
                guard let img = vm.selectedImage else { return }
                await vm.run(model: selectedModel, image: img)
            }
            .disabled(vm.selectedImage == nil)
        } else {
            ModelDownloadPlaceholder(
                "Download \(selectedModel.displayName)",
                message: "Documents are read on this device. Pick another model above, or download this one.",
                models: [RequiredModel(selectedModel, purpose: "Extracts text and structure from photos of documents")]
            )
        }
    }
}

#Preview {
    OCRTab()
}
