import SwiftUI
import AuraCore
import PhotosUI

// MARK: - OCR Tab

public struct OCRTab: View {
    @StateObject private var vm = OCRViewModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var selectedModel: Model = .fastVLM_0_5b_fp16
    
    public init() {}
    
    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    
                    // Image picker
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
                    
                    // Run button
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
                    
                    // Output
                    StatusSection(progress: vm.progress, output: vm.output)
                }
                .padding()
            }
            .navigationTitle("OCR / Document")
            .background(Color.groupedBackground)
        }
    }
}

#Preview {
    OCRTab()
}
