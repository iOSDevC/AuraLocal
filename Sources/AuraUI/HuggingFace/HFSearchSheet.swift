import SwiftUI
import AuraCore

/// Search judged against this device only, showing just what runs, in its own stack. Downloads keep going after
/// the sheet closes because ``ModelDownloadCenter`` is shared.
struct HFSearchSheet: View {
    @State private var search = HFModelSearch(query: "qwen", devices: [DevicePreset.thisDevice()],
                                              compatibleOnly: true)
    @State private var selection: String?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            HFModelSearchList(search: search, selection: $selection)
                .navigationTitle("Hugging Face")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .navigationDestination(item: $selection) { HFRepoDetailView(repoID: $0, search: search) }
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
        .task { search.search() }
        #if os(macOS)
        .frame(minWidth: 560, minHeight: 600)
        #endif
    }
}
