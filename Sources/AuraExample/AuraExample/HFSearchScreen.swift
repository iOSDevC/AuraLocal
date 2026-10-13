import SwiftUI
import AuraCore
import AuraUI

/// Hugging Face search judged against this device, in its own stack. Shown as a sheet: under iPhone's More tab a
/// push inside the Models stack nests a second navigation bar.
struct HFSearchScreen: View {
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
                    ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
                }
        }
        .task { search.search() }
        #if os(macOS)
        .frame(minWidth: 560, minHeight: 600)
        #endif
    }
}

/// Downloaded and paused models, in its own stack for the same reason as ``HFSearchScreen``.
struct DownloadsScreen: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            DownloadedModelsList()
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
                }
        }
        #if os(macOS)
        .frame(minWidth: 520, minHeight: 480)
        #endif
    }
}
