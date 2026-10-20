import SwiftUI
import TonamoxCore
import TonamoxUI

enum AppTab: Hashable {
    case search, downloads, settings
}

/// A thin host for TonamoxUI's Hugging Face browser: search, downloads and the token.
struct ContentView: View {
    @State private var selectedTab = AppTab.search

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab("Search", systemImage: "magnifyingglass", value: AppTab.search) {
                SearchView()
            }
            Tab("Downloads", systemImage: "arrow.down.circle", value: AppTab.downloads) {
                NavigationStack { DownloadedModelsList() }
            }
            Tab("Settings", systemImage: "key", value: AppTab.settings) {
                NavigationStack {
                    Form { HuggingFaceTokenSection() }
                        .formStyle(.grouped)
                        .navigationTitle("Settings")
                }
            }
        }
    }
}

struct SearchView: View {
    @State private var search = HFModelSearch()
    @State private var selection: String?

    var body: some View {
        NavigationSplitView {
            HFModelSearchList(search: search, selection: $selection)
                .navigationTitle("Model Finder")
                #if os(macOS)
                .navigationSplitViewColumnWidth(min: 380, ideal: 440)
                #endif
        } detail: {
            if let selection {
                HFRepoDetailView(repoID: selection, search: search)
            } else {
                ContentUnavailableView(
                    "Pick a model", systemImage: "cpu",
                    description: Text("Search Hugging Face, then select a result to see whether it runs here and why."))
            }
        }
        .task { search.search() }
    }
}
