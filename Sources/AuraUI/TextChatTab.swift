import SwiftUI
import AuraCore

// MARK: - Text Chat Tab

/// Multi-conversation chat. A model that is not on this device shows a ``ModelDownloadPlaceholder`` instead of the
/// chat until the user downloads it; a downloaded one loads on the first message.
public struct TextChatTab: View {
    @StateObject private var vm = TextChatViewModel()
    @State private var selectedModel: Model = .qwen3_1_7b
    @State private var showConversations = false
    @State private var showsSearch = false
    /// Cached: `body` redraws on every streamed token, and the check reads the disk.
    @State private var needsDownload = false
    private let offersHuggingFaceSearch: Bool
    private let autoDownload: Bool

    /// - Parameters:
    ///   - offersHuggingFaceSearch: start the model menu with "Find on Hugging Face…", a sheet that searches, checks
    ///     and downloads models for this device.
    ///   - autoDownload: show the chat for a model that is not downloaded and download it on the first message, as
    ///     the tab did before it showed a download placeholder.
    public init(offersHuggingFaceSearch: Bool = false, autoDownload: Bool = false) {
        self.offersHuggingFaceSearch = offersHuggingFaceSearch
        self.autoDownload = autoDownload
    }

    public var body: some View {
        // Read so the menu is rebuilt when a download finishes or a model is deleted.
        let revision = ModelDownloadCenter.shared.revision
        NavigationStack {
            content
                .navigationTitle(vm.activeConversation?.title ?? "Text Chat")
#if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
#endif
                .toolbar {
                    ToolbarItem(placement: .navigation) {
                        Button { showConversations = true } label: {
                            Image(systemName: "sidebar.left")
                        }
#if !os(macOS)
                        .hoverEffect()
#endif
                    }
                    ToolbarItem(placement: .primaryAction) {
                        Menu {
                            // First: on iPhone the end of this long menu is several screens down.
                            if offersHuggingFaceSearch {
                                Section {
                                    Button("Find on Hugging Face…", systemImage: "magnifyingglass") {
                                        showsSearch = true
                                    }
                                }
                            }
                            // Every downloaded one, uncensored included: textModels leaves those out, but the
                            // user downloaded them.
                            let downloaded = Model.allModels.filter { $0.category == .text && $0.isDownloaded }
                            if !downloaded.isEmpty {
                                Section("Downloaded") {
                                    ForEach(downloaded, id: \.self) { downloadedModel in
                                        modelButton(downloadedModel, blocker: RunBlocker.check(downloadedModel))
                                    }
                                }
                            }
                            Section("MLX Models") {
                                ForEach(Model.textModels.filter { $0.format == .mlx && !$0.isDownloaded },
                                        id: \.self) { m in
                                    modelButton(m)
                                }
                            }
                            let ggufText = Model.ggufModels.filter {
                                guard case .text = $0.purpose else { return false }
                                return !$0.isDownloaded
                            }
                            if !ggufText.isEmpty {
                                Section("GGUF Models (llama.cpp)") {
                                    ForEach(ggufText, id: \.self) { m in
                                        modelButton(m)
                                    }
                                }
                            }
                        } label: {
                            Image(systemName: "square.and.pencil")
                        }
#if !os(macOS)
                        .hoverEffect()
#endif
                    }
                }
                .sheet(isPresented: $showConversations) {
                    ConversationListSheet(vm: vm, isPresented: $showConversations)
                }
                .sheet(isPresented: $showsSearch) { HFSearchSheet() }
                .onAppear {
                    if !selectedModel.isDownloaded { selectedModel = preferredModel ?? selectedModel }
                    refreshNeedsDownload()
                }
                .onChange(of: revision) {
                    resetSelectionIfRemoved()
                    refreshNeedsDownload()
                }
                .onChange(of: selectedModel) { refreshNeedsDownload() }
                // §5.1: Wire FocusedValue actions for Menu Bar commands
                .focusedValue(\.newConversationAction) {
                    Task { await vm.newConversation(model: selectedModel) }
                }
                .focusedValue(\.toggleSidebarAction) {
                    showConversations.toggle()
                }
        }
    }

    @ViewBuilder private var content: some View {
        if !needsDownload {
            ChatView(vm: vm, selectedModel: $selectedModel)
        } else {
            ModelDownloadPlaceholder(
                "Download \(selectedModel.displayName)",
                message: "Chat runs on this device. Pick another model from the menu, or download this one.",
                models: [RequiredModel(selectedModel, purpose: "Chats and keeps your conversations on this device")]
            )
        }
    }

    /// Selecting only starts a conversation; the model loads on send, behind ModelManager's memory guard.
    @ViewBuilder
    private func modelButton(_ m: Model, blocker: RunBlocker? = nil) -> some View {
        Button {
            selectedModel = m
            Task { await vm.newConversation(model: m) }
        } label: {
            HStack {
                Text(blocker.map { "\(m.displayName) (\($0.label))" } ?? m.displayName)
                Spacer()
                if m.isDownloaded {
                    Image(systemName: "checkmark.circle.fill")
                } else {
                    Image(systemName: "arrow.down.circle")
                }
            }
        }
        .disabled(blocker != nil)
    }

    /// A deleted model leaves the registry; sending with it would silently download it again.
    private func resetSelectionIfRemoved() {
        guard ModelRegistry.shared.model(id: selectedModel.id) == nil else { return }
        selectedModel = preferredModel ?? .qwen3_1_7b
    }

    private func refreshNeedsDownload() {
        needsDownload = !autoDownload && !selectedModel.isDownloaded
    }

    private var preferredModel: Model? {
        Model.textModels.first { $0.isDownloaded && RunBlocker.check($0) == nil }
    }
}

#Preview {
    TextChatTab()
}
