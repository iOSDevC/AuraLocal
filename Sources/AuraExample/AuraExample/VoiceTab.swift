import SwiftUI
import Combine
import AuraCore
import AuraUI
import AuraVoice

// MARK: - VoiceTab
//
// Voice has no model type of its own: it talks to a downloaded model that chats from text. Chat's Catalog and
// Hugging Face sheets download more.

struct VoiceTab: View {
    @StateObject private var vm = VoiceTabViewModel()
    @AppStorage("auraExample.voiceModelID") private var storedModelID = ""

    var body: some View {
        // Read so the menu is rebuilt when a download finishes or a model is deleted.
        let revision = ModelDownloadCenter.shared.revision
        NavigationStack {
            Group {
                if let llm = vm.llm {
                    VoiceChatView(llm: llm, conversationID: vm.conversationID)
                        .id(vm.conversationID)
                } else if vm.isLoading {
                    loadingView
                } else if let failure = vm.failure {
                    ContentUnavailableView {
                        Label("Couldn't load the model", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(failure)
                    } actions: {
                        // The menu can't re-pick the failed model: its toggle already reads on.
                        if let failed = vm.activeModel {
                            Button("Try again") { use(failed) }
                                .buttonStyle(.borderedProminent)
                                .accessibilityHint("Loads \(failed.displayName) again")
                        }
                    }
                } else if let preferred = ChatModels.preferred(id: storedModelID, requiring: .text) {
                    ContentUnavailableView {
                        Label("Voice chat", systemImage: "mic")
                    } description: {
                        Text("Talk with a model on this device.")
                    } actions: {
                        Button("Start with \(preferred.displayName)") { use(preferred) }
                            .buttonStyle(.borderedProminent)
                    }
                } else {
                    ContentUnavailableView("No text model on this device", systemImage: "mic.slash",
                                           description: Text("Download one in Chat › Browse the catalog or "
                                                             + "Find on Hugging Face, then come back."))
                }
            }
            .navigationTitle("Voice Chat")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .primaryAction) { modelMenu }
            }
            .task { loadPreferred() }
            .onChange(of: revision) { reloadIfRemoved() }
        }
    }

    private var modelMenu: some View {
        Menu {
            Section("On this device") {
                ForEach(ChatModels.downloaded(requiring: .text), id: \.self) { model in
                    let blocker = ModelBlocker.check(model)
                    Toggle(blocker.map { "\(model.displayName) (\($0.label))" } ?? model.displayName,
                           isOn: Binding { model == vm.activeModel } set: { if $0 { use(model) } })
                    .disabled(blocker != nil)
                }
            }
        } label: {
            HStack(spacing: 4) {
                Text(vm.activeModel?.displayName ?? "Model").font(.caption)
                Image(systemName: "chevron.down").font(.caption2)
            }
            .foregroundStyle(.secondary)
        }
        .disabled(vm.isLoading)
        .accessibilityLabel("Voice model: \(vm.activeModel?.displayName ?? "none")")
    }

    private var loadingView: some View {
        VStack(spacing: 16) {
            ProgressView()
            if !vm.progress.isEmpty {
                Text(vm.progress)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Button("Cancel", role: .cancel) { vm.cancel() }
        }
    }

    private func use(_ model: Model) {
        storedModelID = model.id
        vm.load(model)
    }

    /// Loads the remembered model, or the first one that fits; never downloads.
    private func loadPreferred() {
        guard vm.activeModel == nil,
              let model = ChatModels.preferred(id: storedModelID, requiring: .text) else { return }
        vm.load(model)
    }

    private func reloadIfRemoved() {
        guard let current = vm.activeModel, !current.isDownloaded || ModelRegistry.shared.model(id: current.id) == nil
        else { return }
        vm.unload()
        loadPreferred()
    }
}

// MARK: - VoiceTabViewModel

@MainActor
final class VoiceTabViewModel: ObservableObject {
    @Published private(set) var llm: AuraLocal?
    @Published private(set) var activeModel: Model?
    @Published private(set) var progress = ""
    @Published private(set) var failure: String?
    @Published private(set) var isLoading = false
    @Published private(set) var conversationID: UUID?

    private let store = ConversationStore.shared
    private var loadTask: Task<Void, Never>?

    func load(_ next: Model) {
        loadTask?.cancel()
        llm = nil
        activeModel = next
        failure = nil
        isLoading = true
        loadTask = Task {
            do {
                let loaded = try await ModelManager.shared.load(next) { message in self.progress = message }
                try Task.checkCancellation()
                // A fresh conversation for each voice session.
                let conversation = try await store.createConversation(model: next, title: "Voice session")
                conversationID = conversation.id
                llm = loaded
            } catch is CancellationError {
                // A newer load or Cancel replaced this one.
            } catch {
                failure = error.localizedDescription
            }
            if !Task.isCancelled {
                isLoading = false
                progress = ""
            }
        }
    }

    func cancel() {
        guard let activeModel else { return }
        ModelManager.shared.cancelLoad(activeModel)
        loadTask?.cancel()
        isLoading = false
        progress = ""
        self.activeModel = nil
    }

    func unload() {
        loadTask?.cancel()
        llm = nil
        activeModel = nil
        failure = nil
        isLoading = false
    }
}
