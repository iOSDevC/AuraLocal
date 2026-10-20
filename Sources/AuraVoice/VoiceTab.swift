import SwiftUI
import AuraCore

// MARK: - VoiceTab

struct VoiceTab: View {
    @StateObject private var vm = VoiceTabViewModel()
    @State private var selectedModel: Model = .qwen3_1_7b
    
    var body: some View {
        NavigationStack {
            Group {
                if !selectedModel.isDownloaded {
                    ModelDownloadPlaceholder(
                        "Download \(selectedModel.displayName)",
                        message: "Voice chat listens, answers and speaks on this device.",
                        models: [RequiredModel(selectedModel, purpose: "Answers what you say.")]
                    ) {
                        Task { await vm.load(model: selectedModel) }
                    }
                } else if let llm = vm.llm {
                    VoiceChatView(
                        llm: llm,
                        conversationID: vm.conversationID
                    )
                } else {
                    loadingView
                }
            }
            .navigationTitle("Voice Chat")
#if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
#endif
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Menu {
                        // textModels leaves uncensored models out; a downloaded one is still offered.
                        let uncensored = Model.uncensoredModels.filter { $0.category == .text && $0.isDownloaded }
                        ForEach(Model.textModels + uncensored, id: \.self) { m in
                            Button {
                                selectedModel = m
                            } label: {
                                Label(m.displayName, systemImage: m.isDownloaded ? "checkmark" : "arrow.down")
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Text(selectedModel.displayName)
                                .font(.caption)
                            Image(systemName: "chevron.down")
                                .font(.caption2)
                        }
                        .foregroundStyle(.secondary)
                    }
                }
            }
            // A model on disk loads at once; a missing one waits for the placeholder's Download.
            .task(id: selectedModel) {
                if selectedModel.isDownloaded { await vm.load(model: selectedModel) }
            }
        }
    }
    
    @ViewBuilder
    private var loadingView: some View {
        VStack(spacing: 16) {
            ProgressView()
            if !vm.progress.isEmpty {
                Text(vm.progress)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - VoiceTabViewModel

@MainActor
final class VoiceTabViewModel: ObservableObject {
    @Published var llm: AuraLocal?
    @Published var progress: String = ""
    @Published var conversationID: UUID?
    
    private let store = ConversationStore.shared
    
    func load(model: Model) async {
        llm = nil
        do {
            llm = try await AuraLocal.text(model) { [weak self] p in
                self?.progress = p
            }
            progress = ""
            
            // Create a fresh conversation for this voice session
            let conv = try await store.createConversation(model: model, title: "Voice session")
            conversationID = conv.id
            
        } catch {
            progress = "❌ \(error.localizedDescription)"
        }
    }
}
