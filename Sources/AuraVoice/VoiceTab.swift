import SwiftUI
import AuraCore

// MARK: - VoiceTab

struct VoiceTab: View {
    @StateObject private var vm = VoiceTabViewModel()
    @State private var selectedModel: Model = .qwen3_1_7b
    
    var body: some View {
        NavigationStack {
            Group {
                if let llm = vm.llm {
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
                        ForEach(Model.textModels, id: \.self) { m in
                            Button {
                                Task { await vm.load(model: m) }
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
            .task { await vm.load(model: selectedModel) }
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
