import SwiftUI
import AuraCore

// MARK: - Text Chat Tab
public struct TextChatTab: View {
    @StateObject private var vm = TextChatViewModel()
    @State private var selectedModel: Model = .qwen3_1_7b
    @State private var showConversations = false

    public init() {}

    public var body: some View {
        NavigationStack {
            ChatView(vm: vm, selectedModel: $selectedModel)
                .navigationTitle(vm.activeConversation?.title ?? "Text Chat")
#if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
#endif
                .toolbar {
                    ToolbarItem(placement: .navigation) {
                        Button { showConversations = true } label: {
                            Image(systemName: "sidebar.left")
                        }
                        .hoverEffect()
                    }
                    ToolbarItem(placement: .primaryAction) {
                        Menu {
                            Section("MLX Models") {
                                ForEach(Model.textModels.filter { $0.format == .mlx }, id: \.self) { m in
                                    modelButton(m)
                                }
                            }
                            let ggufText = Model.ggufModels.filter {
                                if case .text = $0.purpose { return true }
                                return false
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
                        .hoverEffect()
                    }
                }
                .sheet(isPresented: $showConversations) {
                    ConversationListSheet(vm: vm, isPresented: $showConversations)
                }
                // §5.1: Wire FocusedValue actions for Menu Bar commands
                .focusedValue(\.newConversationAction) {
                    Task { await vm.newConversation(model: selectedModel) }
                }
                .focusedValue(\.toggleSidebarAction) {
                    showConversations.toggle()
                }
        }
    }

    @ViewBuilder
    private func modelButton(_ m: Model) -> some View {
        Button {
            selectedModel = m
            Task { await vm.newConversation(model: m) }
        } label: {
            HStack {
                Text(m.displayName)
                Spacer()
                if m.isDownloaded {
                    Image(systemName: "checkmark.circle.fill")
                } else {
                    Image(systemName: "arrow.down.circle")
                }
            }
        }
    }
}

#Preview {
    TextChatTab()
}
