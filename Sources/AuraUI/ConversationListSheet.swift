import SwiftUI
import AuraCore

// MARK: - Conversation List Sheet
public struct ConversationListSheet: View {
    @ObservedObject var vm: TextChatViewModel
    @Binding var isPresented: Bool
    
    public var body: some View {
        NavigationStack {
            List {
                if vm.conversations.isEmpty {
                    ContentUnavailableView(
                        "No conversations",
                        systemImage: "bubble.left",
                        description: Text("Use the compose button to start a new chat.")
                    )
                } else {
                    ForEach(vm.conversations) { conv in
                        Button {
                            Task {
                                await vm.selectConversation(conv)
                                isPresented = false
                            }
                        } label: {
                            VStack(alignment: .leading, spacing: 3) {
                                Text(conv.title)
                                    .font(.subheadline.weight(.medium))
                                    .foregroundStyle(.primary)
                                    .lineLimit(1)
                                HStack(spacing: 4) {
                                    Text(conv.model.components(separatedBy: "/").last ?? conv.model)
                                        .font(.caption2).foregroundStyle(.secondary)
                                    Text("·").font(.caption2).foregroundStyle(.secondary)
                                    Text("\(conv.turnCount) turns")
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                            .padding(.vertical, 2)
                        }
                        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                            Button(role: .destructive) {
                                Task { await vm.deleteConversation(conv) }
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                        .listRowBackground(
                            vm.activeConversation?.id == conv.id
                            ? Color.blue.opacity(0.08) : Color.clear
                        )
                    }
                }
            }
            .navigationTitle("Conversations")
#if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
#endif
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") { isPresented = false }
                }
            }
        }
    }
}
