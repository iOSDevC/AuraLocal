import Foundation
import AuraCore
import Combine

// MARK: - Chat Message (UI model)

struct ChatMessage: Identifiable {
    let id: UUID
    let role: Turn.Role
    let text: String
}

// MARK: - TextChatViewModel

@MainActor
final class TextChatViewModel: ObservableObject {
    
    @Published var conversations: [Conversation] = []
    @Published var activeConversation: Conversation?
    @Published var messages: [ChatMessage] = []
    @Published var streamingText: String = ""
    @Published var progress: String = ""
    @Published var isStreaming = false
    
    private let store = ConversationStore.shared
    
    init() {
        Task { await loadConversations() }
    }
    
    // MARK: Conversation management
    
    func loadConversations() async {
        conversations = (try? await store.allConversations()) ?? []
    }
    
    func newConversation(model: Model) async {
        guard let conv = try? await store.createConversation(model: model) else { return }
        activeConversation = conv
        messages = []
        conversations.insert(conv, at: 0)
    }
    
    func selectConversation(_ conv: Conversation) async {
        activeConversation = conv
        let turns = (try? await store.turns(for: conv.id)) ?? []
        messages = turns
            .filter { $0.role != .system }
            .map { ChatMessage(id: $0.id, role: $0.role, text: $0.content) }
    }
    
    func deleteConversation(_ conv: Conversation) async {
        try? await store.deleteConversation(id: conv.id)
        conversations.removeAll { $0.id == conv.id }
        if activeConversation?.id == conv.id {
            activeConversation = nil
            messages = []
        }
    }
    
    // MARK: Send
    
    func send(_ prompt: String, model: Model) async {
        if activeConversation == nil { await newConversation(model: model) }
        guard let convID = activeConversation?.id else { return }

        messages.append(ChatMessage(id: UUID(), role: .user, text: prompt))
        isStreaming = true
        streamingText = ""

        do {
            let llm = try await ModelManager.shared.load(model) { [weak self] p in self?.progress = p }
            progress = ""

            for try await token in llm.stream(prompt, in: convID, store: store) {
                streamingText += token
            }
            
            messages.append(ChatMessage(id: UUID(), role: .assistant, text: streamingText))
            streamingText = ""
            
            // Auto-title after first user message
            if messages.filter({ $0.role == .user }).count == 1 {
                try? await llm.autoTitle(conversationID: convID, store: store)
                await loadConversations()
            }
            
            // Prune long conversations
            try? await llm.summarizeAndPrune(conversationID: convID, store: store)
            
        } catch {
            messages.append(ChatMessage(id: UUID(), role: .assistant, text: "❌ \(error.localizedDescription)"))
        }
        
        isStreaming = false
        progress = ""
    }
}

// MARK: - VisionViewModel

@MainActor
final class VisionViewModel: ObservableObject {
    @Published var selectedImage: PlatformImage?
    @Published var output: String = ""
    @Published var progress: String = ""
    @Published var isLoading = false
    
    func run(model: Model, image: PlatformImage, prompt: String, mode: AuraLocal.VisionRunMode) async {
        isLoading = true
        output = ""

        do {
            let vlm = try await ModelManager.shared.load(model) { [weak self] p in
                self?.progress = p
            }
            progress = ""
            
            switch mode {
                case .standard:
                    output = try await vlm.analyze(prompt, image: image)
                    
                case .stream:
                    for try await token in vlm.streamVision(prompt, image: image) {
                        output += token
                    }
            }
        } catch {
            output = "❌ \(error.localizedDescription)"
        }
        
        isLoading = false
        progress = ""
    }
}

// MARK: - OCRViewModel

@MainActor
final class OCRViewModel: ObservableObject {
    @Published var selectedImage: PlatformImage?
    @Published var output: String = ""
    @Published var progress: String = ""
    @Published var isLoading = false
    
    func run(model: Model, image: PlatformImage) async {
        isLoading = true
        output = ""

        do {
            let ocr = try await ModelManager.shared.load(model) { [weak self] p in
                self?.progress = p
            }
            progress = ""
            
            var result = try await ocr.extractDocument(image)
            
            if case .visionSpecialized(let docTags) = model.purpose, docTags {
                result = AuraLocal.parseDocTags(result)
                output = "📝 Markdown:\n\n\(result)"
            } else {
                output = "⚡ JSON:\n\n\(result)"
            }
        } catch {
            output = "❌ \(error.localizedDescription)"
        }
        
        isLoading = false
        progress = ""
    }
}
