import Foundation
import Combine
import AuraCore

// MARK: - ChatEntry

/// One bubble in the chat. `image` lives only in memory: the store keeps text turns.
struct ChatEntry: Identifiable {
    let id: UUID
    let role: Turn.Role
    let text: String
    var image: PlatformImage?
    var isError = false
    /// A status line such as "Stopped.", not a reply.
    var isNotice = false
}

// MARK: - ChatViewModel

/// Conversations, sending and stopping for ``ChatTab``. Text goes through the history API, so earlier turns are in
/// context; an image turn and a document extraction run without history and are stored as text turns afterwards.
@MainActor
final class ChatViewModel: ObservableObject {

    @Published private(set) var conversations: [Conversation] = []
    @Published private(set) var activeConversation: Conversation?
    @Published private(set) var entries: [ChatEntry] = []
    @Published private(set) var streamingText = ""
    @Published private(set) var progress = ""
    @Published private(set) var isBusy = false
    /// Set while the model loads, before the first token: Stop cancels the load then.
    @Published private(set) var isLoading = false

    /// Stored in front of an image turn's prompt, so the transcript and later context say an image was there.
    static let imageMarker = "[Image] "
    static let defaultImagePrompt = "Describe this image in detail."

    private let store = ConversationStore.shared
    private var work: Task<Void, Never>?
    private var workModel: Model?

    init() {
        Task { await loadConversations() }
    }

    // MARK: Conversations

    func loadConversations() async {
        conversations = (try? await store.allConversations()) ?? []
    }

    func newConversation(model: Model) async {
        guard !isBusy else { return }
        await createConversation(model: model)
    }

    private func createConversation(model: Model) async {
        guard let conversation = try? await store.createConversation(model: model) else { return }
        activeConversation = conversation
        entries = []
        conversations.insert(conversation, at: 0)
    }

    func selectConversation(_ conversation: Conversation) async {
        guard !isBusy else { return }
        activeConversation = conversation
        let turns = (try? await store.turns(for: conversation.id)) ?? []
        entries = turns.filter { $0.role != .system }.map(Self.entry)
    }

    /// The running reply writes its turns into this conversation, so it can't be deleted until it ends.
    func isReplying(in conversation: Conversation) -> Bool {
        isBusy && activeConversation?.id == conversation.id
    }

    func deleteConversation(_ conversation: Conversation) async {
        guard !isReplying(in: conversation) else { return }
        try? await store.deleteConversation(id: conversation.id)
        conversations.removeAll { $0.id == conversation.id }
        if activeConversation?.id == conversation.id {
            activeConversation = nil
            entries = []
        }
    }

    private static func entry(_ turn: Turn) -> ChatEntry {
        ChatEntry(id: turn.id, role: turn.role, text: turn.content)
    }

    // MARK: Sending

    /// Sends a text prompt within the active conversation, streaming the reply.
    func send(_ prompt: String, model: Model) {
        start(model: model) { await self.runText(prompt, model: model) }
    }

    /// Asks about an image; `prompt` empty uses ``defaultImagePrompt``.
    func send(_ prompt: String, image: PlatformImage, model: Model) {
        let question = prompt.isEmpty ? Self.defaultImagePrompt : prompt
        start(model: model) { await self.runImage(question, image: image, model: model) }
    }

    /// Runs the model's document prompt on `image`: Markdown for DocTags models, JSON otherwise.
    func extractDocument(_ image: PlatformImage, model: Model) {
        start(model: model) { await self.runExtraction(image, model: model) }
    }

    /// Cancels the load or the generation. A stopped reply keeps the text generated so far.
    func stop() {
        guard let workModel else { return }
        if isLoading { ModelManager.shared.cancelLoad(workModel) }
        work?.cancel()
    }

    private func start(model: Model, _ body: @escaping @MainActor () async -> Void) {
        guard !isBusy else { return }
        isBusy = true
        workModel = model
        work = Task {
            await body()
            self.isBusy = false
            self.isLoading = false
            self.progress = ""
            self.streamingText = ""
            self.work = nil
            self.workModel = nil
        }
    }

    private func ensureConversation(model: Model) async -> UUID? {
        if activeConversation == nil { await createConversation(model: model) }
        return activeConversation?.id
    }

    private func load(_ model: Model) async throws -> AuraLocal {
        isLoading = true
        defer { isLoading = false }
        let llm = try await ModelManager.shared.load(model) { [weak self] message in self?.progress = message }
        // A cancelled load can still return an instance; don't generate with it.
        try Task.checkCancellation()
        progress = ""
        return llm
    }

    private func runText(_ prompt: String, model: Model) async {
        guard let conversationID = await ensureConversation(model: model) else { return }
        entries.append(ChatEntry(id: UUID(), role: .user, text: prompt))
        do {
            let llm = try await load(model)
            for try await token in llm.stream(prompt, in: conversationID, store: store) {
                streamingText += token
            }
            // Cancelling the iterating task ends the loop without an error; the store saves the partial reply.
            if Task.isCancelled {
                stopped()
                return
            }
            entries.append(ChatEntry(id: UUID(), role: .assistant, text: streamingText))
            streamingText = ""
            await finishExchange(llm: llm, conversationID: conversationID, title: prompt)
            try? await llm.summarizeAndPrune(conversationID: conversationID, store: store)
        } catch {
            // The store saved this text as the reply; keep it on screen.
            if let stop = error as? GenerationInterrupted, !stop.partialText.isEmpty {
                streamingText = stop.partialText
            }
            keepPartialReply()
            report(error)
        }
    }

    private func runImage(_ prompt: String, image: PlatformImage, model: Model) async {
        guard let conversationID = await ensureConversation(model: model) else { return }
        entries.append(ChatEntry(id: UUID(), role: .user, text: prompt, image: image))
        do {
            let llm = try await load(model)
            for try await token in llm.streamVision(prompt, image: image) {
                streamingText += token
            }
            let reply = streamingText
            if Task.isCancelled {
                stopped()
            } else {
                entries.append(ChatEntry(id: UUID(), role: .assistant, text: reply))
                streamingText = ""
            }
            guard !reply.isEmpty else { return }
            await storeExchange(Self.imageMarker + prompt, reply: reply, conversationID: conversationID)
            await finishExchange(llm: llm, conversationID: conversationID, title: prompt)
        } catch {
            keepPartialReply()
            report(error)
        }
    }

    private func runExtraction(_ image: PlatformImage, model: Model) async {
        guard let conversationID = await ensureConversation(model: model) else { return }
        let request = "Extract this document."
        entries.append(ChatEntry(id: UUID(), role: .user, text: request, image: image))
        do {
            let llm = try await load(model)
            progress = "Extracting…"
            let raw = try await llm.extractDocument(image)
            try Task.checkCancellation()
            let reply: String
            if case .visionSpecialized(let docTags) = model.purpose, docTags {
                reply = "Markdown:\n\n" + AuraLocal.parseDocTags(raw)
            } else {
                reply = "JSON:\n\n" + raw
            }
            entries.append(ChatEntry(id: UUID(), role: .assistant, text: reply))
            await storeExchange(Self.imageMarker + request, reply: reply, conversationID: conversationID)
            await finishExchange(llm: llm, conversationID: conversationID, title: "Document")
        } catch {
            report(error)
        }
    }

    // MARK: Transcript

    private func keepPartialReply() {
        guard !streamingText.isEmpty else { return }
        entries.append(ChatEntry(id: UUID(), role: .assistant, text: streamingText))
        streamingText = ""
    }

    /// Keeps what was generated, or says it stopped before the first token.
    private func stopped() {
        if streamingText.isEmpty { report(CancellationError()) } else { keepPartialReply() }
    }

    private func report(_ error: any Error) {
        if error is CancellationError {
            entries.append(ChatEntry(id: UUID(), role: .assistant, text: "Stopped.", isNotice: true))
        } else {
            entries.append(ChatEntry(id: UUID(), role: .assistant, text: error.localizedDescription, isError: true))
        }
    }

    /// The history API stores text turns itself; image and document turns are stored here, once they have a reply.
    private func storeExchange(_ prompt: String, reply: String, conversationID: UUID) async {
        _ = try? await store.appendTurn(Turn(conversationID: conversationID, role: .user, content: prompt))
        _ = try? await store.appendTurn(Turn(conversationID: conversationID, role: .assistant, content: reply))
    }

    /// Titles the conversation after its first exchange: by the model when it chats, else from the prompt.
    private func finishExchange(llm: AuraLocal, conversationID: UUID, title: String) async {
        guard !Task.isCancelled, entries.filter({ $0.role == .user }).count == 1 else { return }
        if llm.model.capabilities.contains(.text) {
            try? await llm.autoTitle(conversationID: conversationID, store: store)
        } else {
            try? await store.updateTitle(String(title.prefix(40)), for: conversationID)
        }
        await loadConversations()
        activeConversation = conversations.first { $0.id == conversationID } ?? activeConversation
    }
}
