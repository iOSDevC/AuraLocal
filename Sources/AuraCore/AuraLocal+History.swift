import Foundation

// MARK: - AuraLocal + History
//
// Extends AuraLocal with stateful, persistent conversation methods.
// The store is injected so you can use ConversationStore.shared or a
// custom instance (e.g. in-memory for tests).

public extension AuraLocal {

    // MARK: - Stateful chat (creates & persists turns automatically)

    /// Send a message within a persistent conversation.
    ///
    /// - Loads the context window from the store (respecting `maxContextTokens`).
    /// - Appends the user turn before inference.
    /// - Streams the response, finalizing the assistant turn when done.
    /// - Returns the complete assistant reply.
    ///
    /// ```swift
    /// let llm   = try await AuraLocal.text(.qwen3_1_7b)
    /// let conv  = try await ConversationStore.shared.createConversation(model: .qwen3_1_7b)
    ///
    /// let reply = try await llm.chat("What is 2+2?", in: conv.id)
    /// let reply2 = try await llm.chat("Why?", in: conv.id) // context includes first exchange
    /// ```
    func chat(
        _ prompt: String,
        in conversationID: UUID,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024,
        maxContextTokens: Int = 3072,
        store: ConversationStore = .shared
    ) async throws -> String {

        // Optionally persist a system prompt on first use
        if let sys = systemPrompt {
            let existing = try await store.turns(for: conversationID)
            if !existing.contains(where: { $0.role == .system }) {
                try await store.appendTurn(Turn(
                    conversationID: conversationID,
                    role: .system,
                    content: sys
                ))
            }
        }

        // Persist the user turn
        try await store.appendTurn(Turn(
            conversationID: conversationID,
            role: .user,
            content: prompt
        ))

        // Build message array from context window
        let context = try await store.contextWindow(
            for: conversationID,
            maxTokens: maxContextTokens
        )
        let messages = context.map { turn -> [String: String] in
            ["role": turn.role.rawValue, "content": turn.content]
        }

        // Run inference with the full context
        let reply = try await chatWithMessages(messages, maxTokens: maxTokens)

        // Persist the assistant reply
        try await store.appendTurn(Turn(
            conversationID: conversationID,
            role: .assistant,
            content: reply
        ))

        return reply
    }

    /// Stream tokens within a persistent conversation.
    ///
    /// The assistant turn is persisted when the stream finishes successfully.
    ///
    /// ```swift
    /// for try await token in llm.stream("Tell me more", in: conv.id) {
    ///     print(token, terminator: "")
    /// }
    /// ```
    func stream(
        _ prompt: String,
        in conversationID: UUID,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024,
        maxContextTokens: Int = 3072,
        store: ConversationStore = .shared
    ) -> AsyncThrowingStream<String, Error> {

        AsyncThrowingStream { continuation in
            Task { @MainActor in
                do {
                    // System prompt (once)
                    if let sys = systemPrompt {
                        let existing = try await store.turns(for: conversationID)
                        if !existing.contains(where: { $0.role == .system }) {
                            try await store.appendTurn(Turn(
                                conversationID: conversationID,
                                role: .system,
                                content: sys
                            ))
                        }
                    }

                    // Persist user turn
                    try await store.appendTurn(Turn(
                        conversationID: conversationID,
                        role: .user,
                        content: prompt
                    ))

                    // Build context
                    let context = try await store.contextWindow(
                        for: conversationID,
                        maxTokens: maxContextTokens
                    )

                    // Build full multi-turn messages from context window
                    let messages = context.map { turn -> [String: String] in
                        ["role": turn.role.rawValue, "content": turn.content]
                    }

                    // Stream with full conversational context
                    var fullReply = ""
                    var lastLength = 0

                    _ = try await engine.generate(
                        messages: messages,
                        maxTokens: maxTokens
                    ) { @MainActor partial in
                        let newText = String(partial.dropFirst(lastLength))
                        lastLength = partial.count
                        if !newText.isEmpty {
                            fullReply += newText
                            continuation.yield(newText)
                        }
                    }

                    // Persist completed assistant turn
                    try await store.appendTurn(Turn(
                        conversationID: conversationID,
                        role: .assistant,
                        content: fullReply
                    ))

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Context-aware one-liner

    /// Load a conversation's context window and send a one-shot message.
    /// Creates a new conversation if `conversationID` is nil.
    @discardableResult
    static func chat(
        _ prompt: String,
        in conversationID: UUID? = nil,
        model: Model = .qwen3_1_7b,
        systemPrompt: String? = nil,
        store: ConversationStore = .shared,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> (reply: String, conversationID: UUID) {

        let convID: UUID
        if let id = conversationID {
            convID = id
        } else {
            let conv = try await store.createConversation(model: model)
            convID = conv.id
        }

        let llm = try await AuraLocal.text(model, onProgress: onProgress)
        let reply = try await llm.chat(
            prompt,
            in: convID,
            systemPrompt: systemPrompt,
            store: store
        )

        return (reply, convID)
    }

    // MARK: - Auto-title

    /// Generate and persist a short title for a conversation based on its first user message.
    func autoTitle(
        conversationID: UUID,
        store: ConversationStore = .shared
    ) async throws {
        let turns = try await store.turns(for: conversationID)
        guard let firstUser = turns.first(where: { $0.role == .user }) else { return }

        let title = try await chat(
            "Summarize this message as a short conversation title (max 6 words, no quotes): \(firstUser.content)",
            maxTokens: 20
        )
        try await store.updateTitle(title.trimmingCharacters(in: .whitespacesAndNewlines), for: conversationID)
    }

    // MARK: - Pruning with summarization

    /// Summarize and prune old turns when the conversation grows beyond a token budget.
    /// Call this periodically (e.g. every 20 turns) to keep inference fast.
    func summarizeAndPrune(
        conversationID: UUID,
        keepLastN: Int = 10,
        maxContextTokens: Int = 4096,
        store: ConversationStore = .shared
    ) async throws {
        let stats = try await store.stats(for: conversationID)
        guard stats.totalTokenEstimate > maxContextTokens else { return }

        let allTurns = try await store.turns(for: conversationID)
        let toPrune  = allTurns.filter { $0.role != .system }.dropLast(keepLastN)
        guard !toPrune.isEmpty else { return }

        let transcript = toPrune
            .map { "[\($0.role.rawValue)]: \($0.content)" }
            .joined(separator: "\n")

        let summary = try await chat(
            "Summarize the following conversation excerpt concisely for an AI assistant's memory:\n\n\(transcript)",
            maxTokens: 256
        )

        try await store.pruneAndSummarize(
            conversationID: conversationID,
            keepLastN: keepLastN,
            summary: summary
        )
    }

    // MARK: - Internal helper

    /// Run inference with a pre-built messages array (used by history methods).
    /// Forwards the full multi-turn context to the engine so the LLM sees
    /// all prior exchanges, not just the system prompt and last user message.
    internal func chatWithMessages(
        _ messages: [[String: String]],
        maxTokens: Int
    ) async throws -> String {
        try await engine.generate(
            messages: messages,
            maxTokens: maxTokens,
            onToken: { _ in }
        )
    }
}
