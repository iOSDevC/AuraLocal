---
layout: docs
title: Conversations & History
parent: Guide
nav_order: 5
description: "Persistent multi-turn chat history with ConversationStore, context window management, auto-title, and summarization."
---

# Conversations & History
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## ConversationStore

`ConversationStore` is an actor-backed SQLite store (no external dependencies). All reads and writes are serialized automatically — safe to call from any task.

```swift
import AuraCore

let store = ConversationStore.shared

// Create a conversation
let conv = try await store.createConversation(
    model: .qwen3_1_7b,
    title: "Finance assistant"
)

// List all conversations
let conversations = try await store.allConversations()

// Full-text search
let results = try await store.search("VAT Mexico")

// Delete
try await store.deleteConversation(id: conv.id)
```

---

## Multi-Turn Chat

Pass a `conversationID` to any `chat` or `stream` call — context is loaded automatically:

```swift
let llm = try await AuraLocal.text(.qwen3_1_7b)

let reply1 = try await llm.chat("What is 2+2?", in: conv.id)
let reply2 = try await llm.chat("Why?", in: conv.id)  // has full context

// Streaming with history
for try await token in llm.stream("Tell me more", in: conv.id) {
    print(token, terminator: "")
}
```

Before each call AuraLocal loads the newest turns that fit `maxContextTokens` (default 3072, estimated at ~4 characters per token). When older turns have to be dropped, it snaps the oldest kept turn to a stride of up to 4 turns so llama.cpp's prompt cache survives, which can drop up to 3 extra old turns.

{: .note }
> MLX models receive the turns with their roles. GGUF models receive them flattened into one prompt: roles are dropped and only the most recent system turn is kept, so on GGUF a `summarizeAndPrune` summary replaces your system prompt.

---

## Auto-Title

Generate a title from the first message automatically:

```swift
try await llm.autoTitle(conversationID: conv.id)
// Updates the conversation title in ConversationStore
```

---

## Context Window Management

When a conversation grows long, `summarizeAndPrune` replaces older turns with a compact LLM-generated summary — preserving semantic continuity without truncation:

```swift
try await llm.summarizeAndPrune(
    conversationID: conv.id,
    keepLastN: 10,           // always keep 10 most recent turns
    maxContextTokens: 4096   // prune when conversation exceeds this
)
```

Nothing calls it for you: `chat(_:in:)`, `stream(_:in:)` and `ModelManager` never prune. Call it yourself after a reply (AuraUI's `TextChatTab` does this after every turn) or before switching models.

---

## One-Liner with Auto-Conversation

```swift
// Creates a new conversation automatically, returns the ID
let (reply, conversationID) = try await AuraLocal.chat(
    "Hello",
    model: .qwen3_1_7b
)
```

---

## SwiftUI Integration

```swift
import SwiftUI
import AuraCore

struct ChatView: View {
    @State private var messages: [Turn] = []
    @State private var llm: AuraLocal?
    let conversationID: UUID

    var body: some View {
        List(messages) { turn in
            Text(turn.content)
        }
        .task {
            llm = try? await AuraLocal.text(.qwen3_1_7b)
            messages = (try? await ConversationStore.shared.turns(for: conversationID)) ?? []
        }
    }

    func send(_ text: String) async {
        guard let llm else { return }
        _ = try? await llm.chat(text, in: conversationID)
        messages = (try? await ConversationStore.shared.turns(for: conversationID)) ?? []
    }
}
```
