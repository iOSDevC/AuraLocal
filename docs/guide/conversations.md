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
import TonamoxCore

let store = ConversationStore.shared

// Create a conversation
let conv = try await store.createConversation(
    model: .qwen3_1_7b,
    title: "Finance assistant"
)

// List all conversations
let conversations = try await store.allConversations()

// Full-text search: turns containing any of the words, best match first
let results = try await store.search("VAT Mexico")

// Delete
try await store.deleteConversation(id: conv.id)
```

---

## Multi-Turn Chat

Pass a `conversationID` to any `chat` or `stream` call — context is loaded automatically:

```swift
let llm = try await Tonamox.text(.qwen3_1_7b)

let reply1 = try await llm.chat("What is 2+2?", in: conv.id)
let reply2 = try await llm.chat("Why?", in: conv.id)  // has full context

// Streaming with history
for try await token in llm.stream("Tell me more", in: conv.id) {
    print(token, terminator: "")
}
```

Before each call Tonamox loads the newest turns that fit `maxContextTokens` (default 3072, estimated at ~4 characters per token). When older turns have to be dropped, it snaps the oldest kept turn to a stride of up to 4 turns, which can drop up to 3 extra old turns. Between moves the window's start stays byte-identical, so the GGUF backends can keep their KV cache and decode only the new turns; a turn on which the start moves prefills the whole window ([Reusing the KV cache]({{ '/guide/backends' | relative_url }}#multi-turn-conversations-gguf)).

The user turn is saved before the model runs. If the call throws, the conversation keeps one reply per prompt:
a `GenerationInterrupted` that carries text (the app moved to the background, or the device guard stopped it)
saves that text as the assistant turn; any other error removes the user turn again.

Cancelling the task that runs a `stream(_:in:)` loop (a Stop button), or leaving the loop early, cancels the
generation and settles the conversation the same way: the text generated so far is saved as the reply, or the user
turn is removed when there was none. Leaving the loop cancels only once nothing else holds the stream; if you keep it
in a variable or property, stop it by cancelling the task. Those writes finish after your loop has ended, so the next
`chat(_:in:)` or `stream(_:in:)` on the same `Tonamox` waits for them before saving its prompt.

On a GGUF model loaded with tools whose call format runs on LocalLLMClient's `LlamaClient` (for example Llama 3.1's
or Mistral's), the reply is cut and saved at once, but llama.cpp keeps decoding until the end of generation or a full
context window, and the next turn on that instance waits for it. With `<tool_call>` JSON templates (Qwen2.5's, and
templates without tool support) the decode stops as without tools: after at most one more token, or when a running
prefill ends. See
[Multi-turn conversations]({{ '/guide/backends' | relative_url }}#multi-turn-conversations-gguf).

{: .note }
> MLX and GGUF models receive the turns with their roles. On GGUF every system turn, a `summarizeAndPrune` summary included, merges into one leading system turn, so the summary is added to your system prompt rather than replacing it. If the window still exceeds the model's context, the GGUF backends drop the oldest exchanges until it fits, keeping the system turn and your new message. See [Multi-turn conversations]({{ '/guide/backends' | relative_url }}#multi-turn-conversations-gguf). MLX trims nothing: past its KV window (`maxKVSize`) it overwrites the oldest tokens without an error, and `llm.lastContextOverflow` then says how many no longer fit. Qwen 3.5 text models, Gemma 3 and 4, and LFM2 ignore the window and leave it `nil` (see [Memory › MLX KV window]({{ '/guide/memory' | relative_url }}#mlx-kv-window)).

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

Nothing calls it for you: `chat(_:in:)`, `stream(_:in:)` and `ModelManager` never prune. Call it yourself after a reply (TonamoxUI's `TextChatTab` does this after every turn) or before switching models.

---

## One-Liner with Auto-Conversation

```swift
// Creates a new conversation automatically, returns the ID
let (reply, conversationID) = try await Tonamox.chat(
    "Hello",
    model: .qwen3_1_7b
)
```

---

## SwiftUI Integration

```swift
import SwiftUI
import TonamoxCore

struct ChatView: View {
    @State private var messages: [Turn] = []
    @State private var llm: Tonamox?
    let conversationID: UUID

    var body: some View {
        List(messages) { turn in
            Text(turn.content)
        }
        .task {
            llm = try? await Tonamox.text(.qwen3_1_7b)
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
