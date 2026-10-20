---
layout: docs
title: TonamoxVoice
parent: API Reference
nav_order: 3
description: "TonamoxVoice API reference — VoiceSession, VoiceButton, VoiceChatView."
---

# TonamoxVoice
{: .no_toc }

Turn-based (half-duplex) voice pipeline: on-device STT → LLM → on-device TTS. The microphone is off
while the assistant thinks and speaks; call `startListening()` again for the next turn. Adds no
packages beyond TonamoxCore, and speech recognition runs on-device (`requiresOnDeviceRecognition`).

```
Microphone → SFSpeechRecognizer → Tonamox.stream() → AVSpeechSynthesizer
```

Sentences stream to TTS **while the LLM is still generating** — the assistant starts speaking after the first complete sentence.

Guide: [Voice Interface]({{ '/guide/voice' | relative_url }}).

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## VoiceSession

```swift
@MainActor
public final class VoiceSession: NSObject, ObservableObject
```

### Initialization

```swift
// conversationID nil = a new conversation, created in `store` on the first utterance
init(llm: Tonamox, conversationID: UUID? = nil, store: ConversationStore = .shared, config: Config = Config())
```

### State

```swift
@Published private(set) var state: VoiceSession.State
@Published private(set) var transcript: String   // live STT transcript
@Published private(set) var response: String     // live LLM response

public enum State: Equatable {   // nested in VoiceSession → VoiceSession.State
    case idle
    case listening
    case thinking(partial: String)
    case speaking(sentence: String)
    case error(String)
}
```

### Control

```swift
func requestPermissions() async -> Bool
func startListening() async throws   // returns without doing anything unless state is .idle
func stopListening() async           // while .listening: stop recording, run the LLM → TTS pipeline
func interrupt()     // stop speaking and the reply being generated, clear the TTS queue, set .idle
func cancel()        // the same, and also stops recording
```

Both stop the reply being generated: the session stops reading its `stream(_:in:)`, which cancels the
generation. As with any stopped history stream, the text generated so far is stored as the assistant turn, and
with none yet the prompt turn is removed ([Multi-Turn Chat]({{ '/guide/conversations' | relative_url }}#multi-turn-chat)).
On a GGUF instance loaded with tools, llama.cpp finishes its decode before the next turn can start.

On iOS, moving the app to the background stops the reply: the session ends in `.error` with the
`GenerationInterrupted` message ("Generation stopped because the app moved to the background."), and the
text generated so far is stored as the assistant turn; with none yet, the prompt turn is removed
([Background Lifecycle]({{ '/guide/memory' | relative_url }}#background-lifecycle-ios)).

### Config

```swift
public struct Config {
    var silenceThreshold: TimeInterval = 1.4      // seconds before triggering LLM
    var maxRecordingDuration: TimeInterval = 30
    var locale: Locale? = nil                     // STT locale; nil = device's preferred language, else en-US
    var speakingRate: Float = AVSpeechUtteranceDefaultSpeechRate   // 0–1
    var maxTokens: Int = 512
    var systemPrompt: String? = nil

    init()
}
```

---

## SwiftUI Components

### VoiceButton

Drop-in microphone button. Manages its own `VoiceSession` internally.

```swift
// Minimal — internal session
VoiceButton(llm: llm)

// Internal session continuing a conversation, with a custom config
VoiceButton(llm: llm, conversationID: conv.id, config: config)

// With external session for state observation
VoiceButton(session: session)
```

### VoiceChatView

Full voice chat UI — transcript bubble, response bubble, and `VoiceButton`.

```swift
// New conversation (created in ConversationStore.shared on the first utterance)
VoiceChatView(llm: llm)

// Continue an existing conversation, with a custom config
VoiceChatView(llm: llm, conversationID: conv.id, config: config)
```

### VoiceTab

`VoiceTab` is **internal** to `TonamoxVoice` — it is **not** public, and nothing is added automatically when you `import TonamoxVoice`. To add voice to your own UI, place a public `VoiceButton` or `VoiceChatView` in your own view hierarchy (for example, as a tab you compose in your own `TabView`).

---

## Permissions

Add to `Info.plist`:

```xml
<key>NSSpeechRecognitionUsageDescription</key>
<string>Used for voice input to the local AI assistant.</string>
<key>NSMicrophoneUsageDescription</key>
<string>Used to capture your voice for the AI assistant.</string>
```
