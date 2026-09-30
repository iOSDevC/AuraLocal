---
layout: docs
title: AuraVoice
parent: API Reference
nav_order: 3
description: "AuraVoice API reference — VoiceSession, VoiceButton, VoiceChatView."
---

# AuraVoice
{: .no_toc }

Turn-based (half-duplex) voice pipeline: on-device STT → LLM → on-device TTS. The microphone is off
while the assistant thinks and speaks; call `startListening()` again for the next turn. Adds no
packages beyond AuraCore, and speech recognition runs on-device (`requiresOnDeviceRecognition`).

```
Microphone → SFSpeechRecognizer → AuraLocal.stream() → AVSpeechSynthesizer
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
init(llm: AuraLocal, conversationID: UUID? = nil, store: ConversationStore = .shared, config: Config = Config())
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
func interrupt()     // stop the current sentence, clear the TTS queue, set .idle
func cancel()        // also stops recording; same effect on TTS as interrupt()
```

{: .warning }
> Neither `interrupt()` nor `cancel()` stops an LLM generation already in progress. It runs to the
> end, its later sentences are still queued and spoken, and the assistant turn is saved to the
> conversation.

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

// Continue an existing conversation
VoiceChatView(llm: llm, conversationID: conv.id)
```

### VoiceTab

`VoiceTab` is **internal** to `AuraVoice` — it is **not** public, and nothing is added automatically when you `import AuraVoice`. To add voice to your own UI, place a public `VoiceButton` or `VoiceChatView` in your own view hierarchy (for example, as a tab you compose in your own `TabView`).

---

## Permissions

Add to `Info.plist`:

```xml
<key>NSSpeechRecognitionUsageDescription</key>
<string>Used for voice input to the local AI assistant.</string>
<key>NSMicrophoneUsageDescription</key>
<string>Used to capture your voice for the AI assistant.</string>
```
