---
layout: docs
title: AuraVoice
parent: API Reference
nav_order: 3
description: "AuraVoice API reference — VoiceSession, VoiceButton, VoiceChatView."
---

# AuraVoice
{: .no_toc }

Full-duplex voice pipeline: on-device STT → LLM → on-device TTS. No external dependencies, no network calls.

```
Microphone → SFSpeechRecognizer → AuraLocal.stream() → AVSpeechSynthesizer
```

Sentences stream to TTS **while the LLM is still generating** — the assistant starts speaking after the first complete sentence.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## VoiceSession

```swift
@MainActor
public final class VoiceSession: ObservableObject
```

### Initialization

```swift
init(llm: AuraLocal, config: Config = .init())
init(llm: AuraLocal, conversationID: UUID, config: Config = .init())
```

### State

```swift
@Published var state: VoiceState
@Published var transcript: String   // live STT transcript
@Published var response: String     // live LLM response

public enum VoiceState {
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
func startListening() async throws
func stopListening() async
func interrupt()     // stop TTS mid-sentence
func cancel()        // cancel everything
```

### Config

```swift
public struct Config {
    var silenceThreshold: TimeInterval = 1.4      // seconds before triggering LLM
    var maxRecordingDuration: TimeInterval = 30
    var speakingRate: Float = 0.5                 // AVSpeechSynthesizer rate (0–1)
    var maxTokens: Int = 512
    var systemPrompt: String? = nil
}
```

---

## SwiftUI Components

### VoiceButton

Drop-in microphone button. Manages its own `VoiceSession` internally.

```swift
// Minimal — internal session
VoiceButton(llm: llm)

// With external session for state observation
VoiceButton(session: session)
```

### VoiceChatView

Full voice chat UI — transcript bubble, response bubble, and `VoiceButton`.

```swift
// Standalone
VoiceChatView(llm: llm)

// With persistent conversation history
VoiceChatView(llm: llm, conversationID: conv.id)
```

### VoiceTab

Ready-to-use tab for `AuraUI`'s `ContentView`. Added automatically when `AuraVoice` is imported.

---

## Permissions

Add to `Info.plist`:

```xml
<key>NSSpeechRecognitionUsageDescription</key>
<string>Used for voice input to the local AI assistant.</string>
<key>NSMicrophoneUsageDescription</key>
<string>Used to capture your voice for the AI assistant.</string>
```
