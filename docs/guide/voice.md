---
layout: docs
title: Voice Interface
parent: Guide
nav_order: 6
description: "Full-duplex voice pipeline with AuraVoice — on-device STT, LLM streaming, and TTS with sentence-level pipelining."
---

# Voice Interface
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## How It Works

```
Microphone
  → SFSpeechRecognizer (on-device, no network)
    → AuraLocal.stream()
      → Sentence splitter
        → AVSpeechSynthesizer (on-device TTS)
```

Key characteristic: TTS **starts speaking after the first complete sentence** while the LLM is still generating the rest. End-to-end latency feels 2–3× lower than waiting for the full response.

Language is auto-detected per utterance via `NLLanguageRecognizer` and mapped to the best available system voice (e.g. `"es"` → `"es-MX"`).

---

## Quick Start

```swift
import AuraVoice

// Drop-in button — manages its own session
VoiceButton(llm: llm)
```

---

## Full Control

```swift
import AuraVoice

@StateObject var session = VoiceSession(llm: llm)

VStack {
    Text(session.transcript)   // live STT
        .foregroundColor(.secondary)
    Text(session.response)     // live LLM response
    VoiceButton(session: session)
}
```

---

## Session States

| State | Description |
|-------|-------------|
| `.idle` | Ready, microphone off |
| `.listening` | Recording, live transcript updating |
| `.thinking(partial:)` | LLM generating, partial response available |
| `.speaking(sentence:)` | TTS playing current sentence |
| `.error(String)` | Something went wrong |

---

## Configuration

```swift
var config = VoiceSession.Config()
config.silenceThreshold     = 1.4    // seconds of silence before LLM triggers
config.maxRecordingDuration = 30     // max recording per utterance
config.speakingRate         = 0.5    // TTS speed (0.0 slow → 1.0 fast)
config.maxTokens            = 512    // max LLM tokens per response
config.systemPrompt         = "You are a concise voice assistant."

let session = VoiceSession(llm: llm, config: config)
```

---

## Persistent Voice Chat

```swift
// Voice session with conversation history
let session = VoiceSession(llm: llm, conversationID: conv.id, config: config)

// Full UI: transcript + response bubbles + button
VoiceChatView(llm: llm, conversationID: conv.id)
```

---

## Manual Control

```swift
// Request mic + speech permissions
let granted = await session.requestPermissions()

// Start listening (silence detection auto-triggers LLM)
try await session.startListening()

// Stop recording manually
await session.stopListening()

// Interrupt TTS mid-sentence
session.interrupt()

// Cancel everything (recording + LLM + TTS)
session.cancel()
```

---

## Required Info.plist Keys

```xml
<key>NSSpeechRecognitionUsageDescription</key>
<string>Used for voice input to the local AI assistant.</string>
<key>NSMicrophoneUsageDescription</key>
<string>Used to capture your voice for the AI assistant.</string>
```

---

## VoiceTab

Use `AuraVoice` in `ContentView` — the Voice tab appears automatically when the module is imported.

```swift
import AuraUI
import AuraVoice  // this import adds the Voice tab to ContentView

@main struct MyApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}
```
