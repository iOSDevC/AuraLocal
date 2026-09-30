---
layout: docs
title: Voice Interface
parent: Guide
nav_order: 6
description: "Turn-based voice pipeline with AuraVoice — on-device STT, LLM streaming, and TTS with sentence-level pipelining."
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

Speech is recognized in one locale: `Config.locale`, else the device's first preferred language, else `en-US`. Each sentence of the reply is language-detected with `NLLanguageRecognizer` to pick the TTS voice (enhanced quality first). A code such as `"es"` becomes the first of the user's preferred languages that starts with it (`"es-MX"` only if listed), otherwise it stays `"es"`.

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

`session.state` is published as `VoiceSession.State` (a nested enum). Its cases:

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
config.locale               = Locale(identifier: "es-MX")   // speech-recognition locale; nil = device language
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

// Stop recording and TTS (an LLM reply in progress keeps generating and speaking)
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

## Adding Voice to Your App

`AuraVoice` does **not** add a tab to your app automatically. The `VoiceTab` view is internal to the module and is not part of the public API — importing `AuraVoice` exposes no tab. To add voice, place the public `VoiceButton` or `VoiceChatView` in your own view hierarchy.

```swift
import AuraVoice

// Drop-in button — manages its own session
VoiceButton(llm: llm)

// Or a full screen: transcript + response bubbles + button
VoiceChatView(llm: llm)
```
