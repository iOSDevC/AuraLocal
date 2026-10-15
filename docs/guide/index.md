---
layout: docs
title: Guide
nav_order: 2
has_children: true
permalink: /guide/
---

# Guide

Step-by-step documentation for integrating AuraLocal into your app.

## On-device ML, not just LLMs

[On-device ML tools]({{ '/guide/ml-tools' | relative_url }}) covers the Vision, NaturalLanguage, SoundAnalysis, Core ML and
Create ML tools in `AuraCore`: what each one does, where it runs, Swift and `aura ml` examples, a
train → ship → classify walkthrough, and how to run your own Core ML model.

## Will this Hugging Face model run?

[Finding compatible models]({{ '/guide/models' | relative_url }}#finding-compatible-models) checks any Hugging
Face repo against the runtimes AuraLocal pins (mlx-swift-lm 3.31.3, llama.cpp b8851) and a device's memory
budget without downloading the weights: from Swift (`ModelCompatibilityChecker`), the CLI (`aura models`) or
the Model Finder example app.

## Images on device

[Image Generation]({{ '/guide/imagegen' | relative_url }}#on-device-iphone-and-mac-core-ml-stable-diffusion) runs Core ML
Stable Diffusion 2.1 base on iPhone and Mac after a one-time ~1.07 GB download. It also covers Apple's Image
Playground sheet (Private Cloud Compute from iOS 27) and FLUX + LoRA on macOS.

## Download models and keep them

[Downloading without loading]({{ '/guide/models' | relative_url }}#downloading-without-loading) covers
`ModelDownloads`: resumable GGUF and MLX downloads, the Hugging Face token, the free-space check, and
`DownloadRecords` for keeping downloads across launches. AuraUI's
[Hugging Face browser]({{ '/api/ui' | relative_url }}#hugging-face-browser) puts search, per-device verdicts and
downloads in your app.

## iOS lifecycle and memory

When an iOS app backgrounds, its running generations stop and throw `GenerationInterrupted` with the text so far
([Background Lifecycle]({{ '/guide/memory' | relative_url }}#background-lifecycle-ios)).
[Device conditions]({{ '/guide/memory' | relative_url }}#device-conditions) is the opt-in thermal guard, and
[iOS]({{ '/guide/memory' | relative_url }}#ios) explains the app-memory-limit budget.

## Constrained and structured output

[Constrained Generation]({{ '/guide/backends' | relative_url }}#constrained-generation-gguf-only) restricts a GGUF
reply to JSON or a GBNF grammar. `AuraSession` enforces a profile's `OutputSchema` by validation and repair
([Structured output]({{ '/api/core' | relative_url }}#structured-output)).

## Ask a bigger model

[Hybrid Inference]({{ '/guide/hybrid' | relative_url }}) escalates to your own llama-server/Ollama box or a
bring-your-own-key Anthropic or OpenAI model; GitHub Models was retired.
[`aura ask`]({{ '/guide/cli' | relative_url }}#ask-a-bigger-model-aura-ask) uses a local server unless you name a
cloud provider or pass a `--base-url` (a public host there counts as cloud).

## Research: distributed inference over USB

[Distributed Inference over USB (research)]({{ '/guide/distributed' | relative_url }}) records an evaluation of
a Mac orchestrating USB-tethered iPhones for speculative decoding and embeddings: measured results, transport
and iOS findings, and what AuraLocal would need. No mesh API ships; the only piece that did is the
multilingual-e5-small embedding provider.
