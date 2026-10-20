---
layout: docs
title: API Reference
nav_order: 3
has_children: true
permalink: /api/
---

# API Reference

Reference for the main public types of TonamoxCore, TonamoxUI, TonamoxVoice and TonamoxDocs. It does not list
every public symbol, and members are listed without the `public` keyword.

Other modules are covered elsewhere:

- The on-device ML tools (Vision, Natural Language, Sound Analysis, Core ML, Create ML): the
  [On-device ML tools]({{ '/guide/ml-tools' | relative_url }}) guide.
- TonamoxImageGen (Core ML Stable Diffusion on device, built in or adopted from Hugging Face with `CoreMLDiffusionInspector`; FLUX via mflux on macOS): the [Image Generation]({{ '/guide/imagegen' | relative_url }}) guide.
- TonamoxAgents (`AgentCrew`): the [Hybrid Inference]({{ '/guide/hybrid' | relative_url }}#per-step-escalation-agent-orchestration) guide.
- TonamoxAppleIntelligence is not documented on this site.

TonamoxAgents and TonamoxAppleIntelligence import `FoundationModels`, so they build only with the Xcode 26 SDK.
TonamoxImageGen uses `ImagePlaygroundOptions`, so it builds only with the 26.4 SDKs (Xcode 26.4).
