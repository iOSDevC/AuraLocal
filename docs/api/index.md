---
layout: docs
title: API Reference
nav_order: 3
has_children: true
permalink: /api/
---

# API Reference

Reference for the main public types of AuraCore, AuraUI, AuraVoice and AuraDocs. It does not list
every public symbol, and members are listed without the `public` keyword.

Other modules are covered elsewhere:

- The on-device ML tools (Vision, Natural Language, Sound Analysis, Core ML, Create ML): the
  [On-device ML tools]({{ '/guide/ml-tools' | relative_url }}) guide.
- AuraImageGen: the [Image Generation]({{ '/guide/imagegen' | relative_url }}) guide.
- AuraAgents (`AgentCrew`): the [Hybrid Inference]({{ '/guide/hybrid' | relative_url }}#per-step-escalation-agent-orchestration) guide.
- AuraAppleIntelligence is not documented on this site.

AuraAgents and AuraAppleIntelligence import `FoundationModels`, so they build only with the Xcode 26 SDK.
