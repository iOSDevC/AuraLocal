---
layout: docs
title: Hybrid Inference
parent: Guide
nav_order: 2
description: "Local-first inference with optional, consent-gated escalation to a bigger local or cloud model — GitHub Models, llama-server/Ollama, Anthropic, or OpenAI — with token-saving compression and cost accounting."
---

# Hybrid Inference (local + remote)
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AuraLocal is **on-device first**. The hybrid line adds an *optional*, **consent-gated**
path to escalate a request to a more powerful model — your own `llama-server`/Ollama box
on the LAN, or a cloud provider (GitHub Models, Anthropic, OpenAI) — only when the local
model isn't enough.

{: .note }
> Escalation is **off by default** (`EscalationPolicy.off`). Nothing leaves the device
> until you set a policy, and every cloud send is shown to the user for approval first.
> The same `AuraLocal.stream()` API drives local and remote — the pipeline is *mixed*,
> not cloud.

The design goal is to **cut the tokens sent to the remote**: stay local by default, and
when you do escalate, compress the context, redact obvious secrets, and cache repeats.

## Token savings — the headline

| Technique | Effect |
|---|---|
| Local-first routing | Most turns never leave the device — **$0**, no tokens sent |
| Selective-context compression | Trims the context toward the remote's budget — **~2–4×** fewer tokens |
| Response cache | Identical requests return instantly — **$0** |
| PII redaction | Strips obvious secrets before the payload is sent |
| Cost ledger | Per-escalation token + dollar accounting, shown as a receipt |

Every escalation prints a receipt — *"via GitHub Models · 800 in / 240 out · compressed 6,000→1,500 (4.0×)"*.

## Detect local providers

Discover a running Ollama (`:11434`) or llama.cpp `llama-server` (`:8080/v1`) and the
models each exposes — a dependency-free `URLSession` probe that never throws (a down
server is a normal result).

```swift
import AuraCore

let providers = await LocalProviderDetector.detectAll()
for p in providers where p.isAvailable {
    print(p.kind, p.baseURL, p.models.map(\.name))
}
```

## Escalate a request

`HybridEscalator` compresses the context toward the remote's budget, optionally redacts
PII, checks the response cache, streams the answer, and records cost.

```swift
import AuraCore

let escalator = HybridEscalator()

// Route by policy: stays local, or escalates / asks consent per the router's rules.
let result = try await escalator.routeAndEscalate(
    policy: EscalationPolicy(mode: .askEachTime, allowCloud: true),
    context: longContext,
    question: "What's the safest fix?",
    localAnswer: draftFromLocalModel,   // enables the low-confidence trigger
    consent: myConsentGate              // presents the payload + projected cost
) { partial in
    // cumulative tokens — same contract as AuraLocal.stream()
}

if let result {
    print(result.answer)
    print(result.usage as Any, result.compression.factor, result.fromCache)
} // nil → the router kept it local
```

The router (`EscalationRouter`) is a pure decision function (rules R1–R7): it stays local
unless there's a size overflow, a low-confidence local answer, or a sensitive domain — and
never escalates silently when a cost cap would be exceeded.

## GitHub Models (BYOK)

Bring your GitHub/Copilot account into the hybrid line via GitHub's official
OpenAI-compatible endpoint (`models.github.ai/inference`). Auth is a **fine-grained PAT
with the `models:read` permission**, stored in the Keychain (never in source, files, or logs).

```swift
// Provider preset — powered by your GitHub token
let provider = OpenAICompatibleProvider.gitHubModels(apiKey: token)
let target = RemoteTarget(provider: provider, modelID: "openai/gpt-4o",
                          contextLength: 128_000, origin: .cloud)
let result = try await HybridEscalator().escalate(
    to: target, context: "", question: "Explain hybrid inference", maxTokens: 512)
```

Save the PAT once (e.g. from a settings screen): `try KeychainStore.save(pat, for: "cloud.github-models")`.
The same BYOK slots exist for `cloud.anthropic` and `cloud.openai`. Works from iOS, macOS, and visionOS.

{: .note }
> The GitHub Models free tier is rate-capped (~15 req/min, ~150/day). AuraLocal treats it
> as an escalation valve — a 429 falls through to the next configured target.

## Per-step escalation (agent orchestration)

Escalation isn't only for the top-level answer. `routeAndEscalate(localAnswer:)` routes
each **sub-task** to the cheapest capable executor: the agent crew drafts a step locally,
and only when that draft looks weak does it transparently escalate to a bigger model —
reusing the same compression, consent, and cost machinery. Fail-closed: any error or a
*stay-local* decision keeps the local draft.

The reusable pipeline lives in the **`AuraAgents`** module (requires iOS 26 / macOS 26):

```swift
import AuraAgents

let crew = AgentCrew(store: .shared, library: .shared)
// Inject the escalation policy + consent gate; the Architect step escalates when weak.
await crew.run(topic: "Q3 security posture", policy: policy, consent: myConsentGate)
```

## Privacy & cost

- **BYOK keys** live only in the Keychain (`WhenUnlockedThisDeviceOnly`), never synced to iCloud.
- The **consent sheet** shows the *exact compressed payload*, projected cost, and the provider's retention note before any cloud send.
- **`PIIRedactor`** strips obvious secrets/PII (code-safe, high-precision).
- **`CostLedger`** records per-escalation token usage and cost; local-network targets are always **$0**.
- **`ResponseCache`** avoids paying twice for identical requests.

## What's included

| Area | Type(s) |
|---|---|
| Discovery | `LocalProviderDetector`, `LocalProviderStatus` |
| Providers | `RemoteLLMProvider`, `OpenAICompatibleProvider` (llama-server / Ollama / OpenAI / GitHub Models), `AnthropicProvider` |
| Transport | `SSELineStream`, `RemoteBackend` (an `InferenceBackend`) |
| Routing | `EscalationRouter` (R1–R7), `EscalationPolicy`, `RoutingDecision`, `HybridEscalator` |
| Compression | `ContextCompressor` + pluggable `SelfInfoScorer` |
| Privacy & cost | `ConsentGate`, `KeychainStore`, `PIIRedactor`, `CostLedger`, `ResponseCache`, `NetworkMonitor` |

See also the [CLI]({{ '/guide/cli' | relative_url }}) — `aura ask` drives this same path headlessly.
