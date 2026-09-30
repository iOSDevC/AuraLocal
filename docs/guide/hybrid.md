---
layout: docs
title: Hybrid Inference
parent: Guide
nav_order: 2.5
description: "Local-first inference with optional, consent-gated escalation to a bigger local or cloud model — llama-server/Ollama, Anthropic, or OpenAI — with token-saving compression and cost accounting."
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
on the LAN, or a cloud provider (Anthropic, OpenAI) — only when the local
model isn't enough.

{: .note }
> Escalation through `routeAndEscalate` is **off by default** (`EscalationPolicy.off`):
> nothing leaves the device until you set a policy, and cloud sends go through your
> `ConsentGate` first. `escalate(to:)` is the low-level call. It checks neither a policy
> nor consent and sends immediately, so gate it yourself.
> `AuraLocal.stream()` drives the **local** backends; escalation runs through
> `HybridEscalator` and an internal remote `InferenceBackend`, which shares the *same
> streaming contract* — so the pipeline is *mixed*, not cloud.

The design goal is to **cut the tokens sent to the remote**: stay local by default, and
when you do escalate, compress the context, optionally redact obvious secrets, and cache repeats.

## Token savings — the headline

| Technique | Effect |
|---|---|
| Local-first routing | Most turns never leave the device — **$0**, no tokens sent |
| Selective-context compression | Trims the context toward the remote's budget — **fewer tokens, input-dependent** (no fixed ratio; a context that already fits is passed through unchanged) |
| Response cache | A repeated request (same provider, model and payload) returns from memory — **$0** |
| PII redaction | Strips obvious secrets before the payload is sent (opt-in: `escalate(to:…, redactPII: true)`) |
| Cost ledger | Per-escalation token + dollar accounting (list prices for Anthropic and OpenAI only) |

Every escalation returns what a receipt needs: `HybridEscalator.Result` carries `providerName`,
`usage` (input/output tokens), `compression` (`originalTokens`, `compressedTokens`, `factor`),
`redactedPIICount` and `fromCache`. The library prints nothing; the Example app's Hybrid tab
formats them as *"via llama-server (local) · &lt;model&gt; · 800 in / 240 out · compressed 6000→1500 (4.0×)"*.

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

Ollama also lists **cloud models** (`deepseek-v4-pro:cloud`, `gpt-oss:120b-cloud`) that it forwards
to `ollama.com`. They carry `remoteHost` and `runsLocally == false`. `bestLocalTarget()` and
`AskTargetResolver.resolve` without a `baseURL` skip them, so the target they pick never forwards
the prompt to Ollama's cloud. A `baseURL` is classified by its host only: a local URL serving a
`:cloud` model named with `model:` still counts as `.localNetwork`.

## Escalate a request

`HybridEscalator` compresses the context toward the remote's budget, checks the response
cache, streams the answer, and records cost. PII redaction runs only when you call
`escalate(to:…, redactPII: true)` directly; `routeAndEscalate` (and so `AgentCrew`) never redacts.

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
} // nil → the router kept it local; a declined offer throws AuraError.escalationDeclined
```

The router (`EscalationRouter`) is a pure decision function (rules R1–R7): it stays local
unless there's a size overflow or a low-confidence local answer (a sensitive domain only
raises the bar for the latter).

- Without a `localAnswer`, only size overflow can fire: the estimated prompt exceeds 90 % of
  `localContextWindow` (default 8192).
- With one, size is not checked. The answer counts as low-confidence when it is under 40
  characters, contains a refusal phrase (English such as "I can't" or "as an AI"; Spanish such
  as "no sé", "no puedo" or "como modelo de lenguaje" — the unaccented "no se" does not count),
  or, for the `.security` / `.medicine` domains, is under 120 characters.
- Cloud targets are always offered to your `ConsentGate`, never escalated silently; only a LAN
  target under `.autoWithConsentMemory` escalates without asking.
- `costCapUSDPerSession` is compared with the projected cost of the current request alone
  (estimated prompt tokens + `maxTokens` at list price); spend already recorded in `CostLedger`
  is not counted. Over the cap, the router still offers, with the reason `.costCapped`.
- Only the first candidate is tried: the LAN box if one is running, otherwise the first cloud
  key found (Anthropic, then OpenAI), or the first of the `targets:` you pass. An error from it, such as HTTP 429, is thrown to the caller;
  there is no fall-through to the next target.

## Cloud targets (BYOK)

`HybridEscalator.cloudTargets(allowCloud:)` builds cloud targets from Keychain keys:
`cloud.anthropic` (`AnthropicProvider`, default model `RemoteTarget.defaultAnthropicModel`,
`claude-sonnet-4-5`) and `cloud.openai` (OpenAI, `RemoteTarget.defaultOpenAIModel`, `gpt-4o`).
Keys are stored in the Keychain, never in source, files, or logs. Save a key once (e.g. from a
settings screen): `try KeychainStore.save(key, for: "cloud.openai")`.
Works from iOS, macOS, and visionOS.

{: .note }
> GitHub retired GitHub Models on 2026-07-30. `OpenAICompatibleProvider.gitHubModels(apiKey:)` and
> the `cloudTargets` overload taking `gitHubModelsModel:` are `unavailable` (a compile error that
> names the replacements), and `cloudTargets` no longer reads the `cloud.github-models` Keychain
> account. A key already saved there stays until you delete it with
> `KeychainStore.delete(for: "cloud.github-models")` (the Example app's Hybrid settings offer a button).

## Pick one target (`AskTargetResolver`)

For a one-shot ask, `AskTargetResolver.resolve` turns a provider choice into a single
`RemoteTarget`. It is what `aura ask` uses, and a pure function: you pass the environment, a
Keychain reader and the detected local providers, so it does no I/O of its own.

```swift
import AuraCore

let target = try AskTargetResolver.resolve(
    .auto,                                        // .auto / .local / .openAI / .anthropic
    environment: ProcessInfo.processInfo.environment,
    readKey: KeychainStore.read(for:),            // read only for .openAI / .anthropic
    localProviders: await LocalProviderDetector.detectAll())
let result = try await HybridEscalator().escalate(
    to: target, context: "", question: prompt, redactPII: !target.isLocalNetwork)
```

Without a `baseURL:`, `.auto` and `.local` return your llama-server, else Ollama, and never a
cloud target; they throw `AskTargetResolver.Failure.noLocalProvider` when neither runs.
`.openAI` / `.anthropic` take the key from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, else the Keychain, and throw
`.missingAPIKey` without one. `baseURL:` (with `model:`) targets any OpenAI-compatible server;
a loopback or private-network host gets the origin `.localNetwork(.llamaServer)`, any other
host `.cloud`. Each `Failure` has an `errorDescription` that says what to do next.

## Per-step escalation (agent orchestration)

Escalation isn't only for the top-level answer. In `AgentCrew`, the Architect step's local
draft goes through `routeAndEscalate(localAnswer:)` and escalates only when it looks weak;
the Extractor, Reviewer and Reporter steps always stay local. It reuses the same compression,
consent, and cost machinery. Fail-closed: any error or a *stay-local* decision keeps the
local draft.

The reusable pipeline lives in the **`AuraAgents`** module (requires iOS 26 / macOS 26 and
the Xcode 26 SDK):

```swift
import AuraAgents

let crew = AgentCrew(store: .shared, library: .shared)
// Inject the escalation policy + consent gate; the Architect step escalates when weak.
await crew.run(topic: "Q3 security posture", policy: policy, consent: myConsentGate)
```

## Privacy & cost

- **BYOK keys** live only in the Keychain (`WhenUnlockedThisDeviceOnly`), never synced to iCloud.
- Your **`ConsentGate`** is called before an offered escalation with the `target` (including
  `target.provider.retentionNote`), the compressed preview (`CompressionResult`) and the projected
  cost. AuraLocal ships no consent UI; the Example app's `UIConsentGate` (Hybrid settings) shows one.
  The preview is compressed with `policy.keepRatio` while the request sent uses 0.5, so the previewed
  context equals the sent context only at the default `keepRatio` of 0.5; the request also carries
  the question and system prompt.
- **`PIIRedactor`** strips obvious secrets/PII (code-safe, high-precision) from the context and
  the question, only on `escalate(to:…, redactPII: true)`.
- **`CostLedger`** records per-escalation token usage and cost. Only `cloud.anthropic` ($3 in /
  $15 out per 1M tokens) and `cloud.openai` ($2.50 / $10) have prices; local-network targets and
  every other provider id (any custom `OpenAICompatibleProvider`, including an
  `AskTargetResolver` base URL) are recorded, and projected, at $0.
- **`ResponseCache`** avoids paying twice for a repeated request. It is keyed on provider id,
  model id and the user payload (context + question), not on the system prompt or `maxTokens`,
  and is held in memory (64 entries, FIFO, cleared on relaunch).

## What's included

| Area | Type(s) |
|---|---|
| Discovery | `LocalProviderDetector`, `LocalProviderStatus` |
| Providers | `RemoteLLMProvider`, `OpenAICompatibleProvider` (llama-server / Ollama / OpenAI), `AnthropicProvider` |
| Transport | internal (`RemoteBackend` adapts a `RemoteLLMProvider` to `InferenceBackend`; not public API) |
| Routing | `EscalationRouter` (R1–R7), `EscalationPolicy`, `RoutingDecision`, `HybridEscalator`, `AskTargetResolver` |
| Compression | `ContextCompressor` + pluggable `SelfInfoScorer` |
| Privacy & cost | `ConsentGate`, `KeychainStore`, `PIIRedactor`, `CostLedger`, `ResponseCache`, `NetworkMonitor` |

See also the [CLI]({{ '/guide/cli' | relative_url }}): `aura ask` resolves its target with
`AskTargetResolver` and sends it through `HybridEscalator.escalate(to:)`.
