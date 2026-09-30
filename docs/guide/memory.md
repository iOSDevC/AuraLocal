---
layout: docs
title: Memory Management
parent: Guide
nav_order: 4
description: "How AuraLocal manages RAM on iOS and macOS — LRU cache, memory pressure, OOM prevention."
---

# Memory Management
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

On-device LLMs can consume significant RAM. AuraLocal has multiple layers of protection against out-of-memory crashes (jetsam on iOS):

1. **`HardwareAnalyzer`** — flags models that won't fit (pure analysis). Enforcement lives in `ModelManager.load`, which refuses a `.tooLarge` model up front by throwing `AuraError.modelTooLarge` **before** downloading it or risking a jetsam kill mid-load.
2. **`ModelManager` LRU cache** — evicts the least-recently-used model when RAM is needed
3. **`MemoryBudgetManager`** (layer-streaming backend only) — checks available memory every 32 tokens during generation and stops early if RAM becomes critical. When the OS reports *unknown* — which on iOS is exactly what happens once the process is at or over its jetsam limit — this is treated as **pressure**, never as "plenty free". Full-load llama.cpp and MLX do not check memory mid-generation.
4. **`BackgroundLifecycle`** — sets `isPaused` when the app is backgrounded (iOS); informational only, nothing in AuraCore reads it. GGUF generation is paused by LocalLLMClient while the app is inactive; MLX is not. See [Background Lifecycle](#background-lifecycle-ios).
5. **Memory pressure listener** — `DispatchSource.makeMemoryPressureSource` + `UIApplication.didReceiveMemoryWarningNotification` trigger immediate eviction of non-active models

---

## LRU Model Cache

`ModelManager.shared` maintains a least-recently-used cache of up to `memoryBudget` models (public, read-only). The budget is computed once, when `ModelManager.shared` is first used, from the memory the process could allocate at that moment (not total device RAM): `min(4, max(1, (available − 2 GB) / 1.5 GB))`, or 1 when the OS reports unknown.

| Available memory at first use | Cache size |
|-----------|------------|
| under ~4.9 GB, or unknown | 1 model (evicts on every switch) |
| ~4.9–6.4 GB | 2 models |
| ~6.4–7.9 GB | 3 models |
| ~7.9 GB or more | 4 models (the cap) |

When you load a model that would exceed the budget, the LRU model is automatically `unload()`ed before the new one loads.

---

## Memory Pressure Response

When iOS sends a memory warning, `ModelManager` immediately evicts all models **except the most recently used**:

```swift
// This happens automatically — you don't need to call it
// But you can manually evict:
ModelManager.shared.evict(.qwen3_1_7b)
ModelManager.shared.evictAll()
```

---

## Per-Generation Budget (MemoryBudgetManager)

During **layer-streaming** inference (`LayerStreamingBackend` only; full-load llama.cpp and MLX do not check), `MemoryBudgetManager` monitors RAM every 32 tokens. It is
**internal** to AuraCore — there is no public singleton to call and no `availableMemoryGB`
property on it. To read available RAM from your own code, use the public `HardwareProfile`:

```swift
// One source of truth. Asks the OS — os_proc_available_memory() on iOS, kernel VM
// stats on macOS — and returns nil when it genuinely cannot tell.
if let bytes = HardwareProfile.availableMemoryBytes() {
    print("\(Double(bytes) / 1_073_741_824) GB available")
} else {
    // iOS reports this exactly when the process is AT or OVER its jetsam limit.
    // Layer-streaming treats it as pressure: context 512, replies capped at 256 tokens.
    // Do not substitute a fraction-of-RAM guess here — that is the bug this replaced.
}

// For display and fit estimates, the convenience property falls back to a rough
// number so the UI always has something to show:
let availableGB = HardwareProfile.current().availableMemoryGB
```

If memory becomes critical mid-generation (checked every 32 tokens), the layer-streaming
backend stops early and returns the partial text generated so far — it does not throw an error.

---

## Platform Memory Budgets

### iOS

What actually fits depends far more on the **entitlements** than on the device, because they set
the app's budget. Streaming is offered only while RAM can still cache a third of the weights —
below that llama.cpp re-reads them from storage every token and decode collapses.

| App budget | Streams | Refused | Notes |
|-----------|---------|---------|-------|
| ~1.5 GB (no entitlement) | up to ~4.5 GB of weights (7B Q4) | **8B and larger** | the strongest argument for adding both entitlements |
| ~3 GB (6 GB device, entitled) | up to ~9 GB of weights (8B, 9B, 12B, Phi-3 Medium 14B Q4) | 32B+ | 1.2B–3B GGUFs load fully |
| ~4.5 GB (8 GB device, entitled) | up to ~13.5 GB (same set) | 32B+ | Mistral 7B also loads fully |

Computed from the shipping catalog with the current fit rules (streaming needs RAM for a third of
the weights) — no 13B model is in the catalog. 32B (18 GB of weights) is refused up to a ~6 GB
budget; at the ~6.4 GB `iphone-17-pro` estimate it rates Streaming.

### macOS

| Mac preset | Largest catalog GGUF that loads fully | GPU layers |
|-----------|----------------------|-----------|
| 16 GB | 14B (Phi-3 Medium) | All |
| 32 GB | 32B (marginal) | All |
| 64 GB | 70B (marginal) | All |

The largest model `HardwareAnalyzer.assess` rates as a full load at each [device preset]({{ '/guide/models' | relative_url }}#device-presets):
the Metal budget is estimated at ≈2/3 of RAM up to 36 GB and ≈3/4 above, and the 32 GB row uses a measured 20 GB.
One size down, each preset streams the larger model instead (32B on 16 GB, 70B on 32 GB).

The context window is derived per model by `HardwareAnalyzer.recommendedContextWindow(for:)` from its
KV cost and the memory left after the weights (1024–32768 tokens on macOS). For example, Llama 3.1 8B
reaches the 32768 ceiling with about 9 GB available and falls to the 1024 floor when little memory is
left after the weights.

---

## Background Lifecycle (iOS)

{: .warning }
> `BackgroundLifecycle` does not pause inference. It sets `isPaused` on backgrounding, but nothing in AuraCore reads it, and it is not automatic: its observers exist only after your code first touches `BackgroundLifecycle.shared`. GGUF generation is paused anyway by LocalLLMClient (llama.cpp waits between tokens from `willResignActive` until `didBecomeActive`); MLX generation keeps running, so cancel it yourself when the app backgrounds.

What it can do is free memory on backgrounding:

```swift
// Touching .shared registers the observers.
BackgroundLifecycle.shared.aggressiveMemorySaving = true
// When set, entering the background calls ModelManager.evictAllButMostRecent(),
// freeing every loaded model except the active one. It reloads lazily on next use.
```

---

## Entitlement

Two entitlements matter, and AuraLocal needs **both**:

- **Increased Memory Limit** — raises the resident (physical RAM) limit before jetsam kills you.
- **Extended Virtual Addressing** — raises the *address-space* limit. llama.cpp loads GGUF weights
  with `use_mmap = true`, and layer-streaming's whole premise is mapping a file far larger than RAM.
  Without this, iOS caps the address space and **terminates the app** when a large mapping hits it,
  so mmap buys much less than it should.

**Always add both for apps using AuraLocal:**

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
```

With the entitlement, the limit is raised to ~3 GB on 6 GB devices and proportionally higher on larger devices.

> **Trade-off worth knowing:** raising the resident limit also makes the app a *bigger* jetsam
> target — iOS preferentially kills high-memory apps once they are backgrounded. Pair it with
> `BackgroundLifecycle.shared.aggressiveMemorySaving = true` so every model except the most recently
> used one is evicted on background.
