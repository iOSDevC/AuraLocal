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
3. **`MemoryBudgetManager`** — checks available memory every 32 tokens during generation and stops early if RAM becomes critical. When the OS reports *unknown* — which on iOS is exactly what happens once the process is at or over its jetsam limit — this is treated as **pressure**, never as "plenty free".
4. **`BackgroundLifecycle`** — pauses inference when app is backgrounded (iOS only)
5. **Memory pressure listener** — `DispatchSource.makeMemoryPressureSource` + `UIApplication.didReceiveMemoryWarningNotification` trigger immediate eviction of non-active models

---

## LRU Model Cache

`ModelManager.shared` maintains a least-recently-used cache. Cache size adapts to device RAM:

| Device RAM | Cache size | Notes |
|-----------|------------|-------|
| < 4 GB | 1 model | Evicts on every switch |
| 4–6 GB | 1–2 models | iPhone 15, base iPad |
| 8–16 GB | 2–4 models | iPad Pro, M-series Mac |
| 32+ GB | 4+ models | Mac Studio / Pro |

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

During llama.cpp inference, `MemoryBudgetManager` monitors RAM every 32 tokens. It is
**internal** to AuraCore — there is no public singleton to call and no `availableMemoryGB`
property on it. To read available RAM from your own code, use the public `HardwareProfile`:

```swift
// One source of truth. Asks the OS — os_proc_available_memory() on iOS, kernel VM
// stats on macOS — and returns nil when it genuinely cannot tell.
if let bytes = HardwareProfile.availableMemoryBytes() {
    print("\(Double(bytes) / 1_073_741_824) GB available")
} else {
    // iOS reports this exactly when the process is AT or OVER its jetsam limit.
    // AuraCore treats it as pressure: allocation refused, context shrunk to 512.
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
| ~1.5 GB (no entitlement) | up to ~4 GB of weights (7B Q4) | **8B and larger** | the strongest argument for adding both entitlements |
| ~3 GB (6 GB device, entitled) | up to ~7.5 GB (8B, 9B, 12B Q4) | 32B+ | |
| ~4.5 GB (8 GB device, entitled) | same, with more headroom | 32B+ | 3B-class models load fully instead of streaming |

Computed from the shipping catalog with the current fit rules — no 13B model is in the catalog, and
32B never fits an iPhone at any budget.

### macOS

| Total RAM | Max model (full load) | GPU layers | Context |
|-----------|----------------------|-----------|---------|
| 8 GB | 7B (tight) | All | 2048 |
| 16 GB | 8B (comfortable) | All | 8192 |
| 32 GB | 13B–14B | All | 8192 |
| 48 GB | 32B | All | 8192 |
| 80+ GB | 70B | All | 8192 |

---

## Background Lifecycle (iOS)

When an iOS app is backgrounded with an active Metal GPU session, the system may kill it for having locked GPU memory. `BackgroundLifecycle` prevents this:

```swift
// Automatic — no setup needed
// Generation is paused when isPaused == true

// Opt into more aggressive saving:
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
> `BackgroundLifecycle.shared.aggressiveMemorySaving = true` so models are evicted on background.
