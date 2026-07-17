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
3. **`MemoryBudgetManager`** — checks `os_proc_available_memory()` every 32 tokens during generation and stops early if RAM becomes critical
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
// AuraCore tracks available memory internally via os_proc_available_memory() (iOS)
// or 60% of physical RAM (macOS), and reduces context automatically when RAM is tight.

let availableGB = HardwareProfile.current().availableMemoryGB   // current available RAM (GB)

// Internally, the adaptive context behaves roughly like:
//   → 2048 tokens on Mac / high-RAM device
//   → 512 tokens on a 6 GB iPhone under pressure
```

If memory becomes critical mid-generation (checked every 32 tokens), the layer-streaming
backend stops early and returns the partial text generated so far — it does not throw an error.

---

## Platform Memory Budgets

### iOS

| Device RAM | App budget | Max model (streaming) | Default context |
|-----------|-----------|----------------------|----------------|
| 6 GB | ~1.5 GB | 13B (Q4_K_M streaming) | 1024 tokens |
| 8 GB | ~2.5 GB | 13B (Q4_K_M streaming) | 2048 tokens |
| 16 GB | ~6 GB | 13B full load | 4096 tokens |

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

Without the `Increased Memory Limit` entitlement, iOS caps your process at ~1.5 GB regardless of device RAM. **Always add this for apps using AuraLocal:**

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
```

With the entitlement, the limit is raised to ~3 GB on 6 GB devices and proportionally higher on larger devices.
