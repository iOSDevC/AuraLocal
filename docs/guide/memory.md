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

1. **`HardwareAnalyzer`** — refuses to load a model that won't fit before attempting it
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

During llama.cpp inference, `MemoryBudgetManager` monitors RAM every 32 tokens:

```swift
// Available memory queried via os_proc_available_memory() on iOS
// On macOS: 60% of physical RAM

let manager = MemoryBudgetManager.shared
print(manager.isUnderPressure)        // true when available < safety threshold
print(manager.availableMemoryGB)      // current available RAM

// Adaptive context — reduces context if RAM is low
let ctx = manager.recommendedContextLength(baseContext: 2048)
// → 2048 on Mac / high-RAM device
// → 512 on 6 GB iPhone under pressure
```

If `isUnderPressure` becomes `true` mid-generation, `LayerStreamingBackend` stops and returns what it has so far with an `AuraError.memoryPressure` error.

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

// For more aggressive saving:
BackgroundLifecycle.shared.aggressiveMemorySaving = true
// → unloads all models when app is backgrounded
// → re-loads on foreground when user resumes
```

---

## Entitlement

Without the `Increased Memory Limit` entitlement, iOS caps your process at ~1.5 GB regardless of device RAM. **Always add this for apps using AuraLocal:**

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
```

With the entitlement, the limit is raised to ~3 GB on 6 GB devices and proportionally higher on larger devices.
