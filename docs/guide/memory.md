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
4. **`BackgroundLifecycle`** — when an iOS app enters the background, cancels every running local generation (text and Core ML images) and refuses new ones until it returns. Installed when a model loads. See [Background Lifecycle](#background-lifecycle-ios).
5. **Memory pressure listener** — `DispatchSource.makeMemoryPressureSource` + `UIApplication.didReceiveMemoryWarningNotification` trigger immediate eviction of non-active models
6. **`DeviceConditions`** (opt-in) — stops a generation on critical heat, and optionally in Low Power Mode or on a low battery. See [Device conditions](#device-conditions).

---

## LRU Model Cache

`ModelManager.shared` maintains a least-recently-used cache of up to `memoryBudget` instances (public, read-only), one per model and sampling setting: a model loaded at two temperature or top-p settings is two cache entries, each a separate load of the model (see [Sampling]({{ '/guide/backends' | relative_url }}#sampling-temperature-and-top-p)). The budget is computed once, when `ModelManager.shared` is first used, from the memory the process could allocate at that moment (not total device RAM): `min(4, max(1, (available − 2 GB) / 1.5 GB))`, or 1 when the OS reports unknown.

| Available memory at first use | Cache size |
|-----------|------------|
| under ~4.9 GB, or unknown | 1 instance (evicts on every switch) |
| ~4.9–6.4 GB | 2 instances |
| ~6.4–7.9 GB | 3 instances |
| ~7.9 GB or more | 4 instances (the cap) |

When you load a model that would exceed the budget, the LRU model is automatically `unload()`ed before the new one loads.
On iOS more can go: models are also evicted while the new one would not fit the app's memory limit beside them
(see [iOS](#ios)).

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

// For display and fit estimates (on iOS: the app's memory limit, see Platform Memory
// Budgets), the convenience property falls back to a rough number so the UI always
// has something to show:
let budgetGB = HardwareProfile.current().availableMemoryGB
```

If memory becomes critical mid-generation (checked every 32 tokens), the layer-streaming
backend stops early and returns the partial text generated so far — it does not throw an error.

---

## Platform Memory Budgets

### iOS

On the running iPhone or iPad the budget is the app's memory limit, read by
`HardwareProfile.memoryLimitBytes()`: `os_proc_available_memory()` plus the app's `phys_footprint`
(`task_vm_info`), capped at 74 % of physical RAM. `<os/proc.h>` defines `os_proc_available_memory()` as
the limit minus `phys_footprint`, so the sum is the limit itself. The cap is there because one 8 GB iPad
reported 8142 MiB available.

- `HardwareProfile.current().availableMemoryGB` carries this figure on iOS, so `DevicePreset.thisDevice()`
  (the Model Finder, `aura models check`), `HardwareAnalyzer.assess` and the refusal in
  `ModelManager.load` agree, and a verdict does not shrink while a model is loaded.
- Because the limit does not shrink, `ModelManager.load` takes the estimated memory of the models it keeps
  cached off it: when the new model would be Too Large beside them, the least recently used are evicted
  first, and the load is refused only when the model is Too Large on its own.
- What is left right now, `HardwareProfile.availableMemoryBytes()`, still drives the safety checks: the
  layer-streaming pressure check, the `ModelManager` cache size, the Core ML image memory gate, and the
  context window (the KV cache is allocated on top of what the app already holds).
- When iOS reports unknown (at the jetsam limit, or not an app), `memoryLimitBytes()` returns `nil`,
  `current()` shows 60 % of RAM for display, and `thisDevice()` labels it an estimate. The safety checks
  still treat unknown as pressure.
- The limit includes what the app already uses for everything else, so a model rated Marginal against it
  can exceed the limit by that amount.
- Metal and Neural Engine memory: `<mach/task_info.h>` splits memory tagged graphics and neural into
  footprint and no-footprint ledgers, so tagged GPU and ANE allocations count toward `phys_footprint`
  unless the allocator marks them no-footprint. Which Metal and ANE allocations land in which ledger is
  not documented and was not measured on a device.

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

On the running Mac, `HardwareProfile.current()` reports the memory the kernel can hand out now (free +
inactive + purgeable pages), not the Metal budget. `HardwareAnalyzer.assess`, `BackendRouter` and the
refusal in `ModelManager.load` therefore judge a model against what is free at that moment, and can rate it
lower, or stream it, while other apps hold memory. `DevicePreset.thisDevice()` (the Model Finder,
`aura models check`) uses Metal's `recommendedMaxWorkingSetSize` instead, so on a Mac the two verdicts can
differ. On iOS both use the app's memory limit.

The context window is derived per model by `HardwareAnalyzer.recommendedContextWindow(for:)` from its
KV cost and the memory left after the weights (1024–32768 tokens on macOS). For example, Llama 3.1 8B
reaches the 32768 ceiling with about 9 GB available and falls to the 1024 floor when little memory is
left after the weights.

### MLX KV window

MLX uses the same window as its KV cache size (`maxKVSize`), fixed when the backend is created. Past it,
MLX keeps the first 4 tokens and overwrites the oldest of the rest without an error; after such a
generation `AuraLocal.lastContextOverflow` says how many tokens no longer fit. Entries without KV metadata
(`kvHeads == 0`) get the fallback window: 8192 or 4096 on a Mac, 2048 or 1024 on iOS.
Architectures that build their own caches, such as Qwen 3.5 text models, Gemma 3 and 4, and LFM2, ignore
the window: their cache still grows with the conversation, and `lastContextOverflow` stays `nil`.

---

## Background Lifecycle (iOS)

When the app enters the background, `BackgroundLifecycle.shared`:

- cancels every generation running through `AuraLocal` (MLX, llama.cpp, layer streaming) or
  `CoreMLDiffusionEngine`. Each call throws `GenerationInterrupted` with `reason == .appBackgrounded`;
  `partialText` holds the text produced so far (a stream has already yielded it, then finishes with the error);
- sets `isPaused`. Until the app returns to the foreground, new generations throw the same error without starting;
- with `aggressiveMemorySaving`, evicts every model except the most recently used one.

Its observers are installed when a model loads or an image generation starts, so there is nothing to set up.
An app that is already in the background when it loads its first model reads as foreground until it returns.

```swift
do {
    reply = try await llm.chat(prompt)
} catch let stop as GenerationInterrupted {
    reply = stop.partialText   // truncated
}

// Optional: free RAM on backgrounding; evicted models reload lazily on next use.
BackgroundLifecycle.shared.aggressiveMemorySaving = true
```

When each engine actually stops:

- **MLX:** at the next token.
- **llama.cpp** (full load and streaming, constrained JSON / GBNF calls included): the call returns at once.
  LocalLLMClient has already paused the decode on `willResignActive`; per its source, that decode runs one
  more token after the app becomes active and then stops. The next chat turn on that model waits for it (with
  tools on LocalLLMClient's `LlamaClient`, for the decode to reach the end of generation); a constrained call runs
  on its own context and can overlap that token. A prompt being prefilled cannot be interrupted.
- **Core ML Stable Diffusion:** at the next denoising step. Loading the model and the final decode cannot
  be interrupted.

`BackgroundLifecycle` observes UIKit's background notifications on iOS only. On macOS and visionOS nothing is
cancelled.

---

## Device conditions

`DeviceConditions` is an opt-in guard for long generations on a hot or nearly flat device. It is assessed
before a generation starts and every `checkInterval` streamed tokens or denoising steps (default 8):

| Reading | Default | Configure |
|---------|---------|-----------|
| `ProcessInfo.thermalState` | warn at `.serious`, block at `.critical` | `warnAtThermalState`, `blockAtThermalState` (`nil` turns either off) |
| Low Power Mode | warn | `lowPowerMode: .normal / .warn / .block` |
| Battery level, unplugged | ignored | `warnBelowBatteryLevel`, `blockBelowBatteryLevel` (0...1) |

At **warn** the generation continues and `onWarning` is called once per transition into warn. At **block**
it is cancelled and the call throws `GenerationInterrupted` with `reason == .deviceConditions(assessment)` and
the text produced so far. A device already at block refuses to start.

```swift
llm.deviceGuard = DeviceConditions(onWarning: { assessment in
    print("Device is \(assessment.causes)")  // e.g. [.thermal(.serious)]
})

let images = CoreMLDiffusionEngine(deviceGuard: DeviceConditions())
```

The battery is read only when your app has turned on `UIDevice.current.isBatteryMonitoringEnabled`. That
switch is process-wide, so AuraLocal never turns it on; without it, or on a Mac, the battery thresholds are
ignored. Image generation reads the battery once when the run starts and the thermal state and Low Power
Mode at every check. `deviceGuard` is set per `AuraLocal` instance, and instances from `ModelManager` are
shared.

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
