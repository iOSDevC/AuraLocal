# Upstream patches for LocalLLMClient

Two patches against [tattn/LocalLLMClient](https://github.com/tattn/LocalLLMClient), the package
that gives AuraLocal its GGUF/llama.cpp backend on **every** Apple platform (the xcframework ships
`ios-arm64`, `ios-simulator`, `macos`, `xros`, `tvos` slices — this is not a Mac-only dependency).

**Status: prepared, verified, NOT submitted.** They live here so they don't get lost, and so the
work is reviewable before anything is pushed anywhere public.

| | |
|---|---|
| Base | `main @ d92cb1f` (2026-08-26) |
| Size | 2 commits · 292 insertions · **0 deletions** |
| Verified | built against upstream with llama.cpp from source; 7 tests in 4 suites pass, including upstream's own pre-existing `ContextTests` |

## Why these exist

llama.cpp supports all three capabilities below. AuraLocal cannot reach any of them, because
`Context.context` — the `llama_context` pointer — is declared `package` in LocalLLMClient, the
`llama` C module is imported `@_implementationOnly`, and no product vends the binary target. There
is no way in from outside the package. Hence: patch upstream, not us.

## What each one unlocks

### `0001` — session KV cache persistence

Adds `LlamaClient.saveState(to:)` / `loadState(from:)` over `llama_state_save_file` /
`llama_state_load_file`.

Long conversations spend most of their time re-prefilling history that has not changed. This is
worst on **iOS**, where the OS suspends and kills apps constantly, so every relaunch pays the whole
prefill again.

The subtlety, and why a naive version would corrupt state: `Context` keeps its own prompt-cache
bookkeeping (`promptCaches`) that drives prefix reuse. Restoring the KV cache while that array
stayed empty leaves the context believing nothing is cached even though positions are occupied — the
next generation would *append* instead of reuse. So `saveState` also writes a sidecar (`<url>.meta`),
and a state file whose sidecar is missing is refused rather than loaded into an inconsistent context.
Multimodal sessions throw instead of silently dropping image chunks.

### `0002` — Flash Attention + KV cache quantization

Adds `Parameter.flashAttention` (`.auto`/`.enabled`/`.disabled`) and `Parameter.kvCacheType`
(`.f16`/`.q8_0`/`.q4_0`), plus `Context.stateSizeBytes`.

Defaults (`.auto`, `.f16`) reproduce llama.cpp's own, so existing callers see no behaviour change.

Measured with upstream's test model at 2048 context, cache filled before measuring:

| KV cache | Size | |
|---|---|---|
| `f16` | 11,531,586 B | |
| `q8_0` | 6,128,706 B | **−46.8 %** |

On memory-limited devices the KV cache — not the weights — is usually what runs out first, so
halving it is the single largest memory win available to the iOS path.

> Measure a **filled** cache. `llama_state_get_size()` only serialises the *used* portion, so an
> empty context reports a ~26 byte header whatever the element type is. The first version of that
> test passed vacuously for exactly this reason.

The two settings are coupled, and the API docs say so: quantizing the KV cache **without** Flash
Attention forces a dequantize on every attention computation and can be slower than not quantizing.

## Applying them

On a fork of LocalLLMClient:

```bash
git checkout -b feature/session-state-persistence
git am /path/to/AuraLocal/contrib/localllmclient/0001-*.patch
git am /path/to/AuraLocal/contrib/localllmclient/0002-*.patch
```

Building upstream from source needs its submodule (llama.cpp):

```bash
git submodule update --init --recursive --depth 1
```

## ⚠️ Do not point AuraLocal at a fork

Tempting, but it re-introduces exactly the failure fixed in `4dfcbb6`: SwiftPM honours only the
**root** package's `Package.resolved`, so a fragile dependency graph that resolves here breaks for
every downstream integrator. Wait for these to land upstream in a tagged release.

Nothing in AuraLocal depends on these patches. The iOS work that shipped without them —
jetsam fail-safe, the `extended-virtual-addressing` entitlement, prefill batch sizing, and the
context-window anchor — is already in the repo and stands on its own.
