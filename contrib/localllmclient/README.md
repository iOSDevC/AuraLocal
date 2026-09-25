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

## Separately blocked on upstream: the vendored llama.cpp build

`Package.swift` in LocalLLMClient hardcodes `let llamaVersion = "b8851"` (2026-04-19) and fetches the
matching xcframework by URL + checksum, so the build number is not something a dependent package can
override. Tag `0.5.0` and `main` both sit on b8851 as of 2026-09-25.

That build cannot load **`qwen35`** GGUFs that carry an MTP block (Qwen3.8-27B and its derivatives,
e.g. `ukisai/Swift-1.5-Qwen3.8-27B-GGUF`). b8851 registers the arch and its pre-tokenizer, so it
looks supported, but it derives layer recurrence arithmetically instead of reading
`qwen35.attention.recurrent_layers`: with `block_count` 65 the MTP block is misclassified as
recurrent, and the load throws on the `blk.64.ssm_*` tensors the file does not contain. The loader
fix landed upstream in llama.cpp PR #24025, first shipped in **b9495** (2026-06-03).

Until LocalLLMClient bumps `llamaVersion` to b9495 or later, that whole family is MLX-only here — see
`swift15_qwen38_27b_mlx` in `models.json`. Standalone llama.cpp and LM Studio at a current build run
the GGUFs today; AuraLocal cannot, and forking to fix it is ruled out above.

### MLX conversions of that family: check the weight prefix first

`mlx-swift-lm`'s `qwen3_5` expects the vision tower under `vision_tower.*` and drops
`vision_tower`/`model.visual` in `sanitize`. Conversions that emit a top-level **`visual.*`** prefix
fall through every rename branch, get `language_model.` prepended, and fail `update(parameters:)` with
hundreds of unhandled keys. Verified against the weight maps: `mlx-community/Qwen3.5-27B-4bit` and
`orcarouter/Qwen3.8-27B-Uncensored-MLX` use `vision_tower.*` and load; `ukisai/Swift-1.5-4bit-MLX` and
`ukisai/Swift-1.5-5bit-MLX` use `visual.*` and do not, which is why the catalog carries the
`-TextOnly` conversion (`language_model.*` only) instead.
