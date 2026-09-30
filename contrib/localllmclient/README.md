# Upstream patches for LocalLLMClient

Three patches against [tattn/LocalLLMClient](https://github.com/tattn/LocalLLMClient), the package
that gives AuraLocal its GGUF/llama.cpp backend on **every** Apple platform (the xcframework ships
`ios-arm64`, `ios-simulator`, `macos`, `xros`, `tvos` slices — this is not a Mac-only dependency).

**Status: prepared, verified, NOT submitted.** They live here so they don't get lost, and so the
work is reviewable before anything is pushed anywhere public.

| | `0001`, `0002` | `0003` |
|---|---|---|
| Base | `main @ d92cb1f` (2026-08-25) | `main @ b68f712` (2026-08-26) |
| Size | 2 commits · 292 insertions · **0 deletions** | 1 commit · 52 insertions · **0 deletions** |
| Verified | built against upstream with llama.cpp from source; 7 tests in 4 suites pass, including upstream's own pre-existing `ContextTests` | built the same way; all 45 Llama tests pass with both upstream test models (SmolVLM-256M as on CI, gemma-4-E4B locally); its 3 new tests fail without the fix |

All three apply cleanly, in order, on `main @ b68f712`.

## Why these exist

llama.cpp supports all three capabilities `0001` and `0002` add. AuraLocal cannot reach any of
them, because `Context.context` — the `llama_context` pointer — is declared `package` in
LocalLLMClient, the `llama` C module is imported `@_implementationOnly`, and no product vends the
binary target. There is no way in from outside the package. Hence: patch upstream, not us.

`0003` is a bug fix, not a capability, and just as unreachable: the sampler chain is an internal
`Context` property, so from outside the only workaround is a new client per generation.

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
LocalLLMClient PR #97 (open since 2026-04-28) also exposes Flash Attention and the KV cache type (as
`flashAttention: Bool` defaulting to `true`, and separate K/V types); reconcile with it before submitting.

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

### `0003` — reset the sampler chain for each generation

`Context` builds one `llama_sampler` chain per client and never resets it, so state leaks from one
generation into the next on the same client:

- **Grammar** (`responseFormat: .json` or `.grammar`): once the first answer completes the grammar,
  it only allows end-of-generation or the grammar's trailing whitespace, so every later answer on
  that client is `""` (or whitespace-only for `.json`).
- **Repetition penalties** count the previous answer's tokens.
- **RNG**: a fixed `seed` reproduces only the client's first generation.

The fix is one `llama_sampler_reset` on the chain in `LlamaClient.textStream(from:)`, which every
generation path goes through (`generateText`, `responseStream`, tool calling, `resumeStream`,
`LLMSession`). That is llama-server's behaviour: fresh sampler state per request. Nothing resets
within a generation, and the prompt cache / KV reuse is untouched.

Reproduced on one client with Qwen2.5-0.5B-Instruct Q4_K_M, two different prompts:

| | 1st answer | 2nd, before | 2nd, after |
|---|---|---|---|
| `.json` | `{"answer": 3}` | `""` | `{"capital": "Paris"}` |
| `.grammar` (`yes`/`no` + `.`) | `yes.` | `""` | `yes.` |
| `LLMSession.streamResponse` + `.json` | `{"answer": 3}` | `""` | `{"answer": 5}` |
| tokens still penalised before the 2nd answer starts | | 20 | 0 |
| same prompt twice, `seed: 42`: identical? | | no | yes |

Answer quality is the 0.5B model's; what matters is that the second answer exists and is valid.

What it unlocks: a constrained client can be kept and reused. Until a release carries the reset,
AuraLocal has to build a dedicated client (model load plus context allocation) for every
constrained call. Unconstrained sessions also stop carrying penalty history between turns.

**Behaviour change to flag when submitting:** with a fixed `Parameter.seed`, the same input on
the same client now gives the same output on every call (before, repeated calls drew different
samples). With the default seed (`nil`) each generation gets a new random seed, as before in effect.

Found while doing this, **not** fixed by `0003` (candidates for separate patches):

- **KV contamination between unrelated chats.** When no cached prefix matches the new `.chat`
  input, `removeCachedChunks` returns without clearing the KV cache, so the new prompt is decoded
  *after* the previous conversation. Measured: position 57 on a used client versus 26 on a fresh
  one for the same prompt. Only multi-turn chats whose history matches the cache byte for byte
  reuse it correctly. `.plain` input clears first and is not affected.
- **Invalid GBNF crashes.** `llama_sampler_init_grammar` returns `NULL`, which is chained anyway and
  dereferenced on the first token. `0003` neither causes nor fixes it: `llama_sampler_reset` skips
  `NULL`.
- **Generation outlives its consumer.** `LlamaClient.responseStream` (the tool-calling path) starts a
  `Task` with no `onTermination`, so a consumer that stops early leaves it decoding to
  end-of-generation; measured, the KV position kept growing (314 → 771 over 4 s) after the consumer
  broke. And prefill is not cancellable: `textStream(from:)` decodes the whole prompt before it
  returns, so a cancelled `LLMSession.streamResponse` keeps prefilling in its detached task. Either
  way the next generation on the same context overlaps a running decode. A candidate patch:
  `onTermination { task.cancel() }` in `responseStream`, plus a cancellation check between prefill
  batches.
- **Sampler order** (read, not measured): penalties run after temperature and top-k/top-p, so at
  temperature 0 they do nothing, and above 0 they only touch the survivors. llama.cpp's default runs
  them first. The grammar is also applied twice per token (on its own, then again inside the chain).

No upstream issue or PR covers any of this (all 68 PR and 33 issue titles, plus a full-text issue
search, checked 2026-09-30).

## Applying them

On a fork of LocalLLMClient:

```bash
git checkout -b feature/session-state-persistence
git am /path/to/AuraLocal/contrib/localllmclient/0001-*.patch
git am /path/to/AuraLocal/contrib/localllmclient/0002-*.patch
git am /path/to/AuraLocal/contrib/localllmclient/0003-*.patch
```

`0003` does not depend on the other two and applies on its own as well.

Building upstream from source needs its submodule (llama.cpp):

```bash
git submodule update --init --recursive --depth 1
```

## Do not point AuraLocal at a fork

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
fix landed upstream in llama.cpp PR #22673 (MTP support, which flags the NextN layers non-recurrent),
first shipped in **b9180** (2026-05-16); PR #24031 (b9488, 2026-06-03) also reads
`qwen35.attention.recurrent_layers` from the GGUF.

Until LocalLLMClient bumps `llamaVersion` to b9180 or later, that whole family is MLX-only here — see
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
