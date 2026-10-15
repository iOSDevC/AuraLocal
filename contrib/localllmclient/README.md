# Upstream patches for LocalLLMClient

Four patches against [tattn/LocalLLMClient](https://github.com/tattn/LocalLLMClient), the package
that gives AuraLocal its GGUF/llama.cpp backend on **every** Apple platform (the xcframework ships
`ios-arm64`, `ios-simulator`, `macos`, `xros`, `tvos` slices — this is not a Mac-only dependency).

**Status: prepared, verified, NOT submitted.** They live here so they don't get lost, and so the
work is reviewable before anything is pushed anywhere public.

| | `0001`, `0002` | `0003` | `0004` |
|---|---|---|---|
| Base | `main @ d92cb1f` (2026-08-25) | `main @ b68f712` (2026-08-26) | `main @ b68f712` (llama.cpp submodule `e365e658`) |
| Size | 2 commits · 292 insertions · **0 deletions** | 1 commit · 52 insertions · **0 deletions** | 1 commit · 184 insertions · 20 deletions (`Context.swift` + a new `PromptCacheTests.swift`) |
| Verified | built against upstream with llama.cpp from source; 7 tests in 4 suites pass, including upstream's own pre-existing `ContextTests` | built the same way; all 45 Llama tests pass with both upstream test models (SmolVLM-256M as on CI, gemma-4-E4B locally); its 3 new tests fail without the fix | upstream's CI setup (`xcodebuild`, `TEST_RUNNER_GITHUB_ACTIONS_TEST=Llama`): 49/49 in Debug (42 + 7 new); 47/49 in Release, where the 2 failures are image tests that fail the same way on unpatched `main`; without the fix 5 of its 7 new tests fail in Debug (the hybrid-model one crashes on the `assert`) and 6 of 7 in Release |

All four apply cleanly, in order, on `main @ b68f712`, and `0004` also applies on its own (re-checked with
`git apply --check` on 2026-10-14). **`0004` does not apply to `edc39ef` (0.5.0), the revision AuraLocal
pins:** its `clear()` hunk sits on `discardPendingTokens()`, which upstream added after 0.5.0 in #101
(`d92cb1f`).

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

### `0004` — keep the KV cache in sync with the prompt cache

`Context` reuses the KV cache across generations by matching the new prompt against its prompt cache
(`promptCaches`). Several paths let the two drift apart, so a prompt is decoded on top of tokens that are
no longer part of the conversation:

- **No cached prefix matches** (an unrelated chat, or a history stored differently from how it was
  generated): `removeCachedChunks` returned without touching the KV cache, so the whole prompt was decoded
  after the previous conversation. It now clears the KV cache and the prompt cache and prefills in full.
- **The cached prefix matches but the KV cache holds a longer tail** (a generation stopped early):
  `llama_memory_seq_rm` was the argument of `assert()`, so release builds kept the stale tail. It now runs
  unconditionally; when it fails (recurrent and hybrid memories cannot drop a partial range, which crashed
  debug builds on the `assert`) the context is cleared and the prompt prefilled in full.
- **`clear()`** emptied the KV cache but kept the prompt cache, so the next prompt could "match" tokens
  that were gone. It now empties both. Behaviour change to flag when submitting: after `clear()` nothing
  is reused.
- **`lastCacheIndex`** stops at the first chunk that diverges (it could report a later image as a match
  after a different text chunk) and puts the uncached rest of a text chunk before the chunks after it.

Reproduced with Qwen2.5-0.5B-Instruct Q4_K_M, the flattened-transcript pattern AuraLocal used, at a
2048 context: KV position after turn 10 was 1586 before the fix and 270 (fresh prompt + reply) after; at a
1024 context turn 8 failed with `context size exceeded[1024 < 1082]` before the fix and ran clean after.
A stopped generation left 19 stale positions in Release before the fix and none after.

What it unlocks: `LLMSession` and `.chat` input become safe for multi-turn chat, unrelated prompts and cut
replies on one client, in release builds too. That is what AuraLocal would need to go back to
`LLMSession` and reuse the cached prefix across turns. Two more things stand in the way of that reuse: the
pin has to move past 0.5.0 (to `d92cb1f` or later, which also makes `clear()` drop tokens a cut generation
left queued in the batch), and `JinjaChatTemplateRenderer` sets `lstripBlocks`/`trimBlocks` on the
`Environment`, which swift-jinja ignores (it reads them only from `Template(_:with:)`), so Qwen-style
templates end the generation prompt with an extra newline and the prefix never matches (candidate
`0005`; fixing it changes the rendered prompt for every model). Until all three land, AuraLocal's GGUF
backends own the context, clear the KV cache and batch before every turn, and prefill the whole history.

Found while doing this, **not** fixed by `0003` or `0004` (candidates for separate patches):
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
  batches. AuraLocal's tool turns wait for that decode before the next turn, which is why a cut tool turn
  delays the one after it.
- **With `extraEOSTokens`,** the stop token's text is added to the prompt cache but never decoded.
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
git am /path/to/AuraLocal/contrib/localllmclient/0004-*.patch
```

`0003` and `0004` do not depend on the others and apply on their own as well.

Running `0004`'s tests in Release needs `ENABLE_TESTABILITY=YES 'SWIFT_ACTIVE_COMPILATION_CONDITIONS=$(inherited) DEBUG'`
(the test target uses `#if DEBUG` helpers); `assert` is still compiled out that way.

Building upstream from source needs its submodule (llama.cpp):

```bash
git submodule update --init --recursive --depth 1
```

## Do not point AuraLocal at a fork

Tempting, but it re-introduces exactly the failure fixed in `4dfcbb6`: SwiftPM honours only the
**root** package's `Package.resolved`, so a fragile dependency graph that resolves here breaks for
every downstream integrator. Wait for these to land upstream in a tagged release.

Nothing in AuraLocal depends on these patches. The iOS work that shipped without them —
jetsam fail-safe, the `extended-virtual-addressing` entitlement and prefill batch sizing — is
already in the repo and stands on its own. The context-window anchor is in the repo too, but it
saves nothing while GGUF turns prefill the whole history; it matters again only once they reuse a
cached prefix (see above).

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
