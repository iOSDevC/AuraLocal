---
layout: docs
title: CLI (aura)
parent: Guide
nav_order: 9
description: "The aura command-line tool — a headless integration harness for the hybrid, native-tool and on-device ML features, plus Homebrew packaging."
---

# CLI — `aura`
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

`aura` is a small **macOS** command-line tool that drives AuraLocal's hybrid and
native-tool features headlessly — useful as an integration reference and a CI smoke test.

Build it from the package:

```sh
swift build -c release --product aura
# or
./scripts/build-cli.sh
```

## Commands

```
aura providers                      # detect Ollama / llama-server + models
aura tools                          # list on-device ML tools by category, with availability
aura ask "<prompt>" [--provider auto|local|openai|anthropic] [--model <id>]
         [--base-url <url>] [--max-tokens N]  # ask a bigger model; local unless you name a cloud API
aura ocr <image>                    # native Vision OCR; ignores EXIF orientation (camera photos: aura ml ocr-lines)
aura ml <subcommand> …              # run an on-device ML tool (see below)
aura models search|check|devices …  # which Hugging Face models AuraLocal can run (see below)
aura imagegen "<prompt>" [--model schnell|dev|<repo>] [--base-model schnell|dev]
              [--lora <file.safetensors> [--lora-scale S]]... [--steps N] [--seed N]
              [--quantize 3|4|6|8] [--low-ram] [--out <dir>]   # FLUX via mflux (macOS; needs uv tool install mflux)
```

- **`providers`** / **`tools`** / **`ocr`** / **`ml`** / **`models`** need no key and no model download.
- **`ask`** needs a running llama-server / Ollama, a named cloud provider with a key, or `--base-url`;
  see [below](#ask-a-bigger-model-aura-ask).
- **`imagegen`** prints the path of the PNG it wrote (`<dir>/image.png`, a temporary folder by default).
  It defaults to `--model schnell` (a gated repo) and `--quantize 4`, and `--lora` takes local files only;
  see [Image Generation]({{ '/guide/imagegen' | relative_url }}).

## Ask a bigger model (`aura ask`)

```
aura ask "<prompt>" [--provider auto|local|openai|anthropic] [--model <id>]
         [--base-url <url>] [--max-tokens N]
```

`--provider` picks where the prompt goes:

| `--provider` | Target | Key |
|---|---|---|
| `auto` (default), `local` | A running `llama-server` (`127.0.0.1:8080/v1`), else Ollama (`localhost:11434`); the model with the largest trained context (Ollama: its first listed model) | none |
| `openai` | `https://api.openai.com/v1`, default model `gpt-4o` | `OPENAI_API_KEY`, else Keychain account `cloud.openai` |
| `anthropic` | Anthropic Messages API, default model `claude-sonnet-4-5` | `ANTHROPIC_API_KEY`, else Keychain account `cloud.anthropic` |

- **`auto` never picks a cloud API.** It also skips Ollama cloud models (`name:cloud`, which Ollama
  forwards to `ollama.com`; `aura providers` marks them). With no on-machine model running, it exits 1
  and lists the options (start llama-server / Ollama, `--provider openai|anthropic`, or `--base-url`).
- **`--model <id>`** overrides the default. For `auto` / `local` it selects the local server that
  serves that model (`llama3` also matches Ollama's `llama3:latest`) and fails if none does.
- **`--base-url <url> --model <id>`** sends to any other OpenAI-compatible server (`chat/completions`
  is appended to the URL, e.g. `http://192.168.1.20:8080/v1`). `--model` is required: llama-server
  accepts any id, other servers need a real one. An optional key comes from `AURA_API_KEY` (sent as
  a Bearer token). A loopback, private-network (`10.*`, `172.16–31.*`, `192.168.*`, IPv6 ULA),
  link-local, `localhost`, `*.local` or `*.home.arpa` host counts as your own machine; any other host
  counts as cloud. The check looks at the host only, so a local server that forwards the model
  elsewhere (an Ollama `:cloud` model named with `--model`) still counts as local. `--base-url` can't
  be combined with `--provider openai|anthropic`, and `--provider local --base-url` accepts only a
  local-network host.
- **`--max-tokens N`** caps the answer (default 512).

```sh
aura ask "Summarize RFC 9110 in one sentence" > answer.txt      # your llama-server / Ollama
aura ask "Review this function: …" --provider anthropic          # sends to Anthropic
aura ask "Hello" --base-url http://192.168.1.20:8080/v1 --model qwen3-32b
```

The answer goes to stdout, so it pipes cleanly. A receipt goes to stderr:
`— via <provider> · <model>[ · <in> in / <out> out][ · cached]`. Token counts appear when the
server reports them; a repeated identical request is served from the in-memory response cache
and ends in `· cached`. `aura ask --help` prints the usage.

For a cloud target, obvious secrets and PII in the prompt (emails, API keys, tokens) are redacted
before sending (`redactPII: true`); a local-network target gets the prompt verbatim. Keychain keys use the
service `dev.auralocal.remote` (`KeychainStore`). Exit codes: 0 success, 1 no target or a failed
request, 2 a usage error (missing prompt, unknown flag or provider, an invalid `--base-url` or one
without `--model`, conflicting options). The choice itself is `AskTargetResolver.resolve` in `AuraCore`; see
[Hybrid Inference]({{ '/guide/hybrid' | relative_url }}).

{: .note }
> `aura ask` used GitHub Models until GitHub retired it on 2026-07-30. `AURA_GITHUB_TOKEN`,
> `GITHUB_TOKEN` and the Keychain account `cloud.github-models` are no longer read.

## On-device ML (`aura ml`)

One subcommand per [on-device ML tool]({{ '/guide/ml-tools' | relative_url }}). Results go to stdout; notes, summaries and
Create ML's training log go to stderr.

```
aura ml classify-image <image> [--max N] [--min 0.1]   # label the scene / objects (Vision)
aura ml barcodes <image> [--symbology QR]... | --list  # QR codes and barcodes
aura ml faces <image>                                  # face boxes and head pose (no identity)
aura ml ocr-lines <image> [--lang es-ES]... [--fast]   # text line by line with boxes
aura ml language "<text>" [--max N] [--only es,en]     # identify the language
aura ml entities "<text>" [--lang es]                  # people, places, organizations
aura ml sentiment "<text>" [--lang es]                 # score from -1 to 1
aura ml similarity "<a>" "<b>" [--lang en]             # sentence-embedding distance
aura ml sounds <audio file> [--max N] [--mean] | --list  # classify everyday sounds
aura ml coreml-describe <model>                        # inputs, outputs, labels, metadata
aura ml coreml-predict <model> <input>=<value>...      # run one prediction
aura ml train-text <csv> --out <Model.mlmodel> [--text-column text] [--label-column label]
                   [--algorithm maxent|crf|static|bert] [--language es]
                   [--holdout 0.2 [--seed 7] | --no-validation]
aura ml classify-text <Model.mlmodel> "<text>" [--max N]
```

`language`, `entities` and `sentiment` read stdin when the text is `-`. Image boxes are normalized
0…1 with the origin at the bottom-left, as Vision reports them.

```sh
$ aura ml train-text expenses.csv --out Expenses.mlmodel --algorithm bert --holdout 0.2 --seed 7 2>/dev/null
/absolute/path/to/Expenses.mlmodel
$ aura ml classify-text Expenses.mlmodel "Pagué el taxi del hotel a la estación"
label  transport
1.00  transport
0.00  housing
0.00  food
```

Training is reproducible: the same CSV and `--seed` give the same split, accuracies and model.

## Model compatibility (`aura models`)

Checks Hugging Face repos against the runtimes AuraLocal pins (mlx-swift-lm 3.31.3, llama.cpp b8851) and a
device budget — see [Finding compatible models]({{ '/guide/models' | relative_url }}#finding-compatible-models).

```
aura models search "<query>" [--format mlx|gguf] [--device <preset>] [--limit N]
aura models check <owner/repo | URL> [--device <preset>] [--json] [--entry]
aura models devices                     # presets and where each memory budget comes from
```

```sh
$ aura models check ukisai/Swift-1.5-Qwen3.8-27B-GGUF --device mac-32gb 2>/dev/null | head -3
ukisai/Swift-1.5-Qwen3.8-27B-GGUF — GGUF · WON'T RUN
Device: Mac, 32 GB (M1 Pro) · budget 20.0 GB (measured)
Model: qwen35 · 65 layers · context 262144 · 22 quants
```

`--entry` prints only the `models.json` entry of a runnable model; `--json` prints the whole report.
`--device` defaults to `this-device` (see `aura models devices`); `--limit` defaults to 15 (1–100).
`search` and `check` query huggingface.co, so they need a network connection.

## Homebrew

`aura` can be packaged as a prebuilt tarball (binary + `llama.framework`, co-located) for a
personal Homebrew tap.

{: .warning }
> The Homebrew tap is not published yet (`github.com/iOSDevC/homebrew-aura` does not resolve),
> and the only released tarball (v0.1.0) predates `aura ml`, `aura models` and `aura imagegen`, and
> its `aura ask` still calls the retired GitHub Models. Build from source (above) to get the commands
> on this page. Once the tap is live:

```sh
brew tap iOSDevC/aura
brew install aura
aura tools
```

See [`docs/HOMEBREW.md`](https://github.com/iOSDevC/AuraLocal/blob/main/docs/HOMEBREW.md)
for the release + tap runbook (`scripts/package-cli.sh` builds the tarball).

{: .note }
> The prebuilt-tarball path is used because Homebrew's install sandbox blocks the SwiftPM
> dependency fetch. macOS arm64; the binary carries only the linker's ad-hoc signature and is not
> notarized.
