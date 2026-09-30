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
aura ask "<prompt>" [--model <id>]  # ask GitHub Models (default openai/gpt-4o)
aura ocr <image>                    # extract text from an image via native Vision OCR
aura ml <subcommand> …              # run an on-device ML tool (see below)
aura models search|check|devices …  # which Hugging Face models AuraLocal can run (see below)
```

- **`ask`** reads a GitHub fine-grained PAT (`models:read`) from `AURA_GITHUB_TOKEN` /
  `GITHUB_TOKEN`, or the Keychain (`cloud.github-models`) — never from source or CI logs.
  The answer goes to stdout; a receipt (provider · tokens · compression) goes to stderr.
- **`providers`** / **`tools`** / **`ocr`** / **`ml`** / **`models`** need no key and no model download.

```sh
export AURA_GITHUB_TOKEN=ghp_…
aura ask "Explain hybrid inference in one paragraph" --model openai/gpt-4o
```

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
aura ml train-text <csv> --out <Model.mlmodel> [--algorithm maxent|crf|static|bert]
                   [--language es] [--holdout 0.2 [--seed 7] | --no-validation]
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

## Homebrew

`aura` ships as a prebuilt tarball (binary + `llama.framework`, co-located) installed via a
personal Homebrew tap:

```sh
brew tap iOSDevC/aura
brew install aura
aura tools
```

See [`docs/HOMEBREW.md`](https://github.com/iOSDevC/AuraLocal/blob/main/docs/HOMEBREW.md)
for the release + tap runbook (`scripts/package-cli.sh` builds the tarball).

{: .note }
> The prebuilt-tarball path is used because Homebrew's install sandbox blocks the SwiftPM
> dependency fetch. macOS arm64; the binary is development-signed (notarize for wider
> distribution).
