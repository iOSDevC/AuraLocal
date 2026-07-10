---
layout: docs
title: CLI (aura)
parent: Guide
nav_order: 9
description: "The aura command-line tool — a headless integration harness for the hybrid + native-tool features, plus Homebrew packaging."
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
aura tools                          # list on-device tools (Vision OCR, embeddings)
aura ask "<prompt>" [--model <id>]  # ask GitHub Models (default openai/gpt-4o)
aura ocr <image>                    # extract text from an image via native Vision OCR
```

- **`ask`** reads a GitHub fine-grained PAT (`models:read`) from `AURA_GITHUB_TOKEN` /
  `GITHUB_TOKEN`, or the Keychain (`cloud.github-models`) — never from source or CI logs.
  The answer goes to stdout; a receipt (provider · tokens · compression) goes to stderr.
- **`providers`** / **`tools`** / **`ocr`** need no key and no model download.

```sh
export AURA_GITHUB_TOKEN=ghp_…
aura ask "Explain hybrid inference in one paragraph" --model openai/gpt-4o
```

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
