---
published: false
---

# Homebrew — the `aura` CLI (personal tap)

Maintainer runbook, kept out of the docs site. For users, the CLI is documented in
[CLI (aura)](guide/cli.md), which builds it from source.

`aura` ships as a **prebuilt tarball** (the `aura` binary + `llama.framework`,
co-located) installed through a personal Homebrew tap. No SwiftPM or network is
needed at install time, which sidesteps Homebrew's build-sandbox blocking the
SwiftPM dependency fetch.

> **Why not build-from-source in the formula?** Homebrew's install sandbox blocks
> network, but `swift build` needs to fetch the pinned SwiftPM dependencies (and
> llama.cpp). Shipping a prebuilt tarball is the reliable path for a personal tap.

## Status

- **v0.1.0 is released**: GitHub release `v0.1.0` (2026-07-10) with the asset
  `aura-v0.1.0-macos-arm64.tar.gz`, whose sha256 is the one in `Formula/aura.rb`.
- **The tap is not published**: `github.com/iOSDevC/homebrew-aura` does not exist, so
  `brew tap iOSDevC/aura` fails until the steps under [Publish the tap](#publish-the-tap) are done.
- **`Formula/aura.rb` installs v0.1.0**, which predates `aura imagegen`, `aura ml` and
  `aura models`: that binary has only `providers`, `tools`, `ask` and `ocr`. The newer
  commands need a new release or a source build (`swift build -c release --product aura`).
- **`aura ask` in v0.1.0 is broken**: that binary only targets GitHub Models, which GitHub retired
  on 2026-07-30. From source (and the next release), `ask` uses a local llama-server/Ollama by
  default, or OpenAI/Anthropic/an OpenAI-compatible `--base-url` when named.

## What's in the repo

- `scripts/package-cli.sh [version]` — builds `aura` (release), checks that the staged
  binary runs `aura tools` without the source tree, and packages `aura` +
  `llama.framework` into `build/aura-v<version>-macos-<arch>.tar.gz`. Prints the
  tarball's `sha256`. The version defaults to `0.1.0`.
- `Formula/aura.rb` — the formula (installs the tarball into `libexec` and links `aura`
  into `bin`; `test do` checks that `aura tools` lists `system.vision.ocr`).
- The `LocalLLMClient` dependency is pinned by `revision:` in `Package.swift`, so
  release builds are reproducible.

## Release a new version

1. **Build the tarball** and note the printed `sha256`:
   ```sh
   ./scripts/package-cli.sh <version>
   # -> build/aura-v<version>-macos-arm64.tar.gz  (prints size + sha256)
   ```
2. **Tag and push**:
   ```sh
   git tag v<version>
   git push origin v<version>
   ```
3. **Create the GitHub release** for `v<version>` and upload
   `build/aura-v<version>-macos-arm64.tar.gz` as a release asset.
4. **Update `Formula/aura.rb`**: `version`, `url` and `sha256`. If the tap exists, copy the
   formula into it and push.

The tag only locates the tarball. Swift package consumers still pin AuraLocal by
`revision:`, because a version requirement cannot resolve (see
[Installation](guide/installation.md)).

## Publish the tap

5. Create a public repo named **`homebrew-aura`** (`github.com/iOSDevC/homebrew-aura`).
6. Copy `Formula/aura.rb` into it at `Formula/aura.rb`, commit, and push.

## Install

Once the tap is published:

```sh
brew tap iOSDevC/aura
brew install aura
aura tools
aura providers
```

## Notes

- **macOS arm64 only** (`depends_on arch: :arm64`). The bundled `llama.framework` is
  universal (arm64 + x86_64); the `aura` binary is arm64. Intel support needs an x86_64
  tarball and an `on_intel` block in the formula.
- **Gatekeeper**: `scripts/package-cli.sh` does not re-sign, so the binary keeps the
  linker's ad-hoc signature and is not notarized. If macOS blocks it, either notarize the
  tarball or run `xattr -dr com.apple.quarantine "$(brew --prefix)/opt/aura"`.
