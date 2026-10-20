---
published: false
---

# Homebrew — the `tonamox` CLI (personal tap)

Maintainer runbook, kept out of the docs site. For users, the CLI is documented in
[CLI (tonamox)](guide/cli.md), which builds it from source.

`tonamox` ships as a **prebuilt tarball** (the `tonamox` binary + `llama.framework`,
co-located) installed through a personal Homebrew tap. No SwiftPM or network is
needed at install time, which sidesteps Homebrew's build-sandbox blocking the
SwiftPM dependency fetch.

> **Why not build-from-source in the formula?** Homebrew's install sandbox blocks
> network, but `swift build` needs to fetch the pinned SwiftPM dependencies (and
> llama.cpp). Shipping a prebuilt tarball is the reliable path for a personal tap.

## Status

- **v0.1.0 is released**: GitHub release `v0.1.0` (2026-07-10) with the asset
  `tonamox-v0.1.0-macos-arm64.tar.gz`, whose sha256 is the one in `Formula/tonamox.rb`.
- **The tap is not published**: `github.com/iOSDevC/homebrew-tonamox` does not exist, so
  `brew tap iOSDevC/tonamox` fails until the steps under [Publish the tap](#publish-the-tap) are done.
- **`Formula/tonamox.rb` installs v0.1.0**, which predates `tonamox imagegen`, `tonamox ml` and
  `tonamox models`: that binary has only `providers`, `tools`, `ask` and `ocr`. The newer
  commands need a new release or a source build (`swift build -c release --product tonamox`).
- **`tonamox ask` in v0.1.0 is broken**: that binary only targets GitHub Models, which GitHub retired
  on 2026-07-30. From source (and the next release), `ask` uses a local llama-server/Ollama by
  default, or OpenAI/Anthropic/an OpenAI-compatible `--base-url` when named.

## What's in the repo

- `scripts/package-cli.sh [version]` — builds `tonamox` (release), checks that the staged
  binary runs `tonamox tools` without the source tree, and packages `tonamox`,
  `llama.framework`, the SwiftPM resource bundles, `LICENSE`, `THIRD_PARTY_NOTICES.md` and
  the dependencies' license files (`licenses/`) into
  `build/tonamox-v<version>-macos-<arch>.tar.gz`. Prints the tarball's `sha256`. The version
  defaults to `0.1.0`. It finds the build products with `swift build --show-bin-path`, which
  also covers the `.build/out/Products/Release` layout of newer SwiftPM.
- `Formula/tonamox.rb` — the formula (installs the tarball into `libexec` and links `tonamox`
  into `bin`; `test do` checks that `tonamox tools` lists `system.vision.ocr`).
- The `LocalLLMClient` dependency is pinned by `revision:` in `Package.swift`, so
  release builds are reproducible.

## Release a new version

The whole release, from CI and the changelog to the tag and the docs, is in [RELEASING.md](RELEASING.md).

1. **Build the tarball** and note the printed `sha256`:
   ```sh
   ./scripts/package-cli.sh <version>
   # -> build/tonamox-v<version>-macos-arm64.tar.gz  (prints size + sha256)
   ```
2. **Tag and push** only the release tag:
   ```sh
   git tag -a v<version> -m "TONAMOX <version>"
   git push origin v<version>
   ```
3. **Create the GitHub release** for `v<version>` and upload
   `build/tonamox-v<version>-macos-arm64.tar.gz` as a release asset.
4. **Update `Formula/tonamox.rb`**: `version`, `url`, `sha256`, `license "AGPL-3.0-or-later"` for
   any tarball built after af056da, and the install lines in [RELEASING.md](RELEASING.md) step 10
   (bundles, license texts, exec wrapper). If the tap exists, copy the formula into it and push.

The tag only locates the tarball. Swift package consumers still pin the package by
`revision:`, because a version requirement cannot resolve (see
[Installation](guide/installation.md)).

## Publish the tap

5. Create a public repo named **`homebrew-tonamox`** (`github.com/iOSDevC/homebrew-tonamox`).
6. Copy `Formula/tonamox.rb` into it at `Formula/tonamox.rb`, commit, and push.

## Install

Once the tap is published:

```sh
brew tap iOSDevC/tonamox
brew install tonamox
tonamox tools
tonamox providers
```

## Notes

- **macOS arm64 only** (`depends_on arch: :arm64`). The bundled `llama.framework` is
  universal (arm64 + x86_64); the `tonamox` binary is arm64. Intel support needs an x86_64
  tarball and an `on_intel` block in the formula.
- **Gatekeeper**: `scripts/package-cli.sh` does not re-sign, so the binary keeps the
  linker's ad-hoc signature and is not notarized. If macOS blocks it, either notarize the
  tarball or run `xattr -dr com.apple.quarantine "$(brew --prefix)/opt/tonamox"`.
