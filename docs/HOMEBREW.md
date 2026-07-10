# Homebrew — the `aura` CLI (personal tap)

`aura` ships as a **prebuilt tarball** (the `aura` binary + `llama.framework`,
co-located) installed through a personal Homebrew tap. No SwiftPM or network is
needed at install time, which sidesteps Homebrew's build-sandbox blocking the
SwiftPM dependency fetch.

> **Why not build-from-source in the formula?** Homebrew's install sandbox blocks
> network, but `swift build` needs to fetch the pinned SwiftPM dependencies (and
> llama.cpp). Shipping a prebuilt tarball is the reliable path for a personal tap.

## What's already prepared

- `scripts/package-cli.sh` — builds `aura` (release) and packages `aura` +
  `llama.framework` into `build/aura-v<version>-macos-<arch>.tar.gz` (verified to
  run standalone). Prints the tarball's `sha256`.
- `Formula/aura.rb` — the formula template (installs the tarball; `test do` runs
  `aura tools`).
- The `LocalLLMClient` dependency is pinned by `revision:` in `Package.swift`, so
  release builds are reproducible.

## Cut the release (your steps — outward actions)

1. **Build the tarball** and note the printed `sha256`:
   ```sh
   ./scripts/package-cli.sh 0.1.0
   # -> build/aura-v0.1.0-macos-arm64.tar.gz  (prints size + sha256)
   ```
2. **Tag & push** (the tag `v0.1.0` already exists locally — just push it):
   ```sh
   git push origin v0.1.0
   ```
3. **Create the GitHub release** for `v0.1.0` and upload
   `build/aura-v0.1.0-macos-arm64.tar.gz` as a release asset.
4. If you rebuilt in step 1, paste the new `sha256` into `Formula/aura.rb`.

## Publish the tap (your steps)

5. Create a repo named **`homebrew-aura`** (e.g. `github.com/iOSDevC/homebrew-aura`).
6. Copy `Formula/aura.rb` into it at `Formula/aura.rb`, commit, and push.

## Install

```sh
brew tap iOSDevC/aura
brew install aura
aura tools
aura providers
aura ask "Explain hybrid inference" --model openai/gpt-4o   # needs AURA_GITHUB_TOKEN
```

## Notes

- **macOS arm64** (the bundled `llama.framework` is universal; the `aura` binary is
  arm64). Add an x86_64 build + a second bottle if Intel support is needed.
- **Gatekeeper**: the binary is development-signed, not notarized. On first run macOS
  may quarantine it — either notarize the tarball, or users run
  `xattr -dr com.apple.quarantine "$(brew --prefix)/opt/aura"`.
- **Token**: `aura ask` reads `AURA_GITHUB_TOKEN` / `GITHUB_TOKEN`, or the Keychain
  (`cloud.github-models`) — never from source or logs.
- Bump: build a new tarball, update `version` + `url` + `sha256`, push a new tag.
