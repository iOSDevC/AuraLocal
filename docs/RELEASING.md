---
published: false
---

# Releasing AuraLocal

Maintainer checklist, kept out of the docs site. The CLI tarball and the Homebrew formula are covered in detail
in [HOMEBREW.md](HOMEBREW.md); this page is the order of operations for a release such as `v1.4.0`.

## Where the tags stand

- When last checked (2026-09-29), the public repository's only tag was `v0.1.0` (97ab9d1, 2026-07-10, MIT).
  `v1.0.0`–`v1.3.0` exist only in the maintainer's clone; their commits are on `origin/main`, the tags are not.
  Check with `git ls-remote --tags origin` before tagging.
- `v1.3.0` (7653c93) is older than `v0.1.0`. [CHANGELOG.md](../CHANGELOG.md) counts from `v1.3.0`.
- Push only the release tag (`git push origin v1.4.0`). `git push --tags` would also publish `v1.0.0`–`v1.3.0`.
- Swift package consumers pin a revision, not a version (see [Installation](guide/installation.md#pin-a-commit)),
  so the tag locates the release and its tarball; it does not change how AuraLocal resolves.

## Before tagging

1. Merge every branch that belongs in the release into `main`, and add each one's entries to the `[Unreleased]`
   section of `CHANGELOG.md`.
2. CI is green on `main`: the `test` and `ios` jobs in `.github/workflows/ci.yml`.
3. Run the workflow by hand (**Actions › CI › Run workflow**) and read the canary jobs' summary: they resolve
   without `Package.resolved`, as an integrator does, and build and test what that resolve returns.
4. Test on a physical iPhone, on iOS 27 if you have one: launch AuraExample (a launch that never reaches `main()`
   is mlx-swift #485), load a GGUF and an MLX model, and generate an image.
5. Check that `THIRD_PARTY_NOTICES.md` lists the versions in `Package.resolved`.
6. In `CHANGELOG.md`, rename `[Unreleased] (planned 1.4.0)` to `[1.4.0] - <date>` and open a new, empty
   `[Unreleased]` above it.

## Tag

7. Tag the release commit and push only that tag:
   ```sh
   git tag -a v1.4.0 -m "AuraLocal 1.4.0"
   git push origin v1.4.0
   ```

## CLI tarball and Homebrew

8. Build the tarball and note its `sha256`: `./scripts/package-cli.sh 1.4.0`. Besides `aura` and
   `llama.framework` it now carries the SwiftPM resource bundles (`AuraLocal_AuraCore.bundle` holds
   `models.json`), `LICENSE`, `THIRD_PARTY_NOTICES.md` and `licenses/<package>/`. The `v0.1.0` tarball had none
   of these.
9. Create the GitHub release `v1.4.0` from the changelog section, state the license (AGPL-3.0-or-later, or
   commercial), and attach `build/aura-v1.4.0-macos-arm64.tar.gz`.
10. Update `Formula/aura.rb` for the new tarball: `version "1.4.0"`, the `v1.4.0` asset `url`, its `sha256`, and
    `license "AGPL-3.0-or-later"`. The formula says MIT today because it installs the MIT `v0.1.0` tarball;
    change the license only together with the URL. Install the bundles next to the binary and the license texts
    with it, and replace the `bin.install_symlink` line with an exec wrapper:
    ```ruby
    libexec.install "aura", "llama.framework", *Dir["*.bundle"]
    bin.write_exec_script libexec/"aura"
    prefix.install "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses"
    ```
    A symlink would not find the bundles: `Bundle.module` looks next to the path that was invoked (`bin/`), and
    without the bundle it calls `fatalError`. The exec wrapper runs the binary from `libexec`.
    These lines are untested: after copying the formula into the tap, run `brew install iOSDevC/aura/aura`,
    `brew test aura`, one `aura` command outside the tap's directory and
    `ls "$(brew --prefix aura)"/libexec/*.bundle` before announcing the release.
11. Update the Status section of `HOMEBREW.md` and, if the tap exists, copy the formula into it.

## Docs

12. Replace the `revision: "<commit SHA from main>"` placeholders in `README.md`,
    `docs/guide/installation.md` and `docs/guide/quickstart.md` with the release commit, for example
    `revision: "<full SHA of v1.4.0>" // v1.4.0`, so a copied snippet resolves to the release. Do the same for
    both occurrences in `docs/_layouts/landing.html` (the escaped `&lt;commit SHA from main&gt;` in the install
    block and the copy button's string), coordinating with whoever owns the landing page.
13. The site rebuilds from `main` on push; check the installation and license pages after it does.

## Credentials

14. Rotate the GitHub personal access token used to push the release, and keep tokens out of remote URLs: use
    a credential helper or SSH instead.
