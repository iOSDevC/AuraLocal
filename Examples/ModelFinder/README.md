# Model Finder

A small iOS 18 / macOS 15 app that searches Hugging Face and tells you, per repo, whether AuraLocal can really
run it on a given device — and if not, why. It is a UI over `ModelCompatibilityChecker` (AuraCore); see
[Finding compatible models](../../docs/guide/models.md#finding-compatible-models).

- Search field, format filter (MLX / GGUF), sort, and a device picker (this device, iPhone classes, Macs — each
  budget labelled measured or estimate, with its source).
- Results show a verdict badge per row. Checks run lazily when a row appears, are cached per repo, and are
  cancelled when the row scrolls away. Changing the device re-evaluates the cache without refetching.
- The detail view lists the findings (blockers first), the per-quant fit table for GGUF repos, a config summary,
  license and gating, **Copy catalog entry** (the exact `models.json` entry) and **Open on Hugging Face**.

## Generate, open, run

The Xcode project is generated with [XcodeGen](https://github.com/yonaskolb/XcodeGen) and committed, so
regenerating is only needed after editing `project.yml`:

```sh
cd Examples/ModelFinder
xcodegen generate            # rewrites ModelFinder.xcodeproj from project.yml
open ModelFinder.xcodeproj   # pick the ModelFinder scheme, then a Mac or an iOS 18+ destination
```

The app depends on the repository root as a local Swift package (product `AuraCore`), so keep it inside the
checkout. AuraCore links llama.cpp, which is why the target sets `SWIFT_OBJC_INTEROP_MODE = objcxx`.
`ModelFinder.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved` is a copy of the root
`Package.resolved` — Xcode does not read a dependency's lockfile — so copy it again when a pin moves.

Command-line builds:

```sh
xcodebuild -project ModelFinder.xcodeproj -scheme ModelFinder -destination 'platform=macOS' build
xcodebuild -project ModelFinder.xcodeproj -scheme ModelFinder -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO build
```

To run on a device, set your team in Signing & Capabilities. The Mac build is sandboxed with outgoing network
access only. Gated repositories need a Hugging Face token saved in the Keychain under
`download.huggingface` (the same account AuraLocal's downloaders read).

The same checks are available headless: `aura models search "<query>"` and `aura models check <repo>`.
