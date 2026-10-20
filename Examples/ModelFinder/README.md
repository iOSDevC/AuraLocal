# Model Finder

A small iOS 18 / macOS 15 app that searches Hugging Face, tells you, per repo, whether Tonamox can really
run it on a given device — and if not, why — and downloads the ones that fit to where Tonamox loads them from.
It is a thin host for TonamoxUI's Hugging Face browser (`HFModelSearch`, `HFModelSearchList`, `HFRepoDetailView`,
`DownloadedModelsList`, `HuggingFaceTokenSection`), which sits on `ModelCompatibilityChecker` and `ModelDownloads`
(TonamoxCore); TonamoxExample shows the same views. See
[Finding compatible models](../../docs/guide/models.md#finding-compatible-models) and
[Downloading without loading](../../docs/guide/models.md#downloading-without-loading).

- Search field, format filter (MLX + GGUF / MLX / GGUF; it starts on GGUF on the Simulator, where MLX can't
  run), sort, a device picker (this device, iPhone classes,
  Macs — each budget labelled measured or estimate, with its source; default this device), and a **Compatible
  only** toggle, which hides only rows already checked as won't run (unchecked rows stay visible).
- Results show a verdict badge per row. Checks run lazily when a row appears and are cached per repo; a row and
  the detail pane share one fetch, which is cancelled only when no view waits on it any more. A new search cancels
  the one in flight. Changing the device re-evaluates the cache without refetching.
- The detail view lists the findings (blockers first), the per-quant fit table for GGUF repos, a config summary,
  license and gating, **Copy catalog entry** (the exact `models.json` entry) and **Open on Hugging Face**.

## Downloads

Repos rated *Runs* or *Runs, with caveats* get a **Download** section in the detail view.

- **GGUF**: a quant picker limited to single-file, top-level quants not rated too large for the selected device
  (split quants never appear). **MLX**: a fixed file set — top-level `*.safetensors`, `*.json`, `*.jinja` of
  `main`.
- Before starting it shows the size and the free space (`Download 415.2 MB · 118 GB free`), or how much is
  already on disk (`Resume (302 MB of 415.2 MB on disk)`). Then a progress bar and **Cancel**.
- One download at a time; other Download buttons are disabled while one runs.
- Errors have their own message: a gated repo (HTTP 401/403 with `X-Error-Code: GatedRepo`, or a rejected token)
  offers **Add a token**, which opens the token form in place, and a link to the repo page, a missing file (404), a repo that is missing, renamed or private (an
  anonymous 401 without an error code; Hugging Face does not tell these apart), not enough space (`Needs X, Y free.`), a size mismatch or a network error (the partial file is kept).
- Files land exactly where the backends read them, under `ModelStorage.rootDirectory`
  (`<Application Support>/AuraLocal/Models`, excluded from backup): GGUF at `<root>/models/<org>/<repo>/<file>.gguf`
  (`Model.resolvedFileURL`), MLX at `<root>/huggingface/hub/models--<org>--<repo>/snapshots/main/` with a
  `.complete` marker written last. Downloads an older build left in `Library/Caches` are still read from there. `ModelManager.shared.load(_:)` finds them without downloading again.
- When a download finishes, the detail view offers **Try it**, and the model is registered
  (`ModelRegistry.register(_:)`), so TonamoxUI's Text and Vision pickers list it.

**Interruptions.** There are no background downloads: the session is a default `URLSession`, so locking the phone
or leaving the app for more than ~30 s (a background-task window) stops the transfer. Each file is written in 32 MB
chunks to `<file>.partial`; **Resume** — in the detail view or the Downloads tab, also after a relaunch —
continues from it, unless the file changed on Hugging Face meanwhile (another ETag): then it starts over. An optional toggle keeps the screen awake while downloading.

**Downloads tab.** A scan of the disk on every visit, so it survives relaunches without a database: finished GGUF
files, paused GGUF `.partial` files, and MLX snapshots (complete or not), with size, Resume, **Delete** (swipe or
context menu; unloads the model first, forgets its record, unregisters it and reports the freed space) and
**Try it**. TonamoxCore's `DownloadRecords` keeps the sized `Model` for each download in `UserDefaults`
(`dev.auralocal.downloadRecords`; records under this app's old `downloadRecords` key are copied once) so Try it
can judge memory; files with neither a record nor a catalog entry show Try it and Resume disabled (size unknown):
delete them, or resume them from their search result. A paused file whose finished copy is already on disk (an
older build's, left in Caches) shows as **Leftover**; Delete removes only that partial.

**Try it** loads the model with `ModelManager`, asks one short question (32 tokens max), shows the reply and the
load/reply times, and always unloads it — also on cancel, on Done and on a memory warning. It is enabled only for
complete downloads with a known size whose `HardwareAnalyzer` fit on this device is not *Too large* (MLX also not
*Streaming*), and never for MLX on the Simulator.

**Gated repos.** Accept the license on the repo page, create a read token, and save it in the **Settings** tab
or from **Add a token** (also the key button) in the detail view. It is stored in the Keychain (service
`dev.auralocal.remote`, account `download.huggingface`) and read by Tonamox's downloaders (`ModelDownloads`,
`GGUFModelDownloader`, `ChunkedDownloader`), by the checker and by TonamoxUI's `ModelSection`; the MLX load path does
not read it. It is sent only to `huggingface.co` and `hf.co`. On iOS that item belongs to this app; other
Tonamox apps don't see it. Search stays anonymous, so private repos don't appear in results. No OAuth sign-in
(it would need a registered Hugging Face OAuth app and URL scheme).

**Simulator.** GGUF downloads and Try it work (llama.cpp on the CPU); MLX can be downloaded but not run there.
"This device" on the Simulator reflects the Mac host's memory, not an iPhone's.

**Real devices.** iOS caps an app's memory well below the phone's RAM. The
`com.apple.developer.kernel.increased-memory-limit` entitlement raises that cap but needs a signing team with the
capability, so it is not in this project; add it in Signing & Capabilities for larger models. Not verified on a
device: memory limits, the background-task window, MLX Try it.

## Generate, open, run

The Xcode project is generated with [XcodeGen](https://github.com/yonaskolb/XcodeGen) and committed, so
regenerating is only needed after editing `project.yml`:

```sh
cd Examples/ModelFinder
xcodegen generate            # rewrites ModelFinder.xcodeproj from project.yml
open ModelFinder.xcodeproj   # pick the ModelFinder scheme, then a Mac or an iOS 18+ destination
```

The app depends on the repository root as a local Swift package (products `TonamoxCore` and `TonamoxUI`; TonamoxUI also
links `TonamoxVoice`), so keep it inside the checkout. TonamoxCore links llama.cpp, which is why the target sets
`SWIFT_OBJC_INTEROP_MODE = objcxx`. After editing `project.yml`, run `xcodegen generate` again and commit the
regenerated project.
`ModelFinder.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved` is a copy of the root
`Package.resolved` — Xcode does not read a dependency's lockfile — so copy it again when a pin moves.

Command-line builds:

```sh
xcodebuild -project ModelFinder.xcodeproj -scheme ModelFinder -destination 'platform=macOS' build
xcodebuild -project ModelFinder.xcodeproj -scheme ModelFinder -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO build
```

To run on a device, set your team in Signing & Capabilities. The Mac build is sandboxed with outgoing network
access only.

For gated repositories the checker uses the same token: without one, `config.json` and file headers come back as
HTTP 401 caveats and the verdict is often *Unknown* or *Runs, with caveats*.

The same checks are available headless: `tonamox models search "<query>"` and `tonamox models check <repo>` — see
[CLI](../../docs/guide/cli.md#model-compatibility-tonamox-models).
