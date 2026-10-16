#!/usr/bin/env bash
# Package the `aura` CLI + its llama.framework into a distributable tarball for a
# Homebrew formula (personal tap). No SwiftPM/network is needed at install time —
# the binary + framework are co-located so the binary's @loader_path rpath finds
# `@rpath/llama.framework` right next to it (no install_name_tool, no re-sign).
# SwiftPM resource bundles sit next to the binary too, where Bundle.module looks for
# a command-line tool, and the license texts travel with the binaries.
#
# Usage: ./scripts/package-cli.sh [version]   (default 0.1.0)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="${1:-0.1.0}"
ARCH="$(uname -m)"                       # arm64
STAGE="$ROOT/build/aura-pkg"
OUT="$ROOT/build/aura-v${VERSION}-macos-${ARCH}.tar.gz"

echo "==> Building aura (release)…"
swift build -c release --product aura >/dev/null

# Swift 6.4's build system writes products to .build/out/Products/Release, not
# .build/<arch>-apple-macosx/release; ask SwiftPM instead of guessing.
BIN_DIR="$(swift build -c release --show-bin-path)"
BIN="$BIN_DIR/aura"
FW="$BIN_DIR/llama.framework"
[ -x "$BIN" ] || { echo "error: aura binary not found at $BIN"; exit 1; }
[ -d "$FW" ]  || { echo "error: llama.framework not found at $FW"; exit 1; }

echo "==> Staging (aura + llama.framework + resource bundles, co-located)…"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$BIN" "$STAGE/aura"
cp -R "$FW" "$STAGE/llama.framework"
for bundle in "$BIN_DIR"/*.bundle; do
  case "$bundle" in *Tests.bundle) continue ;; esac
  [ -d "$bundle" ] && cp -R "$bundle" "$STAGE/"
done

echo "==> Staging license texts…"
cp LICENSE THIRD_PARTY_NOTICES.md "$STAGE/"
for checkout in .build/checkouts/*; do
  name="$(basename "$checkout")"
  for text in "$checkout"/LICENSE* "$checkout"/LICENCE* "$checkout"/NOTICE* "$checkout"/COPYING*; do
    [ -f "$text" ] || continue
    mkdir -p "$STAGE/licenses/$name"
    cp "$text" "$STAGE/licenses/$name/"
  done
done

echo "==> Verifying the staged binary runs standalone…"
( cd "$STAGE" && ./aura tools >/dev/null ) && echo "    ok — runs without the source tree"

echo "==> Creating tarball…"
rm -f "$OUT"
( cd "$STAGE" && tar -czf "$OUT" * )

echo "==> Done: $OUT"
echo "    size:   $(du -h "$OUT" | awk '{print $1}')"
echo "    sha256: $(shasum -a 256 "$OUT" | awk '{print $1}')"
echo "Put that sha256 into Formula/aura.rb before publishing the tap."
