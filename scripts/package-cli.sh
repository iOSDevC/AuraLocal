#!/usr/bin/env bash
# Package the `aura` CLI + its llama.framework into a distributable tarball for a
# Homebrew formula (personal tap). No SwiftPM/network is needed at install time —
# the binary + framework are co-located so the binary's @loader_path rpath finds
# `@rpath/llama.framework` right next to it (no install_name_tool, no re-sign).
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

BIN="$ROOT/.build/release/aura"
FW="$ROOT/.build/${ARCH}-apple-macosx/release/llama.framework"
[ -x "$BIN" ] || { echo "error: aura binary not found at $BIN"; exit 1; }
[ -d "$FW" ]  || { echo "error: llama.framework not found at $FW"; exit 1; }

echo "==> Staging (aura + llama.framework, co-located)…"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$BIN" "$STAGE/aura"
cp -R "$FW" "$STAGE/llama.framework"

echo "==> Verifying the staged binary runs standalone…"
( cd "$STAGE" && ./aura tools >/dev/null ) && echo "    ok — runs without the source tree"

echo "==> Creating tarball…"
rm -f "$OUT"
tar -C "$STAGE" -czf "$OUT" aura llama.framework

echo "==> Done: $OUT"
echo "    size:   $(du -h "$OUT" | awk '{print $1}')"
echo "    sha256: $(shasum -a 256 "$OUT" | awk '{print $1}')"
echo "Put that sha256 into Formula/aura.rb before publishing the tap."
