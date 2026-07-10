#!/usr/bin/env bash
# Build the `aura` integration CLI as a release binary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Building aura (release)…"
swift build -c release --product aura

BIN="$ROOT/.build/release/aura"
echo "==> Done: $BIN"
"$BIN" help
