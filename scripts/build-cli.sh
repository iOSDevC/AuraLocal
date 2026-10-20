#!/usr/bin/env bash
# Build the `tonamox` integration CLI as a release binary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Building tonamox (release)…"
swift build -c release --product tonamox

BIN="$ROOT/.build/release/tonamox"
echo "==> Done: $BIN"
"$BIN" help
