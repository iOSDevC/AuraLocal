#!/usr/bin/env bash
# Prints the sets in Sources/AuraCore/ModelCompatibility/PinnedRuntimes.swift from the pinned sources,
# so a pin bump can be diffed against the tables by hand.
#
# MLX: keys of `LLMTypeRegistry.shared` / `VLMTypeRegistry.shared` (the first `creators:` block of each
#      factory; the VLM file's second block lists processors). Needs `swift package resolve` first.
#      `mlxRopeModelTypes` has no generator: it lists registry entries whose model file hands the config's
#      rope scaling to `initializeRope` (MLXLMCommon/RoPEUtils.swift).
# llama.cpp: `LLM_ARCH_NAMES` of the build LocalLLMClient pins (`llamaVersion` in its Package.swift),
#      minus `clip` (a projector placeholder) and `(unknown)`.
#
# Usage: scripts/regen_pinned_runtimes.sh [llama.cpp build, default b8851]
set -euo pipefail

cd "$(dirname "$0")/.."
build="${1:-b8851}"
checkout=".build/checkouts/mlx-swift-lm/Libraries"

registry_keys() {
    awk '/creators: \[/ { inside = 1; next } inside && /^[[:space:]]*\]/ { exit } inside' "$1" \
        | grep -o '"[^"]*": create(' | cut -d'"' -f2 | sort -u | tr '\n' ' '
    echo
}

echo "== mlxLLMModelTypes"
registry_keys "$checkout/MLXLLM/LLMModelFactory.swift"
echo "== mlxVLMModelTypes"
registry_keys "$checkout/MLXVLM/VLMModelFactory.swift"
echo "== llamaCppArchitectures ($build)"
curl -fsSL "https://raw.githubusercontent.com/ggml-org/llama.cpp/$build/src/llama-arch.cpp" \
    | awk '/LLM_ARCH_NAMES = \{/,/^\};/' | grep -o '"[^"]*"' | tr -d '"' \
    | grep -v -e '^clip$' -e '^(unknown)$' | tr '\n' ' '
echo
