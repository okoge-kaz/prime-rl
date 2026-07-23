#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

MODELS=(
    "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    "PrimeIntellect/INTELLECT-3"
    "PrimeIntellect/MiniMax-M2.5-bf16"
    "Qwen/Qwen3-Coder-480B-A35B-Instruct"
)
if [ $# -gt 0 ]; then
    MODELS=("$@")
fi

cd "${REPO_ROOT}"
for model in "${MODELS[@]}"; do
    echo "=== downloading: ${model} ==="
    uv run --no-sync hf download "${model}"
    echo "done: ${model} -> ${HF_HUB_CACHE}"
done

echo "all done (${#MODELS[@]} models)"
