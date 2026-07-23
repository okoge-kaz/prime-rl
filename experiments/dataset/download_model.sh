#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

MODEL="${1:-Qwen/Qwen3-30B-A3B-Thinking-2507}"

cd "${REPO_ROOT}"
uv run --no-sync hf download "${MODEL}"

echo "done: ${MODEL} -> ${HF_HUB_CACHE}"
