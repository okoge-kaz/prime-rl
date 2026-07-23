#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

cd "${REPO_ROOT}"
uv run --no-sync python - <<'EOF'
from datasets import load_dataset

d = load_dataset("PrimeIntellect/INTELLECT-3-RL", "math", split="train")
print(f"PrimeIntellect/INTELLECT-3-RL [math/train]: {len(d)} rows")
EOF

echo "done: PrimeIntellect/INTELLECT-3-RL (math) -> ${HF_DATASETS_CACHE}"
