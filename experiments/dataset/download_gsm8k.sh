#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

cd "${REPO_ROOT}"
uv run --no-sync python - <<'EOF'
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")
for split, d in ds.items():
    print(f"openai/gsm8k [{split}]: {len(d)} rows")
EOF

echo "done: openai/gsm8k -> ${HF_DATASETS_CACHE}"
