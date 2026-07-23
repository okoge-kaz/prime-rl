#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

DATASET="${1:-PrimeIntellect/R2E-Gym-Subset-Verified}"

cd "${REPO_ROOT}"
uv run --no-sync python - "${DATASET}" <<'EOF'
import sys
from datasets import load_dataset

name = sys.argv[1]
ds = load_dataset(name)
for split, d in ds.items():
    print(f"{name} [{split}]: {len(d)} rows")
EOF

echo "done: ${DATASET} -> ${HF_DATASETS_CACHE}"
