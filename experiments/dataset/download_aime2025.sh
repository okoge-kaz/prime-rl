#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

cd "${REPO_ROOT}"
uv run --no-sync python - <<'EOF'
from datasets import load_dataset

REVISION = "a6ad95f611d72cf628a80b58bd0432ef6638f958"  # AIME25Config.dataset_revision
for subset in ("AIME2025-I", "AIME2025-II"):
    d = load_dataset("opencompass/AIME2025", subset, split="test", revision=REVISION)
    print(f"opencompass/AIME2025 [{subset}/test]: {len(d)} rows")
EOF

echo "done: opencompass/AIME2025 -> ${HF_DATASETS_CACHE}"
