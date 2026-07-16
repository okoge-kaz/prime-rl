#!/bin/bash
# math RL 学習用の INTELLECT-3-RL (math subset) を事前ダウンロードする。
# i3_math_v1 環境 (deps/research-environments/environments/math/i3_math_v1) は
# load_dataset("PrimeIntellect/INTELLECT-3-RL", "math", split="train") を使う。
#
# 実行例 (login node では実行しない):
#   srun -A coreai_horizon_dilations -p cpu_datamover -t 1:00:00 \
#     bash experiments/dataset/download_i3_math.sh

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

cd "${REPO_ROOT}"
uv run --no-sync python - <<'EOF'
from datasets import load_dataset

d = load_dataset("PrimeIntellect/INTELLECT-3-RL", "math", split="train")
print(f"PrimeIntellect/INTELLECT-3-RL [math/train]: {len(d)} rows")
EOF

echo "done: PrimeIntellect/INTELLECT-3-RL (math) -> ${HF_DATASETS_CACHE}"
