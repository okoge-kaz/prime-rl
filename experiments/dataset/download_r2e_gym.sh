#!/bin/bash
# SWE RL の学習 taskset (r2e-gym-v1) が使う HF dataset を事前ダウンロードする。
# load_dataset まで実行するので、生ファイル (hub cache) と処理済み arrow
# (HF_DATASETS_CACHE) の両方がランタイムでオフラインヒットする。
#
# デフォルトは r2e-gym-v1 の DATASET 定数と同じ PrimeIntellect/R2E-Gym-Subset-Verified。
# 他の互換 dataset (R2E-Gym/R2E-Gym-Subset, -RL, -SFT) は引数で指定。
#
# 実行例 (login node では実行しない):
#   srun -A coreai_horizon_dilations -p cpu_datamover -t 1:00:00 \
#     bash experiments/dataset/download_r2e_gym.sh

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
