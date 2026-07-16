#!/bin/bash
# eval 用の AIME2025 dataset を事前ダウンロードする。
# aime25-v1 環境 (deps/research-environments/environments/math/aime25_v1) は
# load_dataset("opencompass/AIME2025", <subset>, split="test", revision=<pin>) を使う。
# revision pin は taskset 側 (AIME25Config.dataset_revision) と揃えること。
#
# 実行例 (login node では実行しない):
#   srun -A coreai_horizon_dilations -p cpu_datamover -t 0:30:00 \
#     bash experiments/dataset/download_aime2025.sh

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

cd "${REPO_ROOT}"
uv run --no-sync python - <<'EOF'
from datasets import load_dataset

REVISION = "a6ad95f611d72cf628a80b58bd0432ef6638f958"  # aime25_v1 の AIME25Config.dataset_revision と同じ pin
for subset in ("AIME2025-I", "AIME2025-II"):
    d = load_dataset("opencompass/AIME2025", subset, split="test", revision=REVISION)
    print(f"opencompass/AIME2025 [{subset}/test]: {len(d)} rows")
EOF

echo "done: opencompass/AIME2025 -> ${HF_DATASETS_CACHE}"
