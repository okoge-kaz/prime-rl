#!/bin/bash
# math RL 動作確認用の gsm8k dataset を事前ダウンロードする。
# gsm8k-v1 環境 (deps/verifiers/environments/gsm8k_v1) は
# load_dataset("openai/gsm8k", "main") を使う。
#
# 実行例 (login node では実行しない):
#   srun -A coreai_horizon_dilations -p cpu_datamover -t 0:30:00 \
#     bash experiments/dataset/download_gsm8k.sh

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
