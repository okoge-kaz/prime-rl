#!/bin/bash
# 学習開始モデル (Qwen3-30B-A3B-Thinking-2507, ~60GB) を HF hub cache に事前ダウンロードする。
# cache 形式で置くので、config の model.name = "Qwen/Qwen3-30B-A3B-Thinking-2507" のまま
# ランタイムがオフラインヒットする。
#
# 実行例 (login node では実行しない):
#   srun -A coreai_horizon_dilations -p cpu_datamover -t 4:00:00 \
#     bash experiments/dataset/download_model.sh

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

MODEL="${1:-Qwen/Qwen3-30B-A3B-Thinking-2507}"

cd "${REPO_ROOT}"
uv run --no-sync hf download "${MODEL}"

echo "done: ${MODEL} -> ${HF_HUB_CACHE}"
