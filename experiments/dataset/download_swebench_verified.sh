#!/bin/bash
# eval 用 taskset (swebench-verified-v1) の harbor データを事前ダウンロードする。
#
# harbor taskset の cache パスは ~/.cache/harbor にハードコードされている
# (deps/verifiers/verifiers/v1/tasksets/harbor/taskset.py の CACHE 定数、環境変数なし)。
# home quota を避けるため、実体を lustre に置いて ~/.cache/harbor から symlink する。
#
# cache ディレクトリ名は taskset.py の cache_dir() と同じ規則:
#   dataset id の "/" を "_" に置換 (selector が dataset のみなので digest なし)
#
# 実行例 (login node では実行しない):
#   srun -A coreai_horizon_dilations -p cpu_datamover -t 1:00:00 \
#     bash experiments/dataset/download_swebench_verified.sh

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

DATASET="swe-bench/swe-bench-verified"
CACHE_NAME="${DATASET//\//_}"

# ~/.cache/harbor -> lustre の symlink を張る
mkdir -p "${HOME}/.cache"
if [ -e "${HOME}/.cache/harbor" ] && [ ! -L "${HOME}/.cache/harbor" ]; then
    echo "ERROR: ${HOME}/.cache/harbor が実ディレクトリとして存在します。" >&2
    echo "中身を ${HARBOR_CACHE_DIR} に移してから削除してください。" >&2
    exit 1
fi
ln -sfn "${HARBOR_CACHE_DIR}" "${HOME}/.cache/harbor"

if [ -d "${HARBOR_CACHE_DIR}/${CACHE_NAME}" ]; then
    echo "already cached: ${HARBOR_CACHE_DIR}/${CACHE_NAME}"
    exit 0
fi

# taskset.py の dataset_dir() と同様に、temp に export してから rename する
# (失敗時に不完全な cache を残さないため)
cd "${REPO_ROOT}"
TMP_EXPORT="$(mktemp -d --tmpdir="${HARBOR_CACHE_DIR}")"
trap 'rm -rf "${TMP_EXPORT}"' EXIT
uv run --no-sync harbor download "${DATASET}" --export -o "${TMP_EXPORT}/export"
mv "${TMP_EXPORT}/export" "${HARBOR_CACHE_DIR}/${CACHE_NAME}"

echo "done: ${DATASET} -> ${HARBOR_CACHE_DIR}/${CACHE_NAME}"
