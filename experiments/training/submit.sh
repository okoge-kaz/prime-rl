#!/bin/bash
# RL 学習ジョブの投入 (login node で実行 OK — config 解決と sbatch 投入のみで重い処理はしない)
#
# 使い方:
#   bash experiments/training/submit.sh math_qwen30b
#   bash experiments/training/submit.sh swe_qwen30b
#   bash experiments/training/submit.sh math_qwen30b --ckpt.resume-step -1   # 再開
#
# 流れ:
#   1. `rl --dry-run` が rl.toml を trainer/orchestrator/inference の subconfig に分割し、
#      コンテナ対応テンプレート (templates/multi_node_rl_container.sbatch.j2) から
#      <output_dir>/rl.sbatch を生成する
#   2. sbatch で投入 (実行はすべて compute node 上の pyxis コンテナ内)
#
# 前提:
#   - .env にキー設定済み (WANDB / HF / PRIME)
#   - モデルとデータセットは experiments/dataset/ のスクリプトで事前ダウンロード済み
#     (dry-run はモデルの pre-download をスキップするため、未ダウンロードだと
#     ジョブ側の初回ロードで HF から落とそうとする)
#   - sqsh: /lustre/fsw/portfolios/coreai/users/kfujii/containers/prime-rl-v0.7.0-cu13-disagg-v3.sqsh
#     (template のデフォルト。PRIME_RL_SQSH 環境変数で差し替え可能)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CKPT_BASE="/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints"

RUN_NAME="${1:?usage: submit.sh <run_name: math_qwen30b|swe_qwen30b> [extra rl CLI args...]}"
shift
CONFIG="${SCRIPT_DIR}/${RUN_NAME}/rl.toml"
[ -f "${CONFIG}" ] || { echo "config not found: ${CONFIG}" >&2; exit 1; }

# PATH に uv が無いシェルから呼ばれても動くようにフォールバック解決する
UV_BIN="$(command -v uv || true)"
[ -n "${UV_BIN}" ] || UV_BIN="${HOME}/.local/bin/uv"
[ -x "${UV_BIN}" ] || { echo "uv not found (PATH にも ${HOME}/.local/bin/uv にも無い)" >&2; exit 1; }

# logs / configs / rollouts / job_%j.log は repo の outputs/ 配下に出す (進捗確認用)。
# compute node からは repo が /lustre/... で見えるため、パスを変換して渡す。
# 大きい checkpoint だけ --ckpt.output-dir で CKPT_BASE に分離する
# (run_name 単位で固定なので、再投入しても --ckpt.resume-step -1 で継続できる)。
#
# サブミットごとに outputs/<日付>-<run_name>/ を切る (フラットな 1 階層)。
# job ID は sbatch が受理して初めて決まる (config にはパスを事前に焼き込む
# 必要がある) ため、投入後に outputs/job-<job_id> -> <日付>-<run_name> の
# symlink を張って job ID でも辿れるようにする。
REPO_ROOT_LUSTRE="${REPO_ROOT/#\/scratch\/fsw/\/lustre\/fsw}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT_BASE="${REPO_ROOT_LUSTRE}/outputs"
RUN_DIR_NAME="${RUN_ID}-${RUN_NAME}"
OUTPUT_DIR="${OUT_BASE}/${RUN_DIR_NAME}"
CKPT_DIR="${CKPT_BASE}/${RUN_NAME}"

cd "${REPO_ROOT}"
set -a
# shellcheck disable=SC1091
source .env
set +a

"${UV_BIN}" run --no-sync rl @ "${CONFIG}" --output-dir "${OUTPUT_DIR}" --ckpt.output-dir "${CKPT_DIR}" --dry-run "$@"

JOB_ID=$(sbatch --parsable "${OUTPUT_DIR}/rl.sbatch")
ln -sfn "${RUN_DIR_NAME}" "${OUT_BASE}/job-${JOB_ID}"
ln -sfn "${RUN_DIR_NAME}" "${OUT_BASE}/${RUN_NAME}-latest"

echo "job id:     ${JOB_ID}"
echo "output dir: ${OUTPUT_DIR}"
echo "ckpt dir:   ${CKPT_DIR}"
echo "job log:    ${OUTPUT_DIR}/job_${JOB_ID}.log"
echo "logs:       ${OUTPUT_DIR}/logs/{trainer,orchestrator,inference}.log"
