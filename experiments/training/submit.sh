#!/bin/bash
# Submit an RL training job. This is safe on a login node.
#
# Usage:
#   bash experiments/training/submit.sh math_qwen30b
#   bash experiments/training/submit.sh swe_qwen30b
#   bash experiments/training/submit.sh math_qwen30b --ckpt.resume-step -1
#
# A named variant overlays a TOML file in the run directory on rl.toml:
#   bash experiments/training/submit.sh swe_intellect3_disagg validation
#   bash experiments/training/submit.sh swe_intellect3_disagg long
#
# `rl --dry-run` resolves component configs and renders <output_dir>/rl.sbatch
# from templates/multi_node_rl_container.sbatch.j2. sbatch then runs every
# component in Pyxis containers on compute nodes.
#
# Prerequisites:
#   - WANDB, HF, and PRIME credentials are configured in .env.
#   - Models and datasets have been downloaded with experiments/dataset scripts.
#   - sqsh: /lustre/fsw/portfolios/coreai/users/kfujii/containers/prime-rl-v0.7.0-cu13-disagg-v4.sqsh
#     PRIME_RL_SQSH may override this template default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CKPT_BASE="/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints"

RUN_NAME="${1:?usage: submit.sh <run_name> [variant] [extra rl CLI args...]}"
shift
CONFIG="${SCRIPT_DIR}/${RUN_NAME}/rl.toml"
[ -f "${CONFIG}" ] || { echo "config not found: ${CONFIG}" >&2; exit 1; }

# Treat a second argument matching a TOML file as a configuration overlay.
VARIANT=""
CONFIG_ARGS=(@ "${CONFIG}")
if [ $# -gt 0 ] && [ -f "${SCRIPT_DIR}/${RUN_NAME}/$1.toml" ]; then
    VARIANT="$1"
    shift
    CONFIG_ARGS+=(@ "${SCRIPT_DIR}/${RUN_NAME}/${VARIANT}.toml")
fi

# Resolve uv from PATH or its standard per-user installation path.
UV_BIN="$(command -v uv || true)"
[ -n "${UV_BIN}" ] || UV_BIN="${HOME}/.local/bin/uv"
[ -x "${UV_BIN}" ] || { echo "uv not found in PATH or ${HOME}/.local/bin/uv" >&2; exit 1; }

# Keep logs, configs, rollouts, and job logs under the repository outputs
# directory. Compute nodes see the repository under /lustre, so paths are
# translated before submission. Large checkpoints use a separate CKPT_BASE
# path that remains stable across submissions for resume-step -1.
#
# Each submission gets outputs/<date>-<run_name>/. After sbatch assigns an ID,
# outputs/job-<job_id> points to that directory.
REPO_ROOT_LUSTRE="${REPO_ROOT/#\/scratch\/fsw/\/lustre\/fsw}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT_BASE="${REPO_ROOT_LUSTRE}/outputs"
RUN_DIR_NAME="${RUN_ID}-${RUN_NAME}${VARIANT:+-${VARIANT}}"
OUTPUT_DIR="${OUT_BASE}/${RUN_DIR_NAME}"
CKPT_DIR="${CKPT_BASE}/${RUN_NAME}${VARIANT:+-${VARIANT}}"

cd "${REPO_ROOT}"
set -a
# shellcheck disable=SC1091
source .env
set +a

"${UV_BIN}" run --no-sync rl "${CONFIG_ARGS[@]}" --output-dir "${OUTPUT_DIR}" --ckpt.output-dir "${CKPT_DIR}" --dry-run "$@"

JOB_ID=$(sbatch --parsable "${OUTPUT_DIR}/rl.sbatch")
ln -sfn "${RUN_DIR_NAME}" "${OUT_BASE}/job-${JOB_ID}"
ln -sfn "${RUN_DIR_NAME}" "${OUT_BASE}/${RUN_NAME}-latest"

echo "job id:     ${JOB_ID}"
echo "output dir: ${OUTPUT_DIR}"
echo "ckpt dir:   ${CKPT_DIR}"
echo "job log:    ${OUTPUT_DIR}/job_${JOB_ID}.log"
echo "logs:       ${OUTPUT_DIR}/logs/{trainer,orchestrator,inference}.log"
