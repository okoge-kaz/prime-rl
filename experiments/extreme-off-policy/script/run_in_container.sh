#!/bin/bash
# Run extreme off-policy phases 2-4 in the training sqsh image, mounts, and /app/.venv
# convention as training.
#
# Invoke this wrapper directly; it calls srun and Pyxis itself.
#
# From an existing allocation:
#     bash experiments/extreme-off-policy/script/run_in_container.sh \
#         python experiments/extreme-off-policy/src/gen_drafts.py --ckpt ... --step 18 ...
#
# From a login node with a new allocation:
#     SRUN_ARGS="-A coreai_horizon_dilations -p interactive -N1 --gpus-per-node=8 -t 04:00:00" \
#         bash experiments/extreme-off-policy/script/run_in_container.sh \
#             python experiments/extreme-off-policy/src/...
#
# Set PRIME_RL_SQSH=<path>.sqsh to override the image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# Compute nodes use the /lustre path because /scratch is not mounted.
PROJECT_DIR="${PROJECT_DIR/#\/scratch\/fsw/\/lustre\/fsw}"

SQSH="${PRIME_RL_SQSH:-/lustre/fsw/portfolios/coreai/users/kfujii/containers/prime-rl-v0.7.0-cu13-disagg-v4.sqsh}"
FLASHINFER_CACHE=/lustre/fsw/portfolios/coreai/users/kfujii/cache/flashinfer
export VLLM_CACHE_ROOT=/lustre/fsw/portfolios/coreai/users/kfujii/cache/vllm
mkdir -p "$FLASHINFER_CACHE" "$VLLM_CACHE_ROOT"
MOUNTS="/lustre/fsw/portfolios/coreai/users/kfujii:/lustre/fsw/portfolios/coreai/users/kfujii"
MOUNTS="$MOUNTS,${PROJECT_DIR}:${PROJECT_DIR},${PROJECT_DIR}/src:/app/src"
MOUNTS="$MOUNTS,${PROJECT_DIR}/packages:/app/packages,${PROJECT_DIR}/deps:/app/deps"
MOUNTS="$MOUNTS,${FLASHINFER_CACHE}:/root/.cache/flashinfer"

# Export .env credentials into the container.
set -a
[ -f "${PROJECT_DIR}/.env" ] && source "${PROJECT_DIR}/.env"
set +a

# Use an overlapping srun from an existing task; otherwise SRUN_ARGS requests
# the allocation from the login node.
exec srun ${SRUN_ARGS:-} \
    --container-image="$SQSH" \
    --container-mounts="$MOUNTS" \
    bash -c 'cd "$0" && export VIRTUAL_ENV=/app/.venv PATH=/app/.venv/bin:$PATH && exec "$@"' \
    "$PROJECT_DIR" "$@"
