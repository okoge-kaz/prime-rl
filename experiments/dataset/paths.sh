#!/bin/bash
# Shared cache paths and .env loading for dataset download scripts.
#
#   hf_checkpoints/hub : Hugging Face model and raw dataset cache
#   datasets/          : processed Arrow cache
#   cache/harbor       : Harbor taskset cache
#   checkpoints/       : training checkpoints

LUSTRE_USER_DIR="/lustre/fsw/portfolios/coreai/users/kfujii"

export HF_HUB_CACHE="${LUSTRE_USER_DIR}/hf_checkpoints/hub"
export HF_DATASETS_CACHE="${LUSTRE_USER_DIR}/datasets"
export HARBOR_CACHE_DIR="${LUSTRE_USER_DIR}/cache/harbor"
export TRAIN_OUTPUT_DIR="${LUSTRE_USER_DIR}/checkpoints"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${HARBOR_CACHE_DIR}" "${TRAIN_OUTPUT_DIR}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT

if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
