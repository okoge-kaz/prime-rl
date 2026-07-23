#!/bin/bash
# Build a CUDA 13.0 prime-rl sqsh with disaggregation via pyxis --container-save.
#
# B300 (sm_103) requires CUDA >=12.9, while the r580 driver caps the toolkit at 13.0.
#
# DeepEP, DeepGEMM, and flash-attn are built from source for CUDA 13.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTAINERS_DIR="/lustre/fsw/portfolios/coreai/users/kfujii/containers"
CACHE_DIR="${CONTAINERS_DIR}/cache"
# v2: DeepEP device-link fix and GDRCopy userland.
# v3: NIXL meta shim fix for job 173696.
# v4: NIXL LIBFABRIC plugin for native EFA KV transfer.
SQSH_FILE="${CONTAINERS_DIR}/prime-rl-v0.7.0-cu13-disagg-v4.sqsh"
BASE_IMAGE="docker://nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04"

if [ -f "${SQSH_FILE}" ]; then
    echo "already exists: ${SQSH_FILE}" >&2
    exit 1
fi

mkdir -p "${CONTAINERS_DIR}" "${CACHE_DIR}/uv"

cd "${REPO_ROOT}"
git submodule update --init --recursive

# Build on a GPU node so the final DeepEP/DeepGEMM import checks have a driver.
srun --account=coreai_horizon_dilations \
    --partition=batch \
    --qos=interactive \
    --job-name=prime-rl-sqsh-build-cu13 \
    --time=02:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=8 \
    --container-image="${BASE_IMAGE}" \
    --container-remap-root \
    --container-writable \
    --container-mounts="${REPO_ROOT}:/workdir,${CACHE_DIR}/uv:/uv-cache" \
    --container-save="${SQSH_FILE}" \
    bash /workdir/experiments/enroot/build_in_container_cuda13.sh \
    || { echo "build failed — removing partial sqsh" >&2; rm -f "${SQSH_FILE}"; exit 1; }

echo "created: ${SQSH_FILE}"
