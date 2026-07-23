#!/bin/bash
# Build a prime-rl sqsh with PD disaggregation via pyxis --container-save.
#
# Rootless podman cannot switch UIDs on this cluster because /etc/subuid is not
# configured. Enroot's seccomp emulation allows apt and source builds, and pyxis
# saves the resulting container state as sqsh.
#
# The build runs through srun on a compute node and produces only a local sqsh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTAINERS_DIR="/lustre/fsw/portfolios/coreai/users/kfujii/containers"
CACHE_DIR="${CONTAINERS_DIR}/cache"
SQSH_FILE="${CONTAINERS_DIR}/prime-rl-v0.7.0-disagg.sqsh"
BASE_IMAGE="docker://nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04"

if [ -f "${SQSH_FILE}" ]; then
    echo "already exists: ${SQSH_FILE}" >&2
    exit 1
fi

mkdir -p "${CONTAINERS_DIR}" "${CACHE_DIR}/uv"

cd "${REPO_ROOT}"
git submodule update --init --recursive

srun --account=coreai_horizon_dilations \
    --partition=cpu \
    --job-name=prime-rl-sqsh-build \
    --time=6:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=200G \
    --container-image="${BASE_IMAGE}" \
    --container-remap-root \
    --container-writable \
    --container-mounts="${REPO_ROOT}:/workdir,${CACHE_DIR}/uv:/uv-cache" \
    --container-save="${SQSH_FILE}" \
    bash /workdir/experiments/enroot/build_in_container.sh \
    || { echo "build failed — removing partial sqsh" >&2; rm -f "${SQSH_FILE}"; exit 1; }

echo "created: ${SQSH_FILE}"
