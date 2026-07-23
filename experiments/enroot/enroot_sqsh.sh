#!/bin/bash
# Import a Docker Hub image into an Enroot sqsh.
#
# Official images do not include PD-disaggregation extras. Override IMAGE when
# importing a non-disaggregated official image.
#   https://hub.docker.com/r/primeintellect/prime-rl/tags
#   - commit-d334ea5 is the amd64 v0.7.0 release image.
#   - Multi-architecture images are hosted on GHCR.

set -euo pipefail

TAG="v0.7.0-disagg"
IMAGE="docker://kazukifujii00/prime-rl:${TAG}"
SQSH_DIR="/lustre/fsw/portfolios/coreai/users/kfujii/containers"
SQSH_FILE="${SQSH_DIR}/prime-rl-${TAG}.sqsh"

# Keep Enroot cache and temporary data on Lustre.
CACHE_DIR="${SQSH_DIR}/cache"
export ENROOT_CACHE_PATH="${CACHE_DIR}/enroot"
export ENROOT_TEMP_PATH="${CACHE_DIR}/tmp"
mkdir -p "${SQSH_DIR}" "${ENROOT_CACHE_PATH}" "${ENROOT_TEMP_PATH}"

# For a private Docker Hub repository, add credentials to:
#   machine auth.docker.io login kazukifujii00 password <access-token>

if [ -f "${SQSH_FILE}" ]; then
    echo "already exists: ${SQSH_FILE}" >&2
    exit 1
fi

enroot import -o "${SQSH_FILE}" "${IMAGE}"

echo "created: ${SQSH_FILE}"
