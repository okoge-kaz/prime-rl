#!/bin/bash
# Build prime-rl with rootless podman and push it to Docker Hub.
#
# Official images:
#   https://hub.docker.com/r/primeintellect/prime-rl/tags
#   - commit-d334ea5 is the amd64 v0.7.0 release image.
#   - Multi-architecture images are on GHCR:
#     https://github.com/PrimeIntellect-ai/prime-rl/pkgs/container/prime-rl
#     ghcr.io/primeintellect-ai/prime-rl
#
# Official images omit DeepEP, DeepGEMM, and NIXL. Set
# INCLUDE_DISAGG_EXTRAS=1 for PD-disaggregated experiments.
#
# Build NIXL from source against the image's UCX 1.19.1 to avoid the bundled-UCX
# prefill-to-decode crash described in issue #2883.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HUB_REPO="docker.io/kazukifujii00/prime-rl"
TAG="v0.7.0-disagg"

# Lustre does not support the xattrs required by the overlay driver. Use /tmp
# for transient podman storage; the final image is pushed to Docker Hub.
CACHE_DIR="/lustre/fsw/portfolios/coreai/users/kfujii/containers/cache"
STORAGE_ROOT="/tmp/${USER}-podman-storage"
export TMPDIR="/tmp/${USER}-podman-tmp"
mkdir -p "${CACHE_DIR}" "${STORAGE_ROOT}" "${TMPDIR}"

# Configure docker.io for unqualified image names and tolerate chown failures
# caused by this cluster's single-ID rootless mapping.
REGISTRIES_CONF="${CACHE_DIR}/registries.conf"
[ -f "${REGISTRIES_CONF}" ] || printf 'unqualified-search-registries = ["docker.io"]\n' > "${REGISTRIES_CONF}"
export CONTAINERS_REGISTRIES_CONF="${REGISTRIES_CONF}"

PODMAN=(podman --root "${STORAGE_ROOT}" --storage-opt ignore_chown_errors=true)

# Missing pretend-version warnings are harmless when submodule git metadata is present.

cd "${REPO_ROOT}"

# Requires initialized deps/ submodules.
git submodule update --init --recursive

"${PODMAN[@]}" build -f Dockerfile.cuda \
    --build-arg INCLUDE_DISAGG_EXTRAS=1 \
    --build-arg INCLUDE_NIXL_FROM_SOURCE=1 \
    -t "${HUB_REPO}:${TAG}" .

# Push to Docker Hub; podman prompts if authentication is missing.
"${PODMAN[@]}" login --get-login docker.io > /dev/null 2>&1 \
    || "${PODMAN[@]}" login docker.io --username kazukifujii00
"${PODMAN[@]}" push "${HUB_REPO}:${TAG}"

# Equivalent commands on a host with Docker:
# docker build -f Dockerfile.cuda --build-arg INCLUDE_DISAGG_EXTRAS=1 --build-arg INCLUDE_NIXL_FROM_SOURCE=1 -t kazukifujii00/prime-rl:v0.7.0-disagg .
# docker push kazukifujii00/prime-rl:v0.7.0-disagg
