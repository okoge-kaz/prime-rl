#!/bin/bash
# Docker Hub のイメージから enroot sqsh を作成する
#
# 自前 build (docker_build.sh) した disagg 入りイメージを kazukifujii00/prime-rl から import する。
#
# 公式のビルド済みイメージも公開 Docker Hub にある:
#   https://hub.docker.com/r/primeintellect/prime-rl/tags
#   - commit-d334ea5 = v0.7.0 リリースコミット (d334ea529)。ただし amd64 のみ
#   - amd64 / arm64 の両対応イメージは GHCR (ghcr.io/primeintellect-ai/prime-rl) にあるが、
#     2026-07-15 時点で v0.7.0 は未公開
#   - PD disaggregation 用の extras (deep-ep / deep-gemm / nixl) を含まないため、
#     disagg 不要な用途なら IMAGE を docker://primeintellect/prime-rl:commit-d334ea5 に
#     差し替えればそのまま使える

set -euo pipefail

TAG="v0.7.0-disagg"
IMAGE="docker://kazukifujii00/prime-rl:${TAG}"
SQSH_DIR="/lustre/fsw/portfolios/coreai/users/kfujii/containers"
SQSH_FILE="${SQSH_DIR}/prime-rl-${TAG}.sqsh"

# home quota 対策: enroot の cache / temp を lustre に向ける
CACHE_DIR="${SQSH_DIR}/cache"
export ENROOT_CACHE_PATH="${CACHE_DIR}/enroot"
export ENROOT_TEMP_PATH="${CACHE_DIR}/tmp"
mkdir -p "${SQSH_DIR}" "${ENROOT_CACHE_PATH}" "${ENROOT_TEMP_PATH}"

# Docker Hub リポジトリが private の場合は ~/.config/enroot/.credentials に
#   machine auth.docker.io login kazukifujii00 password <access-token>
# を記載しておくこと

if [ -f "${SQSH_FILE}" ]; then
    echo "already exists: ${SQSH_FILE}" >&2
    exit 1
fi

enroot import -o "${SQSH_FILE}" "${IMAGE}"

echo "created: ${SQSH_FILE}"
