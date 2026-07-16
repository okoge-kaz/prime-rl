#!/bin/bash
# pyxis の --container-save で prime-rl (PD disaggregation 入り) の sqsh を直接 build する
#
# 背景: rootless podman はこのクラスタでは使えない (/etc/subuid が未設定のため
# コンテナ内で root 以外の UID に遷移できず、apt が _apt への権限降格で死ぬ)。
# enroot は seccomp で setuid 系システムコールを偽装するため、enroot コンテナ内では
# apt もソースビルドもそのまま動く。これを利用して Dockerfile.cuda 相当のビルドを
# compute node (cpu partition) 上のコンテナ内で実行し、ジョブ終了時に pyxis が
# コンテナの状態を sqsh として保存する。
#
# - build は srun 経由で compute node 上で走る (login node では実行されない)
# - この方式では Docker Hub への push はできない (成果物は sqsh のみ)
# - 公開イメージ (docker.io/primeintellect/prime-rl, disagg なし) の import は
#   enroot_sqsh.sh を参照

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
