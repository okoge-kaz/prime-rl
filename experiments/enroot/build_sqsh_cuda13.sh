#!/bin/bash
# pyxis の --container-save で prime-rl の CUDA 13.0 版 sqsh (disagg 入り) を build する
# (build_sqsh.sh の cu13 変種。仕組みの背景コメントはそちらを参照)
#
# CUDA 13.0 を選ぶ理由:
#   - B300 (sm_103) は CUDA >= 12.9 が必要で、cu128 toolchain ではターゲットにできない
#   - ノードのドライバは r580 (= CUDA 13.0 対応上限) なので 13.1+ は使わない
#
# disagg (deep-ep / deep-gemm / nixl / vllm-router) 込み。deep-ep / deep-gemm は
# cu12 prebuilt wheel の代わりにコンテナ内で CUDA 13 に対してソースビルドする。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTAINERS_DIR="/lustre/fsw/portfolios/coreai/users/kfujii/containers"
CACHE_DIR="${CONTAINERS_DIR}/cache"
# v2 (2026-07-16): deep-ep の device-link 修正 (NVSHMEM 'arch' パス問題) + GDRCopy userland 追加
# v3 (2026-07-16): nixl meta shim 修正 (job 173696)。v2 は nixl 修正がスクリプトに
#   入る前にビルドされたため `import nixl` が無い
# v4 (2026-07-16): NIXL の LIBFABRIC plugin 有効化 (EFA ネイティブ KV transfer 用)
SQSH_FILE="${CONTAINERS_DIR}/prime-rl-v0.7.0-cu13-disagg-v4.sqsh"
BASE_IMAGE="docker://nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04"

if [ -f "${SQSH_FILE}" ]; then
    echo "already exists: ${SQSH_FILE}" >&2
    exit 1
fi

mkdir -p "${CONTAINERS_DIR}" "${CACHE_DIR}/uv"

cd "${REPO_ROOT}"
git submodule update --init --recursive

# GPU ノード (batch + interactive QoS) でビルドする: すぐ確保でき、ビルド末尾で
# deep_ep / deep_gemm の import 検証 (要ドライバ) までできる
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
