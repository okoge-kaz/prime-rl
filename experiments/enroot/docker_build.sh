#!/bin/bash
# prime-rl の container image を podman (rootless) で build し、Docker Hub に push する
#
# このクラスタに docker daemon はないが、login node に podman があるので自前 build できる。
#
# 公式のビルド済みイメージは公開 Docker Hub に push されている:
#   https://hub.docker.com/r/primeintellect/prime-rl/tags
#   - commit-d334ea5 = v0.7.0 リリースコミット (d334ea529)。ただし amd64 のみ
#   - amd64 / arm64 の両対応イメージは GHCR にある:
#     https://github.com/PrimeIntellect-ai/prime-rl/pkgs/container/prime-rl
#     (ghcr.io/primeintellect-ai/prime-rl) — ただし 2026-07-15 時点で v0.7.0 は未公開
#   - いずれも PD disaggregation 用の extras (deep-ep / deep-gemm / nixl) を含まない
#     (CI は --build-arg を渡さないため INCLUDE_DISAGG_EXTRAS=0 のデフォルトで build される)
#
# そのため PD disaggregation を使う実験では INCLUDE_DISAGG_EXTRAS=1 で自前 build する。
# なお disagg extras は prime-rl の GitHub releases にあるプリビルト wheel なので、
# build に nvcc は不要 (pyproject.toml の [tool.uv.sources] を参照)。
#
# nixl について (#2883): PyPI の nixl-cu12 wheel は同梱 UCX が prefill→decode の
# KV 転送で segfault する既知バグがあり、UCX 1.19 に対する再 build が必須
# (docs/advanced.md, docs/inference.md)。lock がピンする prime-rl releases の
# nixl_cu12-0.10.1 wheel は調査の結果、UCX 非同梱・外部 UCX 1.19 に動的リンクする
# 再 build 品 (対策済み) だったが、開発者のローカル環境で build されたものなので、
# INCLUDE_NIXL_FROM_SOURCE=1 でイメージ同梱の UCX 1.19.1 (/opt/ucx) に対して
# build し直すのが最も確実。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HUB_REPO="docker.io/kazukifujii00/prime-rl"
TAG="v0.7.0-disagg"

# podman のイメージストレージ:
# lustre は xattr (lsetxattr) 非対応で overlay driver が使えないため、home でも
# lustre でもなく /tmp (tmpfs, ~186G) に置く。build 結果は Docker Hub に push する
# ので、storage がノード再起動で消えても実害はない (build cache が消えるだけ)。
CACHE_DIR="/lustre/fsw/portfolios/coreai/users/kfujii/containers/cache"
STORAGE_ROOT="/tmp/${USER}-podman-storage"
export TMPDIR="/tmp/${USER}-podman-tmp"
mkdir -p "${CACHE_DIR}" "${STORAGE_ROOT}" "${TMPDIR}"

# rootless podman 対策 (このクラスタの設定に起因):
# - /etc/containers/registries.conf に unqualified-search registry が未定義のため、
#   Dockerfile の FROM nvidia/cuda:... のような short-name が解決できない。
#   user 側の registries.conf で docker.io を検索対象にする。
# - /etc/subuid に kfujii の subuid range がなく single-mapping になるため、
#   イメージ内の useradd / chown (appuser 作成など) が失敗し得る。
#   ignore_chown_errors=true で所有権変更の失敗を無視する
#   (enroot は実行時に所有権を潰して起動ユーザーで動かすので実害なし)。
REGISTRIES_CONF="${CACHE_DIR}/registries.conf"
[ -f "${REGISTRIES_CONF}" ] || printf 'unqualified-search-registries = ["docker.io"]\n' > "${REGISTRIES_CONF}"
export CONTAINERS_REGISTRIES_CONF="${REGISTRIES_CONF}"

PODMAN=(podman --root "${STORAGE_ROOT}" --storage-opt ignore_chown_errors=true)

# build 時の WARN について:
# - "missing VERIFIERS_PRETEND_VERSION / RENDERERS_PRETEND_VERSION" は無害。
#   公式 CI も渡しておらず、build context の .git/modules から
#   scripts/docker-editable-pretend-versions.sh がバージョンを導出する。
#   (submodule が git metadata 込みで初期化されていることが前提)

cd "${REPO_ROOT}"

# submodule (deps/) が初期化済みであることが前提
git submodule update --init --recursive

"${PODMAN[@]}" build -f Dockerfile.cuda \
    --build-arg INCLUDE_DISAGG_EXTRAS=1 \
    --build-arg INCLUDE_NIXL_FROM_SOURCE=1 \
    -t "${HUB_REPO}:${TAG}" .

# Docker Hub (kazukifujii00) へ push (未ログインなら対話ログインが走る)
"${PODMAN[@]}" login --get-login docker.io > /dev/null 2>&1 \
    || "${PODMAN[@]}" login docker.io --username kazukifujii00
"${PODMAN[@]}" push "${HUB_REPO}:${TAG}"

# --- 参考: docker が使えるマシンでの同等コマンド ---
# docker build -f Dockerfile.cuda --build-arg INCLUDE_DISAGG_EXTRAS=1 --build-arg INCLUDE_NIXL_FROM_SOURCE=1 -t kazukifujii00/prime-rl:v0.7.0-disagg .
# docker push kazukifujii00/prime-rl:v0.7.0-disagg
