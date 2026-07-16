#!/bin/bash
# Dockerfile.cuda 相当のビルドを enroot コンテナ内で実行する (build_sqsh.sh から起動される)
#
# Dockerfile.cuda (INCLUDE_DISAGG_EXTRAS=1, INCLUDE_NIXL_FROM_SOURCE=1) との差分:
#   - multi-stage の CUDA toolkit トリムなし: devel イメージのまま保存するので
#     sqsh が数 GB 大きい (nvcc は tilelang / deep-gemm の実行時 JIT に必要なので残す)
#   - appuser 作成なし: enroot/pyxis はコンテナを起動ユーザーで実行するため不要
#   - deep-gemm 用 CUDA 13 ランタイムライブラリは docker COPY の代わりに apt で導入
#
# 前提 mounts (build_sqsh.sh が設定):
#   /workdir  = prime-rl リポジトリ (submodule 初期化済み)
#   /uv-cache = uv キャッシュ (lustre 上、ビルド間で再利用され sqsh には含まれない)

set -euxo pipefail

# ENROOT_MOUNT_HOME=yes でユーザーの home がマウントされるため、
# キャッシュ類が home (quota が厳しい) に書かれないよう HOME を退避
export HOME=/root

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:${PATH}

# ── apt (Dockerfile の builder + runtime 両ステージ分をまとめて) ──
# 差分: openssh-server は含めない。enroot がホストの /etc/ssh/ssh_config を
# bind-mount するため openssh-client の conffile 更新が dpkg で失敗する
# (Device or resource busy)。sshd は SLURM/pyxis 運用では不要。
# --force-conf{def,old} は同種の conffile 置き換え失敗の保険 (既存 config を維持)。
apt-get update
apt-get install -y --no-install-recommends \
    -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
    build-essential autoconf automake libtool pkg-config ca-certificates curl sudo git ninja-build \
    libnuma-dev libnl-3-dev libnl-route-3-dev libibverbs-dev librdmacm-dev \
    python3.12 python3.12-venv python3.12-dev \
    wget clang tmux iperf git-lfs gpg iputils-ping net-tools vim \
    ibverbs-providers
ln -sf /usr/bin/python3.12 /usr/local/bin/python

# deep-gemm 用 CUDA 13 ランタイムライブラリ
# (Dockerfile は nvidia/cuda:13.0.1-runtime イメージから COPY しているが、
#  base イメージに CUDA の apt repo が入っているので apt で同等物を導入する)
apt-get install -y --no-install-recommends \
    -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
    cuda-cudart-13-0 cuda-nvrtc-13-0
echo /usr/local/cuda-13.0/targets/x86_64-linux/lib > /etc/ld.so.conf.d/cuda13.conf
ldconfig
ldconfig -p | grep -q libcudart.so.13
ldconfig -p | grep -q libnvrtc.so.13
# cuda-cudart-13-0 のインストールが /usr/local/cuda の symlink を cuda-13.0 に
# 付け替えてしまう (CUDA 13 側には nvcc もヘッダもない) ので 12.8 に戻す
ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
test -x /usr/local/cuda/bin/nvcc
test -f /usr/local/cuda/include/cuda.h

# ── UCX 1.19.1 → /opt/ucx (Dockerfile builder と同一手順) ──
UCX_VERSION=1.19.1
git clone --depth 1 --branch "v${UCX_VERSION}" https://github.com/openucx/ucx.git /tmp/ucx
cd /tmp/ucx
./autogen.sh
./configure --prefix=/opt/ucx --enable-shared --disable-static --disable-doxygen-doc \
    --enable-optimizations --enable-cma --enable-devel-headers --enable-mt \
    --with-verbs --with-cuda=/usr/local/cuda --with-ze=no
make -j"$(nproc)"
make install
rm -rf /tmp/ucx

# ── uv ──
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-installer.sh
INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR=/usr/local/bin sh /tmp/uv-installer.sh
rm /tmp/uv-installer.sh

# ── ソースを /app へ (Dockerfile の COPY 相当) ──
mkdir -p /app/benchmarks
cd /workdir
cp -a pyproject.toml uv.lock README.md src packages deps configs examples scripts /app/
cp -a benchmarks/scripts /app/benchmarks/scripts

# submodule (verifiers / renderers) のバージョンを /workdir の git metadata から導出し、
# git のない /app 側では hatch の fallback-version として埋め込む
eval "$(bash /workdir/scripts/docker-editable-pretend-versions.sh --shell /workdir)"
sed -i "s/fallback-version = \"0.0.0\"/fallback-version = \"${VERIFIERS_PRETEND_VERSION}\"/" /app/deps/verifiers/pyproject.toml
sed -i "s/fallback-version = \"0.0.0\"/fallback-version = \"${RENDERERS_PRETEND_VERSION}\"/" /app/deps/renderers/pyproject.toml

# ── 依存インストール (Dockerfile と同一の uv sync、disagg extra 有効) ──
export UV_PYTHON_PREFERENCE=only-system
export UV_PROJECT_ENVIRONMENT=/app/.venv
export UV_CACHE_DIR=/uv-cache
export UV_COMPILE_BYTECODE=1
export UV_LINK_MODE=copy

cd /app
uv sync --all-packages \
    --extra flash-attn --extra flash-attn-3 --extra flash-attn-cute \
    --extra gpt-oss --extra modelexpress --extra disagg \
    --group mamba-ssm --locked --no-dev

# ── NIXL を /opt/ucx (UCX 1.19) に対してソースビルド (#2883 対策) ──
# nixl の meson は PATH 上に uv があると付属の "nixl-meta" wheel を `uv build` で
# 作ろうとするが、これが Python 3.13 を要求して UV_PYTHON_PREFERENCE=only-system
# (システムは 3.12) の下で失敗する。meta wheel (nixl → nixl-cu12 への依存を張る
# だけの空パッケージ) は不要なので、build の間だけ uv を PATH から隠して meson に
# スキップさせる (find_program('uv', required: false) なので警告のみで通る)。
NIXL_VERSION=0.10.1
VIRTUAL_ENV=/app/.venv uv pip install pip
git clone --depth 1 --branch "${NIXL_VERSION}" https://github.com/ai-dynamo/nixl.git /tmp/nixl
cd /tmp/nixl
for uv_bin in /usr/local/bin/uv /root/.local/bin/uv; do
    if [ -e "${uv_bin}" ]; then mv "${uv_bin}" "${uv_bin}.hidden"; fi
done
PKG_CONFIG_PATH=/opt/ucx/lib/pkgconfig \
    LD_LIBRARY_PATH=/opt/ucx/lib:/opt/ucx/lib/ucx \
    /app/.venv/bin/python -m pip wheel . --no-deps --wheel-dir=/app/deps
for uv_bin in /usr/local/bin/uv /root/.local/bin/uv; do
    if [ -e "${uv_bin}.hidden" ]; then mv "${uv_bin}.hidden" "${uv_bin}"; fi
done
VIRTUAL_ENV=/app/.venv uv pip install --reinstall --no-deps /app/deps/nixl_cu12-*.whl
rm -rf /tmp/nixl /app/deps/nixl_cu12-*.whl

# ── 実行時環境変数 (enroot は起動時にコンテナ内の /etc/environment を反映する) ──
cat >> /etc/environment <<'EOF'
LC_ALL=en_US.UTF-8
CUDA_HOME=/usr/local/cuda
UCX_HOME=/opt/ucx
LD_LIBRARY_PATH=/opt/ucx/lib:/opt/ucx/lib/ucx
PATH=/app/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HF_HUB_ETAG_TIMEOUT=500
HF_HUB_DOWNLOAD_TIMEOUT=300
EOF

# ── sqsh を小さく保つための掃除 ──
apt-get clean autoclean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

echo "container build done"
