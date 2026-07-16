#!/bin/bash
# Dockerfile.cuda13 相当のビルドを enroot コンテナ内で実行する (build_sqsh_cuda13.sh から起動される)
# disagg (deep-ep / deep-gemm / nixl / vllm-router) を含む。
#
# Dockerfile.cuda13 との差分:
#   - multi-stage の CUDA toolkit トリムなし: devel イメージのまま保存するので
#     sqsh が数 GB 大きい (nvcc は tilelang / deep-gemm の実行時 JIT に必要なので残す)
#   - appuser 作成なし: enroot/pyxis はコンテナを起動ユーザーで実行するため不要
#
# build_in_container.sh (cu128 版) との差分:
#   - ベースは nvidia/cuda:13.0.3-cudnn-devel (build_sqsh_cuda13.sh が指定)。
#     toolkit 自体が 13.0 なので、deep-gemm 用 CUDA 13 ランタイム同居ハック
#     (apt 導入 + /usr/local/cuda symlink 巻き戻し) は不要
#   - pyproject.toml を scripts/pyproject-cuda13-patch.sh で cu130 化し、
#     uv sync の前に uv lock で再解決 (COPY 済みの cu128 lock がシードになり
#     無関係な pin のドリフトを抑える)
#   - deep-ep / deep-gemm (cu12 prebuilt しかない) と flash-attn (prebuilt に
#     sm_103 対応コードが無い) は scripts/cuda13-build-wheels.sh でソースビルド
#
# 前提 mounts (build_sqsh_cuda13.sh が設定):
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

# toolkit が sm_103 対応の 13.0 であることを確認
test -x /usr/local/cuda/bin/nvcc
nvcc --version | grep -q "release 13.0"

# ── AWS EFA (libfabric + aws-ofi-nccl) ──
# この cluster (AWS, kernel *-aws) のノード間 RDMA は EFA (rdmap*, driver=efa)。
# NCCL から EFA を使うには aws-ofi-nccl プラグインが必須で、無いと NCCL は
# link_layer=InfiniBand に見える ibp19{8,9}s0f0 (ノード間疎通なし) を選んで
# IBV_WC_RETRY_EXC_ERR で hang する (2026-07-15 に全ノードペアで再現確認)。
#
# EFA installer 1.49.0 (2026-06-27, Ubuntu 24.04 対応) は libfabric を
# /opt/amazon/efa に、aws-ofi-nccl を /opt/amazon/ofi-nccl に同梱インストールする。
# 近年の aws-ofi-nccl は NCCL 2.28.9 でテスト済み・CUDA 13 対応・P6-B300 tuner
# 入り (v1.19+)。カーネルモジュールは host 側にあるため --skip-kmod で
# ユーザースペースのみ入れる。
# 動作確認: 実行時に NCCL_DEBUG=INFO で "NET/OFI Selected Provider is efa"。
EFA_INSTALLER_VERSION=1.49.0
# installer 内部の apt は force-conf オプション無しで openmpi40-aws → openssh-client を
# 入れようとするが、enroot が host の /etc/ssh/ssh_config を bind-mount しているため
# conffile 置き換えが "Device or resource busy" で dpkg ごと失敗する。先に force-conf
# 付きで openssh-client を入れておき、installer の apt に触らせない。
apt-get install -y --no-install-recommends \
    -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
    pciutils openssh-client
curl -fsSL "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
    | tar xz -C /tmp
(cd /tmp/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --mpi=openmpi4)
rm -rf /tmp/aws-efa-installer
# 同梱された plugin の存在確認 (無ければ build を落とす) とバージョン記録
test -e /opt/amazon/ofi-nccl/lib/libnccl-net.so || test -e /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
ls -la /opt/amazon/ofi-nccl/lib/
dpkg -l | grep -iE "efa|ofi" || true

# ── GDRCopy userland (libgdrapi) ──
# aws-ofi-nccl が dlopen("libgdrapi.so") する。無いと起動時に
# "NET/OFI Failed to initialize GDRCopy" 警告 + GIN (GPU-initiated networking)
# が無効化される。kernel module (gdrdrv) と /dev/gdrdrv は host 側にあり
# (2026-07-16 に pool0-0174 で確認済み)、コンテナには lib だけ入れればよい。
# lib のビルドに CUDA/カーネルヘッダは不要。version は host gdrdrv と同系の 2.x。
GDRCOPY_VERSION=2.5.1
git clone --depth 1 --branch "v${GDRCOPY_VERSION}" https://github.com/NVIDIA/gdrcopy /tmp/gdrcopy
make -C /tmp/gdrcopy lib lib_install prefix=/usr/local
ldconfig
test -e /usr/local/lib/libgdrapi.so
rm -rf /tmp/gdrcopy

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

# ── pyproject を CUDA 13 スタックへ書き換え ──
bash /app/scripts/pyproject-cuda13-patch.sh /app/pyproject.toml

# ── 依存インストール ──
export UV_PYTHON_PREFERENCE=only-system
export UV_PROJECT_ENVIRONMENT=/app/.venv
export UV_CACHE_DIR=/uv-cache
export UV_COMPILE_BYTECODE=1
export UV_LINK_MODE=copy

cd /app
uv lock
uv sync --all-packages \
    --extra flash-attn --extra flash-attn-3 --extra flash-attn-cute \
    --extra gpt-oss --extra modelexpress --extra disagg \
    --group mamba-ssm --locked --no-dev

# ── NIXL を /opt/ucx (UCX 1.19) に対してソースビルド (#2883 対策) ──
# nixl の meson は PATH 上に uv があると付属の "nixl-meta" wheel を `uv build` で
# 作ろうとするが、これが Python 3.13 を要求して UV_PYTHON_PREFERENCE=only-system
# (システムは 3.12) の下で失敗する。meta wheel は不要なので、build の間だけ uv を
# PATH から隠して meson にスキップさせる (find_program('uv', required: false))。
# CUDA 13 検出でパッケージ名は nixl_cu13 になる想定だが、glob は cu12/cu13 両対応。
NIXL_VERSION=0.10.1
VIRTUAL_ENV=/app/.venv uv pip install pip
git clone --depth 1 --branch "${NIXL_VERSION}" https://github.com/ai-dynamo/nixl.git /tmp/nixl
cd /tmp/nixl
for uv_bin in /usr/local/bin/uv /root/.local/bin/uv; do
    if [ -e "${uv_bin}" ]; then mv "${uv_bin}" "${uv_bin}.hidden"; fi
done
# libfabric (EFA installer 同梱) も検出させ、LIBFABRIC plugin を有効にする。
# EFA ネイティブの KV transfer (vLLM kv_connector_extra_config backends=["LIBFABRIC"])
# に必要。UCX backend は mlx5 前提で、この cluster では TCP フォールバックしかできない
LIBFABRIC_PC_DIR=$(dirname "$(find /opt/amazon/efa -name 'libfabric.pc' | head -1)")
test -n "$LIBFABRIC_PC_DIR"
PKG_CONFIG_PATH=/opt/ucx/lib/pkgconfig:"$LIBFABRIC_PC_DIR" \
    LD_LIBRARY_PATH=/opt/ucx/lib:/opt/ucx/lib/ucx:/opt/amazon/efa/lib \
    /app/.venv/bin/python -m pip wheel . --no-deps --wheel-dir=/app/deps
for uv_bin in /usr/local/bin/uv /root/.local/bin/uv; do
    if [ -e "${uv_bin}.hidden" ]; then mv "${uv_bin}.hidden" "${uv_bin}"; fi
done
VIRTUAL_ENV=/app/.venv uv pip install --reinstall --no-deps /app/deps/nixl_cu1*-*.whl
# cwd (/tmp/nixl) を消す前に退避 — cwd 消失後の uv は "Current directory does not exist" で死ぬ
cd /app
rm -rf /tmp/nixl /app/deps/nixl_cu1*-*.whl
# vLLM は `import nixl._api` を使う。この名前は meta package (nixl) の shim が
# 提供する (flavor の実体は nixl_cu13/)。meta が欠けると vLLM が実行時に
# "NIXL is not available" になる (job 173696) ので、ここで import を検証する。
LD_LIBRARY_PATH=/opt/ucx/lib:/opt/ucx/lib/ucx \
    /app/.venv/bin/python -c "import nixl._api as m; print('nixl._api:', m.__file__)"
# LIBFABRIC plugin がソースビルド wheel に入ったことを検証 (無ければ build を落とす)
find /app/.venv/lib/python3.12/site-packages -path '*mesonpy.libs/plugins/libplugin_LIBFABRIC.so' | grep -q . \
    || { echo "NIXL libfabric plugin missing — meson が libfabric を検出できていない" >&2; exit 1; }

# ── deep-ep / deep-gemm / flash-attn を CUDA 13 + torch cu130 (sm_103) でソースビルド ──
bash /app/scripts/cuda13-build-wheels.sh /app

# ── 実行時環境変数 (enroot は起動時にコンテナ内の /etc/environment を反映する) ──
cat >> /etc/environment <<'EOF'
LC_ALL=en_US.UTF-8
CUDA_HOME=/usr/local/cuda
UCX_HOME=/opt/ucx
LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/ucx/lib:/opt/ucx/lib/ucx
PATH=/app/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HF_HUB_ETAG_TIMEOUT=500
HF_HUB_DOWNLOAD_TIMEOUT=300
EOF

# ── sqsh を小さく保つための掃除 ──
apt-get clean autoclean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

echo "container build done"
