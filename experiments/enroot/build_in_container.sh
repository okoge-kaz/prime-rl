#!/bin/bash
# Build the Dockerfile.cuda equivalent inside Enroot.
#
# Keep the devel toolkit for runtime JIT, omit appuser, and install the CUDA 13
# runtime through apt. Requires /workdir and /uv-cache mounts.

set -euxo pipefail

# Redirect caches away from the mounted, quota-limited user home.
export HOME=/root

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:${PATH}

# Install builder and runtime apt dependencies while preserving bind-mounted conffiles.
apt-get update
apt-get install -y --no-install-recommends \
    -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
    build-essential autoconf automake libtool pkg-config ca-certificates curl sudo git ninja-build \
    libnuma-dev libnl-3-dev libnl-route-3-dev libibverbs-dev librdmacm-dev \
    python3.12 python3.12-venv python3.12-dev \
    wget clang tmux iperf git-lfs gpg iputils-ping net-tools vim \
    ibverbs-providers
ln -sf /usr/bin/python3.12 /usr/local/bin/python

# Install the CUDA 13 runtime used by DeepGEMM.
apt-get install -y --no-install-recommends \
    -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
    cuda-cudart-13-0 cuda-nvrtc-13-0
echo /usr/local/cuda-13.0/targets/x86_64-linux/lib > /etc/ld.so.conf.d/cuda13.conf
ldconfig
ldconfig -p | grep -q libcudart.so.13
ldconfig -p | grep -q libnvrtc.so.13
# Restore the CUDA 12.8 toolkit symlink after installing the runtime-only CUDA 13 package.
ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
test -x /usr/local/cuda/bin/nvcc
test -f /usr/local/cuda/include/cuda.h

# Build UCX 1.19.1 under /opt/ucx.
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

# Copy the source tree to /app.
mkdir -p /app/benchmarks
cd /workdir
cp -a pyproject.toml uv.lock README.md src packages deps configs examples scripts /app/
cp -a benchmarks/scripts /app/benchmarks/scripts

# Derive dependency versions from /workdir git metadata before copying to /app.
eval "$(bash /workdir/scripts/docker-editable-pretend-versions.sh --shell /workdir)"
sed -i "s/fallback-version = \"0.0.0\"/fallback-version = \"${VERIFIERS_PRETEND_VERSION}\"/" /app/deps/verifiers/pyproject.toml
sed -i "s/fallback-version = \"0.0.0\"/fallback-version = \"${RENDERERS_PRETEND_VERSION}\"/" /app/deps/renderers/pyproject.toml

# Install dependencies with disaggregation extras.
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

# Build NIXL from source against /opt/ucx. Hide uv from Meson to skip the
# optional Python-3.13 nixl-meta build.
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

# Persist runtime environment variables for Enroot.
cat >> /etc/environment <<'EOF'
LC_ALL=en_US.UTF-8
CUDA_HOME=/usr/local/cuda
UCX_HOME=/opt/ucx
LD_LIBRARY_PATH=/opt/ucx/lib:/opt/ucx/lib/ucx
PATH=/app/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HF_HUB_ETAG_TIMEOUT=500
HF_HUB_DOWNLOAD_TIMEOUT=300
EOF

# Remove build artifacts before saving the sqsh.
apt-get clean autoclean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

echo "container build done"
