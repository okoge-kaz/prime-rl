#!/bin/bash
# Build the Dockerfile.cuda13 equivalent inside Enroot, including disaggregation.
#
# Keep the devel toolkit for runtime JIT, omit appuser under Enroot, resolve the
# project for cu130, and source-build components that lack sm_103 wheels.
# Required mounts: /workdir for the repository and /uv-cache for shared cache.

set -euxo pipefail

# Redirect caches away from the mounted, quota-limited user home.
export HOME=/root

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:${PATH}

# Install builder and runtime apt dependencies. Enroot bind-mounts ssh_config,
# so preserve existing conffiles and omit the unnecessary SSH server.
apt-get update
apt-get install -y --no-install-recommends \
    -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
    build-essential autoconf automake libtool pkg-config ca-certificates curl sudo git ninja-build \
    libnuma-dev libnl-3-dev libnl-route-3-dev libibverbs-dev librdmacm-dev \
    libhwloc-dev `# Required by NIXL LIBFABRIC topology detection.` \
    python3.12 python3.12-venv python3.12-dev \
    wget clang tmux iperf git-lfs gpg iputils-ping net-tools vim \
    ibverbs-providers
ln -sf /usr/bin/python3.12 /usr/local/bin/python

# Confirm that the toolkit can target sm_103.
test -x /usr/local/cuda/bin/nvcc
nvcc --version | grep -q "release 13.0"

# ── AWS EFA (libfabric + aws-ofi-nccl) ──
# EFA is the usable inter-node RDMA path. Without aws-ofi-nccl, NCCL selects
# unrelated IB devices and hangs with IBV_WC_RETRY_EXC_ERR. Install userland
# only; the kernel module is provided by the host.
EFA_INSTALLER_VERSION=1.49.0
# Pre-install openssh-client with conffile preservation before the EFA installer.
apt-get install -y --no-install-recommends \
    -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
    pciutils openssh-client
curl -fsSL "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
    | tar xz -C /tmp
(cd /tmp/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --mpi=openmpi4)
rm -rf /tmp/aws-efa-installer
# Fail the build if the EFA NCCL plugin is missing.
test -e /opt/amazon/ofi-nccl/lib/libnccl-net.so || test -e /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
ls -la /opt/amazon/ofi-nccl/lib/
dpkg -l | grep -iE "efa|ofi" || true

# ── GDRCopy userland (libgdrapi) ──
# aws-ofi-nccl dlopens libgdrapi. The host supplies gdrdrv and /dev/gdrdrv;
# build only the matching userland library in the container.
GDRCOPY_VERSION=2.5.1
git clone --depth 1 --branch "v${GDRCOPY_VERSION}" https://github.com/NVIDIA/gdrcopy /tmp/gdrcopy
make -C /tmp/gdrcopy lib lib_install prefix=/usr/local
ldconfig
test -e /usr/local/lib/libgdrapi.so
rm -rf /tmp/gdrcopy

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

# Resolve the project for the CUDA 13 stack.
bash /app/scripts/pyproject-cuda13-patch.sh /app/pyproject.toml

# Install dependencies.
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

# Build NIXL from source against /opt/ucx. Hide uv from Meson to skip the
# optional Python-3.13 nixl-meta build.
NIXL_VERSION=0.10.1
VIRTUAL_ENV=/app/.venv uv pip install pip
git clone --depth 1 --branch "${NIXL_VERSION}" https://github.com/ai-dynamo/nixl.git /tmp/nixl
cd /tmp/nixl
for uv_bin in /usr/local/bin/uv /root/.local/bin/uv; do
    if [ -e "${uv_bin}" ]; then mv "${uv_bin}" "${uv_bin}.hidden"; fi
done
# Expose EFA libfabric so Meson includes the native LIBFABRIC backend.
LIBFABRIC_PC_DIR=$(dirname "$(find /opt/amazon/efa -name 'libfabric.pc' | head -1)")
test -n "$LIBFABRIC_PC_DIR"
PKG_CONFIG_PATH=/opt/ucx/lib/pkgconfig:"$LIBFABRIC_PC_DIR" \
    LD_LIBRARY_PATH=/opt/ucx/lib:/opt/ucx/lib/ucx:/opt/amazon/efa/lib \
    /app/.venv/bin/python -m pip wheel . --no-deps --wheel-dir=/app/deps
for uv_bin in /usr/local/bin/uv /root/.local/bin/uv; do
    if [ -e "${uv_bin}.hidden" ]; then mv "${uv_bin}.hidden" "${uv_bin}"; fi
done
VIRTUAL_ENV=/app/.venv uv pip install --reinstall --no-deps /app/deps/nixl_cu1*-*.whl
# Leave the build directory before deleting it.
cd /app
rm -rf /tmp/nixl /app/deps/nixl_cu1*-*.whl
# vLLM imports nixl._api through the meta-package shim; validate it here.
LD_LIBRARY_PATH=/opt/ucx/lib:/opt/ucx/lib/ucx \
    /app/.venv/bin/python -c "import nixl._api as m; print('nixl._api:', m.__file__)"
# Validate that the source-built wheel contains the LIBFABRIC plugin.
find /app/.venv/lib/python3.12/site-packages -path '*mesonpy.libs/plugins/libplugin_LIBFABRIC.so' | grep -q . \
    || { echo "NIXL LIBFABRIC plugin missing; Meson did not detect libfabric" >&2; exit 1; }

# Build DeepEP, DeepGEMM, and flash-attn for CUDA 13 and sm_103.
bash /app/scripts/cuda13-build-wheels.sh /app

# Persist runtime environment variables for Enroot.
cat >> /etc/environment <<'EOF'
LC_ALL=en_US.UTF-8
CUDA_HOME=/usr/local/cuda
UCX_HOME=/opt/ucx
LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/ucx/lib:/opt/ucx/lib/ucx
PATH=/app/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HF_HUB_ETAG_TIMEOUT=500
HF_HUB_DOWNLOAD_TIMEOUT=300
EOF

# Remove build artifacts before saving the sqsh.
apt-get clean autoclean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

echo "container build done"
