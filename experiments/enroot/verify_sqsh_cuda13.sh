#!/bin/bash
# Validate the CUDA 13 disaggregated sqsh on a GPU node:
# GDRCopy userland, DeepEP sm_103 device-link and kernels, and the NIXL shim.
# Run every check and return the number of failures at the end.
#
# Static checks:
#   srun --account=coreai_horizon_dilations --partition=batch --qos=interactive \
#       --job-name=verify-sqsh --time=00:15:00 --nodes=1 --ntasks=1 --gpus-per-node=1 \
#       --container-image=<sqsh> --container-mounts=$PWD:$PWD \
#       bash $PWD/experiments/enroot/verify_sqsh_cuda13.sh
#
# Add RUN_KERNEL_TEST=1 with eight GPUs for real DeepEP kernel tests:
#   ... --gpus-per-node=8 bash -c "RUN_KERNEL_TEST=1 bash $PWD/experiments/enroot/verify_sqsh_cuda13.sh"

set -uo pipefail

PY=/app/.venv/bin/python
export PATH=/app/.venv/bin:/usr/local/cuda/bin:$PATH

NG=0
section() { echo; echo "===== $1 ====="; }
check() { # check <label> <command...>
    local label="$1"; shift
    if "$@"; then echo "OK: ${label}"; else echo "NG: ${label}"; NG=$((NG + 1)); fi
}

section "0. environment"
hostname
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1

section "1. GDRCopy userland (v2)"
ls -l /usr/local/lib/libgdrapi.so* 2>/dev/null
ls -l /dev/gdrdrv 2>/dev/null
check "libgdrapi dlopen" $PY -c 'import ctypes; ctypes.CDLL("libgdrapi.so")'

section "2. DeepEP device-link architecture (job 171729 regression)"
check "deep_ep/deep_gemm import" $PY -c 'import deep_ep, deep_gemm'
# Find the extension by path because its import name varies by build.
SO=$($PY - <<'EOF' 2>/dev/null || true
import deep_ep, glob, os
pkg = os.path.dirname(deep_ep.__file__)
cands = glob.glob(os.path.join(pkg, "**", "*.so"), recursive=True) \
    + glob.glob(os.path.join(os.path.dirname(pkg), "deep_ep_cpp*.so"))
print(cands[0])
EOF
)
if [ -n "$SO" ]; then
    echo "so: $SO"
    cuobjdump --list-elf "$SO" | grep -o 'sm_[0-9a-f]*' | sort | uniq -c
    # The broken v1 build linked an sm_75 cubin because gencode was missing.
    check "no sm_75 dlink" bash -c "! cuobjdump --list-elf '$SO' | grep -q sm_75"
    check "sm_103 SASS present" bash -c "cuobjdump --list-elf '$SO' | grep -q sm_103"
else
    echo "NG: cannot import deep_ep_cpp to inspect its architecture"; NG=$((NG + 1))
fi

section "3. NIXL meta shim (job 173696 regression)"
check "import nixl._api" $PY -c 'import nixl._api'

section "3b. NIXL LIBFABRIC plugin and EFA HMEM"
check "libplugin_LIBFABRIC.so in source-built wheel" \
    bash -c "find /app/.venv/lib/python3.12/site-packages -path '*mesonpy.libs/plugins/libplugin_LIBFABRIC.so' | grep -q ."
# Confirm that the EFA libfabric build supports CUDA HMEM.
check "libfabric EFA provider reports HMEM" \
    bash -c "/opt/amazon/efa/bin/fi_info -p efa 2>/dev/null | grep -qi hmem || /opt/amazon/efa/bin/fi_info -p efa -c FI_HMEM >/dev/null 2>&1"
# Exercise the same createBackend("LIBFABRIC") API path used by vLLM.
check "nixl_agent createBackend(LIBFABRIC)" $PY -c '
from nixl._api import nixl_agent, nixl_agent_config
a = nixl_agent("verify", nixl_agent_config(backends=["LIBFABRIC"]))
print("LIBFABRIC backend created:", a.backends)
'

if [ "${RUN_KERNEL_TEST:-0}" = "1" ]; then
    section "4. DeepEP intranode kernels (8 GPUs)"
    # Use tests from the same revision as the wheel.
    DEEPEP_REF=29d31c095796f3c8ece47ee9cdcc167051bbeed9
    rm -rf /tmp/DeepEP-test
    git clone https://github.com/deepseek-ai/DeepEP /tmp/DeepEP-test
    git -C /tmp/DeepEP-test checkout "$DEEPEP_REF"
    cd /tmp/DeepEP-test
    # High-throughput prefill path.
    check "test_intranode (HT)" $PY tests/test_intranode.py
    # Low-latency decode path.
    check "test_low_latency (LL)" $PY tests/test_low_latency.py
else
    echo; echo "(Set RUN_KERNEL_TEST=1 and request eight GPUs for kernel tests.)"
fi

section "result"
echo "NG count: $NG"
exit $((NG > 0 ? 1 : 0))
