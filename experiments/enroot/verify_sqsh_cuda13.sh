#!/bin/bash
# cu13-disagg sqsh の検証。コンテナ内・GPU ノードで実行する。
# 検証対象:
#   1. GDRCopy userland (libgdrapi) — dlopen できること (v2 で追加)
#   2. deep_ep の device-link — sm_75 空リンク (job 171729 の原因) が直り
#      sm_103 の SASS が入っていること + 実カーネルが 8 GPU で動くこと (v2 で修正)
#   3. nixl meta shim — `import nixl._api` できること (job 173696 の原因, v3 で修正)
# チェックは途中で止まらず全件実行し、最後に NG 件数で exit code を決める。
#
# 実行例 (静的チェックのみ、~1 分):
#   srun --account=coreai_horizon_dilations --partition=batch --qos=interactive \
#       --job-name=verify-sqsh --time=00:15:00 --nodes=1 --ntasks=1 --gpus-per-node=1 \
#       --container-image=<sqsh> --container-mounts=$PWD:$PWD \
#       bash $PWD/experiments/enroot/verify_sqsh_cuda13.sh
#
# 実カーネルテストまで行う場合は --gpus-per-node=8 で RUN_KERNEL_TEST=1 を付ける:
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

section "2. deep_ep device-link arch (v2, job 171729 の再発チェック)"
check "deep_ep/deep_gemm import" $PY -c 'import deep_ep, deep_gemm'
# 拡張 .so を deep_ep パッケージ周辺から探す (モジュール名はビルドにより
# deep_ep_cpp / deep_ep.deep_ep_cpp 等で揺れるため import 名に依存しない)
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
    # 旧 v1 の壊れた build は device-link cubin が sm_75 (gencode 欠落) だった
    check "no sm_75 dlink" bash -c "! cuobjdump --list-elf '$SO' | grep -q sm_75"
    check "sm_103 SASS present" bash -c "cuobjdump --list-elf '$SO' | grep -q sm_103"
else
    echo "NG: deep_ep_cpp が import できず arch 検査不能"; NG=$((NG + 1))
fi

section "3. nixl meta shim (v3, job 173696 の再発チェック)"
check "import nixl._api" $PY -c 'import nixl._api'

if [ "${RUN_KERNEL_TEST:-0}" = "1" ]; then
    section "4. DeepEP 実カーネル (intranode, 8 GPU)"
    # wheel と同じ rev のテストを使う (scripts/cuda13-build-wheels.sh の DEEPEP_REF)
    DEEPEP_REF=29d31c095796f3c8ece47ee9cdcc167051bbeed9
    rm -rf /tmp/DeepEP-test
    git clone https://github.com/deepseek-ai/DeepEP /tmp/DeepEP-test
    git -C /tmp/DeepEP-test checkout "$DEEPEP_REF"
    cd /tmp/DeepEP-test
    # HT 経路 (prefill 側 = job 171729 で layout.cu が落ちた経路)
    check "test_intranode (HT)" $PY tests/test_intranode.py
    # LL 経路 (decode 側 = NVSHMEM device state 999 で落ちた経路)
    check "test_low_latency (LL)" $PY tests/test_low_latency.py
else
    echo; echo "(実カーネルテストは RUN_KERNEL_TEST=1 + --gpus-per-node=8 で実行)"
fi

section "result"
echo "NG count: $NG"
exit $((NG > 0 ? 1 : 0))
