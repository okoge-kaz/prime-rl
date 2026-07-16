#!/bin/bash
# compute node のホスト環境を検証する (コンテナ無しの srun で実行する):
#   1. GDRCopy — kernel module (gdrdrv) / /dev/gdrdrv / userland lib (libgdrapi)
#   2. EFA — kernel module / RDMA デバイス一覧 / libfabric provider
#   3. GPUDirect RDMA の前提 — nvidia_peermem または dmabuf (kernel >= 5.12)
#   4. aws-ofi-nccl プラグイン — 存在とバージョン (GIN 対応判定の材料)
# read-only チェックのみ。ホストに何も変更しない。
#
# 実行例:
#   srun --account=coreai_horizon_dilations --partition=batch --qos=interactive \
#       --job-name=check-host-efa --time=00:05:00 --nodes=1 --ntasks=1 --gpus-per-node=1 \
#       bash experiments/enroot/check_host_efa_gdrcopy.sh

section() { echo; echo "===== $1 ====="; }

echo "hostname: $(hostname)"
echo "kernel:   $(uname -r)"

section "1. GDRCopy (kernel side)"
if lsmod | grep -q '^gdrdrv'; then
    echo "OK: gdrdrv module loaded"
    modinfo gdrdrv 2>/dev/null | grep -E '^(version|filename):'
else
    echo "NG: gdrdrv module NOT loaded"
fi
if [ -e /dev/gdrdrv ]; then
    ls -l /dev/gdrdrv
else
    echo "NG: /dev/gdrdrv not found"
fi

section "1b. GDRCopy (host userland lib — コンテナへの mount 候補)"
ldconfig -p | grep gdrapi || true
find /usr/lib /usr/lib64 /usr/local /opt -maxdepth 4 -name 'libgdrapi.so*' 2>/dev/null || true
# バージョンは so の SONAME (libgdrapi.so.2.x) で判別する

section "2. EFA kernel module"
if lsmod | grep -q '^efa'; then
    echo "OK: efa module loaded"
    modinfo efa 2>/dev/null | grep -E '^(version|filename):'
else
    echo "NG: efa module NOT loaded"
fi

section "2b. RDMA devices (rdmap* = EFA, ibp* = 別系統)"
if [ -d /sys/class/infiniband ]; then
    for d in /sys/class/infiniband/*; do
        name=$(basename "$d")
        ll=$(cat "$d"/ports/1/link_layer 2>/dev/null || echo '?')
        echo "$name  link_layer=$ll"
    done
    echo "EFA device count: $(ls /sys/class/infiniband | grep -c rdmap)"
else
    echo "NG: /sys/class/infiniband not found"
fi

section "2c. libfabric EFA provider (fi_info)"
FI_INFO=""
for p in /opt/amazon/efa/bin/fi_info fi_info; do
    command -v "$p" >/dev/null 2>&1 && { FI_INFO="$p"; break; }
done
if [ -n "$FI_INFO" ]; then
    "$FI_INFO" --version 2>/dev/null | head -2
    "$FI_INFO" -p efa -t FI_EP_RDM 2>&1 | head -10
else
    echo "NG: fi_info not found (EFA installer 未導入?)"
fi

section "3. GPUDirect RDMA 前提 (nvidia_peermem or dmabuf)"
if lsmod | grep -q nvidia_peermem; then
    echo "OK: nvidia_peermem loaded"
else
    echo "-- nvidia_peermem not loaded (kernel >= 5.12 なら EFA は dmabuf 経路で GPUDirect 可)"
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1

section "4. aws-ofi-nccl plugin"
if [ -d /opt/amazon/ofi-nccl/lib ]; then
    ls -l /opt/amazon/ofi-nccl/lib/
    for so in /opt/amazon/ofi-nccl/lib/libnccl-net*.so; do
        [ -e "$so" ] || continue
        echo "-- strings version tags in $(basename "$so"):"
        strings "$so" | grep -iE 'aws-ofi-nccl.*[0-9]+\.[0-9]+|^NET/OFI' | sort -u | head -5
        echo "-- GIN symbols (GPU-Initiated Networking 対応判定):"
        { nm -D "$so" 2>/dev/null || strings "$so"; } | grep -ci gin || true
    done
else
    echo "NG: /opt/amazon/ofi-nccl not found"
fi
[ -f /opt/amazon/efa_installed_packages ] && { echo "-- efa_installed_packages:"; head -20 /opt/amazon/efa_installed_packages; }

section "summary hints"
echo "gdrdrv + /dev/gdrdrv があれば: コンテナ側に libgdrapi を入れる(or mount)だけで GDRCopy 警告は解消可能"
echo "gdrdrv が無ければ: クラスタ管理者にホスト AMI への gdrcopy 導入を依頼 (コンテナ側では解決不可)"
echo "libgdrapi がホストにあれば: CONTAINER_MOUNTS + LD_LIBRARY_PATH 追加で再ビルド無しに検証可能"
