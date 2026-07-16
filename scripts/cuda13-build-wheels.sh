#!/bin/bash
# B300 (sm_103) で動かない prebuilt wheel を CUDA 13 + torch cu130 でソースビルド
# し直して venv に入れる (Dockerfile.cuda13 / build_in_container_cuda13.sh から使用)
#
# 対象と理由:
#   - DeepEP   1.2.1+29d31c0 (deepseek-ai/DeepEP)
#   - DeepGEMM 2.5.0+891d57b (deepseek-ai/DeepGEMM)
#       公開 prebuilt wheel が cu12 torch 向けビルドのため。rev は cu12 版
#       pyproject の prebuilt wheel と同一
#   - flash-attn (Dao-AILab/flash-attention, upstream main)
#       prebuilt wheel (2.8.3+cu130torch2.11) の fatbin は sm_80/90/100/120 の
#       SASS のみで PTX なし。sm_100 の SASS は sm_103 と非互換なので B300 では
#       カーネルが存在しない。v2.8.3 の setup.py には compute_100f (family) の
#       分岐が無く、main (2.8.4-dev) で追加されたため main の rev を pin して
#       FLASH_ATTN_CUDA_ARCHS=100 → compute_100f,code=sm_100 (sm_10x family で
#       動く SASS) + compute_100 PTX でビルドする
#
# 手順は scripts/install_ep_kernels.sh / scripts/install_deep_gemm.sh と同じだが、
# ビルドノードの GPU に依存しないよう arch は自動検出せず TORCH_CUDA_ARCH_LIST で
# B300 (sm_103) を明示する。
#
# 前提: uv sync 済みの venv (torch cu130 入り)、CUDA 13.0 toolkit、git / curl、
#       GPU ノード (末尾の import / 実行検証にドライバが必要)

set -euxo pipefail

APP_DIR="${1:-/app}"
VENV="${APP_DIR}/.venv"

# 呼び出し元の cwd が消えていても動けるように必ず実在ディレクトリへ移る
cd "${APP_DIR}"

DEEPEP_REF=29d31c095796f3c8ece47ee9cdcc167051bbeed9
DEEPGEMM_REF=891d57b4db1071624b5c8fa0d1e51cb317fa709f
FLASH_ATTN_REF=2402cb0bed7a2185cb9ddbe88fb998656cf73066
NVSHMEM_VER=3.3.24

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# B300 (sm_103) ネイティブ。torch 2.11 の cpp_extension は family 表記 (10.0f) を
# 受け付けない (サポートは 10.0 / 10.0a / 10.3 / 10.3a / ... のみ)。
# arch 依存の accelerated 命令が必要と分かったら 10.3a に上げる。
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.3}"

"${CUDA_HOME}/bin/nvcc" --version | grep -q "release 13"

# CUDA 13 で CCCL (libcu++ / Thrust / CUB) ヘッダは include/cccl/ 配下に移動した。
# nvcc は暗黙に解決するが、deep_ep.cpp のようなホスト側 c++ コンパイルが
# NVSHMEM ヘッダ経由で "cuda/std/tuple" を include するため、CPATH で教える
test -d "${CUDA_HOME}/include/cccl"
export CPATH="${CUDA_HOME}/include/cccl${CPATH:+:${CPATH}}"

VIRTUAL_ENV="${VENV}" uv pip install pip setuptools wheel cmake ninja psutil

# ── NVSHMEM (redist アーカイブ, cuda13 版) — DeepEP のビルド依存 ──
# 実行時の libnvshmem_host.so.3 は torch cu130 が連れてくる nvidia-nvshmem-cu13
# (venv 内) が提供する。ここの redist はビルド時のヘッダ + 静的 device lib 用。
case "$(uname -m)" in
    x86_64)  NVSHMEM_SUBDIR="linux-x86_64" ;;
    aarch64) NVSHMEM_SUBDIR="linux-sbsa" ;;
    *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
NVSHMEM_NAME="libnvshmem-${NVSHMEM_SUBDIR}-${NVSHMEM_VER}_cuda13-archive"
curl -fSL "https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/${NVSHMEM_SUBDIR}/${NVSHMEM_NAME}.tar.xz" \
    -o "/tmp/${NVSHMEM_NAME}.tar.xz"
tar -xf "/tmp/${NVSHMEM_NAME}.tar.xz" -C /tmp
# "-archive" を含むパス名のままにしてはいけない: torch cpp_extension の
# _get_cuda_arch_flags は「フラグ文字列に 'arch' を含むか」で arch 指定済みと
# 判定するため、DeepEP の nvcc_dlink フラグ (-L<NVSHMEM_DIR>/lib) がマッチして
# device-link に -gencode が付かず、sm_75 デフォルトの空リンクになる
# (nvlink warning "SM Arch ('sm_75') not found" → 実行時に最初のカーネルが
# 'unknown error')。install_ep_kernels.sh と同様に 'arch' を含まない名前へ移す。
rm -rf /tmp/nvshmem
mv "/tmp/${NVSHMEM_NAME}" /tmp/nvshmem
export NVSHMEM_DIR=/tmp/nvshmem
case "${NVSHMEM_DIR}" in *arch*) echo "NVSHMEM_DIR must not contain 'arch' (torch dlink arch-flag bug)" >&2; exit 1 ;; esac
export CMAKE_PREFIX_PATH="${NVSHMEM_DIR}/lib/cmake:${CMAKE_PREFIX_PATH:-}"

# ── DeepEP ──
git clone https://github.com/deepseek-ai/DeepEP /tmp/DeepEP
cd /tmp/DeepEP
git checkout "${DEEPEP_REF}"
"${VENV}/bin/python" setup.py bdist_wheel --dist-dir "${APP_DIR}/deps"
VIRTUAL_ENV="${VENV}" uv pip install --no-deps "${APP_DIR}"/deps/deep_ep-*.whl

# ── DeepGEMM (カーネルは実行時 nvrtc JIT。ビルドするのはホスト側拡張のみ) ──
git clone --recurse-submodules https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM
cd /tmp/DeepGEMM
git checkout "${DEEPGEMM_REF}"
git submodule update --init --recursive
"${VENV}/bin/python" -m pip wheel . --no-deps --no-build-isolation --wheel-dir "${APP_DIR}/deps"
VIRTUAL_ENV="${VENV}" uv pip install --no-deps "${APP_DIR}"/deps/deep_gemm-*.whl

# ── flash-attn (compute_100f family + PTX で sm_103 対応) ──
# FLASH_ATTENTION_FORCE_BUILD が無いと setup.py が upstream の prebuilt wheel
# (sm_103 非対応) をダウンロードして済ませてしまうので必ず立てる。
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
cd /tmp/flash-attention
git checkout "${FLASH_ATTN_REF}"
git submodule update --init csrc/cutlass
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="100" NVCC_THREADS=4 \
    "${VENV}/bin/python" -m pip wheel . --no-deps --no-build-isolation --wheel-dir "${APP_DIR}/deps"
VIRTUAL_ENV="${VENV}" uv pip install --reinstall --no-deps "${APP_DIR}"/deps/flash_attn-*.whl
# sm_103 で動くコードが入っていることを検証。compute_100f (family) の cubin は
# cuobjdump 上 "sm_100" と表示され family と区別できないため、PTX の存在を見る
# (compute_100 PTX があれば最悪でも driver JIT で sm_103 上で動く。旧 prebuilt は
# PTX ゼロだったのでこのチェックで弾ける)。実行可否は直後の forward テストで確定。
# 注意: `cuobjdump | grep -q` は grep の早期終了 → SIGPIPE → pipefail で落ちる
# (exit 141) ため、パイプではなくコマンド置換で全出力を受けてから grep する。
grep -q "sm_100" <<<"$("${CUDA_HOME}/bin/cuobjdump" --list-ptx "${VENV}"/lib/python3.12/site-packages/flash_attn_2_cuda*.so)"

# ── import / 実行検証 (GPU ドライバが必要 — GPU ノードでのビルド前提) ──
# cwd がソースツリー (/tmp/flash-attention 等) のままだと venv ではなく cwd 側の
# パッケージを import してしまうので、先に APP_DIR へ移る
cd "${APP_DIR}"
"${VENV}/bin/python" -c "import deep_ep, deep_gemm; print('deep_ep:', deep_ep.__file__); print('deep_gemm:', deep_gemm.__version__)"
# DeepEP は job 171729 で落ちたカーネル (get_dispatch_layout) を実際に起動して
# 検証する。device-link が壊れている (dlink に -gencode が付かない) と import は
# 通るがここで CUDA 'unknown error' になる。DeepEP は 1 ランクをサポートしない
# (runtime.cu 'Unsupported ranks' で Buffer 破棄時に abort) ため 2 ランクで回す。
cat > /tmp/verify_deep_ep.py <<'EOF'
import os
import torch
import torch.distributed as dist
import deep_ep

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=1 << 20)
topk_idx = torch.randint(0, 8, (16, 2), device="cuda", dtype=torch.int64)
buffer.get_dispatch_layout(topk_idx, 8)
torch.cuda.synchronize()
if local_rank == 0:
    print("deep_ep get_dispatch_layout OK")
dist.barrier()
dist.destroy_process_group()
EOF
"${VENV}/bin/torchrun" --standalone --nproc-per-node=2 /tmp/verify_deep_ep.py
rm -f /tmp/verify_deep_ep.py
# flash-attn は B300 実機で forward を 1 回流してカーネルが存在することまで確認
"${VENV}/bin/python" - <<'EOF'
import torch
from flash_attn import flash_attn_func
q, k, v = (torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
out = flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
print("flash_attn forward OK:", tuple(out.shape))
EOF

rm -rf /tmp/DeepEP /tmp/DeepGEMM /tmp/flash-attention "${NVSHMEM_DIR}" "/tmp/${NVSHMEM_NAME}.tar.xz"
rm -f "${APP_DIR}"/deps/deep_ep-*.whl "${APP_DIR}"/deps/deep_gemm-*.whl "${APP_DIR}"/deps/flash_attn-*.whl
