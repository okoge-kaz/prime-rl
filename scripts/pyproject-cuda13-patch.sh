#!/bin/bash
# pyproject.toml を CUDA 13.0 スタック向けに書き換える
# (Dockerfile.cuda13 / experiments/enroot/build_in_container_cuda13.sh から使用)
#
# 変更内容:
#   - torch 系 index を cu128 → cu130 へ (flash-attn prebuilt wheel URL の
#     cu128torch2.11 → cu130torch2.11、flash_attn_3 の test index も同時に置換)
#   - vllm: GitHub release の +cu129 wheel URL 固定を削除し、PyPI の default wheel
#     (CUDA 13 ビルド、torch==2.11.0 を要求) に解決させる。コード互換性のため
#     バージョンは現行 lock と同じ 0.24.0 に固定
#   - cuDNN override を cu13 系へ
#   - disagg extra の cu13 化:
#       * nixl: `import nixl` という名前は meta package (nixl) の shim が提供し、
#         flavor wheel (nixl-cu13) の実体は nixl_cu13/ という別名でインストール
#         される。そのため meta は必須。ただし meta 0.10.1 は base 依存で
#         nixl-cu12 を無条件要求するため、override-dependencies でマーカー無効化
#         して cu13 flavor だけを入れる (shim は nixl_cu13 を優先 import する)。
#         flavor の実体は #2883 対策のソースビルド wheel でのちに上書きされる
#       * deep-ep / deep-gemm: prebuilt wheel が cu12 torch 向けのため依存から
#         除外し、uv sync 後に scripts/cuda13-build-wheels.sh で同じ rev を
#         ソースビルドして venv に入れる

set -euo pipefail

PYPROJECT="${1:?usage: $0 <path-to-pyproject.toml>}"

# index 定義/参照 (pytorch-cu128 → pytorch-cu130, test index 含む) と
# flash-attn prebuilt wheel URL (cu128torch2.11 → cu130torch2.11)
sed -i 's/cu128/cu130/g' "${PYPROJECT}"

# vllm: +cu129 wheel の URL source ブロックを削除し、PyPI default (cu13) を使う
sed -i 's/"vllm>=0.24.0"/"vllm==0.24.0"/' "${PYPROJECT}"
sed -i '/^vllm = \[$/,/^\]$/d' "${PYPROJECT}"

# torch cu130 の cuDNN は nvidia-cudnn-cu13
sed -i 's/"nvidia-cudnn-cu12>=9.15"/"nvidia-cudnn-cu13>=9.15"/' "${PYPROJECT}"

# torch を 2.11.0 に固定 (project.dependencies と override-dependencies の両方)。
# override-dependencies の "torch>=2.9.0" は vllm の torch==2.11.0 pin をも上書き
# するため、放置すると再 lock で cu130 index の最新 (2.13 など) に飛び、
# flash-attn cu130torch2.11 prebuilt wheel / vllm 0.24.0 と ABI 不整合になる
sed -i 's/"torch>=2.9.0",/"torch==2.11.0",/' "${PYPROJECT}"

# disagg extra: deep-ep / deep-gemm を外し (ソースビルドで補う)、
# nixl は meta (import 名 `nixl` の shim) + cu13 flavor の組に置き換え
sed -i '/"deep-ep ; platform_machine == .x86_64.",/d' "${PYPROJECT}"
sed -i '/^    "deep-gemm",$/d' "${PYPROJECT}"
sed -i 's/^    "nixl",$/    "nixl==0.10.1",\n    "nixl-cu13==0.10.1",/' "${PYPROJECT}"
sed -i '/"nixl-cu12 ; platform_machine == .x86_64.",/d' "${PYPROJECT}"

# modelexpress の nixl は meta 経由で cu13 flavor へ
sed -i 's/"nixl\[cu12\]"/"nixl[cu13]"/' "${PYPROJECT}"

# nixl meta の無条件 base 依存 nixl-cu12>=0.10.1 を override で無効化
# (uv の override は成立しないマーカー付き宣言で依存を実質除去できる)
sed -i '/"nvidia-cudnn-cu13>=9.15",/a\    "nixl-cu12 ; sys_platform == '"'"'never'"'"'",' "${PYPROJECT}"

# 参照されなくなった cu12 系の tool.uv.sources エントリを削除
sed -i '/^deep-ep = { url = /d' "${PYPROJECT}"
sed -i '/^deep-gemm = \[$/,/^\]$/d' "${PYPROJECT}"
sed -i '/^nixl-cu12 = { url = /d' "${PYPROJECT}"

# パッチ結果の検証
grep -q 'name = "pytorch-cu130"' "${PYPROJECT}"
grep -q 'cu130torch2.11' "${PYPROJECT}"
grep -q '"vllm==0.24.0"' "${PYPROJECT}"
[ "$(grep -c '"torch==2.11.0",' "${PYPROJECT}")" -eq 2 ]
grep -q '"nixl==0.10.1",' "${PYPROJECT}"
grep -q '"nixl-cu13==0.10.1",' "${PYPROJECT}"
grep -q '"nixl\[cu13\]"' "${PYPROJECT}"
grep -q "nixl-cu12 ; sys_platform == 'never'" "${PYPROJECT}"
for leftover in 'vllm-0.24.0+cu129' 'deep_ep-1.2.1+29d31c0' 'deep_gemm-2.5.0+891d57b' 'nixl_cu12' 'nixl\[cu12\]'; do
    if grep -q "${leftover}" "${PYPROJECT}"; then
        echo "ERROR: cu12 artifact '${leftover}' still present" >&2
        exit 1
    fi
done
echo "patched ${PYPROJECT} for CUDA 13.0 (disagg included)"
