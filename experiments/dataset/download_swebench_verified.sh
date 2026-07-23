#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

DATASET="swe-bench/swe-bench-verified"
CACHE_NAME="${DATASET//\//_}"

# Link Harbor's fixed cache path to Lustre.
mkdir -p "${HOME}/.cache"
if [ -e "${HOME}/.cache/harbor" ] && [ ! -L "${HOME}/.cache/harbor" ]; then
    echo "ERROR: ${HOME}/.cache/harbor exists as a real directory." >&2
    echo "Move its contents to ${HARBOR_CACHE_DIR}, then remove it." >&2
    exit 1
fi
ln -sfn "${HARBOR_CACHE_DIR}" "${HOME}/.cache/harbor"

if [ -d "${HARBOR_CACHE_DIR}/${CACHE_NAME}" ]; then
    echo "already cached: ${HARBOR_CACHE_DIR}/${CACHE_NAME}"
    exit 0
fi

# Export to a temporary directory and rename it atomically.
cd "${REPO_ROOT}"
TMP_EXPORT="$(mktemp -d --tmpdir="${HARBOR_CACHE_DIR}")"
trap 'rm -rf "${TMP_EXPORT}"' EXIT
uv run --no-sync harbor download "${DATASET}" --export -o "${TMP_EXPORT}/export"
mv "${TMP_EXPORT}/export" "${HARBOR_CACHE_DIR}/${CACHE_NAME}"

echo "done: ${DATASET} -> ${HARBOR_CACHE_DIR}/${CACHE_NAME}"
