#!/usr/bin/env bash
# Stage 1: submit three PD arms: P:D 1:1, P:D 1:3, and colocated baseline.
# Run from a login node:
#   bash experiments/rollout-simulation/pd_sweep/run_stage1.sh
#
# Each arm is a Phase 2 replay with orchestrator, trainer, and weight sync, but
# no sandbox. Summarize completed runs with:
#   uv run python experiments/rollout-simulation/summarize_sweep.py \
#       outputs/job-<id1> outputs/job-<id2> ...
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

# The environment-server virtual environment must contain the replay package.
if ! uv run python -c "import swe_replay_v1" 2>/dev/null; then
    echo "[stage1] ERROR: swe_replay_v1 is missing from the virtual environment. Run:" >&2
    echo "  uv pip install -e experiments/rollout-simulation/swe_replay_v1" >&2
    exit 1
fi
echo "[stage1] NOTE: container runs must install the package in the sbatch prologue or image"

bash experiments/training/submit.sh swe_replay_bench            # P:D 1:1 baseline
bash experiments/training/submit.sh swe_replay_bench pd13       # P:D 1:3
bash experiments/training/submit.sh swe_replay_colocated        # Colocated baseline

echo "[stage1] submitted. Check squeue, then pass completed job directories to summarize_sweep.py"
