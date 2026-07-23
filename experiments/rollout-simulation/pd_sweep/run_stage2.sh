#!/usr/bin/env bash
# Stage 2: submit arms that tune the concurrent-rollout load point.
# If the Stage 1 winner is not swe_replay_bench, copy the inflight overlay into
# that configuration directory before running from a login node:
#   bash experiments/rollout-simulation/pd_sweep/run_stage2.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

bash experiments/training/submit.sh swe_replay_bench inflight128   # 512 -> 128

echo "[stage2] submitted. Compare with the inflight-512 baseline using summarize_sweep.py"
