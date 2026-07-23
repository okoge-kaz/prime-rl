#!/usr/bin/env bash
# Stage 0: automate decode-engine simulations that isolate EP/all-to-all overhead.
#
# On a GPU node, this script re-enters the same CUDA 13 container used by
# training jobs. EFA and NCCL are configured in the container, so do not use
# the host virtual environment.
#   bash experiments/rollout-simulation/pd_sweep/run_stage0.sh
#
# Select subsets or override settings through environment variables:
#   ARMS="ep_off agrs" INFLIGHTS="32 64" LIMIT=50 \
#       bash experiments/rollout-simulation/pd_sweep/run_stage0.sh
#   PRIME_RL_SQSH=... bash experiments/rollout-simulation/pd_sweep/run_stage0.sh
#
# Each arm starts an inference server, waits for /v1/models, runs replay_swe.py
# at each concurrency, stops the server, and waits for GPU release. Failed arms
# are skipped. summarize_sweep.py prints the final comparison.
#
# Later runs within an arm use a warm prefix cache. All arms replay the same
# workload and therefore share this condition.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

# Keep caches on Lustre because the container's /root is ephemeral.
LUSTRE_USER=/lustre/fsw/portfolios/coreai/users/kfujii
CACHE_BASE="$LUSTRE_USER/cache"
mkdir -p "$CACHE_BASE/uv" "$CACHE_BASE/vllm" "$CACHE_BASE/flashinfer"
export UV_CACHE_DIR="$CACHE_BASE/uv"
export VLLM_CACHE_ROOT="$CACHE_BASE/vllm"

SQSH="${PRIME_RL_SQSH:-$LUSTRE_USER/containers/prime-rl-v0.7.0-cu13-disagg-v4.sqsh}"
# Match training mounts and overlay the FlashInfer JIT cache at /root/.cache.
CONTAINER_MOUNTS="$LUSTRE_USER:$LUSTRE_USER,$REPO:$REPO,$REPO/src:/app/src,$REPO/packages:/app/packages,$REPO/deps:/app/deps,$CACHE_BASE/flashinfer:/root/.cache/flashinfer"

# Re-execute this script inside the container when necessary.
if [[ ! -d /app/.venv ]]; then
    echo "[stage0] entering container: $SQSH"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        # Inside a Slurm allocation, srun carries the environment into Pyxis.
        exec srun --overlap --container-image="$SQSH" \
            --container-mounts="$CONTAINER_MOUNTS" \
            bash "$REPO/experiments/rollout-simulation/pd_sweep/run_stage0.sh"
    fi
    # On an unallocated GPU node, start Enroot directly.
    if ! enroot list | grep -qx prime-rl-rollout-simulation; then
        echo "[stage0] enroot create (first run only; takes several minutes)"
        enroot create -n prime-rl-rollout-simulation "$SQSH"
    fi
    MOUNT_ARGS=()
    IFS=',' read -ra PAIRS <<< "$CONTAINER_MOUNTS"
    for pair in "${PAIRS[@]}"; do MOUNT_ARGS+=(-m "$pair"); done
    exec enroot start --rw "${MOUNT_ARGS[@]}" \
        -e ARMS="${ARMS:-}" -e INFLIGHTS="${INFLIGHTS:-}" -e LIMIT="${LIMIT:-}" \
        -e RUN_SECONDS="${RUN_SECONDS:-}" \
        -e PORT="${PORT:-}" -e OUT="${OUT:-}" -e PLANS="${PLANS:-}" \
        -e UV_CACHE_DIR="$UV_CACHE_DIR" -e VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
        prime-rl-rollout-simulation \
        bash "$REPO/experiments/rollout-simulation/pd_sweep/run_stage0.sh"
fi

# Inside the container, use /app/.venv directly.
export VIRTUAL_ENV=/app/.venv
export PATH=/app/.venv/bin:$PATH

ARMS="${ARMS:-ep_off deepep_ll agrs flashinfer}"
INFLIGHTS="${INFLIGHTS:-8 32 64 128}"
LIMIT="${LIMIT:-380}"              # Number of plans supplied; RUN_SECONDS stops the run.
RUN_SECONDS="${RUN_SECONDS:-480}"  # Per-measurement steady-state window.
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
PLANS="${PLANS:-outputs/rollout-simulation/job-188448/replay_plans.jsonl}"
TRACES="${TRACES:-$LUSTRE_USER/src/prime-rl-v0.7.0/outputs/20260720-140939-swe_intellect3_disagg-validation/run_default/rollouts/step_1/train/all/traces.jsonl}"
OUT="${OUT:-outputs/rollout-simulation/sweep/stage0-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

if [[ ! -f "$PLANS" ]]; then
    echo "[stage0] building replay plans from $TRACES"
    python experiments/rollout-simulation/build_replay_workload.py "$TRACES" -o "$PLANS"
fi

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill -- -"$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_ready() {
    local arm="$1" deadline=$((SECONDS + 2400))
    echo "[stage0] waiting for server ($arm) ..."
    until curl -sf -o /dev/null "$BASE_URL/models"; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[stage0] server died during startup; tail of log:" >&2
            tail -30 "$OUT/${arm}.server.log" >&2
            return 1
        fi
        if (( SECONDS > deadline )); then
            echo "[stage0] server not ready in 40min" >&2
            return 1
        fi
        sleep 15
    done
    echo "[stage0] server ready"
}

wait_gpu_idle() {
    local deadline=$((SECONDS + 600))
    while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
        if (( SECONDS > deadline )); then
            echo "[stage0] WARN: GPU procs still alive after 10min; continuing" >&2
            break
        fi
        sleep 10
    done
}

for arm in $ARMS; do
    echo "=== [stage0] arm: $arm ==="
    setsid inference @ "experiments/rollout-simulation/pd_sweep/stage0_${arm}.toml" \
        --server.port "$PORT" > "$OUT/${arm}.server.log" 2>&1 &
    SERVER_PID=$!
    if ! wait_ready "$arm"; then
        echo "[stage0] SKIP arm $arm (server failed; see $OUT/${arm}.server.log)" >&2
        kill -- -"$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        wait_gpu_idle
        continue
    fi
    for n in $INFLIGHTS; do
        echo "--- [stage0] $arm inflight=$n"
        python experiments/rollout-simulation/replay_swe.py "$PLANS" \
            --base-url "$BASE_URL" --max-inflight "$n" --limit "$LIMIT" \
            --max-duration-s "$RUN_SECONDS" \
            -o "$OUT/${arm}_n${n}"
    done
    kill -- -"$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    wait_gpu_idle
done

echo "=== [stage0] summary ==="
python experiments/rollout-simulation/summarize_sweep.py "$OUT"/*_n* | tee "$OUT/summary.txt"
echo "[stage0] results in $OUT"
