#!/bin/bash
# Submit a one-node gold-standard measurement that converts production traces,
# verifies them against recorded serving q, and runs acceptance plus fitting.
#
# Staleness zero gives the production decode-versus-prefill noise floor.
# Positive staleness is biased because traces were used for training; use it
# only to compare against offline drafts.
#
# Usage:
#   bash experiments/extreme-off-policy/script/submit_prod_replay.sh math_qwen3_4b_instruct
#
# Environment overrides:
#   PROD_VERSIONS="15 25 35 44"
#   PROD_DELTAS="0"
#   MAX_PER_VERSION=256
#   TRACES_GLOB=...
#   PARTITION=batch QOS=interactive ACCOUNT=coreai_horizon_dilations

set -euo pipefail

RUN_NAME="${1:?usage: submit_prod_replay.sh <run_name (e.g. math_qwen3_4b_instruct)>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO="${REPO/#\/scratch\/fsw/\/lustre\/fsw}"

ACCOUNT="${ACCOUNT:-coreai_horizon_dilations}"
PARTITION="${PARTITION:-batch}"
QOS="${QOS:-interactive}"
PROD_VERSIONS="${PROD_VERSIONS:-15 25 35 44}"
PROD_DELTAS="${PROD_DELTAS:-0}"
MAX_PER_VERSION="${MAX_PER_VERSION:-256}"
TRACES_GLOB="${TRACES_GLOB:-${REPO}/outputs/${RUN_NAME}-latest/run_default/rollouts/step_*/train/all/traces.jsonl}"
GPUS_PER_NODE=8
REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"benchmarking","description":"extreme off-policy prod-trace replay: vLLM verify bursts with CPU conversion/aggregation phases"}}'

WEIGHTS="/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/${RUN_NAME}/weights"
OUT="${REPO}/outputs/extreme-off-policy/${RUN_NAME}-prod"
WRAP="${REPO}/experiments/extreme-off-policy/script/run_in_container.sh"
SRC_DIR="${REPO}/experiments/extreme-off-policy/src"
mkdir -p "$OUT"/{drafts,verify,accept,fit,jobs,logs}

# Reuse the offline microbenchmark when available.
MB="${REPO}/outputs/extreme-off-policy/${RUN_NAME}/microbench.json"

inner="$OUT/jobs/replay_inner.sh"
cat > "$inner" <<EOF
#!/bin/bash
set -u; cd $REPO

# Phase A: idempotent trace-to-draft conversion on CPU.
if ! ls $OUT/drafts/step_*.jsonl.gz > /dev/null 2>&1; then
    python $SRC_DIR/trace_to_drafts.py $TRACES_GLOB -o $OUT/drafts --max-per-version $MAX_PER_VERSION || exit 1
fi

# Phase B: distribute version/staleness verification cells across eight GPUs.
i=0
for v in $PROD_VERSIONS; do
  for d in $PROD_DELTAS; do
    t=\$((v + d))
    [ -f $OUT/drafts/step_\$v.jsonl.gz ] || { echo "skip v=\$v (no draft for this trace version)"; continue; }
    [ -d $WEIGHTS/step_\$t ] || { echo "skip anchor \$t (checkpoint missing)"; continue; }
    if [ ! -f $OUT/verify/anchor_\${t}_src_\${v}.jsonl.gz ]; then
        ( export CUDA_VISIBLE_DEVICES=\$((i % $GPUS_PER_NODE)); \\
          python $SRC_DIR/verify_drafts.py --ckpt $WEIGHTS/step_\$t --anchor \$t \\
              --drafts $OUT/drafts/step_\$v.jsonl.gz --out-dir $OUT/verify \\
        ) > $OUT/logs/verify_v\${v}_d\${d}.log 2>&1 &
        i=\$((i + 1))
        [ \$((i % $GPUS_PER_NODE)) -eq 0 ] && wait
    fi
  done
done
wait

# Check expected outputs after background workers finish.
MISSING=0
for v in $PROD_VERSIONS; do
  for d in $PROD_DELTAS; do
    t=\$((v + d))
    [ -f $OUT/drafts/step_\$v.jsonl.gz ] || continue
    [ -d $WEIGHTS/step_\$t ] || continue
    [ -f $OUT/verify/anchor_\${t}_src_\${v}.jsonl.gz ] || { echo "MISSING: verify v=\$v Δ=\$d (see logs/verify_v\${v}_d\${d}.log)"; MISSING=1; }
  done
done
[ "\$MISSING" -eq 0 ] || exit 1

# Phase C: calibrate noisecal from staleness-zero cells, accept, and fit.
set -e
python $SRC_DIR/accept_rules.py $OUT/verify/anchor_*.jsonl.gz \\
    --tokenizer $WEIGHTS/step_\$(echo $PROD_VERSIONS | awk '{print \$NF}') -o $OUT/accept
python $SRC_DIR/fit_alpha.py $OUT/accept/accept_*.jsonl.gz \\
    \$([ -f $MB ] && echo "--microbench $MB") -o $OUT/fit
EOF

outer="$OUT/jobs/replay.sh"
printf '#!/bin/bash\nset -euo pipefail\nbash %s bash %s\n' "$WRAP" "$inner" > "$outer"
job_id=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" --qos="$QOS" --comment="$REAPER_COMMENT" -N1 \
    --gpus-per-node=$GPUS_PER_NODE -t 02:00:00 \
    -J "extreme-off-policy-replay-$RUN_NAME" -o "$OUT/logs/replay_%j.log" "$outer")

echo "submitted: extreme-off-policy-replay-$RUN_NAME = $job_id"
echo "  versions: $PROD_VERSIONS | deltas: $PROD_DELTAS | max/version: $MAX_PER_VERSION"
echo "  results -> $OUT/fit/   logs -> $OUT/logs/"
