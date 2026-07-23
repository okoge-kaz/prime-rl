#!/bin/bash
# Submit a one-node SWE gold-standard measurement using multi-turn production
# traces from job 188448 and staleness-zero INTELLECT-3 verification.
#
# Measures the SWE numerical noise floor, calibrated delta, complete-turn
# acceptance, and observation-token ratio. Partial errored trajectories are
# included when they retain nodes; empty traces cannot be measured.
#
# Usage:
#   bash experiments/extreme-off-policy/script/submit_swe_replay.sh
#
# Environment overrides:
#   ANCHOR_MODEL=PrimeIntellect/INTELLECT-3
#   PROD_VERSIONS="0"
#   MAX_PER_VERSION=256  TP=8  MAX_MODEL_LEN=131072  BATCH=8
#   TRACES_GLOB=...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO="${REPO/#\/scratch\/fsw/\/lustre\/fsw}"

ACCOUNT="${ACCOUNT:-coreai_horizon_dilations}"
PARTITION="${PARTITION:-batch}"
QOS="${QOS:-interactive}"
ANCHOR_MODEL="${ANCHOR_MODEL:-PrimeIntellect/INTELLECT-3}"
PROD_VERSIONS="${PROD_VERSIONS:-0}"
MAX_PER_VERSION="${MAX_PER_VERSION:-256}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
BATCH="${BATCH:-8}"
TRACES_GLOB="${TRACES_GLOB:-${REPO}/outputs/job-188448/run_default/rollouts/step_*/train/all/traces.jsonl}"
REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"benchmarking","description":"extreme off-policy SWE trace replay: 106B tp8 prefill verify with CPU conversion phases"}}'

OUT="${REPO}/outputs/extreme-off-policy/swe_188448"
WRAP="${REPO}/experiments/extreme-off-policy/script/run_in_container.sh"
SRC_DIR="${REPO}/experiments/extreme-off-policy/src"
mkdir -p "$OUT"/{drafts,verify,accept,fit,jobs,logs}

inner="$OUT/jobs/swe_replay_inner.sh"
cat > "$inner" <<EOF
#!/bin/bash
set -u; cd $REPO

# Phase A: idempotent multi-turn draft conversion on CPU.
if ! ls $OUT/drafts/step_*.jsonl.gz > /dev/null 2>&1; then
    python $SRC_DIR/swe_trace_to_drafts.py $TRACES_GLOB -o $OUT/drafts --max-per-version $MAX_PER_VERSION || exit 1
fi

# Phase B: serial tp$TP verification of sampled positions.
for v in $PROD_VERSIONS; do
    [ -f $OUT/drafts/step_\$v.jsonl.gz ] || { echo "skip v=\$v (draft missing)"; continue; }
    if [ ! -f $OUT/verify/anchor_\${v}_src_\${v}.jsonl.gz ]; then
        python $SRC_DIR/verify_drafts.py --ckpt "$ANCHOR_MODEL" --anchor \$v \\
            --drafts $OUT/drafts/step_\$v.jsonl.gz --out-dir $OUT/verify \\
            --tp $TP --max-model-len $MAX_MODEL_LEN --batch-size $BATCH \\
            > $OUT/logs/verify_v\$v.log 2>&1 || { echo "verify v=\$v FAILED (see logs/verify_v\$v.log)"; exit 1; }
    fi
done

# Phase C: calibrate noisecal from staleness zero, accept, and fit.
set -e
python $SRC_DIR/accept_rules.py $OUT/verify/anchor_*.jsonl.gz -o $OUT/accept
python $SRC_DIR/fit_alpha.py $OUT/accept/accept_*.jsonl.gz -o $OUT/fit
EOF

outer="$OUT/jobs/swe_replay.sh"
printf '#!/bin/bash\nset -euo pipefail\nbash %s bash %s\n' "$WRAP" "$inner" > "$outer"
job_id=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" --qos="$QOS" --comment="$REAPER_COMMENT" -N1 \
    --gpus-per-node=8 -t 04:00:00 \
    -J "extreme-off-policy-swe-replay" -o "$OUT/logs/swe_replay_%j.log" "$outer")

echo "submitted: extreme-off-policy-swe-replay = $job_id"
echo "  anchor: $ANCHOR_MODEL | versions: $PROD_VERSIONS | max/version: $MAX_PER_VERSION"
echo "  results -> $OUT/fit/   logs -> $OUT/logs/"
