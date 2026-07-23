#!/bin/bash
# Submit phases 2-4 from a login node as two whole-node jobs:
#
#   job1: generate drafts for checkpoints in eight-GPU waves.
#   job2 afterok: verify anchors and run the microbenchmark in parallel, then
#                 run acceptance and fitting on CPU.
#
# Stages are idempotent and skip existing outputs. Expected files are checked
# after background workers finish.
#
# Usage:
#   bash experiments/extreme-off-policy/script/submit_analysis.sh math_qwen3_4b_instruct
#
# Environment overrides:
#   PARTITION=batch  QOS=interactive  ACCOUNT=coreai_horizon_dilations
#   GEN_ARGS="--num-prompts 128 --samples 2"

set -euo pipefail

RUN_NAME="${1:?usage: submit_analysis.sh <run_name (e.g. math_qwen3_4b_instruct)>}"
# OUT_NAME selects a separate output namespace while retaining RUN_NAME checkpoints.
OUT_NAME="${OUT_NAME:-$RUN_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO="${REPO/#\/scratch\/fsw/\/lustre\/fsw}"

ACCOUNT="${ACCOUNT:-coreai_horizon_dilations}"
PARTITION="${PARTITION:-batch}"
QOS="${QOS:-interactive}"
GEN_ARGS="${GEN_ARGS:-}"
GPUS_PER_NODE=8
REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"benchmarking","description":"extreme off-policy analysis: vLLM draft/verify waves with CPU aggregation phases between GPU bursts"}}'


WEIGHTS="/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/${RUN_NAME}/weights"
OUT="${REPO}/outputs/extreme-off-policy/${OUT_NAME}"
WRAP="${REPO}/experiments/extreme-off-policy/script/run_in_container.sh"
SRC_DIR="${REPO}/experiments/extreme-off-policy/src"
mkdir -p "$OUT"/{drafts,verify,accept,fit,jobs,logs}

STEPS=($(python3 "$SRC_DIR/steps.py" --print steps))
CELLS=($(python3 "$SRC_DIR/steps.py" --print cells))
ANCHORS=($(printf '%s\n' "${CELLS[@]}" | cut -d: -f1 | sort -nu))
MAX_ANCHOR="${ANCHORS[-1]}"

for s in "${STEPS[@]}"; do
    [ -d "$WEIGHTS/step_$s" ] || { echo "missing checkpoint: $WEIGHTS/step_$s" >&2; exit 1; }
done

# Job 1: draft generation in eight-GPU waves.
inner="$OUT/jobs/drafts_inner.sh"
{
    echo '#!/bin/bash'
    echo "set -u; cd $REPO"
    gpu=0
    for c in "${STEPS[@]}"; do
        echo "[ -f $OUT/drafts/step_$c.jsonl.gz ] || ( export CUDA_VISIBLE_DEVICES=$gpu; python $SRC_DIR/gen_drafts.py --ckpt $WEIGHTS/step_$c --step $c --out-dir $OUT/drafts $GEN_ARGS ) > $OUT/logs/gen_step_$c.log 2>&1 &"
        gpu=$((gpu + 1))
        if [ "$gpu" -eq "$GPUS_PER_NODE" ]; then echo "wait"; gpu=0; fi
    done
    echo "wait"
    echo "MISSING=0"
    for c in "${STEPS[@]}"; do
        echo "[ -f $OUT/drafts/step_$c.jsonl.gz ] || { echo \"MISSING: drafts/step_$c (see logs/gen_step_$c.log)\"; MISSING=1; }"
    done
    echo 'exit $MISSING'
} > "$inner"

outer="$OUT/jobs/drafts.sh"
printf '#!/bin/bash\nset -euo pipefail\nbash %s bash %s\n' "$WRAP" "$inner" > "$outer"
draft_id=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" --qos="$QOS" --comment="$REAPER_COMMENT" -N1 --gpus-per-node=$GPUS_PER_NODE \
    -t 04:00:00 -J "extreme-off-policy-drafts-$OUT_NAME" -o "$OUT/logs/drafts_%j.log" "$outer")

# Job 2: one GPU per anchor, then microbenchmark, acceptance, and fitting.
inner="$OUT/jobs/verify_inner.sh"
{
    echo '#!/bin/bash'
    echo "set -u; cd $REPO"
    gpu=0
    for t in "${ANCHORS[@]}"; do
        chain="true"
        for cell in "${CELLS[@]}"; do
            [ "${cell%%:*}" = "$t" ] || continue
            c="${cell##*:}"
            chain+=" && { [ -f $OUT/verify/anchor_${t}_src_${c}.jsonl.gz ] || python $SRC_DIR/verify_drafts.py --ckpt $WEIGHTS/step_$t --anchor $t --drafts $OUT/drafts/step_$c.jsonl.gz --out-dir $OUT/verify; }"
        done
        echo "( export CUDA_VISIBLE_DEVICES=$gpu; $chain ) > $OUT/logs/verify_anchor_$t.log 2>&1 &"
        gpu=$((gpu + 1))
    done
    echo "[ -f $OUT/microbench.json ] || ( export CUDA_VISIBLE_DEVICES=$gpu; python $SRC_DIR/microbench_prefill_decode.py --ckpt $WEIGHTS/step_$MAX_ANCHOR --drafts $OUT/drafts/step_$MAX_ANCHOR.jsonl.gz -o $OUT/microbench.json ) > $OUT/logs/microbench.log 2>&1 &"
    echo "wait"
    echo "MISSING=0"
    for cell in "${CELLS[@]}"; do
        t="${cell%%:*}"; c="${cell##*:}"
        echo "[ -f $OUT/verify/anchor_${t}_src_${c}.jsonl.gz ] || { echo \"MISSING: verify/anchor_${t}_src_${c} (see logs/verify_anchor_$t.log)\"; MISSING=1; }"
    done
    echo "[ -f $OUT/microbench.json ] || { echo 'MISSING: microbench.json'; MISSING=1; }"
    echo '[ "$MISSING" -eq 0 ] || exit 1'
    echo "set -e"
    echo "python $SRC_DIR/accept_rules.py $OUT/verify/anchor_*.jsonl.gz --tokenizer $WEIGHTS/step_$MAX_ANCHOR -o $OUT/accept"
    echo "python $SRC_DIR/fit_alpha.py $OUT/accept/accept_*.jsonl.gz --microbench $OUT/microbench.json -o $OUT/fit"
} > "$inner"

outer="$OUT/jobs/verify_fit.sh"
printf '#!/bin/bash\nset -euo pipefail\nbash %s bash %s\n' "$WRAP" "$inner" > "$outer"
fit_id=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" --qos="$QOS" --comment="$REAPER_COMMENT" -N1 --gpus-per-node=$GPUS_PER_NODE \
    -t 04:00:00 -J "extreme-off-policy-verify-$OUT_NAME" -o "$OUT/logs/verify_fit_%j.log" \
    --dependency=afterok:"$draft_id" "$outer")

echo "submitted:"
echo "  job1 drafts (1 node, ${#STEPS[@]} checkpoints / 8 GPUs): $draft_id"
echo "  job2 verify+microbench+fit (1 node, afterok):          $fit_id"
echo "results -> $OUT/fit/   logs -> $OUT/logs/"
