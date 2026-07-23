# Extreme Off-Policy feasibility experiment

This directory measures token acceptance when a current policy verifies drafts
from stale checkpoints, then fits a cost model for extreme off-policy RL.
Experiment history, model-selection decisions, and current conclusions belong
in [the experiment note](../notes/extreme-off-policy.md). The preregistered decision
gate is [GO_CRITERIA.md](GO_CRITERIA.md).

The training run is a checkpoint producer. Draft generation, verification,
acceptance rules, and analysis happen offline. Sampling is fixed to
`temperature=1.0` and `top_p=1.0`, which makes the recorded raw log probability
the effective draft probability used by strict acceptance.

`steps.py` is the source of truth for anchor/staleness cells and required
checkpoints:

```bash
uv run experiments/extreme-off-policy/src/steps.py
```

GPU phases should run in the same sqsh image as training. The wrapper configures
srun, Pyxis mounts, and `/app/.venv`:

```bash
bash experiments/extreme-off-policy/script/run_in_container.sh <command> [args...]
```

## Workflow

### 1. Produce checkpoints

Start with a bounded validation run, then launch or resume the full run:

```bash
bash experiments/training/submit.sh math_qwen3_4b_instruct validation
bash experiments/training/submit.sh math_qwen3_4b_instruct
bash experiments/training/submit.sh math_qwen3_4b_instruct --ckpt.resume-step -1
```

The Qwen3-4B-Instruct-2507 configuration retains the checkpoints required by
the measurement grid.

### 2. Generate drafts

Generate each source checkpoint once:

```bash
bash experiments/extreme-off-policy/script/run_in_container.sh \
    python experiments/extreme-off-policy/src/gen_drafts.py \
    --ckpt <weights_dir>/step_<source> --step <source> \
    --out-dir outputs/extreme-off-policy/<run>/drafts
```

Confirm that prompt token IDs match a training rollout trace before launching
the full grid.

### 3. Verify drafts

Verify every source for an anchor checkpoint:

```bash
bash experiments/extreme-off-policy/script/run_in_container.sh \
    python experiments/extreme-off-policy/src/verify_drafts.py \
    --ckpt <weights_dir>/step_<anchor> --anchor <anchor> \
    --drafts outputs/extreme-off-policy/<run>/drafts/step_<source>.jsonl.gz \
    --out-dir outputs/extreme-off-policy/<run>/verify
```

Passing the anchor's own draft file supplies the staleness-zero control.

### 4. Apply acceptance rules and fit the model

```bash
bash experiments/extreme-off-policy/script/run_in_container.sh \
    python experiments/extreme-off-policy/src/microbench_prefill_decode.py \
    --ckpt <weights_dir>/step_<anchor> \
    --drafts outputs/extreme-off-policy/<run>/drafts/step_<anchor>.jsonl.gz \
    -o outputs/extreme-off-policy/<run>/microbench.json

bash experiments/extreme-off-policy/script/run_in_container.sh \
    python experiments/extreme-off-policy/src/accept_rules.py \
    outputs/extreme-off-policy/<run>/verify/anchor_*.jsonl.gz \
    --tokenizer <weights_dir>/step_<anchor> \
    -o outputs/extreme-off-policy/<run>/accept

bash experiments/extreme-off-policy/script/run_in_container.sh \
    python experiments/extreme-off-policy/src/fit_alpha.py \
    outputs/extreme-off-policy/<run>/accept/accept_*.jsonl.gz \
    --microbench outputs/extreme-off-policy/<run>/microbench.json \
    -o outputs/extreme-off-policy/<run>/fit
```

The output includes acceptance by staleness and position, accepted-prefix
survival, selected-token log ratios, token-KL estimates, verification timing,
and predicted end-to-end speedup.

## Components

| Path | Purpose | Execution |
|---|---|---|
| `training/` | Phase 1 checkpoint-producing configurations | `submit.sh` |
| `src/steps.py` | Measurement grid and checkpoint pruning | login node |
| `src/gen_drafts.py` | Generate stale drafts and record `log q(t)` | GPU node |
| `src/verify_drafts.py` | Compute anchor log probabilities and ranks | GPU node |
| `src/accept_rules.py` | Apply acceptance rules post hoc | login node |
| `src/fit_alpha.py` | Fit acceptance and speedup models | login node |
| `src/microbench_prefill_decode.py` | Measure prefill/decode cost ratio | GPU node |
| `script/` | Container and Slurm entrypoints | login or GPU node |
