# Rollout performance simulation

This directory provides a controlled optimization loop for SWE-RL rollout
performance. Running the production workload is expensive because every
iteration normally provisions sandboxes, executes tools, runs the trainer, and
synchronizes weights. Those costs make it difficult to isolate inference and
rollout-scheduling changes.

The simulator derives a repeatable workload from recorded production traces
and replaces the expensive parts that are not needed for a performance
comparison. It is intended to answer systems questions such as:

- Which expert-parallel and all-to-all configuration has the best throughput?
- Which prefill:decode ratio works for the recorded ISL/OSL distribution?
- How much rollout concurrency can the serving stack sustain?
- Does an inference-only improvement survive orchestrator, trainer, and
  weight-sync pressure?

It does not measure model quality, convergence, sandbox correctness, or real
tool execution performance. See
[the experiment note](../notes/rollout_simulation.md) for measured results,
fidelity assumptions, and open questions.

## Two simulation levels

### 1. Inference-only fast loop

`build_replay_workload.py` converts production `traces.jsonl` records into
turn-by-turn replay plans. `replay_swe.py` sends those plans directly to the
real inference router.

This level removes the sandbox, environment server, orchestrator, and trainer.
It preserves recorded prompt token IDs, output lengths, session affinity,
grouped arrivals, and estimated tool-delay gaps. Use it for rapid comparison
of inference backends, P:D topology, and concurrency.

```bash
uv run python experiments/rollout-simulation/build_replay_workload.py \
    outputs/job-188448/run_default/rollouts/step_1/train/all/traces.jsonl \
    -o outputs/rollout-simulation/job-188448/replay_plans.jsonl

uv run python experiments/rollout-simulation/replay_swe.py \
    outputs/rollout-simulation/job-188448/replay_plans.jsonl \
    --base-url http://<router-host>:8000/v1 \
    --max-inflight 256 --group-launch \
    -o outputs/rollout-simulation/replay_ep_off
```

### 2. RL-stack integration simulation

`swe_replay_v1/` is a verifiers v1 plugin that replaces sandbox provisioning
and tool execution with recorded observations, delays, and rewards. Real model
requests still traverse the interception server, router, and inference
engines. The orchestrator, trainer, and weight synchronization remain active
so a fast-loop winner can be checked under the coupled RL system.

- `mode = "open"` reuses recorded assistant messages to keep later prompts
  deterministic.
- `mode = "closed"` feeds actual model responses into later turns while still
  substituting recorded tool observations.
- The taskset never provisions a sandbox.
- Recorded rewards make this a load simulation, not a learning-quality run.

The simulation training configs live under `training/`. Symlinks under
`experiments/training/` expose them to the shared submission entrypoint.

```bash
uv pip install -e experiments/rollout-simulation/swe_replay_v1
bash experiments/training/submit.sh swe_replay_bench
```

Use the integration level only after narrowing the serving choices with the
inference-only loop.

## PD sweep

The staged sweep moves from isolated serving comparisons to the integrated RL
stack:

```bash
bash experiments/rollout-simulation/pd_sweep/run_stage0.sh
bash experiments/rollout-simulation/pd_sweep/run_stage1.sh
bash experiments/rollout-simulation/pd_sweep/run_stage2.sh
```

- Stage 0 runs inference-only EP/all-to-all and inflight simulations.
- Stage 1 submits PD 1:1, PD 1:3, and colocated integration simulations.
- Stage 2 changes the concurrent-rollout load point.

Aggregate driver or job output directories with:

```bash
uv run python experiments/rollout-simulation/summarize_sweep.py \
    <output-dir> [<output-dir> ...]
```

## Fidelity boundaries

- Tool time is estimated from trace intervals and an assumed decode rate.
- Open-loop replay preserves load shape but not response-dependent control
  flow.
- Closed-loop replay uses recorded observations that may not match a newly
  generated tool call.
- Recorded rewards do not validate the quality of new responses.
- The workload represents the source run and must be regenerated when the
  task, model, sampling policy, or rollout distribution changes materially.

## Layout

| Path | Role |
| --- | --- |
| `build_replay_workload.py` | Convert real rollout traces into replay plans |
| `replay_swe.py` | Run the inference-only simulation |
| `swe_replay_v1/` | Replace sandbox/tool execution inside the RL stack |
| `training/` | Integration-simulation training configs |
| `pd_sweep/` | Staged serving and integration comparisons |
| `summarize_sweep.py` | Compare throughput, latency, and deadline misses |
