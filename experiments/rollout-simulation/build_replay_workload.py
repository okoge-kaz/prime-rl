"""Build inference-only replay plans from traces.jsonl.

Each rollout becomes a sequence of turns containing:
  - append_token_ids: tokens appended before issuing the turn
  - max_tokens: the recorded completion length, replayed with ignore_eos
  - tool_sleep_s: the estimated tool time before the turn

Using token IDs stored in trace nodes preserves the input lengths and shared
prefixes without reproducing the chat template. Tool time cannot be separated
directly from the trace timestamps, so it is approximated as:

  tool_sleep_s = max(0, turn_dt - max_tokens / assumed_decode_rate)

The default assumed decode rate, 8.2 tok/s, comes from job 188448.

Usage (I/O only; safe on a login node):
    uv run python experiments/rollout-simulation/build_replay_workload.py \
        outputs/job-188448/run_default/rollouts/step_1/train/all/traces.jsonl \
        -o outputs/rollout-simulation/job-188448/replay_plans.jsonl \
        [--include-errored] [--min-turns 1] [--sample 256] [--seed 0]
"""

import argparse
import json
import random
import statistics
from pathlib import Path


def build_plan(rec: dict) -> dict | None:
    nodes = rec["nodes"]
    if not nodes:
        return None

    turns = []
    pending: list[int] = []  # Tokens appended before the next assistant turn.
    prev_ts: float | None = None
    for node in nodes:
        token_ids = node["token_ids"] or []
        if node["message"]["role"] != "assistant":
            pending.extend(token_ids)
            continue
        usage = node["usage"] or {}
        osl = usage.get("completion_tokens") or len(token_ids)
        dt = node["timestamp"] - prev_ts if prev_ts is not None else None
        turns.append(
            {
                "append_token_ids": pending,
                "max_tokens": osl,
                "turn_dt_s": dt,
                "recorded_prompt_tokens": usage.get("prompt_tokens"),
            }
        )
        prev_ts = node["timestamp"]
        pending = token_ids  # Open loop: append the recorded output to the next prompt.

    if not turns:
        return None

    timing = rec["timing"]
    setup = timing["setup"]
    return {
        "trajectory_id": rec["id"],
        "task_idx": rec["task"]["data"].get("idx"),
        "stop_condition": rec["stop_condition"],
        "errored": bool(rec["errors"]),
        "reward": (rec.get("rewards") or {}).get("reward"),
        "setup_s": max(0.0, setup["end"] - setup["start"]) if setup["end"] > 0 else 0.0,
        "turns": turns,
    }


def attach_tool_sleeps(plan: dict, rate: float) -> None:
    for turn in plan["turns"]:
        dt = turn.pop("turn_dt_s")
        turn["tool_sleep_s"] = round(max(0.0, dt - turn["max_tokens"] / rate), 3) if dt else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--include-errored", action="store_true",
                        help="Include interrupted rollouts, such as 504 failures, as partial load")
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None, help="Number of task groups to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--assumed-decode-rate", type=float, default=8.2,
                        help="Per-request decode tok/s used to estimate tool time (job 188448)")
    args = parser.parse_args()

    plans = []
    for path in args.traces:
        for line in path.open():
            rec = json.loads(line)
            if rec["errors"] and not args.include_errored:
                continue
            plan = build_plan(rec)
            if plan is None or len(plan["turns"]) < args.min_turns:
                continue
            if args.max_turns and len(plan["turns"]) > args.max_turns:
                continue
            attach_tool_sleeps(plan, args.assumed_decode_rate)
            plans.append(plan)

    if args.sample is not None:
        by_task: dict[int, list[dict]] = {}
        for plan in plans:
            by_task.setdefault(plan["task_idx"], []).append(plan)
        rng = random.Random(args.seed)
        tasks = rng.sample(sorted(by_task), min(args.sample, len(by_task)))
        plans = [p for t in tasks for p in by_task[t]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for plan in plans:
            f.write(json.dumps(plan) + "\n")

    turns = [len(p["turns"]) for p in plans]
    osl = [t["max_tokens"] for p in plans for t in p["turns"]]
    isl = [t["recorded_prompt_tokens"] for p in plans for t in p["turns"] if t["recorded_prompt_tokens"]]
    sleeps = [t["tool_sleep_s"] for p in plans for t in p["turns"]]
    ntask = len({p["task_idx"] for p in plans})
    print(f"plans: {len(plans)} (tasks={ntask}) turns/plan mean={statistics.mean(turns):.1f} max={max(turns)}")
    print(f"turns total: {len(osl)}  OSL mean={statistics.mean(osl):.0f} max={max(osl)}")
    print(f"ISL mean={statistics.mean(isl):.0f} max={max(isl)}")
    print(f"tool_sleep mean={statistics.mean(sleeps):.1f}s p90={sorted(sleeps)[int(len(sleeps) * 0.9)]:.1f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
