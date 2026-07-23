"""Convert multi-turn SWE traces to verify_drafts-compatible drafts.

Concatenate node token IDs and record absolute sampled positions, their log q,
and turn numbers. Observation and tool-result tokens are deterministic and need
no verification, so acceptance applies only to model-sampled assistant tokens.
Rollouts with no nodes contain no recoverable token data and are skipped.

Usage (I/O only; safe on a login node):
    uv run experiments/extreme-off-policy/src/swe_trace_to_drafts.py \
        outputs/job-188448/run_default/rollouts/step_*/train/all/traces.jsonl \
        -o outputs/extreme-off-policy/swe_188448/drafts [--max-per-version 256]

Output: <out>/step_<policy_version>.jsonl.gz
    {task_idx, sample_idx, source_step, full_token_ids, sampled_positions,
     turn_idx, lp_q, finish_reason, provenance}
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def convert(trace: dict) -> dict | None:
    nodes = trace.get("nodes") or []
    if not nodes:
        return None  # Empty or failed rollout with no token data.
    full_ids: list[int] = []
    sampled_positions: list[int] = []
    turn_idx: list[int] = []
    lp_q: list[float] = []
    turn = 0
    for node in nodes:
        ids = node["token_ids"]
        if node.get("sampled") and node.get("logprobs"):
            # Sampled generation tokens are the suffix matching the logprob count.
            n_gen = len(node["logprobs"])
            gen_start = len(full_ids) + len(ids) - n_gen
            sampled_positions.extend(range(gen_start, gen_start + n_gen))
            turn_idx.extend([turn] * n_gen)
            lp_q.extend(node["logprobs"])
            turn += 1
        full_ids.extend(ids)
    if not sampled_positions:
        return None
    return {
        "task_idx": trace.get("task"),
        "sample_idx": None,
        "source_step": trace["policy_version"],
        "full_token_ids": full_ids,
        "sampled_positions": sampled_positions,
        "turn_idx": turn_idx,
        "lp_q": lp_q,
        "finish_reason": trace.get("stop_condition"),
        "provenance": "prod_trace_swe",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_files", type=Path, nargs="+")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--max-per-version", type=int, default=256)
    args = parser.parse_args()

    by_version: dict[int, list[dict]] = defaultdict(list)
    n_seen = n_dead = 0
    tok_total = tok_sampled = 0
    for path in args.trace_files:
        with open(path) as f:
            for line in f:
                trace = json.loads(line)
                if trace.get("kind") == "eval":
                    continue
                n_seen += 1
                rec = convert(trace)
                if rec is None:
                    n_dead += 1
                    continue
                tok_total += len(rec["full_token_ids"])
                tok_sampled += len(rec["sampled_positions"])
                v = rec["source_step"]
                if len(by_version[v]) < args.max_per_version:
                    rec["sample_idx"] = len(by_version[v])
                    by_version[v].append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for v, recs in sorted(by_version.items()):
        out = args.out_dir / f"step_{v}.jsonl.gz"
        with gzip.open(out, "wt") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        turns = [max(r["turn_idx"]) + 1 for r in recs]
        print(f"policy_version {v}: {len(recs)} drafts -> {out} (turns/trace mean {sum(turns)/len(turns):.1f})")
    frac = tok_sampled / tok_total if tok_total else 0.0
    print(f"traces: {n_seen} seen, {n_dead} skipped (empty or failed)")
    print(f"token composition: sampled {tok_sampled:,} / total {tok_total:,} = {frac:.1%} "
          f"(remaining {1-frac:.1%} is deterministic observation content)")


if __name__ == "__main__":
    main()
