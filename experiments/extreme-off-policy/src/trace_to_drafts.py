"""Convert production rollout traces to verify_drafts-compatible gold drafts.

Training traces preserve token IDs and per-token log probabilities from the
actual serving path. They provide the best staleness-zero noise control and
avoid offline chat-template mismatch. Later-anchor verification of these
training samples is biased by reuse during learning, so use offline drafts for
the primary positive-staleness measurement.

Usage (I/O only; safe on a login node):
    uv run experiments/extreme-off-policy/src/trace_to_drafts.py \
        outputs/math_qwen3_4b_instruct-latest/run_default/rollouts/step_*/train/all/traces.jsonl \
        -o outputs/extreme-off-policy/math_qwen3_4b_instruct-prod/drafts \
        [--max-per-version 256]

Output: <out>/step_<policy_version>.jsonl.gz using the gen_drafts.py schema.
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def convert(trace: dict) -> dict | None:
    nodes = trace.get("nodes") or []
    # Single-turn math: user prompt followed by the sampled assistant.
    if len(nodes) < 2 or not nodes[1].get("sampled"):
        return None
    prompt_node, gen_node = nodes[0], nodes[1]
    token_ids, logprobs = gen_node["token_ids"], gen_node["logprobs"]
    if not logprobs:
        return None
    # The assistant node may prefix unsampled generation-prompt tokens; sampled
    # generation tokens are the suffix matching the logprob count.
    n_gen = len(logprobs)
    return {
        "task_idx": trace.get("task"),
        "sample_idx": None,  # Assigned sequentially within each version.
        "source_step": trace["policy_version"],
        "prompt_token_ids": list(prompt_node["token_ids"]) + list(token_ids[:-n_gen]),
        "draft_token_ids": list(token_ids[-n_gen:]),
        "lp_q": list(logprobs),
        "finish_reason": gen_node.get("finish_reason"),
        "provenance": "prod_trace",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_files", type=Path, nargs="+")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--max-per-version", type=int, default=256)
    args = parser.parse_args()

    by_version: dict[int, list[dict]] = defaultdict(list)
    n_seen = n_skipped = 0
    for path in args.trace_files:
        with open(path) as f:
            for line in f:
                trace = json.loads(line)
                if trace.get("kind") == "eval":
                    continue
                n_seen += 1
                rec = convert(trace)
                if rec is None:
                    n_skipped += 1
                    continue
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
        print(f"policy_version {v}: {len(recs)} drafts -> {out}")
    print(f"traces: {n_seen} seen, {n_skipped} skipped (multi-turn or missing logprobs)")


if __name__ == "__main__":
    main()
