"""Summarize PD sweep arms in one comparison table.

Inputs may be Phase 1 driver directories containing summary.json or Phase 2
job directories containing logs/inference/*.log. Driver summaries report
throughput, latency, and deadline misses. Job summaries integrate vLLM logger
samples and derive batch duration from orchestrator.log.

Usage (I/O only; safe on a login node):
    uv run python experiments/rollout-simulation/summarize_sweep.py \
        outputs/rollout-simulation/replay_ep_off \
        outputs/rollout-simulation/replay_deepep_ll \
        outputs/job-XXXXXX [...]
"""

import argparse
import json
import re
import statistics
from pathlib import Path

VLLM_LOG = re.compile(
    r"INFO \d\d-\d\d (\d\d):(\d\d):(\d\d) \[loggers\.py:\d+\] Engine \d+: "
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs"
)
# Effective end of step 1: batch completion, or the first weight update.
BATCH_DONE = re.compile(
    r"(\d\d):(\d\d):(\d\d).*(?:Train batch (\d+)/\4 \(100|Updated weights to step 1 )"
)
ORCH_START = re.compile(r"(\d\d):(\d\d):(\d\d).*Starting orchestrator loop")


def hms(h, m, s) -> int:
    return int(h) * 3600 + int(m) * 60 + int(s)


def from_driver(path: Path) -> dict:
    s = json.loads((path / "summary.json").read_text())
    over_key = next(k for k in s if k.startswith("requests_over_"))
    return {
        "kind": "driver",
        "makespan_s": s["wall_s"],
        "gen_tok_s": s["gen_tok_per_s"],
        "per_req_tok_s": s["decode_tok_per_s_per_req"]["p50"],
        "p99_lat_s": s["latency_s"]["p99"],
        "over_deadline": s[over_key],
        "err": s["requests_err"],
    }


def from_job(path: Path) -> dict:
    gen_tok = 0.0
    rate_samples, running_samples = [], []
    for f in sorted((path / "logs" / "inference").glob("node_*.log*")):
        prev = None
        for line in open(f, errors="replace"):
            m = VLLM_LOG.search(line)
            if not m:
                continue
            t = hms(*m.groups()[:3])
            g, r = float(m.group(5)), int(m.group(6))
            if prev is not None and 0 < (t - prev) % 86400 < 60:
                gen_tok += g * ((t - prev) % 86400)
            prev = t
            if g > 0 and r > 0:
                rate_samples.append(g / r)
                running_samples.append(r)
    orch = path / "logs" / "orchestrator.log"
    start = batch_done = None
    if orch.exists():
        for line in open(orch, errors="replace"):
            if start is None and (m := ORCH_START.search(line)):
                start = hms(*m.groups())
            if m := BATCH_DONE.search(line):
                batch_done = hms(*m.groups()[:3])
                break
    step_s = (batch_done - start) % 86400 if start is not None and batch_done is not None else None
    return {
        "kind": "job",
        "makespan_s": step_s,
        "gen_tok_s": None,  # Job arms have different observation windows.
        "gen_tok_total_M": round(gen_tok / 1e6, 1),
        "per_req_tok_s": round(statistics.median(rate_samples), 1) if rate_samples else None,
        "mean_running_per_engine": round(statistics.mean(running_samples), 1) if running_samples else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.runs:
        if (path / "summary.json").is_file():
            row = from_driver(path)
        elif (path / "logs" / "inference").is_dir():
            row = from_job(path)
        else:
            print(f"skip {path}: neither summary.json nor logs/inference exists")
            continue
        rows.append((path.name, row))

    cols = ["kind", "makespan_s", "gen_tok_s", "gen_tok_total_M",
            "per_req_tok_s", "mean_running_per_engine", "p99_lat_s", "over_deadline", "err"]
    used = [c for c in cols if any(c in r for _, r in rows)]
    widths = {c: max(len(c), 10) for c in used}
    name_w = max((len(n) for n, _ in rows), default=4)
    print(f"{'arm':<{name_w}} | " + " | ".join(f"{c:>{widths[c]}}" for c in used))
    print("-" * (name_w + 3 + sum(widths[c] + 3 for c in used)))
    for name, r in rows:
        cells = [f"{r.get(c, ''):>{widths[c]}}" if r.get(c) is not None else " " * widths[c] for c in used]
        print(f"{name:<{name_w}} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
