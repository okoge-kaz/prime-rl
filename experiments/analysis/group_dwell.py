"""Measure GRPO group-completion dwell time from swe.log.

For group_size=G, advantage computation waits for all G sibling rollouts.
A completed rollout's dwell is the final sibling completion time minus its own
completion time. This also works on a live job because swe.log is appended eagerly.

Usage:
    uv run python experiments/analysis/group_dwell.py \
        outputs/job-175771/logs/envs/train/swe.log --group-size 16
"""

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

DONE = re.compile(
    r"(\d\d:\d\d:\d\d).*rollout done: id=\S+ task=(\d+) reward=([\d.]+) turns=(\d+)"
)
START = re.compile(r"(\d\d:\d\d:\d\d).*rollout start: id=\S+ task=(\d+)")


def hms(s: str) -> int:
    h, m, sec = map(int, s.split(":"))
    return h * 3600 + m * 60 + sec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("swe_log", type=Path)
    parser.add_argument("--group-size", type=int, default=16)
    args = parser.parse_args()

    starts: dict[str, list[int]] = defaultdict(list)
    dones: dict[str, list[tuple[int, float]]] = defaultdict(list)
    now = 0
    for raw in args.swe_log.open():
        if m := DONE.search(raw):
            dones[m.group(2)].append((hms(m.group(1)), float(m.group(3))))
            now = max(now, hms(m.group(1)))
        elif m := START.search(raw):
            starts[m.group(2)].append(hms(m.group(1)))
            now = max(now, hms(m.group(1)))

    complete, dwell_all, span_all = 0, [], []
    print(f"{'task':>6} {'done':>7} {'span so far (s)':>16}  state")
    for task in sorted(dones, key=lambda t: -len(dones[t])):
        ts = sorted(t for t, _ in dones[task])
        n = len(ts)
        if n >= args.group_size:
            group = ts[: args.group_size]
            last = group[-1]
            dwell_all.extend(last - t for t in group)
            span_all.append(last - group[0])
            complete += 1
            state = "COMPLETE"
            span = last - group[0]
        else:
            state = f"waiting ({args.group_size - n} more)"
            span = now - ts[0]  # Current lower bound since the first completion.
            dwell_all_live = [now - t for t in ts]
            dwell_all.extend(dwell_all_live)
        print(f"{task:>6} {n:>4}/{args.group_size:<2} {span:>16}  {state}")

    print()
    print(f"groups complete: {complete} | tasks in flight: {len(dones) - complete}")
    if dwell_all:
        print(
            f"buffered dwell (s): mean={statistics.mean(dwell_all):.0f} "
            f"p50={statistics.median(dwell_all):.0f} max={max(dwell_all)}"
            f"{'  (incomplete groups: lower bounds)' if complete == 0 else ''}"
        )
    if span_all:
        print(f"group span first→last done (s): mean={statistics.mean(span_all):.0f} max={max(span_all)}")


if __name__ == "__main__":
    main()
