"""Aggregate and plot SWE-RL wall-clock phases from traces.jsonl.

Convert absolute setup/generation/finalize/scoring timestamps to durations and
combine them with turn and reward statistics. This is I/O-only.

Usage:
    uv run python experiments/analysis/rollout_timing.py \
        outputs/job-174212/run_default/rollouts/step_1/train/all/traces.jsonl \
        [more traces.jsonl ...] [-o outputs/analysis/rollout_timing]

Outputs:
    - phase summary with share, mean, p50, p90, and max
    - <out>/phase_stacked.png
    - <out>/phase_hist.png
    - <out>/rollouts.csv
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

PHASES = ["setup", "generation", "finalize", "scoring"]


def load_rollouts(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.open():
            r = json.loads(line)
            t = r.get("timing") or {}
            if not all(p in t for p in PHASES):
                continue
            rec = {
                "id": r["id"],
                "env": r.get("env_name"),
                "kind": r.get("kind"),
                "reward": (r.get("rewards") or {}).get("solved"),
                "turns": sum(1 for n in r.get("nodes", []) if n["message"]["role"] == "assistant"),
                "tool_calls": sum(
                    len(n["message"].get("tool_calls") or []) for n in r.get("nodes", [])
                ),
                "start": t["start"],
                "source": str(path),
            }
            for p in PHASES:
                # Failed rollouts leave later timestamps at zero.
                phase = t[p]
                rec[p] = phase["end"] - phase["start"] if phase["end"] > 0 and phase["start"] > 0 else 0.0
            rec["total"] = sum(rec[p] for p in PHASES)
            rows.append(rec)
    return rows


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(int(len(values) * q), len(values) - 1)]


def print_summary(rows: list[dict]) -> None:
    total_all = sum(r["total"] for r in rows)
    print(f"rollouts: {len(rows)} | solved: {sum(1 for r in rows if r['reward'] == 1.0)}")
    print(f"turns: mean={statistics.mean(r['turns'] for r in rows):.1f} "
          f"p50={pct([r['turns'] for r in rows], 0.5)} p90={pct([r['turns'] for r in rows], 0.9)}")
    print()
    print(f"{'phase':<12}{'share':>7}{'mean':>9}{'p50':>9}{'p90':>9}{'max':>9}  (seconds)")
    for p in PHASES:
        vals = [r[p] for r in rows]
        share = sum(vals) / total_all * 100
        print(f"{p:<12}{share:>6.1f}%{statistics.mean(vals):>9.1f}"
              f"{pct(vals, 0.5):>9.1f}{pct(vals, 0.9):>9.1f}{max(vals):>9.1f}")
    totals = [r["total"] for r in rows]
    print(f"{'total':<12}{'100.0%':>7}{statistics.mean(totals):>9.1f}"
          f"{pct(totals, 0.5):>9.1f}{pct(totals, 0.9):>9.1f}{max(totals):>9.1f}")


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rollouts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"setup": "#8888cc", "generation": "#cc8844", "finalize": "#44aa44", "scoring": "#aa4444"}

    # Start-time ordering exposes stragglers and phase composition.
    ordered = sorted(rows, key=lambda r: r["start"])
    fig, ax = plt.subplots(figsize=(12, max(3, len(ordered) * 0.12)))
    for i, r in enumerate(ordered):
        left = 0.0
        for p in PHASES:
            ax.barh(i, r[p], left=left, color=colors[p], height=0.8)
            left += r[p]
    ax.set_xlabel("seconds")
    ax.set_ylabel("rollout (start-time order)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in PHASES], labels=PHASES)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_stacked.png", dpi=120)

    fig, axes = plt.subplots(1, len(PHASES), figsize=(4 * len(PHASES), 3), sharey=True)
    for ax, p in zip(axes, PHASES):
        ax.hist([r[p] for r in rows], bins=30, color=colors[p])
        ax.set_title(p)
        ax.set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out_dir / "phase_hist.png", dpi=120)
    print(f"\nwrote: {out_dir}/rollouts.csv, phase_stacked.png, phase_hist.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output directory; omit to print the summary only")
    args = parser.parse_args()

    rows = load_rollouts(args.traces)
    if not rows:
        raise SystemExit("no rollouts with timing found")
    print_summary(rows)
    if args.out is not None:
        write_outputs(rows, args.out)


if __name__ == "__main__":
    main()
