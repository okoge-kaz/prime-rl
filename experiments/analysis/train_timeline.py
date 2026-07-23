"""Plot overlap among async-RL rollouts, trainer steps, and weight updates.

Join three sources from a job output directory:
  - logs/trainer/node_0.log      : "Starting training step N" / "Step N | Xm Ys | ..." (SUCCESS)
  - logs/orchestrator.log        : "Updating weights to step N" / "Updated weights to step N in Xs"
  - run_default/rollouts/step_*/train/all/traces.jsonl : rollout spans and policy versions

Log timestamps are HH:MM:SS in the host timezone, while traces use epoch seconds.
Both are converted to local seconds-of-day. Runs crossing midnight are unsupported.

Usage:
    uv run python experiments/analysis/train_timeline.py outputs/job-174212 \
        [-o outputs/analysis/job-174212-timeline]
"""

import argparse
import json
import re
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TRAINER_START = re.compile(r"(\d\d:\d\d:\d\d).*Starting training step (\d+)")
# The duration is formatted as "2h 35m", "3m 42s", or "42s".
TRAINER_DONE = re.compile(r"(\d\d:\d\d:\d\d).*Step (\d+) \|\s*(?:\d+h\s*)?(?:\d+m\s*)?(?:\d+s)?\s*\|")
WEIGHT_BEGIN = re.compile(r"(\d\d:\d\d:\d\d).*Updating weights to step (\d+)")
WEIGHT_DONE = re.compile(r"(\d\d:\d\d:\d\d).*Updated weights to step (\d+) in ([\d.]+)s")


def hms_to_sec(hms: str) -> float:
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


PHASES_FOR_END = ("setup", "generation", "finalize", "scoring")

# Error colors; successful rollouts retain the policy-version tab10 color.
ERROR_COLORS = {
    "timeout_504": "crimson",
    "tunnel_lost": "magenta",
    "sandbox": "dimgray",
    "other": "saddlebrown",
}


def error_kind(r: dict) -> str | None:
    if not r.get("errors"):
        return None
    msg = str(r["errors"][0])
    if "Tunnel not found" in msg:
        return "tunnel_lost"
    if "504" in msg[:2000]:
        return "timeout_504"
    if "andbox" in msg[:120]:
        return "sandbox"
    return "other"


def epoch_to_sec_of_day(epoch: float) -> float:
    lt = time.localtime(epoch)
    return lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec + (epoch % 1)


def parse_job(job_dir: Path) -> dict:
    trainer_steps = {}  # step -> {start, end}
    for raw in (job_dir / "logs/trainer/node_0.log").open():
        line = ANSI.sub("", raw)
        if m := TRAINER_START.search(line):
            trainer_steps.setdefault(int(m.group(2)), {})["start"] = hms_to_sec(m.group(1))
        elif m := TRAINER_DONE.search(line):
            trainer_steps.setdefault(int(m.group(2)), {})["end"] = hms_to_sec(m.group(1))

    weight_syncs = {}  # step -> {begin, end, duration}
    for raw in (job_dir / "logs/orchestrator.log").open():
        line = ANSI.sub("", raw)
        if m := WEIGHT_BEGIN.search(line):
            weight_syncs.setdefault(int(m.group(2)), {})["begin"] = hms_to_sec(m.group(1))
        elif m := WEIGHT_DONE.search(line):
            weight_syncs.setdefault(int(m.group(2)), {}).update(
                end=hms_to_sec(m.group(1)), duration=float(m.group(3))
            )

    rollouts = []
    for f in sorted(job_dir.glob("run_default/rollouts/step_*/train/all/traces.jsonl")):
        for line in f.open():
            r = json.loads(line)
            t = r.get("timing") or {}
            if "start" not in t or "scoring" not in t:
                continue
            # Failed rollouts leave later phase ends at zero. Use the last
            # positive phase end as the effective completion time.
            end = max(
                (t[p]["end"] for p in PHASES_FOR_END if p in t and t[p]["end"] > 0),
                default=t["start"],
            )
            rollouts.append(
                {
                    "start": epoch_to_sec_of_day(t["start"]),
                    "end": epoch_to_sec_of_day(end),
                    "policy_version": r.get("policy_version", 0),
                    "reward": (r.get("rewards") or {}).get("solved"),
                    "error": error_kind(r),
                }
            )
    return {"trainer_steps": trainer_steps, "weight_syncs": weight_syncs, "rollouts": rollouts}


def print_summary(data: dict) -> None:
    print(f"{'step':>5} {'trainer (s)':>12} {'weight sync (s)':>16} {'inflight rollouts during step':>30}")
    for step in sorted(data["trainer_steps"]):
        ts = data["trainer_steps"][step]
        dur = ts["end"] - ts["start"] if "start" in ts and "end" in ts else float("nan")
        ws = data["weight_syncs"].get(step, {})
        overlap = sum(
            1
            for r in data["rollouts"]
            if "start" in ts and "end" in ts and r["start"] < ts["end"] and r["end"] > ts["start"]
        )
        print(f"{step:>5} {dur:>12.0f} {ws.get('duration', float('nan')):>16.1f} {overlap:>30}")
    versions = sorted({r["policy_version"] for r in data["rollouts"]})
    print(f"rollouts: {len(data['rollouts'])} | policy versions seen: {versions}")


def plot(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    rollouts = sorted(data["rollouts"], key=lambda r: r["start"])
    t0 = min(
        [r["start"] for r in rollouts]
        + [s["start"] for s in data["trainer_steps"].values() if "start" in s]
    )
    cmap = colormaps["tab10"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(rollouts) * 0.06)))
    for i, r in enumerate(rollouts):
        err = r.get("error")
        ax.plot(
            [r["start"] - t0, r["end"] - t0],
            [i, i],
            color=ERROR_COLORS[err] if err else cmap(r["policy_version"] % 10),
            lw=1.2,
            alpha=0.9 if err else 0.6,
        )
    top = len(rollouts)
    for step, ts in data["trainer_steps"].items():
        if "start" in ts and "end" in ts:
            ax.axvspan(ts["start"] - t0, ts["end"] - t0, color="gray", alpha=0.12)
            ax.text((ts["start"] + ts["end"]) / 2 - t0, top * 1.02, f"train step {step}",
                    ha="center", fontsize=8)
    for step, ws in data["weight_syncs"].items():
        if "begin" in ws:
            ax.axvline(ws["begin"] - t0, color="red", lw=1.5, ls="--")
            ax.text(ws["begin"] - t0, top * 1.06, f"sync→v{step}", color="red",
                    ha="center", fontsize=8)
    versions = sorted({r["policy_version"] for r in rollouts if not r.get("error")})
    kinds = [k for k in ERROR_COLORS if any(r.get("error") == k for r in rollouts)]
    handles = [
        plt.Line2D([], [], color=cmap(v % 10), lw=3, label=f"ok (policy v{v})") for v in versions
    ] + [plt.Line2D([], [], color=ERROR_COLORS[k], lw=3, label=k) for k in kinds]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_xlabel("seconds since first event")
    ax.set_ylabel("rollout (ok: policy_version color / error: kind color)")
    ax.set_ylim(-1, top * 1.1)
    fig.tight_layout()
    fig.savefig(out_dir / "train_timeline.png", dpi=120)
    print(f"wrote: {out_dir}/train_timeline.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args()

    data = parse_job(args.job_dir)
    print_summary(data)
    if args.out is not None:
        plot(data, args.out)


if __name__ == "__main__":
    main()
