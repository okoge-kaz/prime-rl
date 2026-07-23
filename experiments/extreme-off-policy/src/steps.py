"""Source of truth for anchor/source cells and required checkpoints.

Generate a draft once per source checkpoint, then verify it for each applicable
anchor. Staleness is anchor minus source, and source reuse reduces generation.

CLI:
    uv run experiments/extreme-off-policy/src/steps.py
    uv run experiments/extreme-off-policy/src/steps.py --prune <weights_dir> [--dry-run]
"""

import argparse
import os
import shutil
from pathlib import Path

# Include early and late anchors to measure the same staleness across training.
# EXTREME_OFF_POLICY_ANCHORS may override this list when a run ends early.
ANCHORS = [
    int(s)
    for s in os.environ.get(
        "EXTREME_OFF_POLICY_ANCHORS", "18 26 34 42 50"
    ).split()
]

# Staleness zero measures the numerical noise floor.
STALENESS = [0, 1, 2, 4, 8, 16, 32]


def measurement_cells() -> list[tuple[int, int]]:
    """Return all valid (anchor, source) pairs with source at least one."""
    cells = []
    for t in ANCHORS:
        for s in STALENESS:
            c = t - s
            if c >= 1:
                cells.append((t, c))
    return cells


def needed_steps() -> list[int]:
    """Return checkpoint steps needed for draft generation or verification."""
    steps = set()
    for t, c in measurement_cells():
        steps.add(t)
        steps.add(c)
    return sorted(steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prune", type=Path, default=None, metavar="WEIGHTS_DIR")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print", dest="print_what", choices=["steps", "cells"], default=None,
                        help="Machine-readable steps or anchor:source cells")
    args = parser.parse_args()

    if args.print_what == "steps":
        print(" ".join(str(s) for s in needed_steps()))
        return
    if args.print_what == "cells":
        print(" ".join(f"{t}:{c}" for t, c in measurement_cells()))
        return

    keep = set(needed_steps())
    print(f"anchors: {ANCHORS}")
    print(f"staleness: {STALENESS}")
    print(f"measurement cells (anchor, source): {measurement_cells()}")
    print(f"needed checkpoint steps ({len(keep)}): {sorted(keep)}")

    if args.prune is None:
        return
    for step_dir in sorted(args.prune.glob("step_*")):
        step = int(step_dir.name.removeprefix("step_"))
        if step in keep:
            continue
        if args.dry_run:
            print(f"would remove {step_dir}")
        else:
            print(f"removing {step_dir}")
            shutil.rmtree(step_dir)


if __name__ == "__main__":
    main()
