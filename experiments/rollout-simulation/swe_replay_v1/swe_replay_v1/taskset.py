"""Replay recorded SWE trajectories against the real inference stack.

The taskset reads tasks from a prior run's traces.jsonl without using a
sandbox. It returns recorded rewards, while SWEReplayHarness substitutes
recorded tool observations and sleeps. Model requests still traverse the real
interception, router, and PD engines, preserving the orchestrator, scheduler,
and weight-sync paths for inference A/B comparisons.

Task data carries only trajectory IDs and the source path. The harness reads
nodes by byte offset from the worker-local source file to avoid bloating wire
messages and output traces.
"""

import json
from pathlib import Path

import verifiers.v1 as vf

__all__ = ["SWEReplayTaskset"]


class SWEReplayConfig(vf.TasksetConfig):
    source: str = ""
    """Source traces.jsonl from rollouts/step_*/train/all/traces.jsonl."""
    include_errored: bool = False
    """Include interrupted trajectories such as 504 failures."""
    min_turns: int = 1
    max_tasks: int | None = None
    """Limit loading to the first N tasks for smoke runs."""


class SWEReplayState(vf.State):
    replay_trajectory_id: str | None = None
    replay_reward: float | None = None


class SWEReplayData(vf.TaskData):
    source: str
    """Path to traces.jsonl as visible from the environment worker."""
    trajectory_ids: list[str]
    """Recorded trajectories claimed by members of this task group."""
    source_task_idx: int
    """Task index in the source run."""


class SWEReplayTask(vf.Task[SWEReplayData, SWEReplayState]):
    NEEDS_CONTAINER = False

    @vf.reward(weight=1.0)
    async def replay_reward(self, trace) -> float:
        """Return the reward recorded for the trajectory claimed by the harness."""
        return trace.state.replay_reward or 0.0


def scan_source(source: Path, include_errored: bool, min_turns: int):
    """Collect trajectory metadata by task in one pass over traces.jsonl.

    Nodes are not retained. Returns {task_idx: [(traj_id, prompt, n_turns)]}.
    """
    by_task: dict[int, list[tuple[str, str, int]]] = {}
    for line in source.open():
        rec = json.loads(line)
        if rec["errors"] and not include_errored:
            continue
        turns = sum(1 for n in rec["nodes"] if n["message"]["role"] == "assistant")
        if turns < min_turns:
            continue
        prompt = next(
            (n["message"]["content"] for n in rec["nodes"] if n["message"]["role"] == "user"),
            None,
        )
        if prompt is None:
            continue
        task_idx = rec["task"]["data"]["idx"]
        by_task.setdefault(task_idx, []).append((rec["id"], prompt, turns))
    return by_task


class SWEReplayTaskset(vf.Taskset[SWEReplayTask, SWEReplayConfig]):
    def load(self) -> list[SWEReplayTask]:
        source = Path(self.config.source)
        if not source.is_file():
            raise FileNotFoundError(f"swe-replay-v1: source not found: {source}")
        by_task = scan_source(source, self.config.include_errored, self.config.min_turns)
        tasks = []
        for i, task_idx in enumerate(sorted(by_task)):
            if self.config.max_tasks is not None and i >= self.config.max_tasks:
                break
            trajs = by_task[task_idx]
            tasks.append(
                SWEReplayTask(
                    SWEReplayData(
                        idx=i,
                        name=f"replay-{task_idx}",
                        prompt=trajs[0][1],
                        source=str(source),
                        trajectory_ids=[t[0] for t in trajs],
                        source_task_idx=task_idx,
                    ),
                    self.config.task,
                )
            )
        return tasks
