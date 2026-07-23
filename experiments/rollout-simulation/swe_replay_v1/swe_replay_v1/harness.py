"""Replay recorded trajectories instead of running a sandbox.

The harness sends chat completions directly to the interception endpoint. Each
turn adds recorded tool results, optionally sleeps for estimated tool time, and
issues a real model request. Open mode discards that response and appends the
recorded assistant message for deterministic prompts. Closed mode appends the
actual response and substitutes only recorded tool observations.

The environment sampling config controls generation length on the training
path. ``replay_swe.py`` is the separate driver for forcing recorded output
lengths. Because trace timestamps combine tool and model time, sleep is
estimated as:

  max(0, turn_dt - recorded_osl / assumed_decode_rate) * tool_scale

The default 8.2 tok/s rate comes from job 188448.
"""

import asyncio
import json
from typing import Literal

import verifiers.v1 as vf
from openai import AsyncOpenAI
from verifiers.v1.runtimes import ProgramResult

__all__ = ["SWEReplayHarness"]

# Match DefaultHarness tool definitions so the renderer reproduces the prompt.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command and return its combined stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        },
    },
}
EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a unique string in a file. old_str must appear exactly once in the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative to cwd or absolute).",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to find (must appear exactly once).",
                },
                "new_str": {"type": "string", "description": "Replacement string."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}
TOOLS = [BASH_TOOL, EDIT_TOOL]


class SWEReplayHarnessConfig(vf.HarnessConfig):
    mode: Literal["open", "closed"] = "open"
    tool_scale: float = 1.0
    """Tool-sleep multiplier; zero disables sleeping."""
    assumed_decode_rate: float = 8.2
    """Per-request decode tok/s used to estimate tool time."""
    max_turns: int | None = None
    request_timeout: float = 3600.0


def _wire(message: dict) -> dict:
    """Convert a recorded node message to the chat-completions wire format."""
    return {k: v for k, v in message.items() if v is not None}


class SWEReplayHarness(vf.Harness[SWEReplayHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MESSAGE_PROMPT = True

    def __init__(self, config: SWEReplayHarnessConfig) -> None:
        super().__init__(config)
        self._offsets: dict[str, dict[str, int]] = {}  # source -> {traj_id: byte offset}
        self._claims: dict[tuple[str, int], int] = {}  # (source, task_idx) -> claim counter
        self._lock = asyncio.Lock()

    async def _claim(self, data) -> dict:
        """Read one recorded trajectory for this task in round-robin order."""
        async with self._lock:
            if data.source not in self._offsets:
                offsets = {}
                with open(data.source, "rb") as f:
                    pos = 0
                    for line in f:
                        # IDs normally occupy a fixed prefix; parse JSON only for nonstandard records.
                        traj_id = line[7:39].decode(errors="replace")
                        if not (line.startswith(b'{"id":"') and traj_id.isalnum()):
                            traj_id = json.loads(line)["id"]
                        offsets[traj_id] = pos
                        pos += len(line)
                self._offsets[data.source] = offsets
            key = (data.source, data.source_task_idx)
            n = self._claims.get(key, 0)
            self._claims[key] = n + 1
        traj_id = data.trajectory_ids[n % len(data.trajectory_ids)]
        with open(data.source, "rb") as f:
            f.seek(self._offsets[data.source][traj_id])
            return json.loads(f.readline())

    async def launch(
        self,
        ctx,
        trace,
        runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        rec = await self._claim(trace.task.data)
        trace.state.replay_trajectory_id = rec["id"]
        trace.state.replay_reward = sum((rec.get("rewards") or {}).values())

        # Turn structure: [leading system/user/tool messages] -> assistant.
        turns: list[dict] = []  # {"pre": [messages], "assistant": message, "sleep_s": float}
        pre: list[dict] = []
        prev_ts: float | None = None
        for node in rec["nodes"]:
            msg = node["message"]
            if msg["role"] == "system":
                continue  # The interception renderer supplies the system prompt.
            if msg["role"] != "assistant":
                pre.append(_wire(msg))
                continue
            osl = (node.get("usage") or {}).get("completion_tokens") or 0
            dt = node["timestamp"] - prev_ts if prev_ts is not None else 0.0
            sleep_s = max(0.0, dt - osl / self.config.assumed_decode_rate)
            turns.append({"pre": pre, "assistant": _wire(msg), "sleep_s": sleep_s})
            prev_ts = node["timestamp"]
            pre = []

        if self.config.max_turns is not None:
            turns = turns[: self.config.max_turns]

        client = AsyncOpenAI(
            base_url=endpoint, api_key=secret, timeout=self.config.request_timeout, max_retries=0
        )
        messages: list[dict] = []
        for i, turn in enumerate(turns):
            messages.extend(turn["pre"])
            if i > 0 and self.config.tool_scale > 0:
                await asyncio.sleep(turn["sleep_s"] * self.config.tool_scale)
            completion = await client.chat.completions.create(
                model=ctx.model, messages=messages, tools=TOOLS
            )
            if trace.stop_condition is not None:
                break  # The interception layer stopped the rollout.
            if self.config.mode == "open":
                messages.append(turn["assistant"])
            else:
                reply = completion.choices[0].message.model_dump(exclude_none=True)
                messages.append(reply)
                calls = reply.get("tool_calls") or []
                if not calls:
                    break
                # Attach recorded observations to the actual call IDs for the next turn.
                nxt = turns[i + 1]["pre"] if i + 1 < len(turns) else []
                obs = [m for m in nxt if m.get("role") == "tool"]
                for call, o in zip(calls, obs):
                    o["tool_call_id"] = call["id"]
                nxt[:] = [m for m in nxt if m.get("role") != "tool"] + obs[: len(calls)]

        await client.close()
        return ProgramResult(exit_code=0, stdout="", stderr="")
