"""Replay a recorded SWE workload against the inference stack.

The driver reads plans.jsonl from build_replay_workload.py and replays each
trajectory through vllm-router in an open loop. It sends recorded token IDs to
/v1/completions, reproduces recorded output lengths with ignore_eos, sleeps for
estimated tool time between turns, and preserves consistent-hash routing with
X-Session-ID. ``--max-inflight`` controls rollout concurrency, while
``--group-launch`` reproduces the grouped arrival pattern used by GRPO.

No sandbox, tunnel, or trainer participates, so the same workload can compare
serving configurations such as EP, all-to-all backends, P:D ratios, and
concurrency.

Usage (CPU only, from a node that can reach the router):
    uv run python experiments/rollout-simulation/replay_swe.py \
        outputs/rollout-simulation/job-188448/replay_plans.jsonl \
        --base-url http://pool0-XXXX:8000/v1 --model PrimeIntellect/INTELLECT-3 \
        --max-inflight 256 --group-launch \
        -o outputs/rollout-simulation/replay_run1

Outputs: per-request measurements in <out>/requests.jsonl, <out>/summary.json,
and a console summary.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import httpx


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0.0


class Replayer:
    def __init__(self, args):
        self.args = args
        headers = {}
        api_key = os.environ.get(args.api_key_var)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(
            base_url=args.base_url, headers=headers, timeout=httpx.Timeout(args.request_timeout)
        )
        self.sem = asyncio.Semaphore(args.max_inflight)
        self.records = []
        self.inflight = 0
        self.done_plans = 0
        self.total_plans = 0
        self.gen_tokens = 0
        self.t0 = 0.0

    async def run_turn(self, plan: dict, turn_idx: int, prompt: list[int], max_tokens: int) -> dict:
        payload = {
            "model": self.args.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": self.args.temperature,
            "stream": True,
        }
        if self.args.logprobs:
            payload["logprobs"] = 0
        headers = {"X-Session-ID": plan["trajectory_id"]}
        t_start = time.monotonic()
        t_first = None
        async with self.client.stream("POST", "/completions", json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data:") and t_first is None:
                    t_first = time.monotonic()
        t_end = time.monotonic()
        self.gen_tokens += max_tokens
        return {
            "trajectory_id": plan["trajectory_id"],
            "task_idx": plan["task_idx"],
            "turn": turn_idx,
            "prompt_tokens": len(prompt),
            "max_tokens": max_tokens,
            "latency_s": round(t_end - t_start, 3),
            "ttft_s": round(t_first - t_start, 3) if t_first else None,
            "start_s": round(t_start - self.t0, 3),
        }

    async def run_plan(self, plan: dict) -> None:
        async with self.sem:
            self.inflight += 1
            try:
                if self.args.setup_scale > 0:
                    await asyncio.sleep(plan["setup_s"] * self.args.setup_scale)
                prompt: list[int] = []
                for i, turn in enumerate(plan["turns"]):
                    if (
                        self.args.max_duration_s
                        and time.monotonic() - self.t0 > self.args.max_duration_s
                    ):
                        break  # Stop issuing turns after the steady-state measurement window.
                    prompt = prompt + turn["append_token_ids"]
                    if len(prompt) + turn["max_tokens"] > self.args.max_model_len:
                        break
                    if i > 0 and self.args.tool_scale > 0:
                        await asyncio.sleep(turn["tool_sleep_s"] * self.args.tool_scale)
                    for attempt in range(self.args.retries + 1):
                        try:
                            rec = await self.run_turn(plan, i, prompt, turn["max_tokens"])
                            self.records.append(rec)
                            break
                        except (httpx.HTTPError, httpx.HTTPStatusError) as e:
                            if attempt == self.args.retries:
                                self.records.append(
                                    {"trajectory_id": plan["trajectory_id"], "turn": i,
                                     "prompt_tokens": len(prompt), "error": repr(e)}
                                )
                            else:
                                await asyncio.sleep(2.0)
            finally:
                self.inflight -= 1
                self.done_plans += 1

    async def progress(self) -> None:
        while True:
            await asyncio.sleep(15)
            dt = time.monotonic() - self.t0
            print(
                f"[{dt:7.0f}s] plans {self.done_plans}/{self.total_plans} inflight={self.inflight} "
                f"requests={len(self.records)} gen_tok/s(cum)={self.gen_tokens / dt:.0f}",
                flush=True,
            )

    async def run(self, plans: list[dict]) -> None:
        self.total_plans = len(plans)
        self.t0 = time.monotonic()
        reporter = asyncio.create_task(self.progress())
        if self.args.group_launch:
            groups: dict[int, list[dict]] = {}
            for plan in plans:
                groups.setdefault(plan["task_idx"], []).append(plan)
            tasks = []
            for i, (_, group) in enumerate(sorted(groups.items())):
                if self.args.group_stagger_s > 0:
                    await asyncio.sleep(self.args.group_stagger_s)
                tasks += [asyncio.create_task(self.run_plan(p)) for p in group]
            await asyncio.gather(*tasks)
        else:
            await asyncio.gather(*(self.run_plan(p) for p in plans))
        reporter.cancel()
        await self.client.aclose()


def summarize(replayer: Replayer, wall_s: float, out: Path) -> None:
    ok = [r for r in replayer.records if "latency_s" in r]
    errs = [r for r in replayer.records if "error" in r]
    lat = [r["latency_s"] for r in ok]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    rate = [r["max_tokens"] / (r["latency_s"] - (r["ttft_s"] or 0)) for r in ok
            if r["latency_s"] - (r["ttft_s"] or 0) > 1]
    deadline = replayer.args.deadline_s
    summary = {
        "wall_s": round(wall_s, 1),
        "plans": replayer.total_plans,
        "requests_ok": len(ok),
        "requests_err": len(errs),
        "gen_tokens": sum(r["max_tokens"] for r in ok),
        "gen_tok_per_s": round(sum(r["max_tokens"] for r in ok) / wall_s, 1),
        "latency_s": {"p50": pct(lat, 0.5), "p90": pct(lat, 0.9), "p99": pct(lat, 0.99), "max": max(lat, default=0)},
        "ttft_s": {"p50": pct(ttft, 0.5), "p90": pct(ttft, 0.9), "p99": pct(ttft, 0.99)},
        "decode_tok_per_s_per_req": {"p50": pct(rate, 0.5), "mean": round(statistics.mean(rate), 2) if rate else 0},
        f"requests_over_{deadline:.0f}s": sum(1 for x in lat if x > deadline),
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "requests.jsonl").open("w") as f:
        for r in replayer.records:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}/requests.jsonl, summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plans", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="PrimeIntellect/INTELLECT-3")
    parser.add_argument("--api-key-var", default="VLLM_API_KEY")
    parser.add_argument("--max-inflight", type=int, default=256)
    parser.add_argument("--group-launch", action="store_true",
                        help="Launch each task group together to reproduce GRPO arrivals")
    parser.add_argument("--group-stagger-s", type=float, default=0.0,
                        help="Delay between group launches for arrival-phase experiments")
    parser.add_argument("--tool-scale", type=float, default=1.0, help="Tool-sleep multiplier; zero disables sleeps")
    parser.add_argument("--setup-scale", type=float, default=0.0, help="Setup-sleep multiplier; default is zero")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logprobs", action="store_true", help="Request logprobs to match orchestrator load")
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--deadline-s", type=float, default=600.0, help="Count requests above this latency as 504 equivalents")
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Replay only the first N plans for a smoke run")
    parser.add_argument("--max-duration-s", type=float, default=None,
                        help="Stop issuing new turns after this steady-state measurement window")
    args = parser.parse_args()

    plans = [json.loads(line) for line in args.plans.open()]
    if args.limit:
        plans = plans[: args.limit]

    replayer = Replayer(args)
    t0 = time.monotonic()
    asyncio.run(replayer.run(plans))
    summarize(replayer, time.monotonic() - t0, args.out)


if __name__ == "__main__":
    main()
