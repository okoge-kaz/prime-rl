"""Phase 3: verify drafts with an anchor checkpoint using offline vLLM prefill.

Prefill prompt and draft token IDs without retokenizing, and record each draft
token's log probability under the anchor plus its rank. Rank one means the
draft token matches the anchor's greedy top choice. Pass the anchor's own draft
file to measure the staleness-zero control.

Usage on a compute node:
    uv run experiments/extreme-off-policy/src/verify_drafts.py \
        --ckpt <ckpt_base>/math_qwen3_4b_instruct/weights/step_50 \
        --anchor 50 \
        --drafts outputs/extreme-off-policy/math_qwen3_4b_instruct/drafts/step_18.jsonl.gz \
        --out-dir outputs/extreme-off-policy/math_qwen3_4b_instruct/verify \
        [--tp 1] [--batch-size 64]

Output: <out-dir>/anchor_<t>_src_<c>.jsonl.gz
    Draft record plus {"anchor_step", "lp_new", "rank_new"}.
"""

import argparse
import gzip
import json
import time
from pathlib import Path


def iter_jsonl(path: Path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Anchor weights/step_N")
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--drafts", type=Path, required=True, help="gen_drafts.py output step_<c>.jsonl.gz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=24576)
    # prompt_logprobs retains a sequence-length buffer; reduce this on OOM.
    parser.add_argument("--batch-size", type=int, default=64)
    # Match gen_drafts.py fp32 lm_head conditions.
    parser.add_argument("--fp32-lm-head", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    records = list(iter_jsonl(args.drafts))
    source_step = records[0]["source_step"]
    print(f"verifying {len(records)} drafts: anchor={args.anchor}, source={source_step} (s={args.anchor - source_step})")

    llm_kwargs = dict(
        model=str(args.ckpt),
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        additional_config={"fp32_lm_head": args.fp32_lm_head},
    )
    if args.fp32_lm_head:
        # Import the worker extension so offline vLLM follows the serving patch path.
        llm_kwargs["worker_extension_cls"] = "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker"
    llm = LLM(**llm_kwargs)
    # Return top-1 plus the actual token's log probability and rank at each position.
    sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"anchor_{args.anchor}_src_{source_step}.jsonl.gz"
    with gzip.open(out_path, "wt") as f:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            # Single-turn records concatenate prompt and draft. Multi-turn
            # records provide full_token_ids and verify only sampled_positions.
            inputs = [
                {"prompt_token_ids": r["full_token_ids"] if "full_token_ids" in r
                 else r["prompt_token_ids"] + r["draft_token_ids"]}
                for r in batch
            ]
            t0 = time.perf_counter()
            outputs = llm.generate(inputs, sampling)
            batch_time = time.perf_counter() - t0
            # Record batch prefill throughput for the verification cost model.
            batch_tokens = sum(len(i["prompt_token_ids"]) for i in inputs)
            verify_tps = batch_tokens / batch_time
            for r, out in zip(batch, outputs):
                if "sampled_positions" in r:
                    positions = r["sampled_positions"]
                    token_at = lambda i: r["full_token_ids"][i]  # noqa: E731
                else:
                    n_prompt = len(r["prompt_token_ids"])
                    positions = range(n_prompt, n_prompt + len(r["draft_token_ids"]))
                    token_at = lambda i: r["draft_token_ids"][i - n_prompt]  # noqa: E731
                lp_new, rank_new = [], []
                for pos in positions:
                    entry = out.prompt_logprobs[pos][token_at(pos)]
                    lp_new.append(entry.logprob)
                    rank_new.append(entry.rank)
                r["anchor_step"] = args.anchor
                r["lp_new"] = lp_new
                r["rank_new"] = rank_new
                r["verify_tps"] = verify_tps
                f.write(json.dumps(r) + "\n")
            print(f"  {min(start + args.batch_size, len(records))}/{len(records)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
