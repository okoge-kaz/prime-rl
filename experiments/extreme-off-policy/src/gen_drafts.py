"""Phase 2: generate drafts from a stale checkpoint with offline vLLM.

Load an HF-compatible weights/step_N checkpoint, generate rollouts for the
i3-math-v1 prompt distribution, and record effective sampling log probability
log q(t) for every sampled token. Temperature and top-p must both equal 1.0 so
raw vLLM log probabilities match the effective sampling distribution.

The same seed and prompt count produce the same prompt set for every checkpoint.

Usage on a compute node:
    uv run experiments/extreme-off-policy/src/gen_drafts.py \
        --ckpt <ckpt_base>/math_qwen3_4b_instruct/weights/step_18 \
        --step 18 \
        --out-dir outputs/extreme-off-policy/math_qwen3_4b_instruct/drafts \
        [--num-prompts 256] [--samples 4] [--max-tokens 16384] [--tp 1]

Output: <out-dir>/step_<N>.jsonl.gz
    {"task_idx", "sample_idx", "source_step", "prompt_token_ids",
     "draft_token_ids", "lp_q", "finish_reason"}
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

# Match the i3-math-v1 prompt construction in deps/research-environments.
DATASET_NAME = "PrimeIntellect/INTELLECT-3-RL"
DATASET_SUBSET = "math"
DATASET_SPLIT = "train"
QUESTION_KEY = "question"
INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"


def load_prompts(filter_column: str, filter_min: float, filter_max: float, num_prompts: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)
    tasks = []
    for i, row in enumerate(rows):
        value = row.get(filter_column)
        if value is None or not (filter_min <= float(value) <= filter_max):
            continue
        tasks.append({"task_idx": i, "prompt": f"{INSTRUCTION}{row[QUESTION_KEY]}"})
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(tasks), size=min(num_prompts, len(tasks)), replace=False)
    return [tasks[i] for i in sorted(picked)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="HF-compatible weights/step_N or model ID")
    parser.add_argument("--step", type=int, required=True, help="Source policy version recorded in output")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4, help="Samples per prompt")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # Match training serving with the fp32 lm_head enabled by default.
    parser.add_argument("--fp32-lm-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-column", default="avg@8_qwen3_4b_instruct_2507")
    # Defaults match the Qwen3-4B-Instruct-2507 training filter.
    parser.add_argument("--filter-min", type=float, default=0.125)
    parser.add_argument("--filter-max", type=float, default=0.625)
    args = parser.parse_args()

    assert args.temperature == 1.0 and args.top_p == 1.0, (
        "temperature and top_p must be 1.0 so raw vLLM logprobs match the "
        "effective sampling distribution"
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    prompts = load_prompts(args.filter_column, args.filter_min, args.filter_max, args.num_prompts, args.seed)
    print(f"{len(prompts)} prompts (filter {args.filter_column} in [{args.filter_min}, {args.filter_max}])")

    tokenizer = AutoTokenizer.from_pretrained(str(args.ckpt))

    def render(prompt: str) -> list[int]:
        # Transformers 5 returns BatchEncoding here; Transformers 4 returns list[int].
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )
        if not isinstance(out, list):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        assert out and isinstance(out[0], int), f"unexpected chat template output: {type(out)}"
        return out

    prompt_token_ids = [render(p["prompt"]) for p in prompts]

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
    sampling = [
        SamplingParams(
            n=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            logprobs=0,  # Return only the sampled token's log probability.
            seed=args.seed * 1_000_003 + p["task_idx"],
        )
        for p in prompts
    ]
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_token_ids],
        sampling,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"step_{args.step}.jsonl.gz"
    n_written = 0
    with gzip.open(out_path, "wt") as f:
        for p, ids, req in zip(prompts, prompt_token_ids, outputs):
            for sample_idx, comp in enumerate(req.outputs):
                lp_q = [d[tid].logprob for tid, d in zip(comp.token_ids, comp.logprobs)]
                f.write(
                    json.dumps(
                        {
                            "task_idx": p["task_idx"],
                            "sample_idx": sample_idx,
                            "source_step": args.step,
                            "prompt_token_ids": list(ids),
                            "draft_token_ids": list(comp.token_ids),
                            "lp_q": lp_q,
                            "finish_reason": comp.finish_reason,
                        }
                    )
                    + "\n"
                )
                n_written += 1
    print(f"wrote {n_written} drafts to {out_path}")


if __name__ == "__main__":
    main()
