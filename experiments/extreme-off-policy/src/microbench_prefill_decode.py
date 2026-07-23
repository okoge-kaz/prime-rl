"""Measure prefill and decode cost with offline vLLM on one GPU setup.

Measure long-prompt prefill throughput with max_tokens=1 and continuation
decode throughput at several batch sizes. Use real sequences from a draft file
when supplied, otherwise use random token sequences.

Usage:
    uv run experiments/extreme-off-policy/src/microbench_prefill_decode.py \
        --ckpt <ckpt_base>/math_qwen3_4b_instruct/weights/step_50 \
        [--drafts outputs/extreme-off-policy/math_qwen3_4b_instruct/drafts/step_50.jsonl.gz] \
        [--tp 1] -o outputs/extreme-off-policy/math_qwen3_4b_instruct/microbench.json
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

PREFILL_BATCH = 64
DECODE_BATCHES = [32, 128, 512]
DECODE_TOKENS = 512
SEQ_LEN = 8192


def load_sequences(drafts: Path | None, vocab_size: int, n: int, seq_len: int) -> list[list[int]]:
    if drafts is not None:
        seqs = []
        with gzip.open(drafts, "rt") as f:
            for line in f:
                r = json.loads(line)
                ids = r["prompt_token_ids"] + r["draft_token_ids"]
                if len(ids) >= 1024:
                    seqs.append(ids[:seq_len])
                if len(seqs) >= n:
                    break
        if len(seqs) >= n:
            return seqs
    rng = np.random.default_rng(0)
    return [rng.integers(0, vocab_size, size=seq_len).tolist() for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    # Disable prefix caching so the prefill measurement remains valid.
    llm = LLM(
        model=str(args.ckpt),
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        additional_config={"fp32_lm_head": True},
        worker_extension_cls="prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
    )
    vocab_size = llm.get_tokenizer().vocab_size
    seqs = load_sequences(args.drafts, vocab_size, PREFILL_BATCH, SEQ_LEN)

    # warmup
    llm.generate([{"prompt_token_ids": seqs[0]}], SamplingParams(max_tokens=1))

    # Prefill every sequence with max_tokens=1.
    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": s} for s in seqs], SamplingParams(max_tokens=1))
    prefill_time = time.perf_counter() - t0
    prefill_tokens = sum(len(s) for s in seqs)
    prefill_tps = prefill_tokens / prefill_time

    # Decode DECODE_TOKENS from short prompts at several batch sizes.
    decode = {}
    for bs in DECODE_BATCHES:
        prompts = [{"prompt_token_ids": s[:256]} for s in (seqs * (bs // len(seqs) + 1))[:bs]]
        sp = SamplingParams(max_tokens=DECODE_TOKENS, min_tokens=DECODE_TOKENS, temperature=1.0, ignore_eos=True)
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        decode[bs] = bs * DECODE_TOKENS / dt

    result = {
        "ckpt": str(args.ckpt),
        "prefill_tokens_per_s": prefill_tps,
        "decode_tokens_per_s_by_batch": decode,
        # Per-token cost ratio for the model, using the largest decode batch.
        "cost_ratio_prefill_over_decode": max(decode.values()) / prefill_tps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
