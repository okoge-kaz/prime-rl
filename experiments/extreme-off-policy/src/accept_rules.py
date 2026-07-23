"""Apply acceptance rules post hoc to verify_drafts.py outputs.

Compute per-token acceptance, accepted-prefix length K, survival, and hazard
from lp_q, lp_new, and rank_new without regenerating drafts.

Rules:
    greedy: accept when the draft token is the anchor's top choice.
    strict: maximal q-aware probability min(1, pi_new/q).
    bubble: deterministic-draft probability pi_new(d).
    len<l>: SPEC-RL lenience min(1, l*pi_new/q).
    noisecal: set l=exp(delta), with delta calibrated from the P99 staleness-zero
              distribution of log q - log pi_new.

Probabilistic rules report analytic expected K and a seeded sampled realization.
Outputs include K/L, full-response acceptance, position bins, selected-token
log ratios, token-KL estimates, and reasoning/final-answer rejection segments.

Usage:
    uv run experiments/extreme-off-policy/src/accept_rules.py \
        outputs/extreme-off-policy/math_qwen3_4b_instruct/verify/anchor_50_src_18.jsonl.gz ... \
        [--tokenizer <ckpt_or_model_dir>] \
        -o outputs/extreme-off-policy/math_qwen3_4b_instruct/accept
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

LENIENCE = [1.2, 1.5, 2.0, 4.0]
NUM_POSITION_BINS = 20


def accept_probs(lp_q: np.ndarray, lp_new: np.ndarray, lenience: float = 1.0) -> np.ndarray:
    """Return per-token acceptance probability min(1, lenience*pi_new/q)."""
    return np.minimum(1.0, lenience * np.exp(lp_new - lp_q))


def expected_prefix_len(p: np.ndarray) -> float:
    """E[K] = Σ_{k=1..T} P(K >= k) = Σ_k Π_{t<k} p(t)."""
    return float(np.cumprod(p).sum())


def sampled_prefix_len(p: np.ndarray, u: np.ndarray) -> int:
    rejected = u >= p
    idx = np.argmax(rejected)
    return int(idx) if rejected.any() else len(p)


def greedy_prefix_len(rank_new: np.ndarray) -> int:
    mismatch = rank_new != 1
    idx = np.argmax(mismatch)
    return int(idx) if mismatch.any() else len(rank_new)


def binned_mean(values: np.ndarray, num_bins: int) -> list[float] | None:
    """Average values over relative-position bins, or return None if too short."""
    if len(values) < num_bins:
        return None
    return [float(chunk.mean()) for chunk in np.array_split(values, num_bins)]


def find_marker_token_idx(tokenizer, token_ids: list[int], marker: str) -> int | None:
    """Find the first token index containing marker with binary-search decoding."""
    text = tokenizer.decode(token_ids)
    if marker not in text:
        return None
    lo, hi = 0, len(token_ids)  # Prefix lo excludes marker; prefix hi includes it.
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if marker in tokenizer.decode(token_ids[:mid]):
            hi = mid
        else:
            lo = mid
    return lo


def segment_boundary(tokenizer, token_ids: list[int]) -> tuple[int | None, int | None, int | None]:
    """Return boundary, think-end, and boxed indices.

    Prefer </think> for reasoning models and \\boxed for non-thinking models.
    """
    think_end = find_marker_token_idx(tokenizer, token_ids, "</think>")
    boxed = find_marker_token_idx(tokenizer, token_ids, "\\boxed")
    return (think_end if think_end is not None else boxed), think_end, boxed


def reject_segment(prefix_len: int, seq_len: int, boxed_idx: int | None) -> str | None:
    if prefix_len >= seq_len:
        return "accepted_full"
    if boxed_idx is None:
        return None
    return "reasoning" if prefix_len < boxed_idx else "final_answer"


def calibrate_noise_delta(paths: list[Path], percentile: float = 99.0) -> float | None:
    """Calibrate delta from the staleness-zero log q - log pi_new percentile."""
    diffs = []
    for path in paths:
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["anchor_step"] != r["source_step"]:
                    break  # Each file has one anchor/source pair.
                diffs.append(np.asarray(r["lp_q"], dtype=np.float64) - np.asarray(r["lp_new"], dtype=np.float64))
    if not diffs:
        return None
    return float(np.percentile(np.concatenate(diffs), percentile))


def process_record(r: dict, rng: np.random.Generator, tokenizer=None, noise_delta: float | None = None) -> dict:
    lp_q = np.asarray(r["lp_q"], dtype=np.float64)
    lp_new = np.asarray(r["lp_new"], dtype=np.float64)
    rank_new = np.asarray(r["rank_new"], dtype=np.int64)
    T = len(lp_q)
    u = rng.random(T)
    log_ratio = lp_new - lp_q

    # Multi-turn SWE records use turn-level boundaries.
    boundary_idx, think_end_idx, boxed_idx = (
        segment_boundary(tokenizer, r["draft_token_ids"])
        if (tokenizer and T and r.get("draft_token_ids"))
        else (None, None, None)
    )

    p_strict = accept_probs(lp_q, lp_new)
    k_strict = sampled_prefix_len(p_strict, u)
    k_greedy = greedy_prefix_len(rank_new)

    out = {
        "task_idx": r["task_idx"],
        "sample_idx": r["sample_idx"],
        "anchor_step": r["anchor_step"],
        "source_step": r["source_step"],
        "staleness": r["anchor_step"] - r["source_step"],
        "len": T,
        "finish_reason": r.get("finish_reason"),
        "verify_tps": r.get("verify_tps"),
        # Monte Carlo sequence-KL estimate and selected-token log-ratio statistics.
        "kl_sum": float(-log_ratio.sum()),
        "kl_per_token": float(-log_ratio.mean()) if T else 0.0,
        "log_ratio_mean": float(log_ratio.mean()) if T else 0.0,
        "log_ratio_p10": float(np.percentile(log_ratio, 10)) if T else 0.0,
        "log_ratio_p90": float(np.percentile(log_ratio, 90)) if T else 0.0,
        # greedy
        "greedy_prefix": k_greedy,
        "greedy_prefix_frac": k_greedy / T if T else 0.0,
        "greedy_alpha": float((rank_new == 1).mean()) if T else 0.0,
        "greedy_full_accept": k_greedy == T,
        # Mean strict acceptance in relative-position bins.
        "strict_alpha_by_bin": binned_mean(p_strict, NUM_POSITION_BINS),
        # Segment rejection boundary: </think> for R1, otherwise \boxed.
        "boxed_token_idx": boxed_idx,
        "think_end_token_idx": think_end_idx,
        "strict_reject_segment": reject_segment(k_strict, T, boundary_idx),
        "greedy_reject_segment": reject_segment(k_greedy, T, boundary_idx),
    }
    # Method B (bubble) uses p(d) directly and is independent of q.
    rule_probs = {"bubble": np.minimum(1.0, np.exp(lp_new))}
    # Noise-calibrated probabilistic lenience uses l=exp(delta).
    if noise_delta is not None:
        rule_probs["noisecal"] = accept_probs(lp_q, lp_new, float(np.exp(noise_delta)))
    for ell in [1.0, *LENIENCE]:
        rule_probs["strict" if ell == 1.0 else f"len{ell:g}"] = accept_probs(lp_q, lp_new, ell)
    for key, p in rule_probs.items():
        k_expected = expected_prefix_len(p)
        k_sampled = sampled_prefix_len(p, u)
        out[f"{key}_alpha"] = float(p.mean()) if T else 0.0
        out[f"{key}_prefix_expected"] = k_expected
        out[f"{key}_prefix_sampled"] = k_sampled
        out[f"{key}_prefix_expected_frac"] = k_expected / T if T else 0.0
        out[f"{key}_prefix_sampled_frac"] = k_sampled / T if T else 0.0
        # Full-response acceptance, analytic and sampled.
        with np.errstate(divide="ignore"):
            out[f"{key}_full_accept_prob"] = float(np.exp(np.log(np.clip(p, 1e-300, 1.0)).sum())) if T else 1.0
        out[f"{key}_full_accept"] = k_sampled == T

    # For multi-turn SWE traces, count turns completed before first rejection.
    if r.get("turn_idx"):
        ti = np.asarray(r["turn_idx"], dtype=np.int64)
        out["turns_total"] = int(ti[-1]) + 1
        for key in ("strict", "noisecal"):
            k = out.get(f"{key}_prefix_sampled")
            if k is None:
                continue
            out[f"{key}_turns_accepted"] = out["turns_total"] if k >= T else int(ti[k])
            out[f"{key}_turns_accepted_frac"] = out[f"{key}_turns_accepted"] / out["turns_total"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verify_files", type=Path, nargs="+")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=None, help="Checkpoint or model ID for segment detection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-delta", type=float, default=None,
                        help="Noisecal delta; defaults to P99 from staleness-zero inputs")
    parser.add_argument("--noise-percentile", type=float, default=99.0)
    args = parser.parse_args()

    noise_delta = args.noise_delta
    if noise_delta is None:
        noise_delta = calibrate_noise_delta(args.verify_files, args.noise_percentile)
    if noise_delta is None:
        print("WARNING: skipping noisecal because no staleness-zero data or --noise-delta was provided")
    else:
        print(f"noisecal delta = {noise_delta:.5f} (equivalent lenience ~= {np.exp(noise_delta):.3f})")

    tokenizer = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.verify_files:
        rng = np.random.default_rng(args.seed)
        out_path = args.out_dir / f"accept_{path.name.removesuffix('.jsonl.gz')}.jsonl.gz"
        n = 0
        with gzip.open(path, "rt") as fin, gzip.open(out_path, "wt") as fout:
            for line in fin:
                fout.write(json.dumps(process_record(json.loads(line), rng, tokenizer, noise_delta)) + "\n")
                n += 1
        print(f"{path.name}: {n} sequences -> {out_path}")


if __name__ == "__main__":
    main()
