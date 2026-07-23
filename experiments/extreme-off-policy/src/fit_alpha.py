"""Phase 4: aggregate acceptance, survival, hazard, and cost-model results.

Read accept_rules.py outputs and report alpha by staleness and anchor,
first-rejection survival and accepted-prefix length, full-response acceptance,
position and segment hazards, selected-token log ratios, token-KL estimates,
verification timing, and predicted end-to-end speedup.

Usage:
    uv run experiments/extreme-off-policy/src/fit_alpha.py \
        outputs/extreme-off-policy/math_qwen3_4b_instruct/accept/accept_anchor_*.jsonl.gz \
        [--microbench outputs/extreme-off-policy/math_qwen3_4b_instruct/microbench.json] \
        -o outputs/extreme-off-policy/math_qwen3_4b_instruct/fit
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RULES = ["greedy", "strict", "noisecal", "bubble", "len1.2", "len1.5", "len2", "len4"]
SEGMENTS = ["reasoning", "final_answer", "accepted_full"]

# Okabe-Ito colorblind-safe palette assigned in a stable order.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with gzip.open(path, "rt") as f:
            rows.extend(json.loads(line) for line in f)
    return rows


def group_by(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return dict(groups)


def summarize_cell(rs: list[dict]) -> dict:
    out = {"n": len(rs), "mean_len": float(np.mean([r["len"] for r in rs]))}
    for rule in RULES:
        out[f"{rule}_alpha"] = float(np.mean([r[f"{rule}_alpha"] for r in rs]))
        if rule == "greedy":
            pfx = np.array([r["greedy_prefix"] for r in rs], dtype=np.float64)
            frac = np.array([r["greedy_prefix_frac"] for r in rs], dtype=np.float64)
            full = np.array([r["greedy_full_accept"] for r in rs], dtype=np.float64)
        else:
            pfx = np.array([r[f"{rule}_prefix_expected"] for r in rs], dtype=np.float64)
            frac = np.array([r[f"{rule}_prefix_expected_frac"] for r in rs], dtype=np.float64)
            full = np.array([r[f"{rule}_full_accept_prob"] for r in rs], dtype=np.float64)
        out[f"{rule}_E_K"] = float(pfx.mean())
        out[f"{rule}_E_K_p50"] = float(np.percentile(pfx, 50))
        out[f"{rule}_E_K_p90"] = float(np.percentile(pfx, 90))
        out[f"{rule}_E_K_over_L"] = float(frac.mean())
        out[f"{rule}_full_accept"] = float(full.mean())
    # Segment rejection breakdown is available when acceptance used a tokenizer.
    segs = [r.get("strict_reject_segment") for r in rs]
    known = [s for s in segs if s is not None]
    if known:
        out["reject_segments"] = {s: known.count(s) / len(known) for s in SEGMENTS if s in known}
    return out


def speedup(mean_len: float, prefix: float, cost_ratio: float, overhead_tokens: float = 0.0) -> float:
    """Compare decoding all L tokens with verifying L and decoding the L-K suffix."""
    L, K = mean_len, min(prefix, mean_len)
    denom = L * cost_ratio + (L - K) + overhead_tokens
    return L / denom if denom > 0 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accept_files", type=Path, nargs="+")
    parser.add_argument("--microbench", type=Path, default=None)
    parser.add_argument("--cost-ratio", type=float, default=None, help="c_prefill/c_decode override")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load(args.accept_files)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Process only rules present in the acceptance outputs.
    global RULES
    RULES = [r for r in RULES if f"{r}_alpha" in rows[0]]

    # Timing inputs from Phase 3 verification and the decode microbenchmark.
    decode_tps = None
    if args.microbench is not None:
        mb = json.loads(args.microbench.read_text())
        decode_tps = max(mb["decode_tokens_per_s_by_batch"].values())
    verify_tps_vals = [r["verify_tps"] for r in rows if r.get("verify_tps")]
    verify_tps = float(np.mean(verify_tps_vals)) if verify_tps_vals else None

    cost_ratio = args.cost_ratio
    if cost_ratio is None and verify_tps and decode_tps:
        cost_ratio = decode_tps / verify_tps  # Per-token c_prefill/c_decode.
    if cost_ratio is None:
        cost_ratio = 0.1
        print("WARNING: using fallback cost ratio 0.1 because no measurement is available")

    # Aggregate per anchor/staleness cell and across anchors.
    cells = {k: summarize_cell(v) for k, v in sorted(group_by(rows, "anchor_step", "staleness").items())}
    by_s = {k[0]: summarize_cell(v) for k, v in sorted(group_by(rows, "staleness").items())}

    noise_floor = 1.0 - by_s.get(0, {}).get("strict_alpha", 1.0)
    print(f"\n== s=0 noise floor (1 - strict alpha): {noise_floor:.4f} ==\n")

    header = ["Δ", "n", "L̄"] + [f"{r}_α" for r in RULES] + ["strict E[K]", "E[K/L]", "full_acc", "segments"]
    print(" | ".join(header))
    for s, c in by_s.items():
        vals = [str(s), str(c["n"]), f"{c['mean_len']:.0f}"]
        vals += [f"{c[f'{r}_alpha']:.4f}" for r in RULES]
        vals += [f"{c['strict_E_K']:.1f}", f"{c['strict_E_K_over_L']:.3f}", f"{c['strict_full_accept']:.3f}"]
        vals += [json.dumps(c.get("reject_segments", {}))]
        print(" | ".join(vals))

    (args.out_dir / "cells.json").write_text(
        json.dumps(
            {f"anchor{t}_s{s}": c for (t, s), c in cells.items()} | {f"pooled_s{s}": c for s, c in by_s.items()},
            indent=2,
        )
    )

    # Timing and predicted speedup table.
    print(f"\n== timing / speedup (strict, c_p/c_d = {cost_ratio:.3f}, verify_tps = {verify_tps}, decode_tps = {decode_tps}) ==")
    speedup_rows = []
    for s, c in by_s.items():
        L, K = c["mean_len"], c["strict_E_K"]
        row = {
            "staleness": s,
            "verification_time_s": L / verify_tps if verify_tps else None,
            "regen_suffix_time_s": (L - K) / decode_tps if decode_tps else None,
            "baseline_decode_time_s": L / decode_tps if decode_tps else None,
            "speedup_normal": speedup(L, K, cost_ratio),
            "speedup_pd_disagg": speedup(L, K, 0.0),
        }
        speedup_rows.append(row)
        vt = f"{row['verification_time_s']:.2f}s" if row["verification_time_s"] else "n/a"
        rt = f"{row['regen_suffix_time_s']:.2f}s" if row["regen_suffix_time_s"] else "n/a"
        print(f"Δ={s:>2}: verify {vt} | regen-suffix {rt} | speedup normal {row['speedup_normal']:.2f}x | PD {row['speedup_pd_disagg']:.2f}x")
    (args.out_dir / "speedup.json").write_text(json.dumps(speedup_rows, indent=2))

    # ---- KL fit ----
    kl = np.array([r["kl_per_token"] for r in rows if r["staleness"] > 0])
    al = np.array([r["strict_alpha"] for r in rows if r["staleness"] > 0])
    mask = (al > 1e-4) & (kl > 0)
    k_fit = float((-np.log(al[mask]) / kl[mask]).mean()) if mask.any() else float("nan")
    print(f"\n== exponential fit: alpha ~= exp(-k * KL_per_token), k ~= {k_fit:.2f} ==")

    # Plots.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s_vals = [s for s in by_s if s > 0]

    def styled(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    # Staleness plots require positive-staleness cells.
    if s_vals:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for rule, color in zip(RULES, COLORS):
            ax.plot(s_vals, [by_s[s][f"{rule}_alpha"] for s in s_vals], marker="o", ms=4, lw=2, color=color, label=rule)
        if 0 in by_s:
            ax.axhline(by_s[0]["strict_alpha"], color="#888888", lw=1, ls="--", label="s=0 floor (strict)")
        ax.set_xscale("log", base=2)
        styled(ax, "staleness Δ (trainer steps)", "token acceptance rate α")
        fig.tight_layout()
        fig.savefig(args.out_dir / "alpha_curves.png", dpi=120)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        anchors = sorted({t for t, _ in cells})
        for t, color in zip(anchors, COLORS):
            ss = sorted(s for (t2, s) in cells if t2 == t and s > 0)
            ax.plot(ss, [cells[(t, s)]["strict_alpha"] for s in ss], marker="o", ms=4, lw=2, color=color, label=f"T={t}")
        ax.set_xscale("log", base=2)
        styled(ax, "staleness Δ (trainer steps)", "strict α")
        fig.tight_layout()
        fig.savefig(args.out_dir / "alpha_by_anchor.png", dpi=120)

        # survival curve S_Δ(k) = Pr(K >= k) (strict sampled, pooled)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for s, color in zip(s_vals, COLORS):
            ks = np.sort([r["strict_prefix_sampled"] for r in rows if r["staleness"] == s])
            surv = 1.0 - np.arange(len(ks)) / len(ks)
            ax.step(ks, surv, lw=2, color=color, label=f"Δ={s}", where="post")
        styled(ax, "k (accepted prefix length)", "S_Δ(k) = Pr(K ≥ k)")
        fig.tight_layout()
        fig.savefig(args.out_dir / "survival.png", dpi=120)

        # CDF of normalized first-rejection position K/L.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for s, color in zip(s_vals, COLORS):
            fr = np.sort([r["strict_prefix_sampled_frac"] for r in rows if r["staleness"] == s])
            ax.plot(fr, np.linspace(0, 1, len(fr)), lw=2, color=color, label=f"Δ={s}")
        styled(ax, "K/L (first rejection, normalized)", "CDF")
        fig.tight_layout()
        fig.savefig(args.out_dir / "first_rejection_cdf.png", dpi=120)

        # Acceptance hazard across 20 relative-position bins.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        any_positive = False
        for s, color in zip(s_vals, COLORS):
            bins = np.array([r["strict_alpha_by_bin"] for r in rows if r["staleness"] == s and r["strict_alpha_by_bin"]])
            if len(bins):
                hazard = 1.0 - bins.mean(axis=0)
                any_positive |= bool((hazard > 0).any())
                ax.plot(np.linspace(0.025, 0.975, bins.shape[1]), hazard, marker="o", ms=3, lw=2, color=color, label=f"Δ={s}")
        # A log scale is invalid when every hazard value is zero.
        if any_positive:
            ax.set_yscale("log")
        styled(ax, "relative token position", "acceptance hazard 1 − ᾱ(pos)")
        fig.tight_layout()
        fig.savefig(args.out_dir / "hazard_by_position.png", dpi=120)

    if len(kl):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(kl, al, s=4, alpha=0.15, color=COLORS[0])
        if np.isfinite(k_fit):
            xs = np.linspace(0, max(kl.max(), 1e-6), 100)
            ax.plot(xs, np.exp(-k_fit * xs), color=COLORS[1], lw=2, label=f"exp(−{k_fit:.1f}·KL)")
        styled(ax, "KL per token (MC est., nats)", "strict α (per sequence)")
        fig.tight_layout()
        fig.savefig(args.out_dir / "kl_alpha.png", dpi=120)

    print(f"\nwrote plots + tables to {args.out_dir}")


if __name__ == "__main__":
    main()
