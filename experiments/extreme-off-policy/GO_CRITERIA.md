# Preregistered GO/NO-GO criteria

Registered: 2026-07-22. Do not change these thresholds after measurement starts.
The experiment history and interpretation are maintained in
[the experiment note](../notes/extreme-off-policy.md).

## Revision 1

The first `math_qwen3_4b_instruct` measurement showed that strict lossless acceptance was
dominated by engine numerical noise at staleness zero. The primary decision
rule therefore changed, with user approval, to noise-calibrated lenience.
Strict acceptance remains a diagnostic, and the numerical GO thresholds remain
unchanged.

## Measurement validity

- V1 — noise floor: strict acceptance for the staleness-zero control must be at
  least 0.98. A lower value requires numerical nondeterminism analysis before
  the measurement can support a decision.
- V2 — policy movement: training reward or evaluation must show a trend over
  the 50 training steps.
- V3 — multiple anchors: each staleness value must have at least two anchors.

## Decision

Use pooled noise-calibrated acceptance across anchors.

- GO: predicted speedup at staleness 8 is at least 1.1x with the measured
  prefill/decode cost ratio.
- STRONG GO: additionally, speedup at staleness 16 is at least 1.1x under the
  measured ratio, or speedup at staleness 32 is at least 1.5x with negligible
  prefill cost in a PD-disaggregated system.
- NO-GO: speedup at staleness 8 is below 1.1x under the measured ratio and below
  1.2x even with negligible PD prefill cost.

## Requirements after GO

The following work is outside this experiment:

1. Re-measure acceptance on multi-turn SWE workloads dominated by observations.
2. Train on accepted and regenerated samples to assess convergence and final
   quality.
3. Design sandbox-state restoration and reward recomputation.
