---
title: "DFlash2 selector and loss-normalization ablations on Qwen3.8-27B"
tags:
  - DFlash2
  - Qwen3.8-27B
  - selector
  - training
---

<div class="entry-byline" markdown>
[![shubhra](https://github.com/shubhra.png?size=64){ .entry-avatar }](https://github.com/shubhra)
**[@shubhra](https://github.com/shubhra)** posted on **2026-10-28**
</div>

# DFlash2 selector and loss-normalization ablations on Qwen3.8-27B

## Summary

DFlash2's draft-model training had two open design questions worth investigating: how the selector should handle a missing gold token among its top-K candidates, and how the per-position selector loss should be normalized. This report covers ablations on both questions on Qwen3.8-27B, followed by a longer training run that incorporates the resulting change.

For the selector miss policy, an earlier, shorter run showed `strict` improving acceptance length over `replace` on every subset, which motivated adding `strict` as a selectable policy in [Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217). A longer 25k-step run afterward shows a smaller, mixed difference between the two policies. For loss normalization, `valid_token_count` and `effective_weight_sum` produce comparable acceptance lengths at 20k training steps, with no consistent winner across subsets. A longer, two-epoch run using the `strict` policy reaches a substantially higher acceptance length than any of the step-capped ablation runs.

## Setup

- Target model: `Qwen/Qwen3.8-27B`.
- Draft model: DFlash2, trained with the `--selector-miss-policy` flag introduced in [Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217).
- Checkpoints evaluated:
  - replace, 25k steps: `qwen38-dflash2-selector-replace-25000steps-20261012_010849/checkpoints/0`
  - strict, 25k steps: `qwen38-dflash2-selector-strict-25000steps-20261009_151402/checkpoints/0`
  - strict, effective-weight-sum normalization, 20k steps: `qwen38-dflash2-selector-strict-effective_weight_sum-20000steps-20261016_155251/checkpoints/0`
  - strict, valid-token-count normalization, 20k steps: `qwen38-dflash2-selector-strict-valid_token_count-20000steps-20261019_011313/checkpoints/0`
  - strict, full two epochs: `nm-research/qwen3.8-27B-speculator.dflash2.updatedparams.strict.ckpt1` (Hugging Face Hub)
- Algorithm settings: block size 8, selector top-K 16.
- Training data: a prepared version of the open-perfectblend dataset mix.
- Training: `torchrun`, 4 GPUs per run, seed 42, 1 epoch capped at 25k steps for the selector-policy ablation and 20k steps for the normalization ablation. The full two-epoch checkpoint is a separate, longer training run with additional parameter changes beyond the miss policy.
- Evaluation: `RedHatAI/speculator_benchmarks` via `evaluate.py ... throughput`.
- Hardware: 8x NVIDIA H200 node (4 GPUs used per training run).

## Reproduction

The strict selector miss policy is implemented in [Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217).

## Results

### Strict versus replace selector policy

When the verifier's target token is absent from the selector's top-K candidates, `replace` inserts the target into the last candidate slot and trains the selector on that position. `strict` leaves the candidate set unchanged and masks the selector loss for that position; the unary loss remains active.

An earlier, shorter run showed `strict` ahead of `replace` on every subset, prompting [Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217) to add `strict` as a selectable policy. The table below extends the comparison to 25k training steps (tokens per round):

| Subset | Replace | Strict | Strict − replace |
|---|---:|---:|---:|
| HumanEval | 3.74 | 3.80 | +0.06 |
| math_reasoning | 4.99 | 5.16 | +0.17 |
| qa | 3.73 | 3.68 | −0.05 |
| question | 3.29 | 3.27 | −0.02 |
| rag | 4.14 | 4.21 | +0.07 |
| summarization | 3.29 | 3.37 | +0.08 |
| tool_call | 3.21 | 3.21 | 0.00 |
| translation | 3.89 | 3.91 | +0.02 |
| writing | 3.25 | 3.23 | −0.02 |

At 25k steps the results are mixed: strict is higher on five subsets, lower on three, and tied on one. Mean acceptance length is approximately 3.73 for replace and 3.76 for strict. Per-position acceptance rates are close between the two policies.

The training-time validation metric `val/unary_candidate_recall_at_16_epoch` (how often the gold token appears in the unary head's top-16 candidates) favors `strict` at both step counts: 0.687 → 0.713 at 5k, and 0.780 → 0.790 at 25k. This metric is more consistent than the downstream acceptance-length comparison above.

### Valid-token-count versus effective-weight-sum normalization

The two loss reductions differ in their denominator:

- `valid_token_count`: divides the weighted loss by the number of valid tokens, using `loss_mask.sum()`.
- `effective_weight_sum`: rescales the loss by `valid.sum() / (valid * decay_weight).sum()`, so the effective denominator is the sum of valid per-position decay weights.

Acceptance length by subset (tokens per round):

#### 20k training steps

| Subset | Valid-token count | Effective-weight sum |
|---|---:|---:|
| HumanEval | 3.66 | 3.64 |
| math_reasoning | 4.81 | 4.87 |
| qa | 3.62 | 3.55 |
| question | 3.17 | 3.22 |
| rag | 4.07 | 4.12 |
| summarization | 3.26 | 3.23 |
| tool_call | 3.14 | 3.14 |
| translation | 3.75 | 3.78 |
| writing | 3.11 | 3.13 |
| Mean | 3.62 | 3.63 |

Mean acceptance length is very similar between the two normalization schemes, and per-position acceptance rates nearly overlap. Subset-level differences are small and mixed in direction, showing no consistent winner between the two normalization methods at this step count.

### Full two-epoch run

This checkpoint reflects a longer, two-epoch training run with additional parameter changes beyond the selector miss policy, so it is not a controlled comparison against the step-capped runs above; it is reported as a separate data point.

| Subset | Acceptance length |
|---|---:|
| HumanEval | 4.38 |
| math_reasoning | 5.64 |
| qa | 4.11 |
| question | 3.77 |
| rag | 4.65 |
| summarization | 3.93 |
| tool_call | 3.70 |
| translation | 4.48 |
| writing | 3.73 |
| Mean | 4.26 |

Mean acceptance length (4.26) is substantially higher than any of the step-capped runs (3.6-3.8 at 20k/25k steps), consistent with additional training and the other parameter changes. Because multiple factors changed together, this run does not isolate the effect of the miss policy on its own.

## Observations

- The `strict` selector miss policy improved acceptance length over `replace` on every subset in an earlier, shorter run, motivating its addition as a selectable training option; at 25k training steps the difference narrows and becomes mixed.
- `val/unary_candidate_recall_at_16_epoch` consistently favors `strict` over `replace` at both 5k and 25k training steps, even where downstream acceptance length does not.
- `valid_token_count` and `effective_weight_sum` loss normalization produce comparable acceptance lengths at 20k training steps, with no consistent subset-level winner.
- The full two-epoch checkpoint reaches a notably higher acceptance length than any step-capped run, but combines several changes at once and should not be read as isolating the selector policy's effect.
- Results above do not include run-to-run variance or confidence intervals; subset-level differences should be interpreted descriptively rather than as statistically significant.

## References

- [DFlash2 strict selector miss policy, Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217)
