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

This report compares two DFlash2 training variants on Qwen3.8-27B: the selector's `replace` versus `strict` gold-miss policy (at 5k and 25k training steps), and the `valid_token_count` versus `effective_weight_sum` loss-normalization scheme (at 20k training steps). The `strict` policy improves acceptance length over `replace` across all displayed subsets at 5k steps; the advantage narrows and becomes mixed at 25k steps. The two normalization schemes produce comparable results at 20k steps, with no consistent winner across subsets. A longer, two-epoch `strict`-policy checkpoint with additional parameter changes reaches a substantially higher acceptance length than any of the step-capped runs, but is not a controlled comparison.

## Setup

- Target model: `Qwen/Qwen3.8-27B`.
- Draft model: DFlash2, trained with the `--selector-miss-policy` flag introduced in [Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217).
- Checkpoints evaluated:
  - replace, 5k steps: `qwen38-dflash2-selector-replace-5k-20261008_002640/checkpoints/0`
  - strict, 5k steps: `qwen38-dflash2-selector-strict-5k-20261008_111207/checkpoints/0`
  - replace, 25k steps: `qwen38-dflash2-selector-replace-25000steps-20261012_010849/checkpoints/0`
  - strict, 25k steps: `qwen38-dflash2-selector-strict-25000steps-20261009_151402/checkpoints/0`
  - strict, effective-weight-sum normalization, 20k steps: `qwen38-dflash2-selector-strict-effective_weight_sum-20000steps-20261016_155251/checkpoints/0`
  - strict, valid-token-count normalization, 20k steps: `qwen38-dflash2-selector-strict-valid_token_count-20000steps-20261019_011313/checkpoints/0`
  - strict, full two epochs: `nm-research/qwen3.8-27B-speculator.dflash2.updatedparams.strict.ckpt1` (Hugging Face Hub)
- Algorithm settings: block size 8, selector top-K 16.
- Training data: a prepared version of the open-perfectblend dataset mix.
- Training: `torchrun`, 4 GPUs per run, seed 42, 1 epoch capped at the step counts listed above for the selector-policy and normalization runs. The full two-epoch checkpoint is a separate, longer training run with additional parameter changes beyond the miss policy.
- Evaluation: `RedHatAI/speculator_benchmarks` via `evaluate.py ... throughput`.
- Hardware: 8x NVIDIA H200 node (4 GPUs used per training run).

## Reproduction

The strict selector miss policy is implemented in [Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217).

## Results

### Strict versus replace selector policy

When the verifier's target token is absent from the selector's top-K candidates, `replace` inserts the target into the last candidate slot and trains the selector on that position. `strict` leaves the candidate set unchanged and masks the selector loss for that position; the unary loss remains active.

Acceptance length by subset (tokens per round):

#### 5k training steps

| Subset | Replace | Strict | Strict − replace |
|---|---:|---:|---:|
| HumanEval | 3.03 | 3.13 | +0.10 |
| math_reasoning | 3.79 | 4.05 | +0.26 |
| qa | 2.74 | 2.91 | +0.17 |
| question | 2.59 | 2.66 | +0.07 |
| rag | 2.85 | 3.06 | +0.21 |
| summarization | 2.38 | 2.56 | +0.18 |
| tool_call | 2.53 | 2.65 | +0.12 |
| translation | 2.39 | 2.57 | +0.18 |
| writing | 2.51 | 2.72 | +0.21 |

Strict is higher on all nine subsets at 5k steps. Mean acceptance length is approximately 2.76 for replace and 2.92 for strict.

#### 25k training steps

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

- The `strict` selector miss policy improves acceptance length over `replace` across all displayed subsets at 5k training steps; the advantage narrows and becomes mixed by 25k steps.
- `valid_token_count` and `effective_weight_sum` loss normalization produce comparable acceptance lengths at 20k training steps, with no consistent subset-level winner.
- The full two-epoch checkpoint reaches a notably higher acceptance length than any step-capped run, but combines several changes at once and should not be read as isolating the selector policy's effect.
- Results above do not include run-to-run variance or confidence intervals; subset-level differences should be interpreted descriptively rather than as statistically significant.

## References

- [DFlash2 strict selector miss policy, Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217)
