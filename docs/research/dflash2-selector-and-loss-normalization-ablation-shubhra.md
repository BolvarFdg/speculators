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

This report compares the DFlash2 selector's `replace` and `strict` gold-miss policies at 5k and 25k training steps, and compares valid-token-count with effective-weight-sum loss normalization at 7k and 20k steps. The 5k strict-selector run improves acceptance length on all nine displayed subsets. At 25k, differences are smaller and mixed. The normalization runs are also close, with no consistent winner in the displayed results.

The results are descriptive comparisons of the shown checkpoints. The figures and complete run provenance will be added before this entry is finalized.

## Setup

- Target model: `Qwen/Qwen3.8-27B`.
- Draft model: DFlash2, trained with [Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217)'s `--selector-miss-policy` flag; checkpoints used for evaluation:
  - replace, 5k steps: `qwen38-dflash2-selector-replace-5k-20261008_002640/checkpoints/0`
  - strict, 5k steps: `qwen38-dflash2-selector-strict-5k-20261008_111207/checkpoints/0`
  - replace, 25k steps: `qwen38-dflash2-selector-replace-25000steps-20261012_010849/checkpoints/0`
  - strict, 25k steps: `qwen38-dflash2-selector-strict-25000steps-20261009_151402/checkpoints/0`
  - 20k effective-weight-sum: `qwen38-dflash2-selector-strict-effective_weight_sum-20000steps-20261016_155251/checkpoints/0`
  - 20k valid-token-count: `qwen38-dflash2-selector-strict-valid_token_count-20000steps-20261019_011313/checkpoints/0`
  - 7k: checkpoint IDs unreliable — see data-integrity note under "7k training steps" below; both eval dirs point at `...-7000steps-20261015_234038_effective_weight_sum/checkpoints/0`.
  - full two-epoch: `nm-research/qwen3.8-27B-speculator.dflash2.updatedparams.strict.ckpt1` (HF Hub; not a clean ablation, see "Full two-epoch run" below).
- Algorithm settings: block size 8, selector top-K 16 (confirmed from training config for the above runs).
- Training data: `data_path` under local prepared data, directory name `prepared_openperfectblend` (i.e. a prepared version of the open-perfectblend dataset mix); exact HF dataset ID/revision still TODO.
- Training: torchrun, 4 GPUs (`--nproc_per_node 4`), seed 42, 1 epoch capped at the step counts above (5k/25k for selector policy; 7k/20k for normalization). The full two-epoch run is a separate, later training pass — its config/results are not in these logs and are still TODO.
- Evaluation: `RedHatAI/speculator_benchmarks`, `evaluate.py ... throughput`; the plotted subsets are listed in the tables below. Exact eval command flags (e.g. `--max-requests`/`--max-concurrency`) TODO — confirm against `evaluate_run.log` for each checkpoint.
- Software and hardware: 8x NVIDIA H200 node (training used 4 of 8); torch 2.13.0+cu130, transformers 5.16.1 (current env — TODO confirm these match what was active at each run's actual training time).

## Reproduction

The strict selector miss policy is implemented in [Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217). Add the training and evaluation commands, configuration files, checkpoint identifiers, and original chart assets before finalizing this report.

<!-- TODO: Add original figures for the 5k strict-vs-replace run, the 25k strict-vs-replace run, and the 7k/20k normalization ablations. -->

## Results

### Strict versus replace selector policy

When the verifier's target token is absent from the selector's top-K candidates, `replace` inserts the target into the last candidate slot and trains the selector on that position. `strict` leaves the candidate set unchanged and masks the selector loss for that position; the unary loss remains active.

Acceptance length by subset (tokens per round, as displayed in the figures):

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

Strict is higher on all nine subsets in the displayed 5k results. The mean acceptance length is approximately 2.76 for replace and 2.92 for strict.

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

The displayed 25k results are mixed: strict is higher on five subsets, lower on three, and tied on one. Mean acceptance length is approximately 3.73 for replace and 3.76 for strict. The plotted per-position acceptance rates are close between the policies.

### Valid-token-count versus effective-weight-sum normalization

The two loss reductions differ in their denominator:

- `valid_token_count`: divides the weighted loss by the number of valid tokens, using `loss_mask.sum()`.
- `effective_weight_sum`: rescales the loss by `valid.sum() / (valid * decay_weight).sum()`, so the effective denominator is the sum of valid per-position decay weights.

Acceptance length by subset (tokens per round, as displayed in the figures):

#### 7k training steps

**Data-integrity issue — not fixed yet.** Checked `evaluate_run.log` for both the `7k_valid_token` and `7k_effective_weight` eval dirs: every `'model':` entry in *both* logs points to the same checkpoint, `qwen38-dflash2-selector-strict-7000steps-20261015_234038_effective_weight_sum/checkpoints/0`. No `valid_token_count`-suffixed 7k checkpoint exists on disk (only an unsuffixed `...-7000steps-20261013_213253`, two days earlier, of unknown normalization setting). So the "valid-token" 7k numbers below are very likely the effective-weight-sum checkpoint evaluated twice (with ordinary request-sampling noise between runs, not a true normalization difference) — the close-but-not-identical numbers are consistent with that, not with a real ablation. **Do not treat this 7k comparison as evidence** until re-run against a confirmed valid-token-count 7k checkpoint. Keeping the numbers below for the record, with this caveat, pending a fresh run.

The valid-token run is also missing the writing subset; the comparison below uses the eight common subsets.

| Subset | Valid-token count | Effective-weight sum |
|---|---:|---:|
| HumanEval | 3.24 | 3.25 |
| math_reasoning | 3.84 | 4.01 |
| qa | 3.02 | 3.06 |
| question | 2.71 | 2.80 |
| rag | 3.21 | 3.02 |
| summarization | 2.65 | 2.78 |
| tool_call | 2.74 | 2.74 |
| translation | 2.75 | 2.68 |
| Mean of common subsets | 3.02 | 3.04 |

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

At both step counts, the displayed mean acceptance lengths are very similar and the per-position acceptance-rate plots nearly overlap. The subset-level differences are mixed; these results do not show a consistent winner between the normalization methods.

### Full two-epoch run

This is a separate experiment from the step-capped 5k/25k/7k/20k ablations above (those are single-epoch, step-limited runs). Checkpoint: `nm-research/qwen3.8-27B-speculator.dflash2.updatedparams.strict.ckpt1` (HF Hub). The `updatedparams` name indicates other hyperparameters changed too, not just the miss policy — so this is **not a clean ablation** against the runs above; treat it as a separate data point, not a controlled strict-vs-replace comparison. No local training log/provenance found for this checkpoint (HF-hub only).

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

Substantially higher acceptance length than any step-capped run above (mean ~4.26 vs. ~3.6–3.8 at 20k/25k) — consistent with more training (full 2 epochs) plus the other, unspecified parameter changes. Because multiple things changed at once, this run cannot isolate the effect of the miss policy on its own.

## Observations

- The 5k strict-selector results favor strict across all nine displayed subsets; the 25k results show smaller, mixed differences.
- Valid-token-count and effective-weight-sum normalization produce similar average acceptance lengths at the plotted 7k and 20k checkpoints, with no consistent subset-level winner.
- The 7k normalization comparison excludes writing because that valid-token run is missing.
- These plots do not include run-to-run variance or confidence intervals; interpret small differences descriptively, not as evidence of statistical significance.
- Training dataset details, exact run configurations, software/hardware versions, the full two-epoch results, and original chart assets remain to be added.

## References

- [DFlash2 strict selector miss policy, Speculators PR #1217](https://github.com/vllm-project/speculators/pull/1217)
