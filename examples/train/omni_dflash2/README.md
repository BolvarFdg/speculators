# Qwen3-Omni (ASR) DFlash2 draft training -- single node, 8 NPU cards

On-policy pipeline from raw ASR JSONL to a trained DFlash2 draft, modeled on
the multi-node dspark/dflash pipeline with all sharding and NFS barriers
removed. Run this end-to-end on one 8-card machine before scaling out.

## Phases

| # | Phase | Cards | What happens |
|---|-------|-------|--------------|
| 1 | convert | CPU | raw ASR jsonl -> speculators `conversations` jsonl; `<audio>` markers become audio parts; rows with missing audio files skipped |
| 2 | regen | 0-7 (TP8 server) | assistant responses regenerated **on-policy** by the verifier itself, with audio; conversations structure preserved |
| 3 | render | 0-7 (TP8 server) | `speculators prepare-data` renders chat-template boundaries into `prepared/` (input_ids + loss_mask + messages) |
| 4 | training | 0-3 extraction server (TP4), 4-7 torchrun DDP | hidden states extracted on demand from the live server and cached under `hidden_states/`; DFlash2 draft trained on them |

Phases are idempotent (`.regen_done` / `.render_done` flags under
`$WORK_DIR`); re-running skips completed work, and training resumes from
`$WORK_DIR/checkpoints` automatically.

## Usage

1. Edit `cluster_env.sh` (MODEL / WORK_DIR / AUDIO_ROOT / RAW_DATA). Use a
   fresh `WORK_DIR` -- the dspark and dflash pipelines share the same layout
   and will overwrite each other's caches.
2. Smoke run first:

   ```bash
   MAX_SAMPLES=200 bash train_dflash2_single.sh
   ```

3. Full run:

   ```bash
   bash train_dflash2_single.sh
   ```

## Prerequisites

- Ascend NPU environment for serving: CANN + torch-npu + vllm-ascend (the
  phase 2/3 regen+render server and the phase 4 extraction server are plain
  `vllm serve` against the verifier checkpoint).
- `pip install qwen-omni-utils librosa soundfile torchvision` (Qwen3-Omni
  processor).
- `speculators` installed from THIS checkout (it carries the Qwen3-Omni
  adaptations: thinker config unwrapping, thinker/talker weight
  disambiguation, MRoPE stripping, NPU-safe trainer timing).
- Launch from a directory that is NOT the speculators repo's parent (the
  checkout shadows the installed package).

## Key knobs (environment variables)

| Variable | Default | Notes |
|----------|---------|-------|
| `MAX_SAMPLES` | 0 (all) | applies to convert + regen; use for smoke runs |
| `BLOCK_SIZE` | 8 | num_speculative_tokens = block_size - 1 = 7 at serving time |
| `TARGET_LAYER_IDS` | `1 12 23 34 45` | same 6-layer coverage as the dspark run; the official z-lab DFlash2 recipe uses 3 layers (`2 24 45`) |
| `MASK_TOKEN_ID` | 151676 | 151669 collides with `<\|audio_start\|>` in the gspo tokenizer |
| `MAX_ANCHORS` | 256 | ASR responses are short |
| `CONV_KERNEL_SIZE / CONV_GROUP_SIZE / SELECTOR_RANK / SELECTOR_TOP_K / SELECTOR_LOSS_ALPHA` | 2 / 16 / 256 / 16 / 1.0 | DFlash2-specific draft knobs |
| `MAX_SEQ` | 4096 | must match between render and training |
| `CG_MODE` | FULL_DECODE_ONLY | NPU graph mode for both servers |
| `SERVER_GPUS / TRAIN_GPUS` | 0-3 / 4-7 | phase 4 split |

## How this differs from the dspark/dflash pipelines

The data phases (1-3) are identical; only the phase 4 training command
changes, in five load-bearing places:

- `--speculator-type dflash2` (grouped-conv + candidate-selector draft).
- **No `--draft-vocab-size`**: DFlash2 requires the full verifier vocab
  (init raises otherwise). No vocab pruning, so no `token_freq.pt`, no
  `check_vocab.py`, and no merge step in single-node runs.
- **No `--sample-from-anchor`**: DFlash2 trains block_size-1 mask slots;
  with block_size=8 that is 7 speculative tokens (within the NPU
  `num_speculative_tokens + 1 <= 16` limit).
- **No `--full-attention-indices`**: DFlash2's recipe is sliding window with
  non-causal masking inside blocks (`sliding_window_non_causal` defaults to
  True for dflash2).
- Loss stays at DFlash2 defaults (KL + fixed-exp-decay); override with
  `--loss-fn` only after the first run validates the pipeline.

## After training

Checkpoints land in `$WORK_DIR/checkpoints/<epoch>/` with a
`checkpoint_best` symlink. They are self-describing speculators models:
serve with

```bash
vllm serve $MODEL --speculative-config \
    '{"method":"dflash","model":"<checkpoint_best path>","num_speculative_tokens":7}'
```

(the `dflash2` speculators model type is rewritten to method `dflash` and
routed to the DFlash2 speculator automatically). On NPU also pass
`--enforce-eager` and V2 model runner settings per the vllm-ascend
speculative-decoding guide.

## Scaling out

The multi-node pipeline lives in `train_parallel/train_parallel_dflash2.sh`
(Huawei Cloud platform env vars, NFS-flag barriers, per-node sharded
regen/render, rank-0 merge, multi-node torchrun DDP). It reuses this
folder's `cluster_env.sh`, `qwen3_omni_asr_prepare_data.py` (shard-aware),
and its own `merge_prepared.py` (no token_freq: DFlash2 trains on the full
verifier vocabulary). Run the single-node pipeline end-to-end before
scaling out.
