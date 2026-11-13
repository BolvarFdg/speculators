#!/bin/bash
# Single-node (8-card NPU) DFlash2 training for the Qwen3-Omni ASR thinker.
#
# Full on-policy pipeline, modeled on the multi-node dspark/dflash pipeline
# but with all sharding/barriers removed: ONE machine, eight cards, four
# phases:
#
#   1  convert    CPU only: raw ASR jsonl -> conversations jsonl
#                 (audio placeholders -> audio parts, missing files skipped)
#   2  regen      TP8 plain server (all 8 cards): regenerate assistant
#                 responses on-policy WITH the audio, conversations kept
#   3  render     same server: speculators prepare-data renders chat
#                 template boundaries -> prepared/ dataset
#   4  training   cards 0-3: hidden-state extraction server (TP4);
#                 cards 4-7: torchrun 4-card DDP training (single node)
#
# Phases are idempotent (.regen_done/.render_done flags in $WORK_DIR);
# re-running skips completed phases. Training itself auto-resumes from
# $WORK_DIR/checkpoints unless --no-resume-from-checkpoint is added.
#
# DFlash2 vs the dspark/dflash runs in this repo -- the training command
# differs in five places, all of which are load-bearing:
#   * --speculator-type dflash2 (conv + candidate-selector draft)
#   * NO --draft-vocab-size: DFlash2 REQUIRES the full verifier vocab
#     (init raises on any other size); no d2t/t2t pruning, no token_freq
#   * NO --sample-from-anchor: DFlash2 trains block_size-1 mask slots
#     (block_size=8 -> 7 speculative tokens, matching the NPU limit
#     num_speculative_tokens+1 <= 16)
#   * NO --full-attention-indices: DFlash2's recipe is sliding window +
#     non-causal within blocks (sliding_window_non_causal defaults True)
#   * loss stays at the DFlash2 defaults (KL + fixed-exp-decay); pass
#     --loss-fn explicitly to override
#
# Usage:
#   1. edit cluster_env.sh for your paths (MODEL / WORK_DIR / AUDIO_ROOT /
#      RAW_DATA)
#   2. bash train_dflash2_single.sh            # full pipeline
#      MAX_SAMPLES=200 bash train_dflash2_single.sh   # smoke run
#
# Launch from any directory EXCEPT the speculators repo's parent (the
# checkout directory shadows the installed package).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAUNCH_VLLM="$REPO_ROOT/scripts/launch_vllm.py"
PREPARE_PY="$SCRIPT_DIR/qwen3_omni_asr_prepare_data.py"
for f in "$LAUNCH_VLLM" "$PREPARE_PY"; do
    [[ -f "$f" ]] || { echo "ERROR: $f not found"; exit 1; }
done

if [[ -d "./speculators" && ! -f "./speculators/__init__.py" ]]; then
    echo "ERROR: current directory contains a 'speculators/' repo checkout." >&2
    echo "       It shadows the installed package; launch from another directory." >&2
    exit 1
fi

# ===== Paths (edit cluster_env.sh, not here) =====
ENV_FILE="$SCRIPT_DIR/cluster_env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
MODEL="${MODEL:-/mnt/nfs/models/qwen3_omni_asr_gspo}"
WORK_DIR="${WORK_DIR:-/mnt/nfs/dflash2}"
RAW_DATA="${RAW_DATA:-$WORK_DIR/raw/asr.jsonl}"
AUDIO_ROOT="${AUDIO_ROOT:-/mnt/nfs/audios}"
CONV_DATA="$WORK_DIR/asr_conversations.jsonl"
ONPOLICY="$WORK_DIR/asr_onpolicy.jsonl"
PREPARED="$WORK_DIR/prepared"
HIDDEN_STATES="$WORK_DIR/hidden_states"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# ===== NPU layout =====
# Phases 2-3 use all 8 cards for the regen/render server; phase 4 splits
# the node: extraction server on 0-3, training on 4-7.
SERVER_GPUS="${SERVER_GPUS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"
VLLM_TP="${VLLM_TP:-4}"
CG_MODE="${CG_MODE:-FULL_DECODE_ONLY}"
PORT="${PORT:-8000}"

# ===== Data prep knobs =====
REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-64}"
REGEN_MAX_TOKENS="${REGEN_MAX_TOKENS:-2048}"
RENDER_NUM_PROC="${RENDER_NUM_PROC:-32}"
MAX_SEQ="${MAX_SEQ:-4096}"

# ===== Training hyperparameters =====
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
NUM_LAYERS="${NUM_LAYERS:-5}"
MAX_ANCHORS="${MAX_ANCHORS:-256}"            # ASR: short responses

# DFlash2 parameters. BLOCK_SIZE is tied to the serving contract:
# num_speculative_tokens = block_size - 1 (no sample-from-anchor), so
# block_size=8 -> 7 speculative tokens, same as the dspark/dflash runs.
BLOCK_SIZE=8
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 12 23 34 45}"   # same wide depth coverage as dspark
MASK_TOKEN_ID="${MASK_TOKEN_ID:-151676}"      # 151669 collides with <|audio_start|>
CONV_KERNEL_SIZE="${CONV_KERNEL_SIZE:-2}"     # dflash2 defaults; tunables below
CONV_GROUP_SIZE="${CONV_GROUP_SIZE:-16}"
SELECTOR_RANK="${SELECTOR_RANK:-256}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-16}"
SELECTOR_LOSS_ALPHA="${SELECTOR_LOSS_ALPHA:-1.0}"
# ========================================

mkdir -p "$WORK_DIR"

LIMIT_ARGS=()
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
    LIMIT_ARGS=(--limit "$MAX_SAMPLES")
fi

start_plain_server() {  # TP8 server for regen/render, all 8 cards
    echo "=== starting plain TP8 server (regen/render) ==="
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        vllm serve "$MODEL" \
        --port "$PORT" --tensor-parallel-size 8 \
        --allowed-local-media-path "$AUDIO_ROOT" \
        --enable-scale-out \
        --max-model-len 8192 --gpu-memory-utilization 0.85 \
        --compilation-config "{\"cudagraph_mode\": \"$CG_MODE\"}" \
        --max-num-seqs 64 --max-cudagraph-capture-size 32 &
    SERVE_PID=$!
    trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
    until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "=== plain server healthy on :$PORT ==="
}

stop_plain_server() {
    if [[ -n "${SERVE_PID:-}" ]]; then
        kill "$SERVE_PID" && wait "$SERVE_PID" 2>/dev/null || true
        trap - EXIT
        echo "=== plain server stopped ==="
    fi
}

# ================= Phase 1: convert =================
if [[ ! -s "$CONV_DATA" ]]; then
    echo "=== phase 1: convert ==="
    python3 "$PREPARE_PY" convert \
        --input "$RAW_DATA" --output "$CONV_DATA" "${LIMIT_ARGS[@]}"
else
    echo "=== phase 1 skipped: $CONV_DATA exists ==="
fi
[[ -s "$CONV_DATA" ]] || {
    echo "ERROR: $CONV_DATA is empty -- convert produced 0 rows." >&2
    echo "       Check the raw jsonl layout and that its audio paths exist on" >&2
    echo "       this machine; then delete the file and re-run." >&2
    exit 1
}

# ================= Phase 2: on-policy regen =================
REGEN_FLAG="$WORK_DIR/.regen_done"
if [[ -f "$REGEN_FLAG" && ! -s "$ONPOLICY" ]]; then
    echo "=== stale regen flag (no $ONPOLICY); re-running phase 2 ==="
    rm -f "$REGEN_FLAG"
fi
if [[ ! -f "$REGEN_FLAG" ]]; then
    start_plain_server
    echo "=== phase 2: regen ==="
    python3 "$PREPARE_PY" regen \
        --input "$CONV_DATA" --output "$ONPOLICY" \
        --endpoint "http://localhost:${PORT}/v1" \
        --temperature 0.0 --max-tokens "$REGEN_MAX_TOKENS" \
        --concurrency "$REGEN_CONCURRENCY" --resume \
        "${LIMIT_ARGS[@]}"
    touch "$REGEN_FLAG"
    echo "=== regen done ==="
    # Phase 3 (render) reuses this server; leave it up.
else
    echo "=== phase 2 skipped: $REGEN_FLAG exists ==="
fi
[[ -s "$ONPOLICY" ]] || {
    echo "ERROR: $ONPOLICY is empty -- regen produced 0 rows." >&2
    echo "       Delete it and the .regen_done flag, then re-run." >&2
    exit 1
}

# ================= Phase 3: render =================
RENDER_FLAG="$WORK_DIR/.render_done"
if [[ -f "$RENDER_FLAG" && ! -f "$PREPARED/dataset_info.json" ]]; then
    echo "=== stale render flag (no $PREPARED/dataset_info.json); re-running phase 3 ==="
    rm -f "$RENDER_FLAG"
fi
if [[ ! -f "$RENDER_FLAG" ]]; then
    [[ -n "${SERVE_PID:-}" ]] || start_plain_server
    echo "=== phase 3: render -> $PREPARED ==="
    speculators prepare-data \
        --model "$MODEL" --data "$ONPOLICY" --output "$PREPARED" \
        --seq-length "$MAX_SEQ" \
        --render-endpoint "http://localhost:${PORT}" \
        --num-preprocessing-workers "$RENDER_NUM_PROC"
    touch "$RENDER_FLAG"
    echo "=== render done ==="
else
    echo "=== phase 3 skipped: $RENDER_FLAG exists ==="
fi

# The plain server is no longer needed; free all 8 cards for phase 4.
stop_plain_server

# ================= Phase 4: extraction server + training =================
echo "=== phase 4: extraction server + training ==="
ASCEND_RT_VISIBLE_DEVICES="$SERVER_GPUS" \
    python "$LAUNCH_VLLM" train "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$HIDDEN_STATES" \
    --provenance-dir "$WORK_DIR" \
    -- --port "$PORT" --tensor-parallel-size "$VLLM_TP" \
       --allowed-local-media-path "$AUDIO_ROOT" \
       --max-model-len 8192 --gpu-memory-utilization 0.85 \
       --compilation-config "{\"cudagraph_mode\": \"$CG_MODE\"}" \
       --max-num-seqs 64 --max-cudagraph-capture-size 32 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "extraction server healthy on :$PORT"

ASCEND_RT_VISIBLE_DEVICES="$TRAIN_GPUS" \
    torchrun \
    --standalone --nproc_per_node="$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$PREPARED" \
    --vllm-endpoint "http://localhost:${PORT}/v1" \
    --save-path "$WORK_DIR/checkpoints" \
    --speculator-type dflash2 --draft-arch qwen3 \
    --block-size "$BLOCK_SIZE" --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --mask-token-id "$MASK_TOKEN_ID" \
    --conv-kernel-size "$CONV_KERNEL_SIZE" \
    --conv-group-size "$CONV_GROUP_SIZE" \
    --selector-rank "$SELECTOR_RANK" \
    --selector-top-k "$SELECTOR_TOP_K" \
    --selector-loss-alpha "$SELECTOR_LOSS_ALPHA" \
    --draft-attn-impl sdpa \
    --max-anchors "$MAX_ANCHORS" --total-seq-len "$MAX_SEQ" \
    --epochs "$EPOCHS" --lr "$LR" \
    --on-missing generate

kill "$SERVE_PID" && wait "$SERVE_PID" 2>/dev/null || true
trap - EXIT
echo "done. Checkpoints: $WORK_DIR/checkpoints (best: checkpoints/checkpoint_best)"
