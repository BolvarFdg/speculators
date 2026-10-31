#!/bin/bash
# Qwen3-Omni (ASR) DSpark draft training on Ascend NPUs -- end-to-end pipeline.
#
# Trains a DSpark draft for an ASR-only Qwen3-Omni target (audio in, text out)
# on data in the raw format:
#   {"messages": [system/user/assistant turns with "<audio>" placeholders],
#    "audios": ["/abs/path.wav"], "ground_truth": "...", ...}
#
# Pipeline (run from the speculators repo root inside the NPU container):
#   Step 0  convert raw JSONL to speculators conversations (offline)
#   Step 1  plain vLLM server -> regenerate responses on-policy (audio refs
#           kept; the official regenerate-responses drops them), then stop it
#   Step 2  hidden-states extraction server (extract_hidden_states), kept up
#   Step 3  speculators prepare-data (renders input_ids/loss_mask/messages)
#   Step 4  DSpark training on separate NPUs, hidden states streamed from the
#           live server (--on-missing generate)
#
# Audio files must be visible inside the container at their JSONL paths, and
# AUDIO_ROOT must cover them (vLLM needs --allowed-local-media-path for
# file:// media).
#
# Usage: copy this script, edit the configuration block, then
#   bash examples/train/qwen3_omni_dspark/dspark_qwen3_omni_asr_npu.sh
#
# The exported checkpoint is served by vLLM with (block size must match):
#   vllm serve $MODEL --tensor-parallel-size 8 \
#     --speculative-config '{"method":"dspark","model":"'"$OUTPUT_DIR"'/checkpoints/final","num_speculative_tokens":7,"draft_sample_method":"greedy"}'

set -euo pipefail

# ============ Configuration ============
MODEL="/path/to/your/final_gspo_qwen3_omni"   # final (gspo) verifier checkpoint
RAW_DATA="./asr_raw.jsonl"                    # messages/audios/ground_truth rows
OUTPUT_DIR="./output/dspark_qwen3_omni_asr"
AUDIO_ROOT="/data"                            # dir covering all audio paths
PORT=8000

# NPU layout: extraction server cards vs. training cards must not overlap.
SERVER_GPUS="0,1,2,3,4,5,6,7"
TRAIN_GPUS="8,9"
NUM_TRAIN_GPUS=2
VLLM_TP=8

MAX_SAMPLES=0            # 0 = full data; set 5000 for a first pipeline run
MAX_SEQ=4096             # ~4.5 min of audio at 13 tok/s + response
REGEN_TEMPERATURE=0.0    # match your serving sampling
REGEN_MAX_TOKENS=2048

# DSpark hyperparameters (must stay aligned with the vLLM serving contract:
# num_speculative_tokens == BLOCK_SIZE, no mrope in the draft, vanilla head).
BLOCK_SIZE=7
NUM_LAYERS=5
TARGET_LAYER_IDS="2 10 18 26 34"
DRAFT_VOCAB_SIZE=32000
MARKOV_RANK=64
MASK_TOKEN_ID=151669
EPOCHS=3
LR=3e-4
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
# ========================================

CONV_DATA="$OUTPUT_DIR/asr_conversations.jsonl"
ONPOLICY_DATA="$OUTPUT_DIR/asr_onpolicy.jsonl"
PREPARED="$OUTPUT_DIR/prepared"

mkdir -p "$OUTPUT_DIR"

wait_healthy() {
    until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "server on port ${PORT} is healthy."
}

LIMIT_ARGS=()
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
    LIMIT_ARGS=(--limit "$MAX_SAMPLES")
fi

# ===== Step 0: raw ASR JSONL -> conversations (offline, checks audio exist) =====
if [[ ! -s "$CONV_DATA" ]]; then
    echo "=== Step 0: converting raw data ==="
    python3 examples/train/qwen3_omni_dspark/qwen3_omni_asr_prepare_data.py convert \
        --input "$RAW_DATA" --output "$CONV_DATA" "${LIMIT_ARGS[@]}"
fi

# ===== Step 1: on-policy response regeneration on a plain server =====
if [[ ! -s "$ONPOLICY_DATA" ]]; then
    echo "=== Step 1: regenerating responses with $MODEL ==="
    ASCEND_RT_VISIBLE_DEVICES="$SERVER_GPUS" \
        vllm serve "$MODEL" \
        --port "$PORT" --tensor-parallel-size "$VLLM_TP" \
        --allowed-local-media-path "$AUDIO_ROOT" \
        --max-model-len 8192 --gpu-memory-utilization 0.85 &
    REGEN_PID=$!
    trap 'kill "$REGEN_PID" 2>/dev/null || true' EXIT
    wait_healthy

    python3 examples/train/qwen3_omni_dspark/qwen3_omni_asr_prepare_data.py regen \
        --input "$CONV_DATA" --output "$ONPOLICY_DATA" \
        --endpoint "http://localhost:${PORT}/v1" \
        --temperature "$REGEN_TEMPERATURE" --max-tokens "$REGEN_MAX_TOKENS" \
        --concurrency 64 --resume "${LIMIT_ARGS[@]}"

    kill "$REGEN_PID" && wait "$REGEN_PID" 2>/dev/null || true
    trap - EXIT
    echo "regeneration server stopped."
fi

# ===== Step 2: hidden-states extraction server (stays up through training) =====
echo "=== Step 2: launching hidden-states extraction server ==="
ASCEND_RT_VISIBLE_DEVICES="$SERVER_GPUS" \
    python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    -- --port "$PORT" --tensor-parallel-size "$VLLM_TP" \
       --allowed-local-media-path "$AUDIO_ROOT" \
       --max-model-len 8192 --gpu-memory-utilization 0.85 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
wait_healthy

# ===== Step 3: prepare-data (render input_ids / loss_mask / messages) =====
if [[ ! -d "$PREPARED" ]]; then
    echo "=== Step 3: preparing data ==="
    if [[ "$MAX_SAMPLES" -gt 0 ]]; then
        speculators prepare-data \
            --model "$MODEL" --data "$ONPOLICY_DATA" --output "$PREPARED" \
            --seq-length "$MAX_SEQ" --max-samples "$MAX_SAMPLES" \
            --render-endpoint "http://localhost:${PORT}"
    else
        speculators prepare-data \
            --model "$MODEL" --data "$ONPOLICY_DATA" --output "$PREPARED" \
            --seq-length "$MAX_SEQ" \
            --render-endpoint "http://localhost:${PORT}"
    fi
else
    echo "=== Step 3 skipped: $PREPARED already exists ==="
fi

# ===== Step 4: DSpark training (hidden states from the live server) =====
echo "=== Step 4: training ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_GPUS" \
    torchrun --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$PREPARED" \
    --vllm-endpoint "http://localhost:${PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --speculator-type dspark --draft-arch qwen3 \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --block-size "$BLOCK_SIZE" --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --full-attention-indices $(seq 0 $((NUM_LAYERS - 1))) \
    --markov-rank "$MARKOV_RANK" --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --mask-token-id "$MASK_TOKEN_ID" \
    --draft-attn-impl sdpa \
    --max-anchors 3072 --total-seq-len "$MAX_SEQ" \
    --epochs "$EPOCHS" --lr "$LR" \
    --loss-fn "$LOSS_FN" \
    --on-missing generate

echo "Done. Checkpoints in $OUTPUT_DIR/checkpoints."
echo "Serving (block size must equal num_speculative_tokens):"
echo "  vllm serve $MODEL --tensor-parallel-size $VLLM_TP \\"
echo "    --speculative-config '{\"method\":\"dspark\",\"model\":\"$OUTPUT_DIR/checkpoints/<final>\",\"num_speculative_tokens\":$BLOCK_SIZE,\"draft_sample_method\":\"greedy\"}'"
