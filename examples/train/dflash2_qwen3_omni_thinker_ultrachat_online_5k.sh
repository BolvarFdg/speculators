#!/bin/bash
# Online DFlash2 Training Script for the Qwen3-Omni thinker (text-first).
#
# Runs the same pipeline as dflash2_qwen3_8b_ultrachat_online_5k.sh against the
# Qwen3-Omni thinker as the verifier: data preparation, vLLM server launch
# (hidden-state extraction), and training with hidden states generated
# on-the-fly from the live server.
#
# This is the text-only path to validate the pipeline for a new verifier
# family: hidden states are extracted from the thinker's text stack with
# plain 1-D positions. Multimodal (image/audio/video) training data requires a
# custom dataset with multimodal parts and is not covered by this script.
#
# Prerequisites (Qwen3-Omni processor deps, on the vLLM host and the training
# host):
#   pip install qwen-omni-utils librosa soundfile torchvision
#
# On Ascend NPU (or other hardware without flex-attention support), add
#   --draft-attn-impl sdpa   (or eager)
# to the torchrun training command below.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dflash2_qwen3_omni_thinker_ultrachat_online_5k.sh

### Example E2E run for DFlash2 on the Qwen3-Omni-30B-A3B-Instruct thinker,
### 5k pre-tokenized UltraChat samples (text-only sanity check).

# Note: With just 5k text-only samples (and none of them on-policy for the
# omni thinker), the draft acceptance will be poor -- the point of this run is
# to verify the pipeline (config resolution, thinker weight loading, hidden
# state extraction, training loop) end to end before investing in omni data.

set -euo pipefail

# ============ Configuration ============
# The thinker is what the draft targets: vLLM serves the thinker-only model
# (Qwen3OmniMoeThinkerForConditionalGeneration) and extracts aux hidden
# states from its text stack.
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
DATASET="hf:inference-optimization/speculators-ci-datasets:tutorial_regen"  # pretokenized text data sharing the Qwen3 vocab; replace with omni on-policy data for real quality
OUTPUT_DIR="./output/dflash2_qwen3_omni_thinker_ultrachat_5k"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=3e-4

# DFlash2-specific parameters
SPECULATOR_TYPE="dflash2"
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=5
# Qwen3-Omni-30B-A3B thinker has 48 text layers; the launch script appends the
# final layer (48) automatically via --include-last-layer. Must match
# vLLM's eagle_aux_hidden_state_layer_ids.
TARGET_LAYER_IDS="2 24 45"

# GPU assignments (online training needs separate GPUs for vLLM and training).
# The 30B-A3B MoE server needs enough GPUs for its weights (TP=4 by default
# here); the draft itself fits comfortably on a single training GPU.
VLLM_GPUS="0,1,2,3"
TRAIN_GPUS="4,5"
NUM_TRAIN_GPUS=2
# =======================================

# Step 1: Prepare data
# The dataset ships pretokenized (input_ids + loss_mask) and shares the Qwen3
# vocabulary with the omni thinker, so prepare-data just packages it -- no
# --render-endpoint (or running vLLM server) is needed. Token frequencies and
# vocab pruning do not apply (DFlash2 requires the full verifier vocab).
echo "=== Step 1: Preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background (hidden-state extraction mode)
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py train "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --provenance-dir "$OUTPUT_DIR" \
    -- --tensor-parallel-size 4 --port "$VLLM_PORT" &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Train against the live vLLM server
echo "=== Step 3: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --on-missing generate

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
