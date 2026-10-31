#!/bin/bash
# Qwen3-Omni (ASR) DSpark training -- Phase A: data preparation.
#
# Runs on the DATA machine. Produces the handoff package consumed by
# train.sh on the training machine:
#   Step 1  convert raw ASR JSONL -> speculators conversations (offline)
#   Step 2  on-policy response regeneration on a plain vLLM server
#   Step 4  speculators prepare-data (render input_ids/loss_mask/messages
#           against the same plain server)
# Steps are idempotent: finished products are skipped on re-run.
#
# Every path is overridable via environment variables, e.g.:
#   MODEL=/mnt/models/omni_gspo RAW_DATA=/mnt/data/asr.jsonl \
#     WORK_DIR=/mnt/share/dspark bash prepare_data.sh
#
# Handoff checklist for the training machine (printed at the end):
#   * $WORK_DIR/prepared/          -- dataset + token_freq.pt
#   * audio files                  -- MUST sit at the SAME absolute paths
#                                      (they are embedded in the dataset's
#                                      messages column)
#   * $MODEL weights               -- extraction server + frozen verifier
#                                      weights both need them
#
# Usage: bash examples/train/qwen3_omni_dspark/prepare_data.sh

set -euo pipefail

# Resolve this script's location so it can be launched from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PREPARE_PY="$SCRIPT_DIR/qwen3_omni_asr_prepare_data.py"
[[ -f "$PREPARE_PY" ]] || { echo "ERROR: $PREPARE_PY not found"; exit 1; }

# ============ Configuration (env-overridable) ============
MODEL="${MODEL:-/path/to/your/final_gspo_qwen3_omni}"   # final gspo verifier
RAW_DATA="${RAW_DATA:-./asr_raw.jsonl}"                 # messages/audios/(ground_truth) rows
WORK_DIR="${WORK_DIR:-./output/dspark_qwen3_omni_asr}"  # artifact root
AUDIO_ROOT="${AUDIO_ROOT:-/data}"                       # covers all audio paths
PORT="${PORT:-8000}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"                        # NPUs for the server
VLLM_TP="${VLLM_TP:-8}"

MAX_SAMPLES="${MAX_SAMPLES:-0}"          # 0 = full data; 5000 for a smoke run
MAX_SEQ="${MAX_SEQ:-4096}"               # audio is ~13 tok/s; keep for train.sh
REGEN_TEMPERATURE="${REGEN_TEMPERATURE:-0.0}"   # match serving sampling
REGEN_MAX_TOKENS="${REGEN_MAX_TOKENS:-2048}"
# ========================================

CONV_DATA="$WORK_DIR/asr_conversations.jsonl"
ONPOLICY_DATA="$WORK_DIR/asr_onpolicy.jsonl"
PREPARED="$WORK_DIR/prepared"

LIMIT_ARGS=()
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
    LIMIT_ARGS=(--limit "$MAX_SAMPLES")
fi

wait_healthy() {
    until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "server on port ${PORT} is healthy."
}

# ===== Step 1: raw ASR JSONL -> conversations (offline) =====
if [[ ! -s "$CONV_DATA" ]]; then
    echo "=== Step 1: converting raw data ==="
    python3 "$PREPARE_PY" \
        convert --input "$RAW_DATA" --output "$CONV_DATA" "${LIMIT_ARGS[@]}"
else
    echo "=== Step 1 skipped: $CONV_DATA exists ==="
fi

# ===== Steps 2 & 4: one plain server serves regen AND rendering =====
echo "=== Launching plain vLLM server for regen + rendering ==="
ASCEND_RT_VISIBLE_DEVICES="$GPUS" \
    vllm serve "$MODEL" \
    --port "$PORT" --tensor-parallel-size "$VLLM_TP" \
    --allowed-local-media-path "$AUDIO_ROOT" \
    --max-model-len 8192 --gpu-memory-utilization 0.85 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
wait_healthy

# ===== Step 2: on-policy response regeneration =====
if [[ ! -s "$ONPOLICY_DATA" ]]; then
    echo "=== Step 2: regenerating responses with $MODEL ==="
    python3 "$PREPARE_PY" \
        regen \
        --input "$CONV_DATA" --output "$ONPOLICY_DATA" \
        --endpoint "http://localhost:${PORT}/v1" \
        --temperature "$REGEN_TEMPERATURE" --max-tokens "$REGEN_MAX_TOKENS" \
        --concurrency 64 --resume "${LIMIT_ARGS[@]}"
else
    echo "=== Step 2 skipped: $ONPOLICY_DATA exists ==="
fi

# ===== Step 4: prepare-data (render input_ids / loss_mask / messages) =====
if [[ ! -d "$PREPARED" ]]; then
    echo "=== Step 4: preparing data ==="
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
    echo "=== Step 4 skipped: $PREPARED exists ==="
fi

kill "$SERVE_PID" && wait "$SERVE_PID" 2>/dev/null || true
trap - EXIT

echo ""
echo "Phase A complete. Handoff package for the training machine:"
echo "  1) dataset:      $PREPARED"
echo "  2) audio files:  must exist at the SAME absolute paths on the train"
echo "                   machine (paths are embedded in the dataset)"
echo "  3) model weights: $MODEL (extraction server + frozen verifier parts)"
echo "  4) environment:  same image with speculators + hs_connectors installed"
echo "Then on the training machine run train.sh with:"
echo "  MODEL=<model path> PREPARED=$PREPARED AUDIO_ROOT=$AUDIO_ROOT \\"
echo "    bash examples/train/qwen3_omni_dspark/train.sh"
echo "(keep MAX_SEQ=$MAX_SEQ identical when launching train.sh)"
