#!/bin/bash
# Multi-instance (cluster) DSpark training on Huawei Cloud.
#
# SUBMIT ONCE with N instances: the platform runs this same script on every
# node; NODE_RANK tells the nodes apart. You do NOT ssh into nodes.
#
# Per-node layout (8 cards):
#   cards 0-3: local hidden-states extraction server (TP4)
#   cards 4-7: torchrun multi-node DDP training (endpoint = LOCAL server;
#              data is sharded by global DP rank so each node only ever
#              requests its own samples -> no cross-node conflicts)
#
# Two operating modes:
#   PREP_IN_JOB=0 (default): data was prepared by an earlier single-instance
#       job running prepare_data.sh. This job trains only.
#   PREP_IN_JOB=1: rank 0 runs the whole data preparation inside this job
#       (convert + regen + prepare-data) while the other nodes poll a
#       ready-flag on shared NFS; then every node trains together.
#
# Usage (identical command on every instance, submitted once):
#   bash train_cluster.sh
#   PREP_IN_JOB=1 RAW_DATA=... bash train_cluster.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAUNCH_VLLM="$REPO_ROOT/scripts/launch_vllm.py"
[[ -f "$LAUNCH_VLLM" ]] || { echo "ERROR: $LAUNCH_VLLM not found"; exit 1; }

if [[ -d "./speculators" && ! -f "./speculators/__init__.py" ]]; then
    echo "ERROR: launch from a directory without a 'speculators/' subdir" >&2
    exit 1
fi

# Shared paths: edit cluster_env.sh once instead of this script.
ENV_FILE="$SCRIPT_DIR/cluster_env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

# ===== Platform env vars (same source as the reference DFlash script) =====
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT="38888"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
PORT="${PORT:-8000}"                       # local extraction server port
PREP_IN_JOB="${PREP_IN_JOB:-0}"
PREP_WAIT_TIMEOUT="${PREP_WAIT_TIMEOUT:-86400}"   # seconds others wait for rank 0

# ===== Shared NFS paths (MUST be identical on every instance) =====
MODEL="${MODEL:-/mnt/nfs/models/qwen3_omni_asr_gspo}"
WORK_DIR="${WORK_DIR:-/mnt/nfs/dspark}"                   # all artifacts live here
RAW_DATA="${RAW_DATA:-$WORK_DIR/raw/asr.jsonl}"          # needed when PREP_IN_JOB=1
AUDIO_ROOT="${AUDIO_ROOT:-/mnt/nfs/audios}"
PREPARED="$WORK_DIR/prepared"
HIDDEN_STATES="$WORK_DIR/hidden_states"
READY_FLAG="$WORK_DIR/.data_ready"

# ===== NPU layout / training hyperparameters (per instance) =====
SERVER_GPUS="${SERVER_GPUS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"
VLLM_TP="${VLLM_TP:-4}"
CG_MODE="${CG_MODE:-FULL_DECODE_ONLY}"
MAX_SEQ="${MAX_SEQ:-4096}"                 # MUST match prepare_data.sh
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
NUM_LAYERS="${NUM_LAYERS:-5}"

# DSpark contract parameters (do not change without updating serving).
BLOCK_SIZE=7
TARGET_LAYER_IDS="2 10 18 26 34"
DRAFT_VOCAB_SIZE=32000
MARKOV_RANK=64
MASK_TOKEN_ID=151669
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
# ========================================

mkdir -p "$WORK_DIR"
echo "node $NODE_RANK/$NNODES  master=$MASTER_ADDR:$MASTER_PORT"

# ---------- Single-job mode: rank 0 prepares, others wait ----------
if [[ "$PREP_IN_JOB" == "1" ]]; then
    if [[ "$NODE_RANK" -eq 0 ]]; then
        rm -f "$READY_FLAG"
        echo "=== rank 0: running data preparation (prepare_data.sh) ==="
        MODEL="$MODEL" RAW_DATA="$RAW_DATA" WORK_DIR="$WORK_DIR" AUDIO_ROOT="$AUDIO_ROOT" \
            GPUS="0,1,2,3,4,5,6,7" VLLM_TP=8 \
            bash "$SCRIPT_DIR/prepare_data.sh"
        touch "$READY_FLAG"
        echo "=== rank 0: data ready ==="
    else
        echo "=== rank $NODE_RANK: waiting for rank 0 to finish data preparation ==="
        waited=0
        until [[ -f "$READY_FLAG" ]]; do
            sleep 30; waited=$((waited + 30))
            if (( waited % 300 == 0 )); then
                echo "  still waiting (${waited}s elapsed)..."
            fi
            [[ $waited -ge "$PREP_WAIT_TIMEOUT" ]] && { echo "ERROR: timed out waiting for .data_ready" >&2; exit 1; }
        done
        echo "=== rank $NODE_RANK: data ready ==="
    fi
fi

# ---------- Pre-flight ----------
[[ -d "$PREPARED" ]] || { echo "ERROR: PREPARED=$PREPARED not found (run prepare_data.sh first, or PREP_IN_JOB=1)"; exit 1; }

# ---------- Local extraction server (cards 0-3) ----------
echo "=== starting local extraction server (node $NODE_RANK) ==="
ASCEND_RT_VISIBLE_DEVICES="$SERVER_GPUS" \
    python "$LAUNCH_VLLM" "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$HIDDEN_STATES" \
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
echo "local extraction server healthy on :$PORT"

# ---------- Multi-node DDP training (cards 4-7) ----------
echo "=== launching multi-node training (node $NODE_RANK/$NNODES) ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_GPUS" \
    torchrun \
    --nnodes="$NNODES" --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    --nproc_per_node="$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$PREPARED" \
    --vllm-endpoint "http://localhost:${PORT}/v1" \
    --save-path "$WORK_DIR/checkpoints" \
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

kill "$SERVE_PID" && wait "$SERVE_PID" 2>/dev/null || true
trap - EXIT
echo "node $NODE_RANK done. Checkpoints: $WORK_DIR/checkpoints"
