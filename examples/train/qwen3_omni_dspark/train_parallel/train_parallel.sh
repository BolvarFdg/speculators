#!/bin/bash
# Parallel-prep DSpark training on Huawei Cloud: ONE submission, N instances,
# every phase fills all nodes.
#
# Phases (barriers via flags on shared NFS; every phase is idempotent --
# re-submitting skips whatever is already done):
#   1  convert          rank 0 only, CPU-only, others wait for the
#                       conversations file
#   2  regen            EVERY node: local TP8 server (all 8 cards) +
#                       sharded regen (rows idx % N == me) -> shard file
#   3  render + merge   EVERY node: local server renders its own shard into
#                       prepared_shard{N}/; rank 0 merges all shards and
#                       verifies (row accounting, token_freq sums)
#   4  training         EVERY node: cards 0-3 local extraction server +
#                       cards 4-7 torchrun multi-node DDP
#
# Platform env vars (same source as the reference DFlash script):
#   NODE_RANK=$VC_TASK_INDEX   NNODES=$MA_NUM_HOSTS
#   MASTER_ADDR=$VC_WORKER_HOSTS   MASTER_PORT=38888
#
# Usage (identical command on every instance, submitted once):
#   bash train_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LAUNCH_VLLM="$REPO_ROOT/scripts/launch_vllm.py"
PREPARE_PY="$SCRIPT_DIR/qwen3_omni_asr_prepare_data.py"
MERGE_PY="$SCRIPT_DIR/merge_prepared.py"
for f in "$LAUNCH_VLLM" "$PREPARE_PY" "$MERGE_PY"; do
    [[ -f "$f" ]] || { echo "ERROR: $f not found"; exit 1; }
done

if [[ -d "./speculators" && ! -f "./speculators/__init__.py" ]]; then
    echo "ERROR: launch from a directory without a 'speculators/' subdir" >&2
    exit 1
fi

# ===== Platform env vars =====
MASTER_ADDR="$VC_WORKER_HOSTS"
MASTER_PORT="38888"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
PORT="${PORT:-8000}"
BARRIER_TIMEOUT="${BARRIER_TIMEOUT:-259200}"   # 72h max wait at any barrier

# ===== Shared NFS paths =====
ENV_FILE="$SCRIPT_DIR/cluster_env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
MODEL="${MODEL:-/mnt/nfs/models/qwen3_omni_asr_gspo}"
WORK_DIR="${WORK_DIR:-/mnt/nfs/dspark}"
RAW_DATA="${RAW_DATA:-$WORK_DIR/raw/asr.jsonl}"
AUDIO_ROOT="${AUDIO_ROOT:-/mnt/nfs/audios}"
CONV_DATA="$WORK_DIR/asr_conversations.jsonl"
SHARD_ONPOLICY="$WORK_DIR/asr_onpolicy.shard${NODE_RANK}.jsonl"
SHARD_PREPARED="$WORK_DIR/prepared_shard${NODE_RANK}"
PREPARED="$WORK_DIR/prepared"
HIDDEN_STATES="$WORK_DIR/hidden_states"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# ===== NPU layout / training hyperparameters =====
SERVER_GPUS="${SERVER_GPUS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"
VLLM_TP="${VLLM_TP:-4}"
CG_MODE="${CG_MODE:-FULL_DECODE_ONLY}"
MAX_SEQ="${MAX_SEQ:-4096}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
NUM_LAYERS="${NUM_LAYERS:-5}"
REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-64}"
RENDER_NUM_PROC="${RENDER_NUM_PROC:-32}"

# DSpark contract parameters (do not change without updating serving).
BLOCK_SIZE=7
TARGET_LAYER_IDS="2 10 18 26 34"
DRAFT_VOCAB_SIZE=32000
MARKOV_RANK=64
MASK_TOKEN_ID=151669
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
# ========================================

mkdir -p "$WORK_DIR"
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NNODES" ]] || { echo "ERROR: NODE_RANK out of range"; exit 1; }
echo "node $NODE_RANK/$NNODES  master=$MASTER_ADDR:$MASTER_PORT  max_samples=$MAX_SAMPLES"

LIMIT_ARGS=()
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
    LIMIT_ARGS=(--limit "$MAX_SAMPLES")
fi

wait_for() {  # wait_for <path> <label>
    local waited=0
    until [[ -e "$1" ]]; do
        sleep 30; waited=$((waited + 30))
        if (( waited % 300 == 0 )); then echo "  waiting for $2 (${waited}s)"; fi
        [[ $waited -ge "$BARRIER_TIMEOUT" ]] && { echo "ERROR: timed out waiting for $2" >&2; exit 1; }
    done
}

start_plain_server() {  # TP8 plain server for regen/render, on all 8 cards
    # Stagger: rank 0 starts immediately, rank N waits N*30s so nodes don't
    # all hammer the NFS for model weights simultaneously (aggregate NFS
    # bandwidth is the bottleneck; 8 concurrent 60GB reads cause NPU
    # runtime memory-allocation timeouts).
    local delay=$(( NODE_RANK * 30 ))
    if [[ $delay -gt 0 ]]; then
        echo "=== [$NODE_RANK] waiting ${delay}s before server start (NFS stagger) ==="
        sleep "$delay"
    fi
    echo "=== [$NODE_RANK] starting plain TP8 server (regen/render) ==="
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
    echo "=== [$NODE_RANK] plain server healthy on :$PORT ==="
}

stop_plain_server() {
    if [[ -n "${SERVE_PID:-}" ]]; then
        kill "$SERVE_PID" && wait "$SERVE_PID" 2>/dev/null || true
        trap - EXIT
        echo "=== [$NODE_RANK] plain server stopped ==="
    fi
}

# ================= Phase 1: convert (rank 0) =================
if [[ ! -s "$CONV_DATA" ]]; then
    if [[ "$NODE_RANK" -eq 0 ]]; then
        echo "=== [$NODE_RANK] phase 1: convert ==="
        python3 "$PREPARE_PY" convert --input "$RAW_DATA" --output "$CONV_DATA" "${LIMIT_ARGS[@]}"
    else
        echo "=== [$NODE_RANK] phase 1: waiting for conversations ==="
        wait_for "$CONV_DATA" "conversations file"
    fi
else
    echo "=== [$NODE_RANK] phase 1 skipped: conversations exist ==="
fi

# ================= Phase 2: sharded regen (all nodes) =================
REGEN_FLAG="$WORK_DIR/.regen_done_${NODE_RANK}"
if [[ ! -f "$REGEN_FLAG" ]]; then
    start_plain_server
    echo "=== [$NODE_RANK] phase 2: sharded regen ($NODE_RANK/$NNODES) ==="
    python3 "$PREPARE_PY" regen \
        --input "$CONV_DATA" --output "$SHARD_ONPOLICY" \
        --endpoint "http://localhost:${PORT}/v1" \
        --temperature 0.0 --max-tokens 2048 \
        --concurrency "$REGEN_CONCURRENCY" --resume \
        --shard-index "$NODE_RANK" --shard-count "$NNODES" \
        "${LIMIT_ARGS[@]}"
    # Phase 3 needs the plain server too (render); keep it running.
else
    echo "=== [$NODE_RANK] phase 2 skipped: $REGEN_FLAG exists ==="
fi

# ================= Phase 3: sharded render + merge =================
RENDER_FLAG="$WORK_DIR/.render_done_${NODE_RANK}"
if [[ ! -f "$RENDER_FLAG" ]]; then
    [[ -n "${SERVE_PID:-}" ]] || start_plain_server
    echo "=== [$NODE_RANK] phase 3: rendering shard $NODE_RANK/$NNODES ==="
    speculators prepare-data \
        --model "$MODEL" --data "$SHARD_ONPOLICY" --output "$SHARD_PREPARED" \
        --seq-length "$MAX_SEQ" \
        --render-endpoint "http://localhost:${PORT}" \
        --num-preprocessing-workers "$RENDER_NUM_PROC"
    printf '{"shard_index": %d, "shard_count": %d}\n' "$NODE_RANK" "$NNODES" \
        > "$SHARD_PREPARED/.shard_meta"
    touch "$RENDER_FLAG"
    echo "=== [$NODE_RANK] render done ==="
else
    echo "=== [$NODE_RANK] phase 3 skipped: $RENDER_FLAG exists ==="
fi

# rank 0 merges all shards; others wait for the final .data_ready flag
if [[ "$NODE_RANK" -eq 0 ]]; then
    echo "=== [0] waiting for all $NNODES shards before merge ==="
    for i in $(seq 0 $((NNODES - 1))); do
        wait_for "$WORK_DIR/.render_done_$i" "shard $i render"
    done
    stop_plain_server
    echo "=== [0] merging $NNODES shards ==="
    python3 "$MERGE_PY" --work-dir "$WORK_DIR" --shard-count "$NNODES"
    touch "$WORK_DIR/.data_ready"
else
    stop_plain_server
    echo "=== [$NODE_RANK] waiting for rank 0 merge ==="
    wait_for "$WORK_DIR/.data_ready" "data_ready flag"
fi
echo "=== [$NODE_RANK] all data ready ==="

# ================= Phase 4: training (existing layout) =================
echo "=== [$NODE_RANK] phase 4: extraction server + training ==="
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
