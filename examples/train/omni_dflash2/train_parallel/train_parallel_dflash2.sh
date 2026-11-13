#!/bin/bash
# Multi-instance DFlash2 training on Huawei Cloud: ONE submission, N
# instances, every phase fills all nodes.
#
# Modeled on the proven qwen3_omni_dspark/train_parallel pipeline; the data
# phases are identical, only the phase 4 training command switches to
# DFlash2 (see the five load-bearing differences below). Unlike the dspark
# merge, DFlash2 trains on the full verifier vocabulary, so no
# token_freq.pt is produced or merged.
#
# Phases (barriers via flags on shared NFS; every phase is idempotent --
# re-submitting skips whatever is already done):
#   1  convert          rank 0 only, CPU-only, others wait for the
#                       conversations file
#   2  regen            EVERY node: local TP8 server (all 8 cards) +
#                       sharded regen (rows idx % N == me) -> shard file
#   3  render + merge   EVERY node: local server renders its own shard into
#                       prepared_shard{N}/; rank 0 merges all shards and
#                       verifies (shard metadata, row accounting)
#   4  training         EVERY node: cards 0-3 local extraction server +
#                       cards 4-7 torchrun multi-node DDP
#
# Platform env vars (same source as the dspark script):
#   NODE_RANK=$VC_TASK_INDEX   NNODES=$MA_NUM_HOSTS
#   MASTER_ADDR=$VC_WORKER_HOSTS   MASTER_PORT=38888
#
# DFlash2 vs the dspark/dflash training commands -- five differences, all
# load-bearing:
#   * --speculator-type dflash2 (conv + candidate-selector draft)
#   * NO --draft-vocab-size: DFlash2 REQUIRES the full verifier vocab
#   * NO --sample-from-anchor: block_size=8 -> 7 speculative tokens
#   * NO --full-attention-indices: sliding window + non-causal in blocks
#   * loss stays at DFlash2 defaults (KL + fixed-exp-decay)
#
# Usage (identical command on every instance, submitted once):
#   bash train_parallel_dflash2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LAUNCH_VLLM="$REPO_ROOT/scripts/launch_vllm.py"
PREPARE_PY="$SCRIPT_DIR/../qwen3_omni_asr_prepare_data.py"
MERGE_PY="$SCRIPT_DIR/merge_prepared.py"
for f in "$LAUNCH_VLLM" "$PREPARE_PY" "$MERGE_PY"; do
    [[ -f "$f" ]] || { echo "ERROR: $f not found"; exit 1; }
done

if [[ -d "./speculators" && ! -f "./speculators/__init__.py" ]]; then
    echo "ERROR: launch from a directory without a 'speculators/' subdir" >&2
    exit 1
fi

# ===== Platform env vars =====
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"   # first host if comma-separated
MASTER_PORT="38888"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
PORT="${PORT:-8000}"
BARRIER_TIMEOUT="${BARRIER_TIMEOUT:-259200}"   # 72h max wait at any barrier

# ===== Shared NFS paths (edit ../cluster_env.sh, not here) =====
ENV_FILE="$SCRIPT_DIR/../cluster_env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
MODEL="${MODEL:-/mnt/nfs/models/qwen3_omni_asr_gspo}"
WORK_DIR="${WORK_DIR:-/mnt/nfs/dflash2}"
RAW_DATA="${RAW_DATA:-$WORK_DIR/raw/asr.jsonl}"
AUDIO_ROOT="${AUDIO_ROOT:-/mnt/nfs/audios}"
CONV_DATA="$WORK_DIR/asr_conversations.jsonl"
SHARD_ONPOLICY="$WORK_DIR/asr_onpolicy.shard${NODE_RANK}.jsonl"
SHARD_PREPARED="$WORK_DIR/prepared_shard${NODE_RANK}"
PREPARED="$WORK_DIR/prepared"
HIDDEN_STATES="$WORK_DIR/hidden_states"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# ===== NPU layout / data prep knobs =====
SERVER_GPUS="${SERVER_GPUS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"
VLLM_TP="${VLLM_TP:-4}"
CG_MODE="${CG_MODE:-FULL_DECODE_ONLY}"
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
# block_size=8 -> 7 speculative tokens (within the NPU
# num_speculative_tokens + 1 <= 16 limit).
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
        if (( waited % 30 == 0 )); then echo "  waiting for $2 (${waited}s)"; fi
        if (( waited >= BARRIER_TIMEOUT )); then
            echo "ERROR: timed out waiting for $2" >&2
            exit 1
        fi
    done
    return 0
}

start_plain_server() {  # TP8 plain server for regen/render, on all 8 cards
    # Stagger: rank 0 starts immediately, rank N waits N*30s so nodes don't
    # all hammer the NFS for model weights simultaneously (aggregate NFS
    # bandwidth is the bottleneck; N concurrent 60GB reads cause NPU
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
        python3 "$PREPARE_PY" convert \
            --input "$RAW_DATA" --output "$CONV_DATA" "${LIMIT_ARGS[@]}"
        [[ -s "$CONV_DATA" ]] || {
            echo "ERROR: $CONV_DATA is empty -- convert produced 0 rows." >&2
            echo "       Check the raw jsonl layout and that its audio paths exist" >&2
            echo "       on this machine; then delete the file and re-submit." >&2
            exit 1
        }
    else
        echo "=== [$NODE_RANK] phase 1: waiting for conversations ==="
        wait_for "$CONV_DATA" "conversations file"
    fi
else
    echo "=== [$NODE_RANK] phase 1 skipped: conversations exist ==="
fi

# ================= Phase 2: sharded regen (all nodes) =================
REGEN_FLAG="$WORK_DIR/.regen_done_${NODE_RANK}"
# Guard: stale flag without its shard file means data was cleaned up.
if [[ -f "$REGEN_FLAG" && ! -s "$SHARD_ONPOLICY" ]]; then
    echo "=== [$NODE_RANK] stale regen flag (no $SHARD_ONPOLICY); re-running phase 2 ==="
    rm -f "$REGEN_FLAG"
fi
if [[ ! -f "$REGEN_FLAG" ]]; then
    start_plain_server
    echo "=== [$NODE_RANK] phase 2: sharded regen ($NODE_RANK/$NNODES) ==="
    python3 "$PREPARE_PY" regen \
        --input "$CONV_DATA" --output "$SHARD_ONPOLICY" \
        --endpoint "http://localhost:${PORT}/v1" \
        --temperature 0.0 --max-tokens "$REGEN_MAX_TOKENS" \
        --concurrency "$REGEN_CONCURRENCY" --resume \
        --shard-index "$NODE_RANK" --shard-count "$NNODES" \
        "${LIMIT_ARGS[@]}"
    [[ -s "$SHARD_ONPOLICY" ]] || {
        echo "ERROR: [$NODE_RANK] $SHARD_ONPOLICY is empty -- regen produced 0 rows." >&2
        exit 1
    }
    touch "$REGEN_FLAG"
    echo "=== [$NODE_RANK] regen done ==="
    # Phase 3 needs the plain server too (render); keep it running.
else
    echo "=== [$NODE_RANK] phase 2 skipped: $REGEN_FLAG exists ==="
fi

# ================= Phase 3: sharded render + merge =================
RENDER_FLAG="$WORK_DIR/.render_done_${NODE_RANK}"
# Guard: a stale flag without its artifact means a previous run's data was
# cleaned up -- drop the flag so this phase actually re-runs.
if [[ -f "$RENDER_FLAG" && ! -f "$SHARD_PREPARED/.shard_meta" ]]; then
    echo "=== [$NODE_RANK] stale render flag (no $SHARD_PREPARED/.shard_meta); re-running phase 3 ==="
    rm -f "$RENDER_FLAG"
fi
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

# rank 0 stops its server FIRST (its own regen+render are done, the server
# would just idle on 8 cards while waiting -- and a long-idling 30B server
# eventually gets killed by the platform/memory watchdog).
# Other ranks stop theirs too, then everyone waits for .data_ready.
stop_plain_server

if [[ "$NODE_RANK" -eq 0 ]]; then
    echo "=== [0] waiting for all $NNODES shards before merge ==="
    for i in $(seq 0 $((NNODES - 1))); do
        wait_for "$WORK_DIR/.render_done_$i" "shard $i render"
    done
    echo "=== [0] merging $NNODES shards ==="
    python3 "$MERGE_PY" --work-dir "$WORK_DIR" --shard-count "$NNODES"
    touch "$WORK_DIR/.data_ready"
else
    echo "=== [$NODE_RANK] waiting for rank 0 merge ==="
    wait_for "$WORK_DIR/.data_ready" "data_ready flag"
fi
echo "=== [$NODE_RANK] all data ready ==="

# ================= Phase 4: training (existing layout) =================
echo "=== [$NODE_RANK] phase 4: extraction server + training ==="
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
echo "node $NODE_RANK done. Checkpoints: $WORK_DIR/checkpoints"
