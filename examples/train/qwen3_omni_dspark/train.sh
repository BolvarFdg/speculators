#!/bin/bash
# Qwen3-Omni (ASR) DSpark training -- Phase B: extraction server + training.
#
# Runs on the TRAIN machine, consuming the handoff package produced by
# prepare_data.sh (Phase A):
#   Step 3  hidden-states extraction server (extract_hidden_states),
#           stays up for the whole training run
#   Step 5  DSpark draft training on separate NPUs; hidden states are
#           fetched on demand from the live server and cached on disk
#           (first fetch persists; later epochs / restarts reuse the cache)
#
# Every path is overridable via environment variables, e.g.:
#   MODEL=/mnt/models/omni_gspo PREPARED=/mnt/share/prepared \
#     AUDIO_ROOT=/mnt/data OUTPUT_DIR=/mnt/share/dspark_train \
#       bash train.sh
#
# Requirements imported from Phase A (verify before launching):
#   * $PREPARED    -- dataset dir from prepare_data.sh
#   * $MODEL       -- same checkpoint used in Phase A
#   * audio files  -- same absolute paths as embedded in the dataset
#   * MAX_SEQ      -- same value as Phase A (token layout must match)
#
# Usage: bash examples/train/qwen3_omni_dspark/train.sh

set -euo pipefail

# Resolve this script's location so it can be launched from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAUNCH_VLLM="$REPO_ROOT/scripts/launch_vllm.py"
[[ -f "$LAUNCH_VLLM" ]] || { echo "ERROR: $LAUNCH_VLLM not found (the speculators repo must stay in the image)"; exit 1; }

# ============ Configuration (env-overridable) ============
MODEL="${MODEL:-/path/to/your/final_gspo_qwen3_omni}"
PREPARED="${PREPARED:-./output/dspark_qwen3_omni_asr/prepared}"  # from Phase A
OUTPUT_DIR="${OUTPUT_DIR:-./output/dspark_qwen3_omni_asr}"       # checkpoints + hidden states
AUDIO_ROOT="${AUDIO_ROOT:-/data}"
PORT="${PORT:-8000}"

# NPU layout: extraction server cards vs. training cards must not overlap.
SERVER_GPUS="${SERVER_GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_GPUS="${TRAIN_GPUS:-8,9}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
VLLM_TP="${VLLM_TP:-8}"

MAX_SEQ="${MAX_SEQ:-4096}"     # MUST match Phase A
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
NUM_LAYERS="${NUM_LAYERS:-5}"

# DSpark contract parameters -- aligned with the vLLM serving validation
# (_validate_qwen3_omni_dspark). Do not change without updating the serving
# side accordingly (num_speculative_tokens == BLOCK_SIZE at serve time).
BLOCK_SIZE=7
TARGET_LAYER_IDS="2 10 18 26 34"
DRAFT_VOCAB_SIZE=32000
MARKOV_RANK=64
MASK_TOKEN_ID=151669
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
# ========================================

# ===== Pre-flight checks =====
[[ -d "$PREPARED" ]] || { echo "ERROR: PREPARED=$PREPARED not found (run prepare_data.sh first or set PREPARED)"; exit 1; }
[[ -d "$MODEL" || -f "$MODEL" ]] || { echo "ERROR: MODEL=$MODEL not found"; exit 1; }

python3 - "$PREPARED" <<'EOF'
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from datasets import load_from_disk

ds = load_from_disk(sys.argv[1])
missing, checked = [], 0
for row in ds.select(range(min(5, len(ds)))):
    for msg in row.get("messages") or []:
        if not isinstance(msg.get("content"), list):
            continue
        for part in msg["content"]:
            url = (part.get("audio_url") or {}).get("url", "")
            if url.startswith("file://"):
                path = Path(unquote(urlparse(url).path))
                checked += 1
                if not path.exists():
                    missing.append(str(path))
if missing:
    print("ERROR: audio files referenced by the dataset are missing on this "
          "machine; mount them at the SAME absolute paths:")
    for p in missing[:10]:
        print("  ", p)
    sys.exit(1)
print(f"pre-flight OK: dataset rows={len(ds)}, audio paths checked={checked}")
EOF

# ===== Step 3: hidden-states extraction server (up through training) =====
echo "=== Step 3: launching hidden-states extraction server ==="
ASCEND_RT_VISIBLE_DEVICES="$SERVER_GPUS" \
    python "$LAUNCH_VLLM" "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    -- --port "$PORT" --tensor-parallel-size "$VLLM_TP" \
       --allowed-local-media-path "$AUDIO_ROOT" \
       --max-model-len 8192 --gpu-memory-utilization 0.85 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "extraction server on port ${PORT} is healthy."

# ===== Step 5: DSpark training (hidden states fetched on demand) =====
echo "=== Step 5: training ==="
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
echo "Serving (num_speculative_tokens must equal BLOCK_SIZE=$BLOCK_SIZE):"
echo "  vllm serve $MODEL --tensor-parallel-size $VLLM_TP \\"
echo "    --speculative-config '{\"method\":\"dspark\",\"model\":\"$OUTPUT_DIR/checkpoints/<final>\",\"num_speculative_tokens\":$BLOCK_SIZE,\"draft_sample_method\":\"greedy\"}'"
