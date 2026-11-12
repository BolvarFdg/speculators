# Shared path configuration for the single-node DFlash2 training pipeline.
#
# This is the ONLY file you need to edit for paths: the training script
# sources it. One-off knobs (MAX_SAMPLES, EPOCHS, ...) stay on the command
# line or environment and keep their in-script defaults.

# Verifier checkpoint (Qwen3-Omni ASR final gspo checkpoint) on shared NFS.
# Must be the SAME checkpoint used for serving/regen/extraction/training.
MODEL=/mnt/nfs/models/qwen3_omni_asr_gspo

# All pipeline artifacts under this tree: asr_conversations.jsonl,
# asr_onpolicy.jsonl, prepared/, hidden_states/, checkpoints/.
# NOTE: use a DIFFERENT WORK_DIR from the dspark/dflash pipelines to avoid
# overwriting each other's checkpoints and hidden-state caches.
WORK_DIR=/mnt/nfs/dflash2

# Directory covering all audio files (server --allowed-local-media-path root)
AUDIO_ROOT=/mnt/nfs/audios

# Raw ASR jsonl (messages/audios/ground_truth rows)
RAW_DATA=/mnt/nfs/data/all_asr.jsonl

# Training hyperparameters (DFlash2-specific defaults). Overridable via the
# environment so smoke runs can override without editing this file.
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
NUM_LAYERS="${NUM_LAYERS:-5}"
