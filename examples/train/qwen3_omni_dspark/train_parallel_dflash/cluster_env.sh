# Shared path configuration for the DFlash training pipeline.
#
# This is the ONLY file you need to edit for paths: the training script
# sources it. One-off knobs (MAX_SAMPLES, EPOCHS, ...) stay on the command
# line and keep their in-script defaults.

# 30B verifier (final gspo checkpoint) on shared NFS
MODEL=/mnt/nfs/models/qwen3_omni_asr_gspo

# All pipeline artifacts on shared NFS: asr_conversations.jsonl,
# asr_onpolicy.jsonl, prepared/, hidden_states/, checkpoints/ all live
# under this tree, visible to every instance.
# NOTE: use a DIFFERENT WORK_DIR from the DSpark pipeline to avoid
# overwriting each other's checkpoints and hidden-state caches.
WORK_DIR=/mnt/nfs/dflash

# Directory covering all audio files
AUDIO_ROOT=/mnt/nfs/audios

# Raw ASR jsonl (messages/audios/ground_truth rows; only needed when
# running with PREP_IN_JOB=1)
RAW_DATA=/mnt/nfs/data/all_asr.jsonl

# Training hyperparameters (DFlash-specific defaults)
DRAFT_VOCAB_SIZE=64000
EPOCHS=3
LR=3e-4
NUM_LAYERS=5
