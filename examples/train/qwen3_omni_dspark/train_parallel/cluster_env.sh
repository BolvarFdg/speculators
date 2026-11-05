# Shared path configuration for the qwen3_omni_dspark pipeline.
#
# This is the ONLY file you need to edit for paths: prepare_data.sh and
# train_cluster.sh both source it. One-off knobs (MAX_SAMPLES, PREP_IN_JOB,
# EPOCHS, ...) stay on the command line and keep their in-script defaults.

# 30B verifier (final gspo checkpoint) on shared NFS
MODEL=/mnt/nfs/models/qwen3_omni_asr_gspo

# All pipeline artifacts on shared NFS: asr_conversations.jsonl,
# asr_onpolicy.jsonl, prepared/, hidden_states/, checkpoints/ all live
# under this tree, visible to every instance.
WORK_DIR=/mnt/nfs/dspark

# Directory covering all audio files
AUDIO_ROOT=/mnt/nfs/audios

# Raw ASR jsonl (messages/audios/ground_truth rows; only needed when
# running with PREP_IN_JOB=1)
RAW_DATA=/mnt/nfs/data/all_asr.jsonl
