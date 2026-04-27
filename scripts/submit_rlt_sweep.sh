#!/bin/bash
set -euo pipefail

BETAS=(0.01 0.05 0.1 0.5)

for beta in "${BETAS[@]}"; do
    job_name=$(printf "rlt-sparse-b%s-g5" "$beta" | tr '.' 'p')
    echo "Submitting ${job_name}"
    sbatch \
      --job-name "${job_name}" \
      --export=ALL,\
BETA="${beta}",\
Q_LOSS_WEIGHT_MAX="1.0",\
Q_LOSS_WEIGHT_INCREMENT="0.05",\
Q_CURRICULUM_START_SUCCESS_RATE="0.3",\
REF_ACTION_DROPOUT_PROB="0.5",\
ACTOR_OUTPUT_VARIANCE="0.05",\
GRAD_UPDATES_PER_TRANSITION="5",\
WARMUP_EPISODES="200",\
TOTAL_EPISODES="1000",\
EVAL_FREQ="100",\
EVAL_EPISODES="10",\
EVAL_VIDEOS_TO_SAVE="1",\
BATCH_SIZE="256",\
ACTOR_LR="3e-4",\
CRITIC_LR="3e-4",\
GAMMA="0.99",\
TAU="0.005",\
SPARSE_REWARD="1",\
TIME_PENALTY="0" \
      scripts/train_rlt_ac.sh
done
