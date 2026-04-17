#!/usr/bin/env bash
# recap_train.sh — full RECAP pipeline from SFT checkpoint to eval
#
# Usage (Git Bash / WSL / Linux):
#   bash scripts/recap_train.sh
#
# Override any variable at the top before running.
# All phases are gated: each phase only runs if its output doesn't already
# exist, so you can re-run safely after a failure.

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
SFT_CKPT="./checkpoints/libero_sft/checkpoints/020000/pretrained_model"
EXPERT_DATASET="lerobot/libero_spatial_image"
ROLLOUTS_DIR="./data/libero_spatial_recap_rollouts"
MIXED_DIR="./data/libero_spatial_recap_mixed"
RECAP_OUTPUT="./checkpoints/libero_recap"
RECAP_CKPT="${RECAP_OUTPUT}/checkpoints/010000/pretrained_model"

N_ROLLOUT_EPISODES=500
RECAP_STEPS=10000
BATCH_SIZE=32
SEED=0
DEVICE="cuda"

# ---------------------------------------------------------------------------
# PHASE 2 — collect rollouts with advantage labels
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 2: collect rollouts (${N_ROLLOUT_EPISODES} episodes) ==="
if [ -d "${ROLLOUTS_DIR}" ]; then
    echo "Rollout dataset already exists at ${ROLLOUTS_DIR} — skipping."
else
    python -m lerobot.scripts.lerobot_collect_rollouts \
        --policy.path="${SFT_CKPT}" \
        --env.type=libero \
        --env.task=libero_spatial \
        --output_repo_id="${ROLLOUTS_DIR}" \
        --n_episodes="${N_ROLLOUT_EPISODES}" \
        --seed="${SEED}"
fi

# ---------------------------------------------------------------------------
# PHASE 3 — build mixed dataset (expert A_pos + rollouts mixed)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 3: build mixed RECAP dataset ==="
if [ -d "${MIXED_DIR}" ]; then
    echo "Mixed dataset already exists at ${MIXED_DIR} — skipping."
else
    python scripts/build_recap_dataset.py \
        --expert_repo_id="${EXPERT_DATASET}" \
        --rollouts_repo_id="${ROLLOUTS_DIR}" \
        --output_repo_id="${MIXED_DIR}"
fi

# ---------------------------------------------------------------------------
# PHASE 4 — RECAP fine-tuning
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 4: RECAP fine-tuning (${RECAP_STEPS} steps) ==="
if [ -d "${RECAP_CKPT}" ]; then
    echo "RECAP checkpoint already exists at ${RECAP_CKPT} — skipping."
else
    python -m lerobot.scripts.lerobot_train \
        --policy.path="${SFT_CKPT}" \
        --policy.use_advantage_conditioning=true \
        --policy.use_transformer_rlt=false \
        --dataset.repo_id="${MIXED_DIR}" \
        --batch_size="${BATCH_SIZE}" \
        --steps="${RECAP_STEPS}" \
        --output_dir="${RECAP_OUTPUT}"
fi

# ---------------------------------------------------------------------------
# PHASE 5 — eval RECAP (A_pos at inference, the default)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 5: eval RECAP checkpoint ==="
# On Linux/WSL add: MUJOCO_GL=osmesa before python if running headless
python -m lerobot.scripts.lerobot_eval \
    --policy.path="${RECAP_CKPT}" \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.n_episodes=50 \
    --output_dir="./eval_results/libero_recap"

echo ""
echo "=== Done. Results at ./eval_results/libero_recap/eval_info.json ==="
