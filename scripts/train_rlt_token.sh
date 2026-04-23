#!/bin/bash
#SBATCH --job-name=peg-sft
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260117p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH --output=/ocean/projects/cis260117p/shared/logs/peg-sft_%j.out
#SBATCH --error=/ocean/projects/cis260117p/shared/logs/peg-sft_%j.err

REPO=/ocean/projects/cis260117p/$USER/lerobot
OCEAN=/ocean/projects/cis260117p/shared
JOB=rlt-no-x-attn

export HF_LEROBOT_HOME=$OCEAN/data
export HF_HOME=$OCEAN/hf_cache
export WANDB_DIR=$OCEAN/wandb
export WANDB_CACHE_DIR=$OCEAN/wandb/cache

module load anaconda3
conda activate rlt
cd $REPO

lerobot-train \
  --policy.path="/ocean/projects/cis260117p/shared/checkpoints/peg-sft-c10/checkpoints/last/pretrained_model" \
  --policy.load_vlm_weights=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --policy.training_mode="reconstruction" \
  --policy.train_state_proj=false \
  --policy.use_transformer_rlt=true \
  --dataset.repo_id=lerobot/metaworld_mt50 \
  --dataset.episodes="[1750,1751,1752,1753,1754,1755,1756,1757,1758,1759,1760,1761,1762,1763,1764,1765,1766,1767,1768,1769,1770,1771,1772,1773,1774,1775,1776,1777,1778,1779,1780,1781,1782,1783,1784,1785,1786,1787,1788,1789,1790,1791,1792,1793,1794,1795,1796,1797,1798,1799]" \
  --batch_size=32 \
  --steps=10000 \
  --log_freq=100 \
  --save_freq=5000 \
  --num_workers=4 \
  --output_dir=$OCEAN/checkpoints/$JOB \
  --wandb.enable=true \
  --wandb.entity=idl_34 \
  --wandb.project=rlt-smolvla \
  --job_name=$JOB
