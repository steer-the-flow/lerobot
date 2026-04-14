# SmolVLA Training — Step by Step

---

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ".[smolvla,libero,metaworld]"

huggingface-cli login
wandb login

export HF_LEROBOT_HOME=<PATH TO DATA FOLDER>
```

---

## LIBERO Spatial SFT baseline

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --batch_size=32 \
  --steps=20000 \
  --log_freq=100 \
  --save_freq=5000 \
  --num_workers=8 \
  --output_dir=./checkpoints/libero_sft \
  --wandb.enable=true \
  --wandb.entity=idl_34 \
  --wandb.project=rlt-smolvla \
  --job_name=libero-sft
```

Eval:
```bash
MUJOCO_GL=osmesa lerobot-eval \
  --policy.path=./checkpoints/libero_sft/checkpoints/020000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=50 \
  --eval.batch_size=2 \
  --policy.device=cuda \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}' \
  --output_dir=./eval_results/libero_sft \
  --wandb.enable=true \
  --wandb.entity=idl_34 \
  --wandb.project=rlt-smolvla \
  --job_name=eval-libero-sft
```

---

## MetaWorld Peg Insertion SFT 

Episodes 1750-1799 are the 50 peg-insert-side-v3 demos in metaworld_mt50.

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/metaworld_mt50 \
  --dataset.episodes="[1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799]" \
  --batch_size=32 \
  --steps=20000 \
  --log_freq=100 \
  --save_freq=5000 \
  --num_workers=8 \
  --output_dir=./checkpoints/peg_sft \
  --wandb.enable=true \
  --wandb.entity=idl_34 \
  --wandb.project=rlt-smolvla \
  --job_name=peg-sft
```

Eval:
```bash
MUJOCO_GL=osmesa lerobot-eval \
  --policy.path=./checkpoints/peg_sft/checkpoints/020000/pretrained_model \
  --env.type=metaworld \
  --env.task_name=peg-insert-side-v3 \
  --eval.n_episodes=50 \
  --eval.batch_size=2 \
  --policy.device=cuda \
  --output_dir=./eval_results/peg_sft \
  --wandb.enable=true \
  --wandb.entity=idl_34 \
  --wandb.project=rlt-smolvla \
  --job_name=eval-peg-sft
```

---
