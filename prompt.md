# RECAP Implementation Prompt

## Context

You are working in a fork of HuggingFace LeRobot (`steer-the-flow/lerobot`) that has been modified to support a SmolVLA-based research project. The goal is to implement **RECAP** (Reward-Conditioned Action Policy) on top of the existing SmolVLA flow-matching action expert.

The project currently has two completed components:
1. A behavior cloning baseline trained on `lerobot/libero_spatial_image` for 20k steps achieving ~62% success on LIBERO-Spatial.
2. A `TransformerRLT` module that compresses VLM prefix embeddings into a bottleneck token `z_rl` via reconstruction. **RLT is unrelated to RECAP and must be left intact.**

You are adding RECAP as a *parallel* training and inference path. RLT must remain functional and selectable via existing config flags.

## What RECAP is

RECAP fine-tunes a flow-matching VLA by conditioning the action expert on a binary advantage token `A ∈ {A_pos, A_neg}` derived from rollout outcomes. The training data mixes:
- Expert demonstrations → labeled `A_pos` (label=1)
- Successful rollouts from the BC baseline → labeled `A_pos` (label=1)
- Failed rollouts from the BC baseline → labeled `A_neg` (label=0)

Training uses the standard flow-matching MSE loss; the only change is that the action expert receives an additional conditioning token derived from the advantage label. At inference, the policy is always prompted with `A_pos`, steering generation toward successful action distributions.

There is no critic, no value function, no reward shaping, no policy gradient. RECAP is supervised fine-tuning with a learned discriminator built into the conditioning context.

## Architectural decision: where does the advantage token go?

Add the advantage token to the **prefix stream** (VLM side), not the suffix (noisy-action) stream.

Reasoning:
- The suffix carries noisy actions that get denoised by the flow-matching objective. Mixing a categorical conditioning token into the denoising target is conceptually wrong and creates a leakage path where the network can memorize action values keyed off advantage.
- The prefix is where conditioning context lives — image tokens, language tokens, state token. The advantage token is conditioning context. It belongs there.
- Architecturally this mirrors how the state token is added: project to VLM hidden dim, append as a single token to the prefix sequence, let the action expert cross-attend to it through the existing interleaved attention pattern. No new attention machinery needed.
- This means the existing prefix attention mask (bidirectional within prefix, expert cross-attends to all of prefix) handles the new token correctly with zero changes.

## Files you will modify

Inspect each before editing. The repo layout:

```
src/lerobot/policies/smolvla/
├── configuration_smolvla.py    # add RECAP config flags
├── modeling_smolvla.py         # add advantage embedding, modify prefix construction, plumb labels
├── smolvlm_with_expert.py      # likely no changes if you handle this in modeling_smolvla
├── transformer_rlt.py          # DO NOT TOUCH
└── ARCHITECTURE.md             # update with RECAP section
```

You will also create:
```
src/lerobot/scripts/lerobot_collect_rollouts.py    # rollout collection
scripts/recap_train.sh                             # end-to-end pipeline script
```

And modify the dataset pipeline to plumb `advantage_label` from dataset → batch → model forward call. Find the relevant transform / collate code by grepping for how `task` (the language instruction) flows through, and follow the same path.

## Implementation steps

### Step 1: Configuration

In `configuration_smolvla.py`, add to `SmolVLAConfig`:

```python
# RECAP (advantage-conditioned policy fine-tuning)
use_advantage_conditioning: bool = False
# Default advantage label at inference when none is provided.
# 1 = A_pos (steer toward success). Always 1 for RECAP inference.
recap_inference_advantage: int = 1
# Embedding dimension matches VLM hidden size since the token is appended to the prefix.
# No separate hyperparameter needed; reuses self.vlm_hidden_size.
```

Add an assertion in `__post_init__`:
```python
if self.use_advantage_conditioning and self.use_transformer_rlt:
    # Both can technically coexist but training_mode logic gets ambiguous.
    # Force the user to pick.
    raise ValueError(
        "use_advantage_conditioning and use_transformer_rlt cannot both be True. "
        "RECAP and RLT are independent research tracks; enable one at a time."
    )
```

### Step 2: Model changes in `modeling_smolvla.py`

In the `VLAFlowMatching` class `__init__` (the inner model class, not the outer policy):

```python
# Advantage conditioning: a learnable embedding for binary outcome labels.
# Embedding output has VLM hidden dim so it can be appended to the prefix
# alongside image, language, and state tokens.
if self.config.use_advantage_conditioning:
    self.advantage_embedding = nn.Embedding(2, self.config.vlm_hidden_size)
    # Initialize small to avoid disrupting the pretrained prefix distribution
    # at the start of fine-tuning.
    nn.init.normal_(self.advantage_embedding.weight, std=0.02)
```

In `embed_prefix` (or whatever method assembles the prefix tokens — find it by grepping for where the state token is appended to the image+language token sequence):

```python
def embed_prefix(self, images, language_tokens, state, advantage_label=None):
    # ... existing image, language, state token assembly ...
    # prefix_tokens: (batch, M, vlm_hidden)
    # prefix_attention_mask: (batch, M)

    if self.config.use_advantage_conditioning:
        if advantage_label is None:
            # Inference path: default to A_pos
            advantage_label = torch.full(
                (prefix_tokens.shape[0],),
                self.config.recap_inference_advantage,
                dtype=torch.long,
                device=prefix_tokens.device,
            )
        # advantage_label: (batch,) of int64 in {0, 1}
        adv_token = self.advantage_embedding(advantage_label)  # (batch, vlm_hidden)
        adv_token = adv_token.unsqueeze(1)                      # (batch, 1, vlm_hidden)
        prefix_tokens = torch.cat([prefix_tokens, adv_token], dim=1)
        # Extend attention mask to mark the advantage token as valid
        adv_mask = torch.ones(
            prefix_tokens.shape[0], 1,
            dtype=prefix_attention_mask.dtype,
            device=prefix_attention_mask.device,
        )
        prefix_attention_mask = torch.cat([prefix_attention_mask, adv_mask], dim=1)

    return prefix_tokens, prefix_attention_mask
```

In the main `forward` method, accept `advantage_label` from the batch and pass it through. Find every call site of `embed_prefix` (likely 2-3 places: the action training path, possibly an inference path, possibly the RLT reconstruction path) and update them. **For the RLT reconstruction path, pass `advantage_label=None` regardless of config — RLT trains on prefix embeddings and shouldn't see the advantage token.** Actually, given the assertion in step 1, RLT and RECAP can't coexist, so this is moot — but be defensive.

In the outer `SmolVLAPolicy.forward`, extract `advantage_label` from the batch dict if present:
```python
advantage_label = batch.get("advantage_label", None)
# Pass through to self.model.forward(...)
```

### Step 3: Dataset plumbing

Find the data transform pipeline. Grep for how `observation.state` or `action` get transformed from the raw LeRobotDataset format into the batch dict the model consumes. There's likely a transform class or a `__getitem__` method that constructs the dict.

Add `advantage_label` to the passed-through fields:
- If the dataset has an `advantage_label` column (RECAP fine-tuning datasets will), pass it through as `torch.long`.
- If the dataset does not have it (vanilla BC datasets), do nothing — the model will default to `A_pos` at inference if `use_advantage_conditioning=True` and the label is missing during training, but this should be a hard error during training. Add an assertion in the model forward:
```python
if self.config.use_advantage_conditioning and self.training and advantage_label is None:
    raise ValueError(
        "RECAP training requires 'advantage_label' in batch but it is missing. "
        "Use a RECAP-formatted dataset built from collect_rollouts.py."
    )
```

The LeRobotDataset format stores per-frame features in Parquet. Adding `advantage_label` requires:
- Constructing the new dataset with `advantage_label` as an integer feature.
- Per-episode label propagation: every frame in an episode gets the same label (the episode's outcome). Handle this in `collect_rollouts.py` (Step 4), not in the transform.

### Step 4: `lerobot_collect_rollouts.py`

New script. Skeleton:

```python
# src/lerobot/scripts/lerobot_collect_rollouts.py
"""
Roll out a trained SmolVLA policy in LIBERO and save the resulting episodes
as a LeRobotDataset with an `advantage_label` field per frame.

Each episode receives a single label propagated to all its frames:
  1 (A_pos) if the LIBERO success signal fires by episode end
  0 (A_neg) otherwise

Usage:
  lerobot-collect-rollouts \
    --policy.path=./checkpoints/libero_sft/checkpoints/020000/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial \
    --output_repo_id=./data/libero_spatial_recap_rollouts \
    --n_episodes_per_task=50 \
    --max_steps=300
"""

import draccus
import torch
from dataclasses import dataclass
from pathlib import Path

from lerobot.policies.factory import make_policy_for_eval
from lerobot.envs.factory import make_env
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# ... other imports as needed; mirror lerobot_eval.py


@dataclass
class CollectRolloutsConfig:
    policy_path: str
    env_type: str = "libero"
    env_task: str = "libero_spatial"
    output_repo_id: str = "./data/recap_rollouts"
    n_episodes_per_task: int = 50
    max_steps: int = 300
    device: str = "cuda"
    seed: int = 0


def main(cfg: CollectRolloutsConfig):
    policy = load_policy(cfg.policy_path, device=cfg.device)
    policy.eval()

    env = make_env(type=cfg.env_type, task=cfg.env_task)

    # Get task list. For LIBERO suites, this is 10 tasks.
    task_ids = env.get_task_ids()  # adapt to actual API

    dataset = create_empty_lerobot_dataset(
        repo_id=cfg.output_repo_id,
        features={
            # Mirror the schema of lerobot/libero_spatial_image plus advantage_label
            "observation.images.image": {...},       # copy from source dataset meta
            "observation.images.wrist_image": {...},
            "observation.state": {"dtype": "float32", "shape": (8,)},
            "action": {"dtype": "float32", "shape": (7,)},
            "advantage_label": {"dtype": "int64", "shape": (1,)},
        },
    )

    pos_count, neg_count = 0, 0

    for task_id in task_ids:
        for episode_idx in range(cfg.n_episodes_per_task):
            obs, info = env.reset(task_id=task_id, seed=cfg.seed + episode_idx)
            episode_buffer = []
            success = False

            for step in range(cfg.max_steps):
                with torch.no_grad():
                    batch = preprocess_obs(obs, device=cfg.device)
                    # During rollout collection, do NOT pass an advantage_label.
                    # The trained baseline has use_advantage_conditioning=False so
                    # the advantage_embedding doesn't exist. If you're collecting
                    # with a RECAP-trained policy, the model defaults to A_pos.
                    action = policy.select_action(batch).cpu().numpy().squeeze()

                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_buffer.append({
                    "observation.images.image": obs["agentview_image"],
                    "observation.images.wrist_image": obs["wrist_image"],
                    "observation.state": obs["state"],
                    "action": action,
                    # advantage_label filled in below after we know the outcome
                })

                obs = next_obs
                if info.get("success", False):
                    success = True
                    break
                if terminated or truncated:
                    break

            label = 1 if success else 0
            if success:
                pos_count += 1
            else:
                neg_count += 1

            for frame in episode_buffer:
                frame["advantage_label"] = label
                dataset.add_frame(frame)
            dataset.save_episode(task=env.get_task_description(task_id))

            print(
                f"task={task_id} ep={episode_idx} "
                f"success={success} pos_total={pos_count} neg_total={neg_count}"
            )

    dataset.consolidate()
    print(f"Done. Pos: {pos_count}, Neg: {neg_count}, Pos rate: {pos_count/(pos_count+neg_count):.3f}")


if __name__ == "__main__":
    cfg = draccus.parse(CollectRolloutsConfig)
    main(cfg)
```

**Critical implementation notes**:
- Do NOT trust the field names above — `observation.images.image`, the obs dict keys from LIBERO, the `select_action` API, `add_frame`/`save_episode` calls — these are pseudocode. Open `lerobot_eval.py` and the LIBERO env wrapper and copy the exact API patterns.
- The LeRobotDataset v3 API has specific requirements for episode boundaries and feature schemas. Read `src/lerobot/datasets/lerobot_dataset.py` for the actual `add_frame`/`save_episode`/`consolidate` flow.
- Aim for roughly balanced pos/neg. If the BC baseline is at 62% and you collect 500 episodes, you get ~310 pos / ~190 neg. That's fine. If your baseline is >85% successful, you'll need to either collect way more episodes to get enough negatives, or subsample the positives when building the mixed dataset.
- Add an `entry_point` in `pyproject.toml` so `lerobot-collect-rollouts` becomes a CLI command.

### Step 5: Mixed dataset construction

Decide whether to do this:
- (a) **In a separate script** that takes the original `lerobot/libero_spatial_image` and the rollout dataset, adds `advantage_label=1` to all expert frames, and produces a merged LeRobotDataset.
- (b) **At training time** via a config flag that loads two datasets and interleaves.

Option (a) is simpler and more reproducible. Write `scripts/build_recap_dataset.py` that merges `lerobot/libero_spatial_image` (all labeled `A_pos`) with the rollout dataset (mixed labels). Use LeRobotDataset's merge utilities — grep for `merge` in the datasets module.

### Step 6: Training pipeline

The existing `lerobot-train` should mostly work once Steps 2 and 3 are done, since the dataset just has an extra column that flows through. Add to `claude.md` and `TODO.md`:

```bash
# Phase 1: BC baseline (already exists, no changes)
lerobot-train \
  --policy.type=smolvla \
  --policy.use_advantage_conditioning=false \
  --policy.use_transformer_rlt=false \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --batch_size=32 --steps=20000 \
  --output_dir=./checkpoints/libero_sft

# Phase 2: collect rollouts
lerobot-collect-rollouts \
  --policy_path=./checkpoints/libero_sft/checkpoints/020000/pretrained_model \
  --env_type=libero --env_task=libero_spatial \
  --output_repo_id=./data/libero_spatial_recap_rollouts \
  --n_episodes_per_task=50

# Phase 3: build mixed dataset
python scripts/build_recap_dataset.py \
  --expert_repo_id=lerobot/libero_spatial_image \
  --rollouts_repo_id=./data/libero_spatial_recap_rollouts \
  --output_repo_id=./data/libero_spatial_recap_mixed

# Phase 4: RECAP fine-tuning
lerobot-train \
  --policy.path=./checkpoints/libero_sft/checkpoints/020000/pretrained_model \
  --policy.use_advantage_conditioning=true \
  --policy.use_transformer_rlt=false \
  --dataset.repo_id=./data/libero_spatial_recap_mixed \
  --batch_size=32 --steps=10000 \
  --output_dir=./checkpoints/libero_recap

# Phase 5: eval (existing eval script works; RECAP defaults to A_pos at inference)
MUJOCO_GL=osmesa lerobot-eval \
  --policy.path=./checkpoints/libero_recap/checkpoints/010000/pretrained_model \
  --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=50 \
  --output_dir=./eval_results/libero_recap
```

### Step 7: Sanity checks before running long jobs

Before kicking off a 10k-step RECAP fine-tuning run, verify:

1. **Embedding actually loads.** Print `policy.model.advantage_embedding.weight.shape` after loading from a RECAP-enabled checkpoint. Should be `(2, vlm_hidden_size)`.

2. **Gradient flow.** Run one training step on the mixed dataset and check `policy.model.advantage_embedding.weight.grad` is non-None and non-zero.

3. **Steering works.** This is the cheap, decisive ablation. Run inference twice on the same initial state — once with `recap_inference_advantage=1`, once with `=0`. Action chunks should differ. If they don't, conditioning isn't reaching the action expert and something in Step 2 is wrong.

4. **Loss decreases.** RECAP fine-tuning loss should decrease similarly to BC. If it spikes or plateaus high, check that the advantage label is actually varying across the batch (not always the same value due to a dataset bug).

### Step 8: ARCHITECTURE.md

Add a "RECAP" section parallel to the RLT section. Same diagram style. Make explicit that RECAP and RLT are mutually exclusive and target different problems (policy conditioning vs. state representation).

## What NOT to do

- Do not modify `transformer_rlt.py`. It's a parallel research track.
- Do not touch the VLM backbone. Train_expert_only=True throughout, same as the BC baseline. The advantage_embedding is a new trainable module but the VLM stays frozen.
- Do not implement PPO, GRPO, value functions, or any critic. RECAP is supervised fine-tuning with conditioning, not online RL.
- Do not add the advantage token to the suffix stream (see architectural decision above).
- Do not propagate the binary reward through bellman backups or temporal smoothing — uniform per-episode propagation, exactly as the paper describes. The credit assignment risk is acknowledged in the paper's Section 6.
- Do not pretend RECAP works on small data scales without verification — the data efficiency ablation (50%, 25%) is part of the paper, but only run it after the full-data RECAP shows a real success-rate lift over the BC baseline.

## Acceptance criteria

You're done with the implementation when:
1. `use_advantage_conditioning=False` produces bitwise-identical training behavior to the current code (RECAP is a clean opt-in).
2. `use_advantage_conditioning=True` with a RECAP dataset trains without errors and the advantage embedding receives gradients.
3. Inference with `recap_inference_advantage=1` vs `=0` produces measurably different action chunks on identical initial states.
4. Full pipeline (collect → mix → train → eval) runs end-to-end on at least one LIBERO-Spatial task without manual intervention.

You're done with the *experiment* when you have eval success rates for: BC baseline (already have, 62%), RECAP fine-tuned (target: >62%), A_neg ablation (should be substantially worse than A_pos), and 50%/25% data ablation pairs.

---

That's the prompt. Hand it to a coding agent (Claude Code, Cursor, etc.) with the repo open, or give it to a teammate. A few notes on using it:

- The pseudocode for `collect_rollouts.py` deliberately doesn't try to be runnable — the LIBERO env wrapper API and the LeRobotDataset v3 add_frame API are easy to get wrong from memory, and the right move is to crib from `lerobot_eval.py` and the existing dataset code rather than guess.
- The "where does the advantage token go" decision is the one design call that will haunt you if you get it wrong, which is why I argued it explicitly in the prompt rather than leaving it open.
- The mutual exclusion with RLT (Step 1 assertion) is a deliberate choice to avoid having two RL-flavored conditioning paths fighting for the same `training_mode` slot. If your team disagrees, drop the assertion and let them coexist — but then someone has to write the four-way combinatorial test matrix.

If you want, after you've read this and decided whether the prompt matches what you actually want, I can also generate the actual diffs for Steps 1, 2, and the assertion in Step 3 — those are the ones I'd be confident writing without inventing API surface, since I have the real `configuration_smolvla.py` and `modeling_smolvla.py` extracted in `/home/claude/lerobot`. The collect_rollouts script genuinely benefits from a human (or agent) cross-referencing `lerobot_eval.py` line by line, so I'd leave that one as a prompt rather than try to write it cold.