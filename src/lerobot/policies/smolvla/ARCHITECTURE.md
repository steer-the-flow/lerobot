# SmolVLA + TransformerRLT Architecture

## Overview

SmolVLA is a Vision-Language-Action (VLA) model that combines a pretrained vision-language model (VLM) with a lightweight action expert to predict robot actions. TransformerRLT is a small encoder-decoder transformer added on top of SmolVLA to learn a compact **RL token** (`z_rl`) — a single embedding that summarizes the full perceptual context and can be used as a state representation for reinforcement learning.

---

## SmolVLA Architecture

SmolVLA is composed of two interleaved transformer stacks that run in parallel over every layer:

```
Images ──► SigLIP Vision Encoder ──► Connector/Resampler ──► image tokens
                                                                    │
Language tokens ──► Embedding layer ──► language tokens             │
                                                                    │
State ──► Linear(state_dim, vlm_hidden) ──► state token             │
                                                                    │
                        ┌───────────────────────────────────────────┘
                        │            PREFIX (VLM stream)
                        ▼
          ┌─────────────────────────┐
          │   VLM Text Model        │  SmolVLM2-500M backbone
          │   (Llama-style)         │  hidden_size H (e.g. 2048)
          │   16 transformer layers │  bfloat16
          │   RoPE + RMSNorm        │
          └────────────┬────────────┘
                       │ KV cache / cross-attention
                       ▼
          ┌─────────────────────────┐        Noisy actions x_t ──► action_in_proj
          │   Action Expert         │  ◄───── Timestep t ──────────► time MLP
          │   (smaller Llama)       │                    SUFFIX (expert stream)
          │   hidden = 0.75 × H     │
          │   same depth as VLM     │
          └────────────┬────────────┘
                       │
                  action_out_proj
                       │
                       ▼
               Predicted velocity v_t
               (batch, chunk_size, action_dim)
```

### Key components

| Component | Details |
|---|---|
| **Backbone** | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| **Vision encoder** | SigLIP, images normalized to `[-1, 1]`, resized to `512×512` |
| **Connector** | Learned resampler projecting patch features to the text hidden dim |
| **VLM text model** | Llama-style transformer, 16 layers, hidden size `H` (≈2048), bfloat16 |
| **Action expert** | Smaller Llama-style model, hidden size `0.75 × H`, same layer count |
| **Attention mode** | Interleaved self-attention (every 2 layers) and cross-attention (expert ← VLM) |
| **State projection** | `Linear(max_state_dim, H)` → 1 token appended to prefix |
| **Action input** | `Linear(max_action_dim, expert_hidden)` fused with sinusoidal timestep via MLP |
| **Action output** | `Linear(expert_hidden, max_action_dim)` |

### VLM + Expert joint forward

The two streams (prefix = VLM, suffix = expert) are processed together through each layer. The attention mask is structured so that:
- Prefix tokens (image, language, state) attend only among themselves (bidirectional within prefix).
- Suffix tokens (noisy actions) attend to the full prefix via cross-attention and to previous suffix positions causally.

This lets the expert condition on the rich VLM representations without re-running the VLM at every denoising step.

### Flow-matching training objective

SmolVLA is trained with **flow matching**: given a clean action `a`, a noise sample `ε`, and a random timestep `t ~ Beta(1.5, 1.0)`:

```
x_t = t·ε + (1−t)·a          # noisy action at time t
u_t = ε − a                   # target velocity
loss = MSE(v_t, u_t)          # predicted vs. target velocity
```

At inference, actions are denoised from pure noise over `num_steps=10` Euler steps.

---

## TransformerRLT Architecture

TransformerRLT is a lightweight encoder-decoder transformer that distills the VLM's `M` final-layer prefix token embeddings `z = {z_1, ..., z_M}` into a single **RL token** `z_rl`. The bottleneck design forces `z_rl` to encode all task-relevant information.

```
VLM prefix outputs z = [z_1, ..., z_M]     shape: (batch, M, H)
         │
         ▼
   input_proj: Linear(H, d_model)           project to RLT internal dim
         │
         ▼
   Positional Encoding (batch-first)
         │
         │   Append learned <rl> token e_rl to the END
         ▼
   [z_1, z_2, ..., z_M, e_rl]              shape: (batch, M+1, d_model)
         │
         ▼
   ┌─────────────────────┐
   │   RLT Encoder       │  3 layers, nhead=8, d_model=512, Pre-LN
   └──────────┬──────────┘
              │
         enc_out[:, -1, :]   ← last sequence position = z_rl
              │
              ▼
           z_rl               shape: (batch, d_model)      → used for RL
              │
              │  (used as cross-attention memory in decoder)
              │
   Teacher-forced decoder input:
   [BOS, z_1, z_2, ..., z_{M-1}]           shape: (batch, M, d_model)
   + causal mask
              │
              ▼
   ┌─────────────────────┐
   │   RLT Decoder       │  3 layers, nhead=8, d_model=512, Pre-LN
   │   cross-attn ← z_rl │  memory = z_rl (the bottleneck)
   └──────────┬──────────┘
              │
         output_proj: Linear(d_model, H)
              │
              ▼
          z_recon             shape: (batch, M, H)
```

### Why this works as a bottleneck

The decoder must reconstruct all `M` embeddings `z_recon ≈ z` using **only** `z_rl` as its cross-attention memory. If `z_rl` loses any information, reconstruction quality degrades. The MSE reconstruction loss therefore pushes `z_rl` to be a maximally informative summary of the full state-language context — a learned compression that is useful as a state representation for a downstream RL value function.

### RLT default hyperparameters

| Parameter | Value |
|---|---|
| `input_dim` | VLM hidden size `H` (set automatically from backbone config) |
| `d_model` | 512 |
| `nhead` | 8 |
| `num_encoder_layers` | 3 |
| `num_decoder_layers` | 3 |
| `dim_ff` | 2048 |
| `dropout` | 0.1 |

---

## Two-Phase Training

The two components are trained in **separate phases**, controlled by `SmolVLAConfig.training_mode`.

### Phase 1 — Action training (`training_mode="action"`)

Standard SmolVLA behavioral cloning via flow matching. TransformerRLT is not used.

```
Batch (images, language, state, actions)
        │
        ▼
embed_prefix()  ──►  embed_suffix(noisy_actions, t)
        │                        │
        └──────────┬─────────────┘
                   ▼
        VLM + Expert joint forward
                   │
             suffix_out[:, -chunk_size:]
                   │
            action_out_proj
                   │
              MSE(v_t, u_t)   ◄── training loss
```

Frozen components (default): VLM backbone (`train_expert_only=True`), vision encoder (`freeze_vision_encoder=True`).
Trained components (default): action expert, action projections, state projection.

### Phase 2 — RLT reconstruction training (`training_mode="reconstruction"`)

TransformerRLT is trained to reconstruct VLM prefix embeddings from `z_rl`. The VLA action head is completely skipped — no noisy actions, no expert stream.

```
Batch (images, language, state)    ← actions not needed
        │
        ▼
embed_prefix()
        │
        ▼
VLM prefix-only forward            ← no suffix/expert stream
        │
  prefix_out (batch, M, H)         ← VLM final-layer embeddings z
        │
  detach() if train_expert_only    ← freeze VLM; or allow gradients to finetune
        │
        ▼
TransformerRLT.forward(z)
        ├──► z_rl    (batch, d_model)
        └──► z_recon (batch, M, H)
        │
  masked MSE loss over valid (non-padding) positions:
  loss = Σ ||z_recon_i − z_i||² / num_valid_tokens
        │
        ▼
  backprop → update TransformerRLT weights
             (+ VLM weights if train_expert_only=False)
```

### Gradient flow summary

| Component | Phase 1 (action) | Phase 2 (reconstruction) |
|---|---|---|
| Vision encoder | frozen (default) | frozen (default) |
| VLM text model | frozen (default) | frozen if `train_expert_only=True`; trainable if `False` |
| Action expert | trained | not used |
| Action projections | trained | not used |
| TransformerRLT | not used | trained |

---

## Using `z_rl` for RL

After Phase 2, `z_rl` can be used as the state representation for a downstream RL value/policy head. At inference, call `TransformerRLT.encode(z)` to get `z_rl` without running the decoder:

```python
# Get VLM prefix embeddings
prefix_out = vlm_prefix_forward(images, language, state)

# Extract RL token (no decoder pass needed at inference)
z_rl = transformer_rlt.encode(prefix_out.float())   # (batch, d_model)

# Feed to downstream value head / RL policy
value = value_head(z_rl)
```

---

---

## RECAP: Advantage-Conditioned Policy Fine-Tuning

RECAP (from π0.6) fine-tunes the action expert by conditioning it on a binary
**advantage token** derived from rollout outcomes.  It is a *parallel* research
track to RLT and targets a different problem: improving policy success rate by
teaching the expert to distinguish successful from failed action distributions,
rather than learning a compact state representation.

> **RECAP and RLT are mutually exclusive.** Setting both
> `use_advantage_conditioning=True` and `use_transformer_rlt=True` raises a
> `ValueError` in `SmolVLAConfig.__post_init__`.

### Core idea

```
advantage_label ∈ {0, 1}
  1 = A_pos  →  this episode was successful
  0 = A_neg  →  this episode failed

Training: pass ground-truth label from rollout outcome.
Inference: always use label=1 (A_pos) to steer toward success.
```

The advantage token is a **prefix-side conditioning token**, not a suffix
token. It is appended to the prefix stream alongside image, language, and
state tokens so the action expert can cross-attend to it through the existing
interleaved attention pattern.  No new attention machinery is required.

### Where the token goes

```
Images ──► SigLIP ──► image tokens
Language ──► Embed ──► language tokens         PREFIX
State ──► state_proj ──► state token           (VLM stream)
advantage_label ──► advantage_embedding ──► advantage token  ← NEW

         ┌──────────────────────────────────────┐
         │   VLM Text Model (frozen)            │
         └─────────────────┬────────────────────┘
                           │ KV cache / cross-attention
                           ▼
         ┌──────────────────────────────────────┐
         │   Action Expert (trained)            │
         │   cross-attends to full prefix incl. │
         │   the advantage token                │
         └─────────────────┬────────────────────┘
                           ▼
                    predicted velocity v_t
```

Concretely, `advantage_embedding = nn.Embedding(2, vlm_hidden_size)` is added
to `VLAFlowMatching.__init__`.  In `embed_prefix`, after the state token is
assembled, the advantage token is appended to the `embs` / `pad_masks` /
`att_masks` lists (att_mask=1, same as state) before `torch.cat`.

### Data labeling

Every frame in an episode gets the **same label** as the episode outcome
(uniform per-episode credit assignment, as in the π0.6 paper):

```
label = 1  if episode.success else 0
```

The `lerobot-collect-rollouts` script handles this: it records episodes and
writes `advantage_label` per frame after the episode terminates.

### Training data

A mixed dataset combining:
- Expert demonstrations → labeled A_pos (label=1)
- Successful rollouts from the BC baseline → labeled A_pos (label=1)
- Failed rollouts from the BC baseline → labeled A_neg (label=0)

Use `scripts/build_recap_dataset.py` to merge the expert dataset with the
rollout dataset into a single LeRobotDataset for fine-tuning.

### RECAP training flow

```
Batch (images, language, state, actions, advantage_label)
        │
        ▼
embed_prefix(advantage_label=label)
        │    ← image tokens + lang tokens + state token + advantage token
        ▼
embed_suffix(noisy_actions, t)
        │
        └──────────── VLM + Expert joint forward ────────────┘
                                │
                         action_out_proj
                                │
                          MSE(v_t, u_t)   ← standard flow-matching loss
```

The only training change is the extra prefix token.  The loss is identical to
the BC baseline; the advantage signal enters purely through conditioning.

### Gradient flow

| Component | RECAP training |
|---|---|
| Vision encoder | frozen |
| VLM text model | frozen (`train_expert_only=True`) |
| Action expert | trained |
| Action projections | trained |
| `advantage_embedding` | trained ← new parameter |
| TransformerRLT | not present (`use_transformer_rlt=False`) |

### Sanity checks (run before full training)

1. `policy.model.advantage_embedding.weight.shape` == `(2, vlm_hidden_size)`.
2. After one training step: `policy.model.advantage_embedding.weight.grad` is non-None.
3. Inference with `recap_inference_advantage=1` vs `=0` produces different action chunks on the same initial state.
4. RECAP loss decreases at a similar rate to the BC baseline.

### Five-phase pipeline

```bash
# Phase 1: BC baseline (existing)
lerobot-train \
  --policy.type=smolvla \
  --policy.use_advantage_conditioning=false \
  --policy.use_transformer_rlt=false \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --batch_size=32 --steps=20000 \
  --output_dir=./checkpoints/libero_sft

# Phase 2: collect rollouts with advantage labels
lerobot-collect-rollouts \
  --policy.path=./checkpoints/libero_sft/checkpoints/020000/pretrained_model \
  --env.type=libero --env.task=libero_spatial \
  --output_repo_id=./data/libero_spatial_recap_rollouts \
  --n_episodes=500

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

# Phase 5: eval (defaults to A_pos at inference)
MUJOCO_GL=osmesa lerobot-eval \
  --policy.path=./checkpoints/libero_recap/checkpoints/010000/pretrained_model \
  --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=50
```

---

## File Map

| File | Role |
|---|---|
| [configuration_smolvla.py](configuration_smolvla.py) | All hyperparameters including RLT config, `training_mode`, and RECAP flags |
| [modeling_smolvla.py](modeling_smolvla.py) | `SmolVLAPolicy` (outer wrapper) and `VLAFlowMatching` (core model); training branching logic; advantage embedding |
| [smolvlm_with_expert.py](smolvlm_with_expert.py) | `SmolVLMWithExpertModel`: interleaved VLM + action expert transformer |
| [transformer_rlt.py](transformer_rlt.py) | `TransformerRLT`: encoder-decoder bottleneck for learning `z_rl` (RLT track only) |
| [processor_smolvla.py](processor_smolvla.py) | Image and text preprocessing |
| [../../scripts/lerobot_collect_rollouts.py](../../scripts/lerobot_collect_rollouts.py) | Rollout collection with advantage labels (RECAP track) |
| [../../../../scripts/build_recap_dataset.py](../../../../scripts/build_recap_dataset.py) | Merge expert demos + rollouts into mixed RECAP dataset |
