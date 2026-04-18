#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collect rollouts from a trained SmolVLA policy and save as a LeRobotDataset
with per-episode advantage_label (1 = A_pos / success, 0 = A_neg / failure).

This script mirrors the structure of lerobot_eval.py but instead of computing
metrics it records every episode frame to a new dataset that can be used for
RECAP fine-tuning (build_recap_dataset.py + lerobot-train).

Usage:
  lerobot-collect-rollouts \\
    --policy.path=./checkpoints/libero_sft/checkpoints/020000/pretrained_model \\
    --env.type=libero \\
    --env.task=libero_spatial \\
    --output_repo_id=./data/libero_spatial_recap_rollouts \\
    --n_episodes=500

The output dataset will have the same feature schema as the source LIBERO
training dataset plus an `advantage_label` int64 feature (shape (1,)) that is
broadcast to every frame of an episode based on that episode's success signal.
"""

import logging
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import einops
import numpy as np
import torch
from tqdm import trange

from lerobot import envs, policies  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs import EnvConfig
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

logger = logging.getLogger(__name__)


@dataclass
class CollectRolloutsConfig:
    """Configuration for the rollout collection pipeline.

    Mirrors EvalPipelineConfig for the policy and environment args so that the
    same checkpoint path / env settings can be copy-pasted from the eval command.
    """

    env: EnvConfig
    policy: PreTrainedConfig | None = None

    # Where to write the output LeRobotDataset.
    output_repo_id: str = "./data/recap_rollouts"
    output_dir: str | None = None

    # Total number of episodes to collect across all tasks in the suite.
    n_episodes: int = 500

    # Number of parallel envs per task. Keep at 1 for simplicity; increase if
    # you have memory headroom and want faster collection.
    batch_size: int = 1

    seed: int = 0
    use_amp: bool = False

    # Optional camera rename map: env feature name → policy feature name.
    # e.g. '{"observation.images.wrist_image": "observation.images.image2"}'
    rename_map: dict | None = None

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides
            )
            self.policy.pretrained_path = Path(policy_path)

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ---------------------------------------------------------------------------
# Dataset feature helpers
# ---------------------------------------------------------------------------

def _obs_to_numpy_frame(obs_preprocessed: dict, action: np.ndarray, task: str) -> dict:
    """Convert a single-env preprocessed observation + action into a frame dict
    suitable for LeRobotDataset.add_frame.

    obs_preprocessed contains tensors of shape (1, ...) from the env_preprocessor
    stage (after LiberoProcessorStep, before policy normalization).

    Images are float32 (1, C, H, W) in [0,1] — convert to uint8 (H, W, C).
    State is float32 (1, state_dim).
    """
    frame = {"task": task}

    for key, val in obs_preprocessed.items():
        if not isinstance(val, torch.Tensor):
            continue
        arr = val.squeeze(0).cpu().numpy()  # remove batch dim

        if key.startswith(OBS_IMAGES):
            # (C, H, W) float32 [0,1] → (H, W, C) uint8
            arr = einops.rearrange(arr, "c h w -> h w c")
            arr = (arr * 255).clip(0, 255).astype(np.uint8)

        frame[key] = arr

    frame[ACTION] = action.astype(np.float32)
    return frame


def _build_features_from_first_frame(frame: dict, fps: int) -> dict:
    """Infer the LeRobotDataset features dict from the first collected frame."""
    features = {}
    for key, val in frame.items():
        if key in ("task",):
            continue
        arr = np.asarray(val)
        if key.startswith(OBS_IMAGES):
            # val is already (H, W, C) uint8 at this point
            features[key] = {
                "dtype": "image",
                "shape": tuple(arr.shape),
                "names": ["height", "width", "channel"],
            }
        else:
            features[key] = {
                "dtype": str(arr.dtype),
                "shape": tuple(arr.shape),
                "names": None,
            }
    features["advantage_label"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    return features


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_episodes(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    dataset: LeRobotDataset,
    n_episodes: int,
    start_seed: int,
    use_amp: bool,
    device: torch.device,
) -> tuple[int, int]:
    """Run rollouts and save all episodes to *dataset* with advantage_label.

    Returns (pos_count, neg_count) — number of successful / failed episodes.
    """
    pos_count = 0
    neg_count = 0
    episode_idx = 0

    # We run one episode at a time (batch_size=1) so that each episode gets its
    # own advantage label before calling save_episode.
    progbar = trange(n_episodes, desc="Collecting rollouts")

    # Determine max_steps from the env's metadata.
    max_steps = env.call("_max_episode_steps")[0]

    for ep_i in progbar:
        policy.reset()
        seed = start_seed + ep_i
        observation, info = env.reset(seed=[seed])

        # Capture env_preprocessed observations for saving (NOT the normalized
        # policy inputs — we want raw enough data to reproduce the training dist).
        episode_frames: list[dict] = []
        task_str: str = ""
        success = False

        done = np.array([False])

        for _step in range(max_steps):
            if np.all(done):
                break

            # 1. Convert to tensors, rename keys.
            obs_tensor = preprocess_observation(observation)

            # 2. Add task string from env.
            obs_with_task = add_envs_task(env, deepcopy(obs_tensor))
            task_str = obs_with_task.get("task", "")

            # 3. Env-specific preprocessing (LiberoProcessorStep: flip imgs, build state vec).
            obs_env = env_preprocessor(obs_with_task)

            # 4. Save this frame (env-preprocessed, not normalized).
            #    advantage_label added below after episode ends.
            episode_frames.append(deepcopy(obs_env))

            # 5. Policy preprocessing (normalization) + inference.
            obs_policy = preprocessor(deepcopy(obs_env))
            with torch.inference_mode(), (
                torch.autocast(device_type=device.type) if use_amp else nullcontext()
            ):
                action_policy = policy.select_action(obs_policy)

            # 6. Postprocess action (unnormalize).
            action_post = postprocessor(action_policy)
            action_env = env_postprocessor({ACTION: action_post})[ACTION]

            action_numpy = action_env.to("cpu").numpy()
            assert action_numpy.ndim == 2, "Expected (batch, action_dim)"

            # 7. Step the env.
            observation, reward, terminated, truncated, info = env.step(action_numpy)

            # Check success via final_info (Gymnasium >= 1.0 vectorized API).
            if "final_info" in info:
                final_info = info["final_info"]
                if isinstance(final_info, dict):
                    ep_success = bool(final_info.get("is_success", False))
                else:
                    ep_success = False
                if ep_success:
                    success = True

            done = np.array(terminated) | np.array(truncated) | done

            # Also capture actions for this step so we can save them.
            episode_frames[-1][ACTION] = action_numpy[0]  # remove batch dim

        label = 1 if success else 0
        if success:
            pos_count += 1
        else:
            neg_count += 1

        # Build dataset schema on the very first episode.
        if episode_idx == 0 and len(episode_frames) > 0:
            first_frame = _obs_to_numpy_frame(episode_frames[0], episode_frames[0][ACTION], task_str)
            if "advantage_label" not in first_frame:
                first_frame["advantage_label"] = np.array([label], dtype=np.int64)
            features = _build_features_from_first_frame(first_frame, fps=dataset.fps)
            # Warn if features differ from dataset schema (they should match if dataset
            # was created from the source dataset's metadata).
            logger.info("Inferred features from first frame: %s", list(features.keys()))

        # Save every frame of this episode to the dataset.
        for obs_env in episode_frames:
            frame = _obs_to_numpy_frame(obs_env, obs_env[ACTION], task_str)
            frame["advantage_label"] = np.array([label], dtype=np.int64)
            dataset.add_frame(frame)

        dataset.save_episode()
        episode_idx += 1

        total = pos_count + neg_count
        pos_rate = pos_count / total if total > 0 else 0.0
        progbar.set_postfix({"success_rate": f"{pos_rate:.2%}", "pos": pos_count, "neg": neg_count})

    return pos_count, neg_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@parser.wrap()
def main(cfg: CollectRolloutsConfig):
    import lerobot.envs  # noqa: F401
    import lerobot.policies  # noqa: F401

    init_logging()
    logging.info("Starting rollout collection")

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path(cfg.output_repo_id)

    # Build envs (single env per task; vectorized wrapper still used for API compat).
    logging.info("Creating environment: %s", cfg.env.type)
    envs = make_env(cfg.env, n_envs=cfg.batch_size)

    # Load policy.
    logging.info("Loading policy from: %s", cfg.policy.pretrained_path)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env, policy_cfg=cfg.policy
    )

    # Determine action dimension from the policy config.
    action_dim = getattr(cfg.policy, "max_action_dim", 7)

    # Create the output dataset.
    # Feature schema: images as (H, W, C) uint8, state as float32, action as float32,
    # advantage_label as int64 (1,). We'll get the exact image shape from the env.
    # For LIBERO default: observation height/width = 256.
    obs_height = getattr(cfg.env, "observation_height", 256)
    obs_width = getattr(cfg.env, "observation_width", 256)

    # Build a minimal feature schema.  Image keys are derived from camera names in
    # the env config; we use the LIBERO defaults here.  If your env has different
    # cameras, update camera_keys accordingly.
    camera_keys_map = {
        "agentview_image": f"{OBS_IMAGES}.image",
        "robot0_eye_in_hand_image": f"{OBS_IMAGES}.image2",
    }
    camera_output_keys = list(camera_keys_map.values())

    # State dim: eef_pos(3) + axis_angle(3) + gripper_qpos(2) = 8 for LIBERO.
    state_dim = getattr(cfg.policy, "state_feature_dim", 8)

    features = {}
    for cam_key in camera_output_keys:
        features[cam_key] = {
            "dtype": "image",
            "shape": (obs_height, obs_width, 3),
            "names": ["height", "width", "channel"],
        }
    features[OBS_STATE] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": None,
    }
    features[ACTION] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": None,
    }
    features["advantage_label"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }

    fps = cfg.env.fps
    dataset = LeRobotDataset.create(
        repo_id=cfg.output_repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        use_videos=False,  # save as PNG frames; easier to inspect
    )

    logging.info("Collecting %d episodes → %s", cfg.n_episodes, output_dir)
    with torch.no_grad():
        pos_count, neg_count = collect_episodes(
            env=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            n_episodes=cfg.n_episodes,
            start_seed=cfg.seed,
            use_amp=cfg.use_amp,
            device=device,
        )

    dataset.finalize()

    total = pos_count + neg_count
    pos_rate = pos_count / total if total > 0 else 0.0
    logging.info(
        "Done. Episodes: %d  Pos (A_pos): %d  Neg (A_neg): %d  Success rate: %.1f%%",
        total, pos_count, neg_count, pos_rate * 100,
    )
    logging.info("Dataset saved to: %s", output_dir)
    envs.close()


if __name__ == "__main__":
    main()
