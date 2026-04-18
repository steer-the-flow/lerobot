#!/usr/bin/env python
"""Build a mixed RECAP dataset from expert demonstrations + policy rollouts.

Expert demonstrations are relabeled as A_pos (advantage_label=1).
Rollout episodes keep their original advantage_label (1=success, 0=failure).
The two datasets are then merged into a single LeRobotDataset ready for
RECAP fine-tuning via lerobot-train.

Usage:
  python scripts/build_recap_dataset.py \\
    --expert_repo_id=lerobot/libero_spatial_image \\
    --rollouts_repo_id=./data/libero_spatial_recap_rollouts \\
    --output_repo_id=./data/libero_spatial_recap_mixed

The script uses modify_features to add advantage_label=1 to the expert
dataset (which was collected without labels), then merge_datasets to
combine it with the already-labeled rollout dataset.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_tools import merge_datasets, modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert_repo_id",
        default="lerobot/libero_spatial_image",
        help="HF Hub repo ID or local path of the expert BC dataset (no advantage_label yet).",
    )
    parser.add_argument(
        "--expert_root",
        default=None,
        help="Local root directory for the expert dataset (if not on HF Hub).",
    )
    parser.add_argument(
        "--rollouts_repo_id",
        default="./data/libero_spatial_recap_rollouts",
        help="Local path to the rollout dataset produced by lerobot-collect-rollouts.",
    )
    parser.add_argument(
        "--rollouts_root",
        default=None,
        help="Root dir for rollout dataset (defaults to rollouts_repo_id itself).",
    )
    parser.add_argument(
        "--output_repo_id",
        default="./data/libero_spatial_recap_mixed",
        help="Repo ID / path for the merged output dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Local root directory for the output dataset (defaults to output_repo_id).",
    )
    args = parser.parse_args()

    init_logging()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_repo_id)
    expert_root = Path(args.expert_root) if args.expert_root else None
    rollouts_root = Path(args.rollouts_root) if args.rollouts_root else Path(args.rollouts_repo_id)

    # ------------------------------------------------------------------
    # Step 1: Load expert dataset and add advantage_label=1 to all frames.
    # ------------------------------------------------------------------
    logger.info("Loading expert dataset: %s", args.expert_repo_id)
    expert_ds = LeRobotDataset(repo_id=args.expert_repo_id, root=expert_root)
    n_expert = len(expert_ds)
    logger.info("Expert dataset has %d frames.", n_expert)

    # Build advantage_label array: every expert frame gets label 1 (A_pos).
    expert_labels = np.ones(n_expert, dtype=np.int64).reshape(-1, 1)

    labeled_expert_dir = output_dir / "expert_labeled"
    logger.info("Adding advantage_label=1 to expert dataset → %s", labeled_expert_dir)
    labeled_expert_ds = modify_features(
        dataset=expert_ds,
        add_features={
            "advantage_label": (
                expert_labels,
                {"dtype": "int64", "shape": (1,), "names": None},
            ),
        },
        output_dir=labeled_expert_dir,
        repo_id=args.expert_repo_id + "_labeled",
    )
    logger.info("Expert dataset labeled. Frames: %d", len(labeled_expert_ds))

    # ------------------------------------------------------------------
    # Step 2: Load rollout dataset (already has advantage_label).
    # ------------------------------------------------------------------
    logger.info("Loading rollout dataset: %s", args.rollouts_repo_id)
    rollout_ds = LeRobotDataset(repo_id=args.rollouts_repo_id, root=rollouts_root)
    n_rollout = len(rollout_ds)

    # Count pos/neg in rollout dataset.
    adv_labels = rollout_ds.hf_dataset["advantage_label"]
    pos = sum(1 for lbl in adv_labels if (lbl[0] if hasattr(lbl, "__len__") else lbl) == 1)
    neg = n_rollout - pos
    logger.info(
        "Rollout dataset: %d frames (A_pos=%d, A_neg=%d, pos_rate=%.1f%%)",
        n_rollout, pos, neg, 100.0 * pos / n_rollout if n_rollout > 0 else 0.0,
    )

    # ------------------------------------------------------------------
    # Step 3: Merge the two datasets.
    # ------------------------------------------------------------------
    logger.info("Merging datasets → %s", output_dir)
    merged_ds = merge_datasets(
        datasets=[labeled_expert_ds, rollout_ds],
        output_repo_id=args.output_repo_id,
        output_dir=output_dir,
    )

    n_merged = len(merged_ds)
    logger.info(
        "Merged dataset: %d frames total (expert=%d, rollout=%d).",
        n_merged, n_expert, n_rollout,
    )
    logger.info("Done. Output dataset at: %s", output_dir)


if __name__ == "__main__":
    main()
