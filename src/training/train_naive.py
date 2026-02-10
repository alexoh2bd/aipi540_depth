# Generated with assistance from Claude (Anthropic) via Claude Code
# https://github.com/anthropics/claude-code
"""
Naive baselines for depth estimation.

Two baselines:
1. Mean Depth Predictor: predicts the global training-set mean for every pixel.
2. Vertical Gradient Predictor: predicts a linear gradient from 1 (top) to 0
   (bottom), encoding the prior that depth increases toward the horizon.

Usage:
    uv run train --naive [--save_path checkpoints/naive.pt] [--evaluate]
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.dataset import DepthDataset
from src.utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train naive depth baselines")
    parser.add_argument("--save_path", type=str, default="checkpoints/naive.pt",
                        help="Path to save the naive model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Also run evaluation after computing baselines")
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def evaluate_baselines(mean_depth, img_size):
    """Evaluate both naive baselines on the test set."""
    test_ds = DepthDataset(
        split="test",
        global_img_size=img_size,
        multi_view=False,
    )

    # Accumulators for both baselines
    mean_acc = {"abs_rel": 0.0, "sq_diff": 0.0, "n_pixels": 0,
                "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
    grad_acc = {"abs_rel": 0.0, "sq_diff": 0.0, "n_pixels": 0,
                "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}

    for idx in tqdm(range(len(test_ds)), desc="Evaluating"):
        # Load full-resolution depth for proper evaluation
        _, depth_path = test_ds.samples[idx]
        depth_pil = Image.open(depth_path)
        depth_arr = np.array(depth_pil, dtype=np.float32) / test_ds.depth_max
        depth_gt = torch.from_numpy(depth_arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        depth_gt = depth_gt.clamp(0, 1)
        H, W = depth_arr.shape

        # --- Mean Depth Baseline ---
        mean_pred = torch.full_like(depth_gt, mean_depth)
        m = compute_metrics(mean_pred, depth_gt)
        mean_acc["abs_rel"] += m["abs_rel_sum"]
        mean_acc["sq_diff"] += m["sq_diff_sum"]
        mean_acc["delta1"] += m["delta1_sum"]
        mean_acc["delta2"] += m["delta2_sum"]
        mean_acc["delta3"] += m["delta3_sum"]
        mean_acc["n_pixels"] += m["n_pixels"]

        # --- Vertical Gradient Baseline ---
        # Linear gradient: 1 at top (far/horizon) -> 0 at bottom (near/ground)
        gradient = torch.linspace(1, 0, H).view(1, 1, H, 1).expand(1, 1, H, W)
        m = compute_metrics(gradient, depth_gt)
        grad_acc["abs_rel"] += m["abs_rel_sum"]
        grad_acc["sq_diff"] += m["sq_diff_sum"]
        grad_acc["delta1"] += m["delta1_sum"]
        grad_acc["delta2"] += m["delta2_sum"]
        grad_acc["delta3"] += m["delta3_sum"]
        grad_acc["n_pixels"] += m["n_pixels"]

    # Print results
    print("\n" + "=" * 50)
    print("NAIVE BASELINE RESULTS")
    print("=" * 50)

    for name, acc in [("Mean Depth Predictor", mean_acc),
                      ("Vertical Gradient", grad_acc)]:
        n = acc["n_pixels"]
        print(f"\n--- {name} ---")
        print(f"  AbsRel:        {acc['abs_rel'] / n:.4f}")
        print(f"  RMSE:          {(acc['sq_diff'] / n) ** 0.5:.4f}")
        print(f"  Delta < 1.25:  {acc['delta1'] / n:.4f}")
        print(f"  Delta < 1.25²: {acc['delta2'] / n:.4f}")
        print(f"  Delta < 1.25³: {acc['delta3'] / n:.4f}")

    return mean_acc, grad_acc


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # --- Compute mean depth from training set ---
    print("Computing mean depth from training set...")
    train_ds = DepthDataset(
        split="train",
        global_img_size=args.img_size,
        multi_view=False,
    )

    depth_sum = 0.0
    pixel_count = 0

    for idx in tqdm(range(len(train_ds)), desc="Scanning training depths"):
        img_path, depth_path = train_ds.samples[idx]
        depth = Image.open(depth_path)
        depth_arr = np.array(depth, dtype=np.float64)
        depth_arr = depth_arr / train_ds.depth_max
        depth_sum += depth_arr.sum()
        pixel_count += depth_arr.size

    mean_depth = depth_sum / pixel_count
    print(f"Mean depth value: {mean_depth:.6f}")

    # Save both baselines as a minimal "model"
    model_data = {
        "type": "naive",
        "mean_depth": mean_depth,
        "pixel_count": pixel_count,
    }
    torch.save(model_data, args.save_path)
    print(f"Saved naive model to {args.save_path}")

    if args.evaluate:
        evaluate_baselines(mean_depth, args.img_size)


if __name__ == "__main__":
    main()
