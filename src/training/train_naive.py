"""
Naive baseline for depth estimation: mean depth predictor.

Computes the mean depth value across the training set and predicts
that constant value for every pixel. This is the simplest possible
baseline, analogous to a "mean predictor" in classification.

Usage:
    uv run train --naive [--save_path checkpoints/naive.pt]
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.dataset import DepthDataset, collate_depth
from src.utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train naive depth baseline (mean predictor)")
    parser.add_argument("--save_path", type=str, default="checkpoints/naive.pt",
                        help="Path to save the naive model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Also run evaluation after computing mean")
    parser.add_argument("--img_size", type=int, default=224)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

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

    # Save as a minimal "model"
    model_data = {
        "type": "naive_mean",
        "mean_depth": mean_depth,
        "pixel_count": pixel_count,
    }
    torch.save(model_data, args.save_path)
    print(f"Saved naive model to {args.save_path}")

    if args.evaluate:
        print("\nEvaluating on test set...")
        test_ds = DepthDataset(
            split="test",
            global_img_size=args.img_size,
            multi_view=False,
        )

        metrics_acc = {"abs_rel": 0.0, "sq_diff": 0.0, "n_pixels": 0,
                       "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}

        for idx in tqdm(range(len(test_ds)), desc="Evaluating"):
            result = test_ds[idx]
            # Handle both patched (val/test) and single (train) returns
            if len(result) == 3:
                _, depth_patches, _ = result
                depth_gt = depth_patches
            else:
                _, depth_gt = result
                depth_gt = depth_gt.unsqueeze(0)

            pred = torch.full_like(depth_gt, mean_depth)
            m = compute_metrics(pred, depth_gt)
            metrics_acc["abs_rel"] += m["abs_rel_sum"]
            metrics_acc["sq_diff"] += m["sq_diff_sum"]
            metrics_acc["delta1"] += m["delta1_sum"]
            metrics_acc["delta2"] += m["delta2_sum"]
            metrics_acc["delta3"] += m["delta3_sum"]
            metrics_acc["n_pixels"] += m["n_pixels"]

        n = metrics_acc["n_pixels"]
        print("\n=== Naive Baseline Results ===")
        print(f"Abs Rel: {metrics_acc['abs_rel'] / n:.4f}")
        print(f"RMSE:    {(metrics_acc['sq_diff'] / n) ** 0.5:.4f}")
        print(f"Delta1:  {metrics_acc['delta1'] / n:.4f}")
        print(f"Delta2:  {metrics_acc['delta2'] / n:.4f}")
        print(f"Delta3:  {metrics_acc['delta3'] / n:.4f}")


if __name__ == "__main__":
    main()
