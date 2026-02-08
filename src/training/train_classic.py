"""
Classical (non-deep-learning) baseline for depth estimation.

Uses a Random Forest regressor trained on hand-crafted image features
to predict per-patch mean depth. Features include color statistics,
gradient magnitudes, and spatial position.

This satisfies the rubric requirement for a classical ML model.

Usage:
    uv run train --classic [--save_path checkpoints/classic_rf.pt]
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import joblib

from src.data.dataset import DepthDataset
from src.utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train classical ML depth baseline")
    parser.add_argument("--save_path", type=str, default="checkpoints/classic_rf.joblib")
    parser.add_argument("--patch_size", type=int, default=32,
                        help="Patch size for feature extraction")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max training patches to use (for speed)")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in random forest")
    parser.add_argument("--evaluate", action="store_true",
                        help="Also run evaluation after training")
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def extract_features(img_arr, row_frac, col_frac):
    """
    Extract hand-crafted features from an RGB image patch.

    Args:
        img_arr: (H, W, 3) numpy array, float [0, 1]
        row_frac: vertical position fraction [0, 1]
        col_frac: horizontal position fraction [0, 1]

    Returns:
        feature vector (1D numpy array)
    """
    features = []

    # Color statistics per channel
    for c in range(3):
        channel = img_arr[:, :, c]
        features.extend([
            channel.mean(),
            channel.std(),
            np.percentile(channel, 25),
            np.percentile(channel, 75),
        ])

    # Grayscale gradient magnitudes
    gray = img_arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    features.extend([
        grad_mag.mean(),
        grad_mag.std(),
        grad_mag.max(),
    ])

    # Spatial position (strong prior: closer objects at bottom)
    features.extend([row_frac, col_frac])

    # Intensity statistics
    features.extend([
        gray.mean(),
        gray.std(),
    ])

    return np.array(features, dtype=np.float32)


def build_dataset(split, patch_size, max_samples):
    """Extract features and targets from depth dataset."""
    ds = DepthDataset(split=split, global_img_size=224, multi_view=False)

    X_list = []
    y_list = []
    count = 0

    for idx in tqdm(range(len(ds.samples)), desc=f"Extracting features ({split})"):
        if count >= max_samples:
            break

        img_path, depth_path = ds.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path), dtype=np.float32) / ds.depth_max

        H, W = img.shape[:2]
        for row in range(0, H - patch_size + 1, patch_size):
            for col in range(0, W - patch_size + 1, patch_size):
                if count >= max_samples:
                    break

                img_patch = img[row:row+patch_size, col:col+patch_size]
                depth_patch = depth[row:row+patch_size, col:col+patch_size]

                row_frac = (row + patch_size / 2) / H
                col_frac = (col + patch_size / 2) / W

                feat = extract_features(img_patch, row_frac, col_frac)
                target = depth_patch.mean()

                X_list.append(feat)
                y_list.append(target)
                count += 1

    return np.array(X_list), np.array(y_list)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)

    print("Building training features...")
    X_train, y_train = build_dataset("train", args.patch_size, args.max_samples)
    print(f"Training set: {X_train.shape[0]} patches, {X_train.shape[1]} features")

    print(f"\nTraining Random Forest (n_estimators={args.n_estimators})...")
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    rf.fit(X_train, y_train)

    joblib.dump(rf, args.save_path)
    print(f"Saved model to {args.save_path}")

    # Training score
    train_pred = rf.predict(X_train)
    train_mse = np.mean((train_pred - y_train) ** 2)
    print(f"Training MSE: {train_mse:.6f}")

    if args.evaluate:
        print("\nBuilding test features...")
        X_test, y_test = build_dataset("test", args.patch_size, args.max_samples)
        print(f"Test set: {X_test.shape[0]} patches")

        test_pred = rf.predict(X_test)

        # Convert to tensors for metric computation
        pred_t = torch.tensor(test_pred).float().view(-1, 1, 1, 1)
        gt_t = torch.tensor(y_test).float().view(-1, 1, 1, 1)
        m = compute_metrics(pred_t, gt_t)

        n = m["n_pixels"]
        print("\n=== Classical ML Baseline Results ===")
        print(f"Abs Rel: {m['abs_rel_sum'] / n:.4f}")
        print(f"RMSE:    {(m['sq_diff_sum'] / n) ** 0.5:.4f}")
        print(f"Delta1:  {m['delta1_sum'] / n:.4f}")
        print(f"Delta2:  {m['delta2_sum'] / n:.4f}")
        print(f"Delta3:  {m['delta3_sum'] / n:.4f}")


if __name__ == "__main__":
    main()
