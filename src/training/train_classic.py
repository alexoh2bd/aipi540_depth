"""
Classical (non-deep-learning) baseline for depth estimation.

Uses a Random Forest regressor trained on hand-crafted image features
to predict per-patch mean depth. Features include color statistics,
gradient magnitudes, texture descriptors, and spatial position.

At evaluation time, the model predicts depth for every patch in the test
images and tiles them back into full-resolution depth maps for proper
pixel-level metric computation.

Usage:
    uv run train --classic [--save_path checkpoints/classic_rf.joblib] [--evaluate]
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

    Features (21 total):
        - Color statistics: mean, std, 25th/75th percentile per RGB channel (12)
        - Gradient information: magnitude mean/std/max on grayscale (3)
        - Spatial position: row fraction, column fraction (2)
        - Intensity statistics: grayscale mean, std (2)
        - Texture: Laplacian variance (edge density), local entropy approximation (2)

    Args:
        img_arr: (H, W, 3) numpy array, float [0, 1]
        row_frac: vertical position fraction [0, 1]
        col_frac: horizontal position fraction [0, 1]

    Returns:
        feature vector (1D numpy array)
    """
    features = []

    # Color statistics per channel (12 features)
    for c in range(3):
        channel = img_arr[:, :, c]
        features.extend([
            channel.mean(),
            channel.std(),
            np.percentile(channel, 25),
            np.percentile(channel, 75),
        ])

    # Grayscale gradient magnitudes (3 features)
    gray = img_arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    features.extend([
        grad_mag.mean(),
        grad_mag.std(),
        grad_mag.max(),
    ])

    # Spatial position (2 features) -- strong prior: closer objects at bottom
    features.extend([row_frac, col_frac])

    # Intensity statistics (2 features)
    features.extend([
        gray.mean(),
        gray.std(),
    ])

    # Texture: Laplacian variance (edge density indicator) (1 feature)
    laplacian = (
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1] - gray[2:, 1:-1]
        - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    features.append(laplacian.var() if laplacian.size > 0 else 0.0)

    # Texture: local entropy approximation via histogram uniformity (1 feature)
    hist, _ = np.histogram(gray, bins=16, range=(0, 1))
    hist = hist / hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    features.append(entropy)

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


def evaluate_full_images(rf, patch_size):
    """
    Evaluate the Random Forest on full test images by tiling patch predictions
    back into full-resolution depth maps, then computing pixel-level metrics.
    """
    ds = DepthDataset(split="test", global_img_size=224, multi_view=False)

    metrics_acc = {"abs_rel": 0.0, "sq_diff": 0.0, "n_pixels": 0,
                   "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}

    for idx in tqdm(range(len(ds.samples)), desc="Evaluating (full images)"):
        img_path, depth_path = ds.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        depth_gt = np.array(Image.open(depth_path), dtype=np.float32) / ds.depth_max

        H, W = img.shape[:2]
        pred_map = np.zeros((H, W), dtype=np.float32)

        # Predict per-patch and tile into full image
        for row in range(0, H - patch_size + 1, patch_size):
            for col in range(0, W - patch_size + 1, patch_size):
                img_patch = img[row:row+patch_size, col:col+patch_size]
                row_frac = (row + patch_size / 2) / H
                col_frac = (col + patch_size / 2) / W

                feat = extract_features(img_patch, row_frac, col_frac).reshape(1, -1)
                pred_val = rf.predict(feat)[0]
                pred_map[row:row+patch_size, col:col+patch_size] = pred_val

        # Handle remaining pixels at edges (if image isn't evenly divisible)
        # Fill with nearest patch prediction
        last_row = (H // patch_size) * patch_size
        last_col = (W // patch_size) * patch_size
        if last_row < H:
            pred_map[last_row:, :last_col] = pred_map[last_row-1:last_row, :last_col]
        if last_col < W:
            pred_map[:last_row, last_col:] = pred_map[:last_row, last_col-1:last_col]
        if last_row < H and last_col < W:
            pred_map[last_row:, last_col:] = pred_map[last_row-1, last_col-1]

        # Convert to tensors and compute metrics
        pred_t = torch.from_numpy(pred_map).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        gt_t = torch.from_numpy(depth_gt).unsqueeze(0).unsqueeze(0).clamp(min=1e-6)

        m = compute_metrics(pred_t, gt_t)
        metrics_acc["abs_rel"] += m["abs_rel_sum"]
        metrics_acc["sq_diff"] += m["sq_diff_sum"]
        metrics_acc["delta1"] += m["delta1_sum"]
        metrics_acc["delta2"] += m["delta2_sum"]
        metrics_acc["delta3"] += m["delta3_sum"]
        metrics_acc["n_pixels"] += m["n_pixels"]

    n = metrics_acc["n_pixels"]
    print("\n" + "=" * 50)
    print("CLASSICAL ML BASELINE RESULTS (Full Image)")
    print("=" * 50)
    print(f"  AbsRel:        {metrics_acc['abs_rel'] / n:.4f}")
    print(f"  RMSE:          {(metrics_acc['sq_diff'] / n) ** 0.5:.4f}")
    print(f"  Delta < 1.25:  {metrics_acc['delta1'] / n:.4f}")
    print(f"  Delta < 1.25²: {metrics_acc['delta2'] / n:.4f}")
    print(f"  Delta < 1.25³: {metrics_acc['delta3'] / n:.4f}")

    return metrics_acc


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

    # Feature importances
    feature_names = []
    for c_name in ["R", "G", "B"]:
        feature_names.extend([f"{c_name}_mean", f"{c_name}_std", f"{c_name}_p25", f"{c_name}_p75"])
    feature_names.extend(["grad_mean", "grad_std", "grad_max"])
    feature_names.extend(["row_frac", "col_frac"])
    feature_names.extend(["gray_mean", "gray_std"])
    feature_names.extend(["laplacian_var", "entropy"])

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nFeature Importances (top 10):")
    for rank, i in enumerate(sorted_idx[:10]):
        print(f"  {rank+1}. {feature_names[i]:15s} {importances[i]:.4f}")

    if args.evaluate:
        evaluate_full_images(rf, args.patch_size)


if __name__ == "__main__":
    main()
