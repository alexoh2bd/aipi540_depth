"""
Evaluation script for depth estimation on the DDOS test set.

Processes full-resolution images (resized to be divisible by 16 for ViT)
instead of breaking into small 224x224 patches, leveraging the ViT's
dynamic_img_size=True support for much better spatial coherence.
"""

import argparse
import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.model import DepthViT
from src.utils.metrics import compute_metrics
from src.data.dataset import DepthDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--img_size", type=int, default=1024,
                        help="Max dimension to resize images to (will be rounded to multiple of 16)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="test_results")
    parser.add_argument("--model_name", type=str, default="vit_small_patch16_224.augreg_in21k")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 recommended for full-res images)")
    parser.add_argument("--num_vis", type=int, default=5, help="Number of samples to visualize")
    return parser.parse_args()


class FullResTestDataset(Dataset):
    """
    Test dataset that loads full-resolution images, resized to max_size
    while maintaining aspect ratio and ensuring dimensions are divisible by 16.
    """

    def __init__(self, base_dataset: DepthDataset, max_size=1024):
        """
        Args:
            base_dataset: A DepthDataset instance (used just to get sample paths)
            max_size: Maximum dimension (H or W). Will round to multiple of 16.
        """
        self.samples = base_dataset.samples
        self.max_size = max_size
        self.depth_max = base_dataset.depth_max
        self.PATCH = 16  # ViT patch size

        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        print(f"FullResTestDataset: {len(self.samples)} images, max_size={max_size}")

    def _round16(self, x):
        """Round to nearest multiple of 16."""
        return max(self.PATCH, round(x / self.PATCH) * self.PATCH)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path)  # 16-bit or 32-bit depth

        W_orig, H_orig = img.size  # PIL uses (W, H)

        # Scale to max_size while keeping aspect ratio, round to multiples of 16
        scale = min(self.max_size / W_orig, self.max_size / H_orig)
        new_w = self._round16(int(W_orig * scale))
        new_h = self._round16(int(H_orig * scale))

        img = img.resize((new_w, new_h), Image.BILINEAR)
        depth = depth.resize((new_w, new_h), Image.NEAREST)

        # To tensor + normalize
        img_t = transforms.functional.to_tensor(img)  # (3, H, W)
        img_t = transforms.functional.normalize(img_t, mean=self.mean, std=self.std)

        depth_arr = np.array(depth, dtype=np.float32)
        depth_t = torch.from_numpy(depth_arr).unsqueeze(0)  # (1, H, W)
        depth_t = (depth_t / self.depth_max).clamp(0, 1)

        return img_t, depth_t, (H_orig, W_orig)


def load_model(model_path, model_name, img_size, device):
    """Load the trained ViT depth model."""
    print(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Remove _orig_mod prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v

    # Count decoder layers to determine num_upsample
    decoder_layers = [k for k in new_state_dict.keys() if k.startswith("decoder.net.")]
    num_upsample = len(set(k.split('.')[2] for k in decoder_layers))
    if num_upsample == 0:
        num_upsample = 5  # Default fallback
    print(f"Detected {num_upsample}-layer decoder in checkpoint")

    model = DepthViT(
        model_name=model_name,
        img_size=img_size,
        pretrained=True,
        num_upsample=num_upsample
    ).to(device)

    model.load_state_dict(new_state_dict, strict=True)
    model.to(torch.bfloat16)
    model.eval()

    return model


@torch.no_grad()
def predict_full_res(model, img_t, device):
    """
    Run the model on a full-resolution image and resize the output to match.

    The decoder may output at a different spatial size (since it was trained at
    a fixed img_size). We use F.interpolate to match the input resolution.

    Args:
        model: DepthViT with dynamic_img_size=True
        img_t: (1, 3, H, W) input tensor
        device: torch device

    Returns:
        depth_pred: (1, 1, H, W) predicted depth, same spatial size as input
    """
    _, _, H, W = img_t.shape

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        # The ViT encoder handles arbitrary (H,W) divisible by patch_size=16
        # The decoder will output at some size, we then resize to (H, W)
        patch_features, _ = model.forward_features(img_t)

        # Decoder forward (outputs at whatever size the decoder produces)
        raw_depth = model.decoder.net(patch_features)
        raw_depth = model.decoder.head(raw_depth)

    # Resize to match the input spatial dims
    depth_pred = F.interpolate(raw_depth.float(), size=(H, W), mode='bilinear', align_corners=False)
    return depth_pred


def visualize(img, depth_gt, depth_pred, naive_pred, save_path):
    """Create 4-panel visualization: input, GT, naive, model prediction."""
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img.cpu() * std + mean
    img = img.clamp(0, 1)

    depth_gt = depth_gt.cpu().squeeze(0)
    depth_pred = depth_pred.cpu().squeeze(0)
    naive_pred = naive_pred.cpu().squeeze(0)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(img.permute(1, 2, 0))
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis('off')

    im1 = axes[1].imshow(depth_gt, cmap='inferno')
    axes[1].set_title("Ground Truth Depth", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(naive_pred, cmap='inferno')
    axes[2].set_title("Naive (Gradient)", fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(depth_pred, cmap='inferno')
    axes[3].set_title("Model Prediction", fontsize=14)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {save_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load Model ---
    model = load_model(args.model_path, args.model_name, args.img_size, device)

    # --- Load Dataset ---
    print("Loading test dataset...")
    base_ds = DepthDataset(
        split="test",
        global_img_size=224,  # not used for full-res, just for parent init
        multi_view=False,
    )

    test_ds = FullResTestDataset(base_ds, max_size=args.img_size)

    # Since images may be different sizes, batch_size should be 1
    # for full-res evaluation (or we could pad, but 1 is simplest)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # --- Evaluation ---
    metrics_acc = {"abs_rel": 0.0, "sq_diff": 0.0, "n_pixels": 0,
                   "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
    naive_acc = {"abs_rel": 0.0, "sq_diff": 0.0, "n_pixels": 0,
                 "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}

    visualized_count = 0

    print(f"\nStarting evaluation on {len(test_ds)} images "
          f"(max_size={args.img_size}, batch_size={args.batch_size})...\n")

    with torch.no_grad():
        for batch_idx, (images, depths, orig_shapes) in enumerate(
                tqdm.tqdm(loader, desc="Evaluating")):

            images = images.to(device, dtype=torch.bfloat16)
            depths = depths.to(device, dtype=torch.float32)

            # Full-res prediction
            preds = predict_full_res(model, images, device)  # (B, 1, H, W)

            B = images.shape[0]
            for i in range(B):
                pred_i = preds[i:i+1]       # (1, 1, H, W)
                depth_i = depths[i:i+1]     # (1, 1, H, W)
                img_i = images[i]           # (3, H, W)

                H, W = pred_i.shape[2], pred_i.shape[3]

                # Naive baseline: linear gradient (top=far=1, bottom=near=0)
                naive_pred = torch.linspace(1, 0, H, device=device).view(1, 1, H, 1).expand(1, 1, H, W)

                # Compute metrics
                m = compute_metrics(pred_i, depth_i)
                nm = compute_metrics(naive_pred, depth_i)

                for key in ["abs_rel_sum", "sq_diff_sum", "delta1_sum", "delta2_sum", "delta3_sum"]:
                    short = key.replace("_sum", "")
                    metrics_acc[short] = metrics_acc.get(short, 0.0) + m[key]
                    naive_acc[short] = naive_acc.get(short, 0.0) + nm[key]
                metrics_acc["n_pixels"] += m["n_pixels"]
                naive_acc["n_pixels"] += nm["n_pixels"]

                # Visualize a few samples
                if visualized_count < args.num_vis:
                    visualize(
                        img_i.float(), depth_i.squeeze(0), pred_i.squeeze(0).squeeze(0),
                        naive_pred.squeeze(0).squeeze(0),
                        os.path.join(args.save_dir, f"result_{visualized_count}.png")
                    )
                    visualized_count += 1

    # --- Report ---
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    n = metrics_acc["n_pixels"]
    if n > 0:
        model_abs_rel = metrics_acc["abs_rel"] / n
        model_rmse = (metrics_acc["sq_diff"] / n) ** 0.5
        model_d1 = metrics_acc.get("delta1", 0) / n
        model_d2 = metrics_acc.get("delta2", 0) / n
        model_d3 = metrics_acc.get("delta3", 0) / n
        print(f"\n  MODEL:")
        print(f"    AbsRel:  {model_abs_rel:.4f}")
        print(f"    RMSE:    {model_rmse:.4f}")
        print(f"    δ<1.25:  {model_d1:.4f}")
        print(f"    δ<1.25²: {model_d2:.4f}")
        print(f"    δ<1.25³: {model_d3:.4f}")

    nn = naive_acc["n_pixels"]
    if nn > 0:
        naive_abs_rel = naive_acc["abs_rel"] / nn
        naive_rmse = (naive_acc["sq_diff"] / nn) ** 0.5
        naive_d1 = naive_acc.get("delta1", 0) / nn
        naive_d2 = naive_acc.get("delta2", 0) / nn
        naive_d3 = naive_acc.get("delta3", 0) / nn
        print(f"\n  NAIVE BASELINE (gradient):")
        print(f"    AbsRel:  {naive_abs_rel:.4f}")
        print(f"    RMSE:    {naive_rmse:.4f}")
        print(f"    δ<1.25:  {naive_d1:.4f}")
        print(f"    δ<1.25²: {naive_d2:.4f}")
        print(f"    δ<1.25³: {naive_d3:.4f}")

    print(f"\n  Total pixels evaluated: {n:,}")
    print(f"  Visualizations saved to: {args.save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
