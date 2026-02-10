"""
Inference script for depth estimation on arbitrary-sized images.

Supports all model types:
- ViT (deep learning): Chunks images into patches, runs inference, stitches back
- Naive baseline: Predicts constant mean depth or vertical gradient
- Random Forest (classic): Extracts hand-crafted features per patch, predicts, tiles back
"""

import argparse
import os
import sys
sys.path.append(os.getcwd())

# MPS fallback: timm uses bicubic+antialias interpolation for position embedding
# resampling, whose backward pass isn't implemented on MPS yet.
# Must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from src.models.model import DepthViT


def parse_args():
    parser = argparse.ArgumentParser(description="Run depth estimation on arbitrary-sized images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch size for chunking (model input size)")
    parser.add_argument("--overlap", type=int, default=32, help="Overlap between patches for smoother blending")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (auto-detects cuda > mps if omitted)")
    parser.add_argument("--model_name", type=str, default="vit_small_patch16_224.augreg_in21k")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing patches")
    parser.add_argument("--colormap", type=str, default="inferno", help="Colormap for depth visualization")
    parser.add_argument("--save_raw", action="store_true", help="Save raw depth map as .npy")
    parser.add_argument("--naive_mode", type=str, default="gradient", choices=["mean", "gradient"],
                        help="Naive prediction mode: 'mean' (constant) or 'gradient' (vertical)")
    parser.add_argument("--rf_patch_size", type=int, default=32,
                        help="Patch size for Random Forest feature extraction")
    return parser.parse_args()


def detect_model_type(model_path):
    """Detect model type from the checkpoint file.

    Returns one of: 'classic', 'naive', 'vit'
    """
    if model_path.endswith(".joblib"):
        return "classic"

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and checkpoint.get("type") == "naive":
        return "naive"

    return "vit"


def load_vit_model(model_path, model_name, device, amp_dtype):
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
        num_upsample = 4  # Default fallback
    print(f"Detected {num_upsample}-layer decoder in checkpoint")

    model = DepthViT(
        model_name=model_name,
        img_size=224,  # Standard ViT patch size
        pretrained=True,
        num_upsample=num_upsample
    ).to(device)

    model.load_state_dict(new_state_dict, strict=True)
    model.to(amp_dtype)
    model.eval()

    return model


def chunk_image(image_tensor, patch_size, overlap):
    """
    Chunk an image tensor into overlapping patches.

    Args:
        image_tensor: (C, H, W) tensor
        patch_size: Size of each patch
        overlap: Overlap between adjacent patches

    Returns:
        patches: List of (C, patch_size, patch_size) tensors
        positions: List of (y, x) positions for each patch
        padded_shape: (H_padded, W_padded) shape after padding
    """
    C, H, W = image_tensor.shape
    stride = patch_size - overlap

    # Calculate padding needed
    pad_h = (patch_size - (H - patch_size) % stride) % stride if H > patch_size else patch_size - H
    pad_w = (patch_size - (W - patch_size) % stride) % stride if W > patch_size else patch_size - W

    # For small images, ensure at least one patch
    if H <= patch_size:
        pad_h = patch_size - H
    if W <= patch_size:
        pad_w = patch_size - W

    # Pad image (right and bottom)
    padded = F.pad(image_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
    _, H_pad, W_pad = padded.shape

    patches = []
    positions = []

    y = 0
    while y + patch_size <= H_pad:
        x = 0
        while x + patch_size <= W_pad:
            patch = padded[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))

            # Move to next position
            if x + patch_size >= W_pad:
                break
            x += stride
            # Adjust last column to align with edge
            if x + patch_size > W_pad:
                x = W_pad - patch_size

        if y + patch_size >= H_pad:
            break
        y += stride
        # Adjust last row to align with edge
        if y + patch_size > H_pad:
            y = H_pad - patch_size

    return patches, positions, (H_pad, W_pad)


def stitch_patches(patches, positions, output_shape, patch_size, overlap):
    """
    Stitch depth patches back together with weighted blending.

    Args:
        patches: List of (1, patch_size, patch_size) depth tensors
        positions: List of (y, x) positions
        output_shape: (H, W) target output shape
        patch_size: Size of each patch
        overlap: Overlap between patches

    Returns:
        depth_map: (1, H, W) stitched depth map
    """
    H, W = output_shape
    device = patches[0].device
    dtype = patches[0].dtype

    # Create output tensor and weight accumulator
    depth_sum = torch.zeros(1, H, W, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(1, H, W, device=device, dtype=torch.float32)

    # Create blending weights (linear ramp for overlap regions)
    weights = create_blend_weights(patch_size, overlap, device)

    for patch, (y, x) in zip(patches, positions):
        patch_float = patch.float()

        # Handle edge cases where patch extends beyond output
        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        patch_h = y_end - y
        patch_w = x_end - x

        depth_sum[:, y:y_end, x:x_end] += patch_float[:, :patch_h, :patch_w] * weights[:, :patch_h, :patch_w]
        weight_sum[:, y:y_end, x:x_end] += weights[:, :patch_h, :patch_w]

    # Avoid division by zero
    weight_sum = torch.clamp(weight_sum, min=1e-6)
    depth_map = depth_sum / weight_sum

    return depth_map


def create_blend_weights(patch_size, overlap, device):
    """
    Create blending weights for smooth patch stitching.
    Uses linear ramps in overlap regions.
    """
    if overlap == 0:
        return torch.ones(1, patch_size, patch_size, device=device)

    # Create 1D ramp
    ramp = torch.ones(patch_size, device=device)
    ramp[:overlap] = torch.linspace(0, 1, overlap, device=device)
    ramp[-overlap:] = torch.linspace(1, 0, overlap, device=device)

    # Create 2D weights
    weights = ramp.unsqueeze(0) * ramp.unsqueeze(1)
    return weights.unsqueeze(0)


def preprocess_image(image_path):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(image)
    return tensor, original_size


def postprocess_depth(depth_tensor, original_size):
    """Crop and resize depth map back to original image size."""
    # depth_tensor: (1, H_padded, W_padded)
    # original_size: (W, H)
    W_orig, H_orig = original_size

    # Crop to remove padding
    depth_cropped = depth_tensor[:, :H_orig, :W_orig]

    return depth_cropped


def visualize_depth(image_path, depth_map, save_path, colormap='inferno'):
    """Create visualization comparing input image and predicted depth."""
    # Load original image
    image = Image.open(image_path).convert("RGB")

    depth_np = depth_map.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    im = axes[1].imshow(depth_np, cmap=colormap)
    axes[1].set_title("Predicted Depth")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def run_inference(model, image_tensor, args, device, amp_dtype):
    """
    Run inference on an arbitrary-sized image.

    Args:
        model: Trained ViT depth model
        image_tensor: (C, H, W) preprocessed image tensor
        args: Command line arguments
        device: Torch device
        amp_dtype: dtype for mixed precision

    Returns:
        depth_map: (1, H, W) predicted depth map
    """
    C, H, W = image_tensor.shape

    # Chunk the image into patches
    patches, positions, padded_shape = chunk_image(
        image_tensor, args.patch_size, args.overlap
    )

    print(f"Image size: {H}x{W}, Padded: {padded_shape[0]}x{padded_shape[1]}")
    print(f"Created {len(patches)} patches with {args.overlap}px overlap")

    # Stack patches for batched inference
    patches_tensor = torch.stack(patches).to(device, dtype=amp_dtype)

    # Run inference in batches
    depth_patches = []
    with torch.no_grad():
        for i in range(0, len(patches_tensor), args.batch_size):
            batch = patches_tensor[i:i + args.batch_size]
            with autocast(device_type=device.type, dtype=amp_dtype):
                batch_depth = model(batch, return_embedding=False)
            depth_patches.extend([d for d in batch_depth])

    # Stitch patches back together
    depth_map = stitch_patches(
        depth_patches, positions, padded_shape, args.patch_size, args.overlap
    )

    return depth_map


def predict_naive(image_path, naive_mode, mean_depth):
    """Generate a naive depth prediction for an image.

    Args:
        image_path: Path to the input image
        naive_mode: 'mean' for constant prediction, 'gradient' for vertical gradient
        mean_depth: Mean depth value from the naive checkpoint

    Returns:
        depth_map: (1, H, W) numpy-backed tensor
        original_size: (W, H) tuple
    """
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    if naive_mode == "mean":
        depth_map = np.full((1, H, W), mean_depth, dtype=np.float32)
    else:  # gradient
        gradient = np.linspace(1.0, 0.0, H, dtype=np.float32).reshape(1, H, 1)
        depth_map = np.broadcast_to(gradient, (1, H, W)).copy()

    return torch.from_numpy(depth_map), (W, H)


def predict_rf(image_path, rf_model, rf_patch_size):
    """Generate a Random Forest depth prediction for an image.

    Args:
        image_path: Path to the input image
        rf_model: Loaded sklearn RandomForestRegressor
        rf_patch_size: Patch size used for feature extraction

    Returns:
        depth_map: (1, H, W) tensor
        original_size: (W, H) tuple
    """
    from src.training.train_classic import extract_features

    image = Image.open(image_path).convert("RGB")
    img_arr = np.array(image, dtype=np.float32) / 255.0
    H, W = img_arr.shape[:2]

    pred_map = np.zeros((H, W), dtype=np.float32)

    features_list = []
    coords_list = []
    for row in range(0, H - rf_patch_size + 1, rf_patch_size):
        for col in range(0, W - rf_patch_size + 1, rf_patch_size):
            img_patch = img_arr[row:row + rf_patch_size, col:col + rf_patch_size]
            row_frac = (row + rf_patch_size / 2) / H
            col_frac = (col + rf_patch_size / 2) / W
            features_list.append(extract_features(img_patch, row_frac, col_frac))
            coords_list.append((row, col))

    if features_list:
        X_img = np.array(features_list)
        preds = rf_model.predict(X_img)
        for (row, col), pred_val in zip(coords_list, preds):
            pred_map[row:row + rf_patch_size, col:col + rf_patch_size] = pred_val

    # Handle remaining pixels at edges
    last_row = (H // rf_patch_size) * rf_patch_size
    last_col = (W // rf_patch_size) * rf_patch_size
    if last_row < H:
        pred_map[last_row:, :last_col] = pred_map[last_row - 1:last_row, :last_col]
    if last_col < W:
        pred_map[:last_row, last_col:] = pred_map[:last_row, last_col - 1:last_col]
    if last_row < H and last_col < W:
        pred_map[last_row:, last_col:] = pred_map[last_row - 1, last_col - 1]

    depth_map = torch.from_numpy(pred_map).unsqueeze(0)  # (1, H, W)
    return depth_map, (W, H)


def process_single_image(model, image_path, args, device, amp_dtype):
    """Process a single image and save results."""
    print(f"\nProcessing: {image_path}")

    # Preprocess
    image_tensor, original_size = preprocess_image(image_path)

    # Run inference
    depth_map = run_inference(model, image_tensor, args, device, amp_dtype)

    # Postprocess (crop to original size)
    depth_map = postprocess_depth(depth_map, original_size)

    # Create output filename
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Save visualization
    vis_path = os.path.join(args.output_dir, f"{basename}_depth.png")
    visualize_depth(image_path, depth_map, vis_path, args.colormap)

    # Optionally save raw depth
    if args.save_raw:
        raw_path = os.path.join(args.output_dir, f"{basename}_depth.npy")
        np.save(raw_path, depth_map.squeeze().cpu().numpy())
        print(f"Saved raw depth to {raw_path}")

    return depth_map


def process_single_image_naive(image_path, mean_depth, args):
    """Process a single image with the naive baseline."""
    print(f"\nProcessing (naive/{args.naive_mode}): {image_path}")

    depth_map, original_size = predict_naive(image_path, args.naive_mode, mean_depth)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(args.output_dir, f"{basename}_depth.png")
    visualize_depth(image_path, depth_map, vis_path, args.colormap)

    if args.save_raw:
        raw_path = os.path.join(args.output_dir, f"{basename}_depth.npy")
        np.save(raw_path, depth_map.squeeze().cpu().numpy())
        print(f"Saved raw depth to {raw_path}")

    return depth_map


def process_single_image_rf(image_path, rf_model, args):
    """Process a single image with the Random Forest model."""
    print(f"\nProcessing (RF, patch={args.rf_patch_size}): {image_path}")

    depth_map, original_size = predict_rf(image_path, rf_model, args.rf_patch_size)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(args.output_dir, f"{basename}_depth.png")
    visualize_depth(image_path, depth_map, vis_path, args.colormap)

    if args.save_raw:
        raw_path = os.path.join(args.output_dir, f"{basename}_depth.npy")
        np.save(raw_path, depth_map.squeeze().cpu().numpy())
        print(f"Saved raw depth to {raw_path}")

    return depth_map


def collect_image_paths(image_path):
    """Return a list of image file paths from a file or directory."""
    if os.path.isfile(image_path):
        return [image_path]
    elif os.path.isdir(image_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        files = [
            os.path.join(image_path, f)
            for f in sorted(os.listdir(image_path))
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        print(f"Found {len(files)} images in {image_path}")
        return files
    else:
        print(f"Error: {image_path} is not a valid file or directory")
        sys.exit(1)


def main():
    args = parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32
    print(f"Using device: {device} (dtype: {amp_dtype})")

    os.makedirs(args.output_dir, exist_ok=True)

    model_type = detect_model_type(args.model_path)
    print(f"Detected model type: {model_type}")

    image_paths = collect_image_paths(args.image_path)

    if model_type == "naive":
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        mean_depth = checkpoint["mean_depth"]
        print(f"Loaded naive model (mean_depth={mean_depth:.4f})")
        for path in image_paths:
            try:
                process_single_image_naive(path, mean_depth, args)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    elif model_type == "classic":
        import joblib
        rf_model = joblib.load(args.model_path)
        print(f"Loaded Random Forest from {args.model_path}")
        for path in image_paths:
            try:
                process_single_image_rf(path, rf_model, args)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    else:  # vit
        model = load_vit_model(args.model_path, args.model_name, device, amp_dtype)
        for path in image_paths:
            try:
                process_single_image(model, path, args, device, amp_dtype)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
