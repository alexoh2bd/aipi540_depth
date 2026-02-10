"""
Flask server for depth estimation inference.

Supports three model types:
- naive: Vertical gradient baseline (no checkpoint needed)
- rf: Random Forest on hand-crafted features (requires classic.joblib)
- deeplearning: ViT + decoder (requires deeplearning.pt)

Run with:
    gunicorn -c server/gunicorn.conf.py server.app:app
"""

import base64
import io
import logging
import os
import sys
import time
import traceback

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Manual CORS (no flask-cors dependency) ───────────────────────────────────

ALLOWED_ORIGINS = {
    "https://aipi540-frontend.vercel.app",
}


@app.before_request
def handle_preflight():
    """Return 200 for all OPTIONS preflight requests."""
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        origin = request.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Max-Age"] = "86400"
        return response


@app.after_request
def add_cors_headers(response):
    """Add CORS headers to every response."""
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ── Global state ─────────────────────────────────────────────────────────────

device = None
amp_dtype = None
models = {}

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MAX_DIM = 1280
COLORMAP = matplotlib.colormaps["inferno"]


# ── Model loading ────────────────────────────────────────────────────────────


def load_models():
    """Load all available model checkpoints at startup."""
    global device, amp_dtype

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    logger.info("Device: %s  dtype: %s", device, amp_dtype)

    # Naive: always available
    models["naive"] = True
    logger.info("Naive model: ready")

    # Random Forest
    rf_path = os.path.join(CHECKPOINT_DIR, "classic.joblib")
    if os.path.isfile(rf_path):
        try:
            import joblib
            models["rf"] = joblib.load(rf_path)
            logger.info("RF model: loaded from %s", rf_path)
        except Exception:
            logger.exception("Failed to load RF model from %s", rf_path)
    else:
        logger.warning("RF checkpoint not found: %s", rf_path)

    # Deep learning (ViT)
    dl_path = os.path.join(CHECKPOINT_DIR, "deeplearning.pt")
    if os.path.isfile(dl_path):
        try:
            from src.models.model import DepthViT

            checkpoint = torch.load(dl_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            # Strip _orig_mod. prefix left by torch.compile
            clean = {}
            for k, v in state_dict.items():
                clean[k.removeprefix("_orig_mod.")] = v

            # Detect decoder depth from checkpoint keys
            decoder_keys = [k for k in clean if k.startswith("decoder.net.")]
            num_upsample = len({k.split(".")[2] for k in decoder_keys}) or 4

            model = DepthViT(
                model_name="vit_small_patch16_224.augreg_in21k",
                img_size=224,
                pretrained=False,
                num_upsample=num_upsample,
            )
            model.load_state_dict(clean, strict=True)
            model.backbone.set_grad_checkpointing(False)
            model.to(device, dtype=amp_dtype)
            model.eval()
            models["deeplearning"] = model
            logger.info("Deep learning model: loaded (%d-layer decoder)", num_upsample)
        except Exception:
            logger.exception("Failed to load deep learning model from %s", dl_path)
    else:
        logger.warning("Deep learning checkpoint not found: %s", dl_path)


# ── Helpers ──────────────────────────────────────────────────────────────────


def colorize_depth(depth_np):
    """Convert a depth map (H, W) to a base64 PNG data URI."""
    vmin, vmax = float(depth_np.min()), float(depth_np.max())
    if vmax - vmin > 1e-6:
        normalized = (depth_np - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(depth_np)

    rgba = COLORMAP(normalized)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def cap_size(pil_image):
    """Resize image if either dimension exceeds MAX_DIM."""
    w, h = pil_image.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        pil_image = pil_image.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )
    return pil_image


# ── Inference functions ──────────────────────────────────────────────────────


def infer_naive(pil_image):
    """Vertical gradient: 1.0 (top/far) to 0.0 (bottom/near)."""
    W, H = pil_image.size
    gradient = np.linspace(1.0, 0.0, H, dtype=np.float32).reshape(H, 1)
    return np.broadcast_to(gradient, (H, W)).copy()


def infer_rf(pil_image, rf_model, patch_size=32):
    """Random Forest inference with hand-crafted features."""
    from src.training.train_classic import extract_features

    img_arr = np.array(pil_image, dtype=np.float32) / 255.0
    H, W = img_arr.shape[:2]
    pred_map = np.zeros((H, W), dtype=np.float32)

    features_list, coords_list = [], []
    for row in range(0, H - patch_size + 1, patch_size):
        for col in range(0, W - patch_size + 1, patch_size):
            patch = img_arr[row:row + patch_size, col:col + patch_size]
            row_frac = (row + patch_size / 2) / H
            col_frac = (col + patch_size / 2) / W
            features_list.append(extract_features(patch, row_frac, col_frac))
            coords_list.append((row, col))

    if features_list:
        preds = rf_model.predict(np.array(features_list))
        for (row, col), val in zip(coords_list, preds):
            pred_map[row:row + patch_size, col:col + patch_size] = val

    # Replicate edge pixels
    last_row = (H // patch_size) * patch_size
    last_col = (W // patch_size) * patch_size
    if last_row < H:
        pred_map[last_row:, :last_col] = pred_map[last_row - 1:last_row, :last_col]
    if last_col < W:
        pred_map[:last_row, last_col:] = pred_map[:last_row, last_col - 1:last_col]
    if last_row < H and last_col < W:
        pred_map[last_row:, last_col:] = pred_map[last_row - 1, last_col - 1]

    return pred_map


def infer_deeplearning(pil_image, model, patch_size=224, overlap=32, batch_size=16):
    """ViT inference with patch chunking and blended stitching."""
    import torch
    from torch.amp import autocast
    from torchvision import transforms
    from src.test.inference import chunk_image, stitch_patches

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    W_orig, H_orig = pil_image.size
    tensor = normalize(pil_image)  # (3, H, W)

    patches, positions, padded_shape = chunk_image(tensor, patch_size, overlap)
    patches_tensor = torch.stack(patches).to(device, dtype=amp_dtype)

    depth_patches = []
    with torch.no_grad():
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i:i + batch_size]
            with autocast(device_type=device.type, dtype=amp_dtype):
                out = model(batch, return_embedding=False)
            depth_patches.extend(list(out))

    depth_map = stitch_patches(depth_patches, positions, padded_shape, patch_size, overlap)
    return depth_map[0, :H_orig, :W_orig].cpu().float().numpy()


# ── Routes ───────────────────────────────────────────────────────────────────


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "service": "depth-estimation"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": list(models.keys()),
        "device": str(device),
    })


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "available": list(models.keys()),
        "default": next(
            (m for m in ("deeplearning", "rf", "naive") if m in models), "naive"
        ),
    })


@app.route("/predict-depth", methods=["POST"])
def predict_depth():
    # ── Validate request ──
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send as multipart form field 'image'."}), 400

    model_type = request.form.get("model")
    if not model_type:
        model_type = next(
            (m for m in ("deeplearning", "rf", "naive") if m in models), "naive"
        )

    if model_type not in models:
        return jsonify({
            "error": f"Model '{model_type}' is not available.",
            "available": list(models.keys()),
        }), 400

    try:
        pil_image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not decode uploaded image."}), 400

    pil_image = cap_size(pil_image)

    # ── Run inference ──
    t0 = time.time()
    try:
        if model_type == "naive":
            depth_np = infer_naive(pil_image)
        elif model_type == "rf":
            depth_np = infer_rf(pil_image, models["rf"])
        else:
            depth_np = infer_deeplearning(pil_image, models["deeplearning"])
    except Exception:
        logger.exception("Inference failed for model=%s", model_type)
        return jsonify({"error": "Inference failed. Check server logs."}), 500

    elapsed = time.time() - t0
    depth_uri = colorize_depth(depth_np)

    return jsonify({
        "depth_map": depth_uri,
        "model": model_type,
        "width": pil_image.size[0],
        "height": pil_image.size[1],
        "inference_time_s": round(elapsed, 3),
    })


# ── Startup ──────────────────────────────────────────────────────────────────

try:
    load_models()
except Exception:
    logger.exception("load_models() crashed — only naive model will be available")
    models.setdefault("naive", True)

logger.info("Server ready. Available models: %s", list(models.keys()))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
