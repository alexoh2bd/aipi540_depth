"""
Download the DDOS (Depth from Driving Open Scenes) dataset from HuggingFace.

Downloads to data/DDOS/ by default. The dataset contains paired RGB images
and 16-bit depth maps of residential neighborhoods.

Usage:
    uv run setup
"""

import os
import sys


DATASET_REPO = "benediktkol/DDOS"
DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data", "DDOS")


def get_data_root(data_dir=None):
    """
    Return the resolved data root path for the DDOS dataset.

    Priority:
        1. Explicit data_dir argument
        2. LEDEEP_DATA_DIR environment variable
        3. Default: data/DDOS/ in the project root
    """
    if data_dir:
        return data_dir
    return os.environ.get("LEDEEP_DATA_DIR", DEFAULT_DATA_DIR)


def is_dataset_ready(data_dir):
    """Check if the dataset has already been downloaded."""
    data_path = os.path.join(data_dir, "data", "train", "neighbourhood")
    return os.path.isdir(data_path)


def download_dataset(data_dir=None, force=False):
    """
    Download the DDOS dataset from HuggingFace.

    Args:
        data_dir: Where to download (default: data/DDOS/)
        force: Re-download even if already present
    """
    data_dir = get_data_root(data_dir)

    if is_dataset_ready(data_dir) and not force:
        print(f"Dataset already exists at {data_dir}")
        print("Use --force to re-download.")
        return data_dir

    print("=" * 60)
    print("DDOS Dataset Download")
    print("=" * 60)
    print(f"Source:      huggingface.co/datasets/{DATASET_REPO}")
    print(f"Destination: {data_dir}")
    print()
    print("This dataset contains paired RGB + depth images of")
    print("residential neighborhoods. Download size is ~3-5 GB.")
    print("=" * 60)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required. Install with:")
        print("  uv add huggingface_hub")
        sys.exit(1)

    print(f"\nDownloading to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)

    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        local_dir=data_dir,
    )

    if is_dataset_ready(data_dir):
        print(f"\nDataset downloaded successfully to {data_dir}")
    else:
        print(f"\nWarning: Download completed but expected directory structure")
        print(f"not found. Check {data_dir} manually.")

    return data_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download the DDOS dataset")
    parser.add_argument("--data_dir", type=str, default=None,
                        help=f"Download location (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if already present")
    args = parser.parse_args()

    download_dataset(data_dir=args.data_dir, force=args.force)


if __name__ == "__main__":
    main()
