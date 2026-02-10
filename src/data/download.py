# Generated with assistance from Claude (Anthropic) via Claude Code
# https://github.com/anthropics/claude-code
"""
Download the DDOS (Depth from Driving Open Scenes) dataset from HuggingFace.

Downloads to data/DDOS/ by default. The dataset contains paired RGB images
and 16-bit depth maps of residential neighborhoods.

Supports resumable downloads with automatic retry on network failures.

Usage:
    uv run setup
"""

import os
import sys
import time


DATASET_REPO = "benediktkol/DDOS"
DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data", "DDOS")


def _make_total_aware_tqdm(total_bytes, total_files, initial_bytes):
    """Create a tqdm subclass that knows the total download size upfront.

    snapshot_download's progress bar starts with total=0 and incrementally
    discovers file sizes. By providing the pre-computed totals (and crediting
    already-downloaded bytes) we get an accurate overall progress bar from
    the start, even on resume.
    """
    from tqdm.auto import tqdm as _tqdm_base

    class _TotalAwareTqdm(_tqdm_base):
        def __init__(self, *args, **kwargs):
            self._lock_total = False
            # Strip hf_tqdm-specific kwargs that base tqdm doesn't accept
            kwargs.pop("name", None)
            desc = str(kwargs.get("desc", ""))

            if "incomplete total" in desc:
                kwargs["total"] = total_bytes
                kwargs["initial"] = initial_bytes
                kwargs["desc"] = "Downloading"
                super().__init__(*args, **kwargs)
                self._lock_total = True
            elif "Fetching" in desc:
                kwargs["total"] = total_files
                kwargs["desc"] = f"Fetching {total_files} files"
                super().__init__(*args, **kwargs)
            else:
                super().__init__(*args, **kwargs)

        def __setattr__(self, name, value):
            if name == "total" and getattr(self, "_lock_total", False):
                return
            super().__setattr__(name, value)

    return _TotalAwareTqdm

MAX_RETRIES = 5
INITIAL_BACKOFF = 5  # seconds
MAX_BACKOFF = 120  # seconds


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


def download_dataset(data_dir=None, force=False, max_retries=MAX_RETRIES):
    """
    Download the DDOS dataset from HuggingFace.

    Args:
        data_dir: Where to download (default: data/DDOS/)
        force: Re-download even if already present
        max_retries: Maximum number of retry attempts on failure
    """
    data_dir = get_data_root(data_dir)

    print("=" * 60)
    print("DDOS Dataset Download")
    print("=" * 60)
    print(f"Source:      huggingface.co/datasets/{DATASET_REPO}")
    print(f"Destination: {data_dir}")
    print()
    print("This dataset contains paired RGB + depth images of")
    print("residential neighborhoods.")
    print("=" * 60)

    if is_dataset_ready(data_dir) and not force:
        print(f"\nDataset directory already exists at {data_dir}")
        print("Verifying completeness and resuming any incomplete downloads...")

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required. Install with:")
        print("  uv add huggingface_hub")
        sys.exit(1)

    print(f"\nDownloading to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Tip: Set HF_TOKEN environment variable for higher rate limits.")
        print("  Get a token at https://huggingface.co/settings/tokens")
        print()

    # Enable hf_transfer for much faster downloads (Rust-based parallel chunked transfers)
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("Using hf_transfer for accelerated download.")
    except ImportError:
        print("Tip: Install hf_transfer for faster downloads: uv add hf_transfer")

    # Pre-fetch the complete file listing so snapshot_download knows the
    # total size upfront (avoids a progress bar with a moving total).
    print("\nFetching file manifest...")
    api = HfApi()
    all_files = list(api.list_repo_tree(
        DATASET_REPO, repo_type="dataset", recursive=True, token=token,
    ))
    total_size = 0
    file_count = 0
    downloaded_bytes = 0
    for f in all_files:
        if hasattr(f, "size") and f.size:
            total_size += f.size
            file_count += 1
            if os.path.isfile(os.path.join(data_dir, f.rfilename)):
                downloaded_bytes += f.size
    print(f"Found {file_count} files ({total_size / (1024 ** 3):.1f} GB)")
    if downloaded_bytes > 0:
        print(f"Already downloaded: {downloaded_bytes / (1024 ** 3):.1f} GB")

    tqdm_cls = _make_total_aware_tqdm(total_size, file_count, downloaded_bytes)

    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                DATASET_REPO,
                repo_type="dataset",
                local_dir=data_dir,
                token=token,
                max_workers=32,
                tqdm_class=tqdm_cls,
            )
            break
        except KeyboardInterrupt:
            print("\nDownload cancelled by user.")
            print("Run 'uv run setup' again to resume where you left off.")
            sys.exit(1)
        except Exception as e:
            if attempt == max_retries:
                print(f"\nDownload failed after {max_retries} attempts: {e}")
                print("Run 'uv run setup' again to resume where you left off.")
                sys.exit(1)
            backoff = min(INITIAL_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
            print(f"\nDownload interrupted (attempt {attempt}/{max_retries}): {e}")
            print(f"Retrying in {backoff}s... (already-downloaded files will be skipped)")
            time.sleep(backoff)

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
    parser.add_argument("--retries", type=int, default=MAX_RETRIES,
                        help=f"Max retry attempts on failure (default: {MAX_RETRIES})")
    args = parser.parse_args()

    download_dataset(data_dir=args.data_dir, force=args.force, max_retries=args.retries)


if __name__ == "__main__":
    main()
