"""
CLI entry points for uv run scripts.

Usage:
    uv run setup                              # Download the DDOS dataset
    uv run train --naive
    uv run train --classic
    uv run train --deeplearning [--epochs 50 --bs 16 ...]
    uv run evaluate --model_path checkpoints/model.pt [...]
    uv run infer --model_path checkpoints/model.pt --image_path img.jpg [...]
"""

import sys


def train():
    # Parse just the mode flag, forward everything else to the appropriate module
    modes = {"--naive", "--classic", "--deeplearning"}
    selected = None

    for arg in sys.argv[1:]:
        if arg in modes:
            selected = arg
            break

    if selected is None:
        print("Usage: uv run train --naive|--classic|--deeplearning [extra args...]")
        print()
        print("Modes:")
        print("  --naive          Train naive baseline (mean depth predictor)")
        print("  --classic        Train classical ML model (random forest)")
        print("  --deeplearning   Train LeJEPA deep learning model")
        sys.exit(1)

    # Remove the mode flag from argv so the downstream argparse doesn't choke
    sys.argv.remove(selected)

    if selected == "--naive":
        from src.training.train_naive import main
        main()
    elif selected == "--classic":
        from src.training.train_classic import main
        main()
    elif selected == "--deeplearning":
        from src.training.train_deeplearning import main
        main()


def setup():
    from src.data.download import main
    main()


def evaluate():
    from src.test.evaluate import main
    main()


def infer():
    from src.test.inference import main
    main()
