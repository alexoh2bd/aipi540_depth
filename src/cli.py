"""
CLI entry points for uv run scripts.

Usage:
    uv run setup                              # Download the DDOS dataset
    uv run train --naive
    uv run train --classic
    uv run train --supervised [--epochs 50 --bs 16 ...]
    uv run train --deeplearning [--epochs 50 --bs 16 ...]
    uv run evaluate --model_path checkpoints/model.pt [...]
    uv run infer --model_path checkpoints/model.pt --image_path img.jpg [...]
"""

import sys


def train():
    # Parse just the mode flag, forward everything else to the appropriate module
    modes = {"--naive", "--classic", "--supervised", "--deeplearning"}
    selected = None

    for arg in sys.argv[1:]:
        if arg in modes:
            selected = arg
            break

    if selected is None:
        print("Usage: uv run train --naive|--classic|--supervised|--deeplearning [extra args...]")
        print()
        print("Modes:")
        print("  --naive          Naive baseline (predicts mean depth)")
        print("  --classic        Classical ML baseline (random forest on hand-crafted features)")
        print()
        print("  --supervised     Deep learning: standard single-view depth supervision with")
        print("                   SIGReg embedding regularization. Simpler, faster to train,")
        print("                   and supports both ViT and ResNet backbones. Good starting")
        print("                   point and ablation baseline.")
        print()
        print("  --deeplearning   Deep learning: multi-view LeJEPA self-supervised learning")
        print("                   combined with depth supervision. Uses global + local crops")
        print("                   to learn view-invariant representations. Slower to train but")
        print("                   produces richer features. ViT only.")
        sys.exit(1)

    # Remove the mode flag from argv so the downstream argparse doesn't choke
    sys.argv.remove(selected)

    if selected == "--naive":
        from src.training.train_naive import main
        main()
    elif selected == "--classic":
        from src.training.train_classic import main
        main()
    elif selected == "--supervised":
        from src.training.train_supervised import main
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
