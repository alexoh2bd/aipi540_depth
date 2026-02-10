#!/bin/bash
#SBATCH --job-name=test_depth
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1

set -e

mkdir -p logs
mkdir -p test_results

echo "Starting Testing..."
echo "GPU: $SLURM_GPUS"
echo "Node: $(hostname)"

uv run evaluate \
    --model_path checkpoints/deeplearning.pt \
    --save_dir test_results \
    --num_workers 4 \
    --img_size 1024

echo "Testing complete!"
